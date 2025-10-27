"""
Data Writing and Visualization Utilities

This module provides comprehensive functionality for writing experimental results,
creating visualizations, and managing output data for the anonymization methods
comparison framework.

Key features:
- Cluster preprocessing and visualization
- Results plotting with multiple chart types (Plotly and matplotlib)
- JSON data storage and management
- Temporary data persistence
- Dataset size estimation and storage

Functions:
    preprocess_clusters: Prepare cluster data for visualization
    plot_results: Create comprehensive visualizations of clustering results
    process_data: Process experimental data for plotting
    create_plots: Generate comparison plots from results (supports Plotly and matplotlib)
    store_general_metrics: Save experimental metrics to JSON
    create_json_file: Initialize experimental configuration files
    write_to_json: Append results to JSON files
    write_to_pkl_tmp: Save temporary data to pickle files
    store_generated_data: Cache generated datasets

Note:
    The module defaults to using Plotly for interactive visualizations with automatic
    fallback to matplotlib if Plotly is not available.
"""

import os
import json
import pickle
from datetime import datetime
from ucimlrepo import fetch_ucirepo

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm.auto import tqdm 
from sklearn.manifold import TSNE
import logging
logger = logging.getLogger(__name__)

# Plotly imports with fallback to matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Use module logger for warnings
    logger.warning("Plotly not available. Falling back to matplotlib.")


os.environ["LOKY_MAX_CPU_COUNT"] = str(max(os.cpu_count() - 2, 1))
sns.set_palette("colorblind")

def _extract_risk_value(risk_dict, name="risk"):
    if isinstance(risk_dict, dict):
        # Prefer 'risk' key if present
        if name in risk_dict:
            return risk_dict[name]
        # Otherwise, try first float/int value
        for v in risk_dict.values():
            if isinstance(v, (float, int)):
                return v
    elif isinstance(risk_dict, (float, int)):
        return risk_dict
    return np.nan


def preprocess_clusters(clusters, instance, pbar):
    """
    Preprocess cluster data for visualization.
    
    Prepares cluster data by extracting relevant features, applying one-hot
    encoding to categorical variables, and normalizing numeric features.
    
    Parameters:
        clusters (list): List of cluster data arrays
        instance: Algorithm instance containing QI specification
        pbar: Progress bar for status updates
        
    Returns:
        numpy.ndarray: Preprocessed data ready for visualization
    """
    pbar.set_postfix_str("Preprocessing clusters")
    num_idx = instance.QI.get_numerical_columns()
    cat_idx = instance.QI.get_categorical_columns()
    selected_idx = sorted(set(np.concatenate((num_idx, cat_idx)).tolist()))

    # Extract only numerical and categorical columns
    data = np.vstack(clusters)[:, selected_idx]
    col_names = [f"col_{i}" for i in selected_idx]
    df = pd.DataFrame(data, columns=col_names)

    # Determine column names by type
    num_cols = [f"col_{i}" for i in num_idx if i in selected_idx]
    cat_cols = [f"col_{i}" for i in cat_idx if i in selected_idx]

    # One-hot encoding (converts to float)
    if cat_cols:
        pbar.set_postfix_str("Preprocessing clusters: One-hot encoding") 
        df = pd.get_dummies(df, columns=cat_cols, dtype=float)

    # Min-Max normalization of numeric columns
    if num_cols:
        pbar.set_postfix_str("Preprocessing clusters: Min-Max normalization")
        df[num_cols] = df[num_cols].apply(
            lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0.0
        )

    # Convert to NumPy array
    processed = df.to_numpy(dtype=float)

    # Reconstruir clusters
    pbar.set_postfix_str("Preprocessing clusters: Reconstructing clusters")
    sizes = np.fromiter((len(c) for c in clusters), dtype=int)
    return np.split(processed, np.cumsum(sizes)[:-1])



def plot_results(clusters, instance, centroids=None, images_dir=".", file_name="plot", method="Clustering", figsize=(20, 18), big_dim_type="t-SNE"):
    """
    Create comprehensive visualizations of clustering results.
    
    Generates multiple visualization plots including scatter plots, histograms,
    and dimensionality reduction plots for high-dimensional data.
    
    Parameters:
        clusters (list): List of cluster data arrays
        instance: Algorithm instance containing QI specification
        centroids (numpy.ndarray, optional): Cluster centroids
        images_dir (str): Directory to save images
        file_name (str): Base name for output files
        method (str): Method name for plot titles
        figsize (tuple): Figure size for plots
        big_dim_type (str): Dimensionality reduction method ("t-SNE")
        
    Returns:
        None: Saves plots to specified directory
    """
    n_clusters = len(clusters)
    pbar = tqdm(total= 3 + n_clusters, desc="Creating plots", leave=False)
    init_time = datetime.now()
    
    plt.ioff()
    if not clusters or not all(len(cluster) > 0 for cluster in clusters):
        raise ValueError("Clusters must be a non-empty list of non-empty arrays.")
    
    pbar.set_postfix_str("Converting data to NumPy arrays")
    clusters = [np.asarray(cluster) for cluster in clusters]
    clusters = preprocess_clusters(clusters, instance, pbar)
    original_dims = clusters[0].shape[1]
    pbar.update(1)
    
    if original_dims > 3 and big_dim_type == "pairplot":
        pbar.set_postfix_str("Creating pairplot by features")
        data = []
        for idx, sub_cluster in enumerate(clusters):
            for point in sub_cluster:
                row = {f"Feature {i + 1}": point[i] for i in range(original_dims)}
                row["Cluster"] = idx + 1
                data.append(row)
        df = pd.DataFrame(data)
        fig = sns.pairplot(df, hue="Cluster", palette=sns.color_palette(n_colors=len(clusters))).figure
        pbar.update(n_clusters)
        fig.suptitle(f"{method} ({len(clusters)} clusters)", fontsize=10, fontweight='bold', y=1.02)
    elif big_dim_type != "t-SNE":
        raise ValueError(f"Invalid big_dim_type: {big_dim_type}. Use 't-SNE' or 'pairplot'.")
    else:
        if original_dims > 2:
            pbar.set_postfix_str("Reducing dimensions with t-SNE")
            cluster_sizes = [len(c) for c in clusters]
            total = sum(cluster_sizes)

            centroid_points = np.asarray(centroids) if centroids is not None else None
            all_points = np.vstack(clusters + [centroid_points] if centroid_points is not None else clusters)

            reduced = TSNE(n_components=2, perplexity=35).fit_transform(all_points)
            clusters = np.split(reduced[:total], np.cumsum(cluster_sizes)[:-1])
            centroids = reduced[total:] if centroid_points is not None else None  # dims = 3
            # fig = plt.figure(figsize=figsize)
            # ax = fig.add_subplot(111, projection='3d')

        dims = 2
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        pbar.set_postfix_str("Plotting clusters")
        logger.debug(f"Plotting {len(clusters)} clusters")
        # Log a sample point from the first cluster for debugging
        try:
            logger.debug(f"Sample point (first cluster): {clusters[0][0]}")
        except Exception:
            logger.debug("No sample point available for clusters")
        for i, cluster in enumerate(clusters):
            ax.scatter(*cluster.T[:dims], marker='.', s=20)
            if centroids is not None:
                ax.text(*centroids[i][:dims], str(i + 1), fontsize=8, fontweight='bold', ha='center', va='center')
            pbar.update(1)
        ax.set_title(f"{method} ({n_clusters} clusters)", fontsize=10, fontweight='bold')
    pbar.update(1)

    pbar.set_postfix_str("Saving plot")
    fig.savefig(os.path.join(images_dir, f"{file_name}.png"), bbox_inches='tight')
    plt.close(fig)

    end_time = datetime.now()
    pbar.set_postfix_str(f"Time: {end_time - init_time} [ST: {init_time.strftime('%Y-%m-%d %H:%M:%S')}, ET: {end_time.strftime('%Y-%m-%d %H:%M:%S')}]")
    pbar.update(1)
    pbar.close()

def process_data(data, skip_models, allowed_models=None):
    """
    Process experimental data for plotting.
    
    Processes raw experimental results by calculating statistics,
    filtering out specified models, and preparing data for visualization.
    
    Parameters:
        data (dict): Raw experimental data from JSON results
        skip_models (list): List of model names to exclude from plots
        allowed_models (list or dict): If list, models to include. If dict, maps original names (keys) to display names (values)
        
    Returns:
        tuple: (processed_data, model_names) ready for plotting
    """
    # Determine model name mapping
    model_name_map = {}
    if isinstance(allowed_models, dict):
        model_name_map = allowed_models
        allowed_keys = set(allowed_models.keys())
    elif isinstance(allowed_models, list):
        allowed_keys = set(allowed_models)
    else:
        allowed_keys = None
    
    records = []
    estimated_records = []
    k_values_set = set()
    
    for dataset_size_str, experiments in data.items():
        dataset_size = int(dataset_size_str)
        for k_value_str, results in experiments.items():
            k_value = int(k_value_str)
            k_values_set.add(k_value)
            for result in results:
                original_model_name = result["Model"]
                
                # Skip if in skip_models
                if original_model_name in skip_models:
                    continue
                
                # Skip if allowed_models is specified and model not in it
                if allowed_keys is not None and original_model_name not in allowed_keys:
                    continue
                
                # Use mapped name if available, otherwise use original
                display_model_name = model_name_map.get(original_model_name, original_model_name)
                
                times = result.get("Elapsed_Times", [])
                avg_time = np.nan  # Default to NaN
                is_estimated = False
                
                if times:
                    avg_time = np.mean(times)
                else:
                    info = result.get("Information", "")
                    if "Estimated time to would be done it" in info:
                        try:
                            estimated_time = float(info.split(":")[-1].split()[0])
                            # Convert to seconds (adding 1000 as an offset)
                            avg_time = 1000 + (estimated_time * 3600)
                        except ValueError:
                            estimated_time = None
                        is_estimated = (estimated_time is not None)
                
                record = {
                    "Dataset_Size": dataset_size,
                    "K-Value": k_value,
                    "Model": display_model_name,
                    "Time": avg_time,
                    "NCP": result.get("NCP", np.nan),

                }
                if "Mean_Distance" in result:
                    record["Mean_Distance"] = result.get("Mean_Distance", np.nan)
                records.append(record)
                if is_estimated:
                    estimated_records.append(record)
    
    k_values = np.array(sorted(k_values_set))
    return pd.DataFrame(records), pd.DataFrame(estimated_records), k_values


def create_plots(file_path, images_dir, file_name, log_scale=False, skip_models=[], allowed_models=None, use_plotly=False, print_up_lines=False):
    """
    Generate comparison plots from experimental results.
    
    Creates visualization plots comparing different anonymization methods
    across various metrics like execution time and quality measures.
    
    Parameters:
        file_path (str): Path to results JSON file
        images_dir (str): Directory to save generated plots
        file_name (str): Base name for output files
        log_scale (bool): Whether to use logarithmic scale for time plots
        skip_models (list): List of model names to exclude from plots
        allowed_models (list or dict): If list, models to include. If dict, maps original names (keys) to display names (values)
        use_plotly (bool): Whether to use Plotly (True) or matplotlib (False)
        
    Returns:
        None: Saves plots to specified directory
    """
    # Check if Plotly is available and user wants to use it
    if use_plotly and not PLOTLY_AVAILABLE:
        logging.getLogger(__name__).info("Plotly not available, falling back to matplotlib")
        use_plotly = False
    
    with open(file_path, 'r') as f:
        data = json.load(f)[1]

    df_real, df_estimated, k_values = process_data(data, skip_models, allowed_models)
    dataset_sizes = sorted(df_real["Dataset_Size"].unique())

    use_mean_distance = "Mean_Distance" in df_real.columns

    metrics = [
        ("Time", "Time (seconds)", None), #"Execution Time"),
        ("NCP", "NCP (Normalized Certainty Penalty)", None), #, "Normalized Certainty Penalty")
    ]
    x_symbol = "K"
    if use_mean_distance:
        metrics.insert(1, ("Mean_Distance", "Mean Distance", "Mean Distance to Centroids"))

    for dataset_size in dataset_sizes:
        subset_real = df_real[df_real["Dataset_Size"] == dataset_size]
        subset_estimated = (df_estimated[df_estimated["Dataset_Size"] == dataset_size]
                            if not df_estimated.empty else pd.DataFrame())

        # Create one plot per metric (separate files)
        for metric_tuple in metrics:
            metric_list = [metric_tuple]
            metric_name = metric_tuple[0]
            metric_file_name = f"{file_name}_{metric_name}"

            if use_plotly:
                _create_plotly_plots(subset_real, subset_estimated, k_values, metric_list,
                                   dataset_size, images_dir, metric_file_name, log_scale, x_symbol, print_up_lines)
            else:
                _create_matplotlib_plots(subset_real, subset_estimated, k_values, metric_list,
                                         dataset_size, images_dir, metric_file_name, log_scale, x_symbol, print_up_lines)


def _create_plotly_plots(subset_real, subset_estimated, k_values, metrics, 
                        dataset_size, images_dir, file_name, log_scale, x_symbol, print_up_lines=True):
    """Create plots using Plotly."""
    # Calculate appropriate spacing and size
    num_plots = len(metrics)
    spacing = 0.1
    
    fig = make_subplots(
        rows=1, cols=num_plots,
        subplot_titles=[title for _, _, title in metrics],
        horizontal_spacing=spacing
    )
    
    # Use a qualitative palette; fall back to Plotly default if unavailable
    colors = px.colors.qualitative.Set1 if hasattr(px.colors.qualitative, 'Set1') else px.colors.qualitative.Plotly
    # Define a set of linestyles to reuse when colors are exhausted
    # Plotly supports dash styles: 'solid', 'dash', 'dot', 'dashdot'
    dash_styles = ['solid', 'dash', 'dot', 'dashdot']
    # Define a rich set of marker symbols for colorblind-friendly distinction
    marker_symbols = [
        'circle', 'square', 'diamond', 'cross', 'x',
        'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right',
        'star', 'hexagon', 'pentagon', 'hourglass', 'bowtie',
        'circle-open', 'square-open', 'diamond-open', 'triangle-up-open', 'triangle-down-open',
        'triangle-left-open', 'triangle-right-open', 'star-open'
    ]
    
    for col_idx, (metric, y_label, title) in enumerate(metrics, 1):
        unique_models = subset_real["Model"].unique()
        num_models = len(unique_models)

        if len(k_values) > 2 and num_models > 0:
            line_space = np.linspace(1, len(k_values) - 2, num_models, dtype=int)
            sampled_k = k_values[line_space]
        else:
            sampled_k = k_values

        for model_idx, (model, annot_k) in enumerate(zip(unique_models, sampled_k)):
            model_data = subset_real[subset_real["Model"] == model]
            # Cycle colors first; when colors are exhausted reuse colors with different dash styles
            style_round = model_idx // len(colors)
            color = colors[ model_idx % len(colors)]
            dash = dash_styles[style_round % len(dash_styles)]
            symbol = marker_symbols[model_idx % len(marker_symbols)]

            # Add line plot with dash style
            fig.add_trace(
                go.Scatter(
                    x=model_data["K-Value"],
                    y=model_data[metric],
                    mode='lines+markers',
                    name=model if col_idx == 1 else None,  # Show legend only for first subplot
                    line=dict(color=color, width=1.5, dash=dash),
                    marker=dict(color=color, size=8, symbol=symbol, line=dict(color=color, width=1)),
                    showlegend=(col_idx == 1),
                    legendgroup=model
                ),
                row=1, col=col_idx
            )
            
            
            # Add annotation with better positioning
            if print_up_lines and not model_data.empty and len(model_data) > 0:
                model_data_sorted = model_data.sort_values("K-Value")
                if len(model_data_sorted) > 0:
                    idx = (model_data_sorted["K-Value"] - annot_k).abs().idxmin()
                    last_x = model_data_sorted.loc[idx, "K-Value"]
                    last_y = model_data_sorted.loc[idx, metric]
                    
                    # Only add annotation if the values are valid
                    if not (np.isnan(last_x) or np.isnan(last_y)):
                        fig.add_annotation(
                            x=last_x, y=last_y,
                            text=model,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor=color,
                            font=dict(size=10),
                            row=1, col=col_idx
                        )

        # Add estimated points
        if print_up_lines and not subset_estimated.empty:
            for model in subset_estimated["Model"].unique():
                model_data = subset_estimated[subset_estimated["Model"] == model]
                if metric not in model_data.columns:
                    continue
                valid_model_data = model_data.dropna(subset=[metric])
                if not valid_model_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_model_data["K-Value"],
                            y=valid_model_data[metric],
                            mode='markers',
                            name="Point Estimated" if col_idx == 1 else None,
                            marker=dict(color='black', size=10, symbol='x'),
                            showlegend=(col_idx == 1),
                            legendgroup="estimated"
                        ),
                        row=1, col=col_idx
                    )

        # Update axes
        fig.update_xaxes(title_text=x_symbol, tickfont=dict(size=24), row=1, col=col_idx, showgrid=True)
        fig.update_yaxes(title_text=y_label, row=1, col=col_idx, showgrid=True)
        
        if log_scale and metric == "Time":
            fig.update_yaxes(type="log", row=1, col=col_idx)

    # Calculate appropriate width and height
    # Allow overriding via environment variables for very wide outputs

    PLOTLY_PER_PLOT_WIDTH = int(os.environ.get('PLOTLY_PER_PLOT_WIDTH', 1100))
    PLOTLY_MAX_WIDTH = int(os.environ.get('PLOTLY_MAX_WIDTH', 3000))
    PLOTLY_HEIGHT = int(os.environ.get('PLOTLY_HEIGHT', 700))
    PLOTLY_MAX_HEIGHT = int(os.environ.get('PLOTLY_MAX_HEIGHT', 2000))

    plot_width = min(PLOTLY_PER_PLOT_WIDTH * num_plots, PLOTLY_MAX_WIDTH)
    plot_height = min(PLOTLY_HEIGHT * num_plots, PLOTLY_MAX_HEIGHT)

    # Update layout with better sizing
    fig.update_layout(
        # title=dict(
        #     text=f"Dataset Size: {dataset_size}",
        #     x=0.5,
        #     xanchor='center'
        # ),
        height=plot_height,
        width=plot_width,
        showlegend=True,
        legend=dict(
            orientation="v",
            # yanchor="top",
            y=1,
            # xanchor="left",
            x=1
        ),
        margin=dict(l=80, r=120, t=80, b=80)
    )
    
    # Save interactive HTML (dynamic) and attempt PNG export (static)
    try:
        html_path_dir = os.path.join(images_dir, "dynamic_images")
        os.makedirs(html_path_dir, exist_ok=True)
        html_path = os.path.join(html_path_dir, f"{file_name}_x_{dataset_size}.html")
        fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    except Exception as e:
        logger.warning(f"Failed to save interactive Plotly HTML: {e}")

    # Attempt to save PNG (requires kaleido or orca). If it fails, we keep the HTML.
    png_path = os.path.join(images_dir, f"{file_name}_x_{dataset_size}.png")
    try:
        fig.write_image(
            png_path,
            format="png",
            width=plot_width,
            height=plot_height,
            scale=int(os.environ.get('PLOTLY_IMAGE_SCALE', 1))
        )
    except Exception as e:
        logger.warning(f"Failed to save Plotly PNG: {e}")
        logger.info(f"Interactive HTML saved at: {html_path}")


def _create_matplotlib_plots(subset_real, subset_estimated, k_values, metrics, 
                           dataset_size, images_dir, file_name, log_scale, x_symbol, print_up_lines=True):
    """Create plots using matplotlib (original implementation)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(12 * len(metrics), 10))

    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, y_label, title) in zip(axes, metrics):
        unique_models = subset_real["Model"].unique()
        num_models = len(unique_models)

        if len(k_values) > 2 and num_models > 0:
            line_space = np.linspace(1, len(k_values) - 2, num_models, dtype=int)
            sampled_k = k_values[line_space]
        else:
            sampled_k = k_values

        # Prepare matplotlib color cycle and linestyles.
        prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
        base_colors = None
        if prop_cycle is not None:
            try:
                base_colors = prop_cycle.by_key().get('color', None)
            except Exception:
                base_colors = None
        if not base_colors:
            # Fallback to seaborn colorblind palette
            base_colors = sns.color_palette('colorblind', n_colors=max(4, len(unique_models)))
        linestyles = ['-', '--', ':', '-.']
        # Distinct marker shapes for accessibility (repeats if many models)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', 'd', 'p', '1', '2', '3', '4', '8']

        for model_idx, (model, annot_k) in enumerate(zip(unique_models, sampled_k)):
            model_data = subset_real[subset_real["Model"] == model]
            style_round = model_idx // len(base_colors)
            color = base_colors[model_idx % len(base_colors)]
            linestyle = linestyles[style_round % len(linestyles)]
            marker = markers[model_idx % len(markers)]

            ax.plot(
                model_data["K-Value"],
                model_data[metric],
                label=model,
                linestyle=linestyle,
                color=color,
                lw=1.5,
                zorder=1,
                marker=marker,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.3,
                markeredgecolor=color
            )
            if print_up_lines and not model_data.empty:
                model_data_sorted = model_data.sort_values("K-Value")
                idx = (model_data_sorted["K-Value"] - annot_k).abs().idxmin()
                last_x = model_data_sorted.loc[idx, "K-Value"]
                last_y = model_data_sorted.loc[idx, metric]
                ax.annotate(model, xy=(last_x, last_y), xytext=(0,15),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=7, arrowprops=dict(arrowstyle='->'))

        if print_up_lines and not subset_estimated.empty:
            printed_estimated_label = False
            for model in subset_estimated["Model"].unique():
                model_data = subset_estimated[subset_estimated["Model"] == model]
                if metric not in model_data.columns:
                    continue
                valid_model_data = model_data.dropna(subset=[metric])
                if not valid_model_data.empty:
                    ax.scatter(
                        valid_model_data["K-Value"],
                        valid_model_data[metric],
                        color='black',
                        label="Point Estimated" if not printed_estimated_label else None,
                        marker='X', s=60, zorder=2
                    )
                    printed_estimated_label = True

        ax.set_title(title)
        ax.set_xlabel(x_symbol, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.legend(fontsize=16)
        ax.grid(True)
        
        ax.tick_params(axis='both', labelsize=15)
        ax.ticklabel_format(style="plain", axis="both")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"{file_name}_x_{dataset_size}.png"), bbox_inches='tight')
    plt.close()


def plot_mdav_ld_mixed_phase_times_mean(file_path, images_dir, file_name="mdav_ld_mixed_phase_times", skip_models=[], allowed_models=None):
    """
    Create a stacked-bar chart of phase times for MDAV_LD_Mixed models.

    This utility scans the results JSON for model entries whose name contains
    the substring provided in `contains`, extracts phase timings from
    `Aditional_Information` (or the correctly-spelled `Additional_Information`),
    and produces a stacked bar chart saved to the images directory.

    Parameters
    ----------
    file_path : str
        Path to the results JSON file (the same format used by create_plots).
    images_dir : str
        Directory where the resulting PNG will be saved.
    file_name : str
        Base name for the saved PNG file (no extension).
    skip_models : list
        List of model names to exclude from the plot.
    allowed_models : list or dict
        If list, models to include. If dict, maps original names (keys) to display names (values).

    Returns
    -------
    dict
        Dictionary with per-phase min/max statistics.
    """
    # Determine model name mapping
    model_name_map = {}
    if isinstance(allowed_models, dict):
        model_name_map = allowed_models
        allowed_keys = set(allowed_models.keys())
    elif isinstance(allowed_models, list):
        allowed_keys = set(allowed_models)
    else:
        allowed_keys = None
    
    # Read JSON (some result files embed the mapping at index 1)
    with open(file_path, 'r') as f:
        raw = json.load(f)
    data = raw[1] if isinstance(raw, list) and len(raw) > 1 else raw

    # walk through structure to find entries with 'Model'
    def _walk(obj):
        if isinstance(obj, dict):
            if 'Model' in obj and ("Aditional_Information" in obj or "Additional_Information" in obj):
                yield obj
            for v in obj.values():
                yield from _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from _walk(it)

    PHASE_KEYS = ["Preprocess", "Training", "Reduction", "MDAV", "Generalization"]

    def _extract(entry):
        info = entry.get("Aditional_Information") or entry.get("Additional_Information")
        if not isinstance(info, dict):
            return None
        phases = {}
        for k in PHASE_KEYS:
            v = info.get(k)
            if isinstance(v, (int, float)):
                phases[k] = float(v)
        return phases if phases else None

    # Collect all entries for MDAV_LD_Mixed models (may appear multiple times for different k/repeats)
    models_raw = {}
    for entry in _walk(data):
        original_name = entry.get('Model')
        if not original_name:
            continue
        if skip_models and original_name in skip_models:
            continue
        
        # Skip if allowed_models is specified and model not in it
        if allowed_keys is not None and original_name not in allowed_keys:
            continue
        
        if 'MDAV_LD_Mixed' not in original_name:
            continue
        phases = _extract(entry)
        if not phases:
            continue
        
        # Use mapped name if available, otherwise use original
        display_name = model_name_map.get(original_name, original_name)
        
        # accumulate per-model lists of phase dicts
        models_raw.setdefault(display_name, []).append(phases)

    if not models_raw:
        logger.warning(f"No MDAV_LD_Mixed models with phase timings found in {file_path}")
        return None

    # Compute average per-phase across all recorded entries (k, repeats) for each model
    aggregated = []  # list of (model_name, {phase: mean_val})
    for model_name, phase_list in models_raw.items():
        mean_phases = {}
        for pk in PHASE_KEYS:
            vals = [p.get(pk, 0.0) for p in phase_list if isinstance(p.get(pk, None), (int, float))]
            mean_phases[pk] = float(np.mean(vals)) if vals else 0.0
        aggregated.append((model_name, mean_phases))

    # Sort methods by total mean time (descending)
    aggregated.sort(key=lambda it: sum(it[1].get(pk, 0.0) for pk in PHASE_KEYS), reverse=True)

    # Use all found models (sorted); if many, limit to a reasonable number
    max_display = 200  # keep high default; notebook/user can still use skip_models
    selected = aggregated[:max_display]
    model_names = [m for m, _ in selected]

    # prepare phase matrix (aligned to PHASE_KEYS)
    phase_values = {k: [] for k in PHASE_KEYS}
    for _, ph in selected:
        for k in PHASE_KEYS:
            phase_values[k].append(ph.get(k, 0.0))

    # Plot horizontal stacked bars: methods on Y, phases along X
    n_methods = len(selected)
    fig_height = max(6, n_methods * 0.35)
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    lefts = [0.0] * n_methods
    colors = sns.color_palette('colorblind', n_colors=len(PHASE_KEYS))
    for idx, k in enumerate(PHASE_KEYS):
        vals = phase_values[k]
        ax.barh(list(range(n_methods)), vals, left=lefts, label=k, color=colors[idx % len(colors)])
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(list(range(n_methods)))
    ax.set_yticklabels(model_names, fontsize=14)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    # No title as requested

    # Place legend below the plot, centered
    ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.13), ncol=min(5, len(PHASE_KEYS)), fontsize=14)

    ax.grid(axis='x', linestyle=':', alpha=0.4)

    # Adjust left margin dynamically to avoid clipping long labels
    try:
        longest = max(len(s) for s in model_names)
        left_margin = min(0.08 + longest * 0.006, 0.25)
    except Exception:
        left_margin = 0.08
    # Apply subplots adjustments
    fig.subplots_adjust(left=left_margin, bottom=0.18)

    # Shrink only the plotting area (axes) horizontally so the graphic part is slightly narrower
    try:
        pos = ax.get_position()  # Bbox in figure coordinates [x0, y0, width, height]
        # Reduce axes width to 85% to make the graphic area slightly narrower, but keep a sensible minimum
        # new_width = max(0.4, pos.width * 0.9)
        ax.set_position([pos.x0, pos.y0,  pos.width * 0.7, pos.height])
    except Exception:
        # If anything fails, continue without changing axes position
        pass

    os.makedirs(images_dir, exist_ok=True)
    # Save with tight bbox to avoid clipping labels/legend
    fig.savefig(os.path.join(images_dir, f"{file_name}.png"), bbox_inches='tight')
    plt.close(fig)



###############################
### Storing general metrics ###
###############################
# Convert NumPy types to native Python types
def _convert_to_native(obj):
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def store_general_metrics(file_path, methods=None):
    """
    Store and analyze general metrics from experimental results.
    
    Processes experimental data from a JSON file, calculating average metrics
    for different anonymization methods and dataset sizes. Generates summary
    statistics and prints formatted results.
    
    Parameters:
        file_path (str): Path to the JSON file containing experimental results
        methods (list, optional): List of method names to analyze. If None,
                                 automatically detects all methods in the file.
                                 
    Returns:
        None: Prints results and updates the JSON file with summary statistics
        
    Note:
        When methods=None, the function will scan the entire dataset to find
        all unique method names and process them automatically. This is useful
        when you want to analyze all available methods without manually 
        specifying them.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)[1]

    # If methods is None, automatically detect all methods in the file
    if methods is None:
        methods = set()
        for dataset_size, experiments in data.items():
            for experiment in experiments.values():
                for result in experiment:
                    model = result.get("Model")
                    if model:
                        methods.add(model)
        methods = sorted(list(methods))  # Convert to sorted list for consistent ordering
        logger.info(f"Auto-detected methods: {methods}")

        # Check if any methods were found
        if not methods:
            logger.warning("No methods found in the data file.")
            return

    # Detectar si Mean_Distance y RL_sobo existen en algún resultado
    try:
        sample_result = next(iter(next(iter(data.values())).values()))[0]
        use_mean_distance = "Mean_Distance" in sample_result
    except (StopIteration, IndexError, KeyError):
        logger.warning("No valid data found in the file.")
        return


    aggregated_data = { 
        "Time": {method: [] for method in methods},
        "NCP": {method: [] for method in methods}
    }
    if use_mean_distance:
        aggregated_data["Mean_Distance"] = {method: [] for method in methods}

    aggregated_by_size = {}
    ignored = 0 
    
    for dataset_size, experiments in data.items():
        if dataset_size not in aggregated_by_size:
            aggregated_by_size[dataset_size] = { 
                method: {"Time": [], "NCP": []} for method in methods
            }
            if use_mean_distance:
                for method in methods:
                    aggregated_by_size[dataset_size][method]["Mean_Distance"] = []

        for k, sub_experiments in experiments.items():
            for result in sub_experiments:
                if result.get("Number_of_Clusters", 0) < 1:
                    ignored += 1
                    continue
                model = result.get("Model")
                if model not in methods:
                    continue
                elapsed_times = result.get("Elapsed_Times", None)
                if elapsed_times is None:
                    raise ValueError(f"Elapsed_Times missing for model {model} in dataset size {dataset_size} with k={k}") 
                time_val = np.mean(elapsed_times)

                ncp = result.get("NCP", None)
                if ncp is None:
                    raise ValueError(f"NCP missing for model {model} in dataset size {dataset_size} with k={k}")

                aggregated_data["Time"][model].append(time_val)
                aggregated_data["NCP"][model].append(ncp)

                aggregated_by_size[dataset_size][model]["Time"].append(time_val)
                aggregated_by_size[dataset_size][model]["NCP"].append(ncp)

                if use_mean_distance:
                    mean_distance = result.get("Mean_Distance", np.nan)
                    aggregated_data["Mean_Distance"][model].append(mean_distance)
                    aggregated_by_size[dataset_size][model]["Mean_Distance"].append(mean_distance)
        
    overall_results = {
        method: {
            "NCP": np.mean(aggregated_data["NCP"][method]) if aggregated_data["NCP"][method] else np.nan,
            "Time": np.mean(aggregated_data["Time"][method]) if aggregated_data["Time"][method] else np.nan,
            "Ignored Metrics": ignored
        }
        for method in methods
    }
    if use_mean_distance:
        for method in methods:
            overall_results[method]["Distance to Centroids"] = (
                np.mean(aggregated_data["Mean_Distance"][method])
                if aggregated_data["Mean_Distance"][method] else np.nan
            )
    # Record-linkage (SoBo) metrics removed

    by_dataset_size = {}
    for size, methods_data in aggregated_by_size.items():
        by_dataset_size[size] = {}
        for method in methods:
            times = methods_data[method]["Time"]
            ncp_vals = methods_data[method]["NCP"]
            method_result = {
                "NCP": np.mean(ncp_vals) if ncp_vals else np.nan,
                "Time": np.mean(times) if times else np.nan,
            }
            if use_mean_distance:
                md_vals = methods_data[method]["Mean_Distance"]
                method_result["Distance to Centroids"] = np.mean(md_vals) if md_vals else np.nan
            by_dataset_size[size][method] = method_result

    results_dict = {
        "Average_overall": overall_results,
        "Average_by_dataset_size": by_dataset_size
    }
    write_to_json(file_path, results_dict)

    return results_dict

##########################
###  Storing functions ###
##########################

def get_column_types(df: pd.DataFrame) -> list:
    """
    Return a list indicating for each column in df:
      - 'numerical' if dtype is numeric,
      - 'categorical' if dtype is object or category,
      - None otherwise.
    """
    types = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            types.append('numerical')
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == object:
            types.append('categorical')
        else:
            types.append(None)
    return types

def create_json_file(file_path, repetitions, n_iterations, k_iterations, k_list, 
                     csv_path=None, n_features=None, n_instances=None, ucimlrepo_id=None, 
                     QI_names=None, column_names=None, SA_names=None, generalization_technique=None, verbose=False):
    """
    Create or validate experimental configuration JSON file.
    
    Loads data from UCI ML repository or CSV file, extracts metadata,
    and creates a JSON configuration file for the experiment.
    
    Parameters:
        file_path (str): Path to save the JSON configuration
        repetitions (int): Number of repetitions per experiment
        n_iterations (int): Number of dataset size iterations
        k_iterations (int): Number of k-value iterations
        k_list (list): Range of k values [min, max]
        csv_path (str, optional): Path to CSV data file
        n_features (int, optional): Number of features (required if no data source)
        n_instances (int, optional): Number of instances (required if no data source)
        ucimlrepo_id (int, optional): UCI ML repository dataset ID
        QI_names (list, optional): Names of quasi-identifier columns
        column_names (list, optional): Names of all columns
        SA_names (list, optional): Names of sensitive attribute columns
        generalization_technique (list, optional): Generalization technique per column
        
    Returns:
        tuple: (n_features, repetitions, n_iterations, k_iterations, n_list, k_list,
                ucimlrepo_id, column_names, QI_types, SA_types, generalization_technique)
    """
    
    # Skip if file already exists
    if os.path.exists(file_path):
        return n_features, repetitions, n_iterations, k_iterations, None, k_list, ucimlrepo_id, column_names, None, None, generalization_technique
    
    # ============================================================================
    # Load data from different sources
    # ============================================================================
    
    if ucimlrepo_id is not None:
        # Load from UCI ML Repository
        data = fetch_ucirepo(id=ucimlrepo_id).data
        n_instances, n_features = data.features.shape
        
        # Validate or extract column names
        column_names_tmp = list(data.features.columns)
        if column_names is not None and column_names != column_names_tmp:
            raise ValueError(f"Column names mismatch. Expected: {column_names_tmp}, Got: {column_names}")
        column_names = column_names_tmp
        
        # Extract column types
        types_uciml = get_column_types(data.features)
        QI_types, SA_types = _process_attribute_types(
            types_uciml, column_names, QI_names, SA_names
        )
        
        # Align generalization technique with QI types
        if generalization_technique is not None:
            generalization_technique = [
                generalization_technique[i] if QI_types[i] is not None else None 
                for i in range(n_features)
            ]
    
    elif csv_path is not None:
        # Load from CSV file
        data = pd.read_csv(csv_path)
        n_instances, n_features = data.shape
        
        if column_names is None:
            column_names = list(data.columns)
        
        types = get_column_types(data)
        QI_types, SA_types = _process_attribute_types(
            types, column_names, QI_names, SA_names
        )
    
    else:
        # Manual configuration (no data source)
        _validate_manual_parameters(n_features, column_names, QI_names)
        
        # TODO: Create QI_types and SA_types based on input data or additional parameters
        QI_types = [None] * n_features if n_features else []
        SA_types = []
    
    # ============================================================================
    # Validate and prepare configuration
    # ============================================================================
    
    if n_instances is None:
        raise ValueError("n_instances must be provided to compute N_Metrics")
    
    if n_features is None:
        raise ValueError("n_features must be determined from data source or provided explicitly")
    
    n_instances_int = int(n_instances)
    
    # Set default generalization technique if not provided
    if generalization_technique is None:
        generalization_technique = [None] * n_features
    
    # Build compact representations for JSON
    QI_triplet = _build_attribute_triplet(QI_types, generalization_technique)
    SA_duplet = _build_attribute_duplet(SA_types)
    
    # Calculate instance range for iterations
    n_list = (
        [int(n_instances_int * 0.1), n_instances_int] 
        if n_iterations > 1 
        else [n_instances_int]
    )
    
    # ============================================================================
    # Create JSON configuration
    # ============================================================================
    
    config = {
        "N_Features": n_features,
        "Repetitions": repetitions,
        "N_Metrics": [n_iterations, n_list],
        "K_Metrics": [k_iterations, k_list],
        "ucimlrepo_id": ucimlrepo_id,
        "columns": column_names,
        "QI": QI_triplet,
        "sensitive_attributes_names": SA_duplet,
    }
    
    # Save configuration
    with open(file_path, 'w') as f:
        json.dump([config, {}, None], f, indent=2)
    if verbose:
        print(f"{'='*80}")
        print(f"Configuration file created: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        print(f"Features         : {n_features}")
        print(f"Instances        : {n_instances_int}")
        print(f"Repetitions      : {repetitions}")
        print(f"Iterations       : N={n_iterations}, K={k_iterations}")
        print(f"QI attributes    : {len([qi for qi in QI_types if qi is not None])}/{n_features}")
        print(f"SA attributes    : {len([sa for sa in SA_types if sa is not None])}/{n_features}")
        print(f"Data source      : {'UCI ML Repo' if ucimlrepo_id else 'CSV' if csv_path else 'Manual'}")
        print(f"{'='*80}\n")
        
    return (n_features, repetitions, n_iterations, k_iterations, n_list, k_list,
            ucimlrepo_id, column_names, QI_types, SA_types, generalization_technique)


# ============================================================================
# Helper functions for create_json_file
# ============================================================================

def _process_attribute_types(types, column_names, QI_names, SA_names):
    """Process and map attribute types for QI and SA columns."""
    if QI_names is not None:
        if len(QI_names) > len(column_names):
            raise ValueError("QI_names length exceeds column_names length")
        QI_types = [
            types[column_names.index(qi)] if qi in QI_names else None 
            for qi in column_names
        ]
    else:
        QI_types = types
    
    if SA_names is not None:
        if len(SA_names) > len(column_names):
            raise ValueError("SA_names length exceeds column_names length")
        SA_types = [
            types[column_names.index(sa)] if sa in SA_names else None 
            for sa in column_names
        ]
    else:
        SA_types = []
    
    return QI_types, SA_types


def _validate_manual_parameters(n_features, column_names, QI_names):
    """Validate required parameters for manual configuration."""
    missing = []
    if n_features is None:
        missing.append('n_features')
    if column_names is None:
        missing.append('column_names')
    if QI_names is None:
        missing.append('QI_names')
    
    if missing:
        raise ValueError(
            f"Missing required parameters for manual configuration: {', '.join(missing)}"
        )
    
    if len(column_names) != len(QI_names):
        raise ValueError(
            f"Length mismatch: column_names has {len(column_names)} items "
            f"but QI_names has {len(QI_names)} items"
        )


def _build_attribute_triplet(QI_types, generalization_technique):
    """Build compact (index, type, technique) triplet for QI attributes."""
    triplet = []
    for idx, (qi, gt) in enumerate(zip(QI_types, generalization_technique)):
        if qi is not None:
            triplet.append((idx, qi, gt))
    return triplet


def _build_attribute_duplet(SA_types):
    """Build compact (index, type) duplet for SA attributes."""
    duplet = []
    for idx, sa in enumerate(SA_types):
        if sa is not None:
            duplet.append((idx, sa))
    return duplet
        
    # Do nothing if the file already exists

def write_to_json(file_path, data, n=None, k=None):
    """
    Append experimental results to a JSON file.
    
    Adds new experimental results to the existing JSON structure,
    organizing data by sample size (n) and anonymization parameter (k).
    
    Parameters:
        file_path (str): Path to the JSON results file
        data (dict): Experimental result data to append
        n (int, optional): Number of samples for this experiment
        k (int, optional): Anonymization parameter for this experiment
        
    Returns:
        None: Updates the JSON file in place
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        raise ValueError(f"File {file_path} not found or empty.")

    if n is not None and k is not None:        
        entry = existing_data[1].setdefault(str(n), {}).setdefault(str(k), [])
        entry.append(data)
        entry.sort(key=lambda x: x.get("Model", ""))
    elif n is None and k is None:
        existing_data[2] = data
    else:
        raise ValueError("Either both n and k or neither should be provided.")

    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2, default=_convert_to_native)


def write_to_pkl_tmp(file_path, elapsed_times, key):
    with open(file_path, 'wb') as f:
        pickle.dump([key, elapsed_times], f)
        
def store_generated_data(file_path, data, verbose=False):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:
            raise FileExistsError(f"File {file_path} already exists.")
        os.remove(file_path)

    if verbose:
        init_time = datetime.now()
        logger.info("Storing generated data...")

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    if verbose:
        end_time = datetime.now()
        logger.info(f"Done. Time: {end_time - init_time} [ST: {init_time.strftime('%Y-%m-%d %H:%M:%S')}, ET: {end_time.strftime('%Y-%m-%d %H:%M:%S')}]")
  