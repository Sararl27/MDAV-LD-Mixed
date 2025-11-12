# MDAV-LD-Mixed

This repository contains the implementation and experimental setup for the **MDAV-LD-Mixed** algorithm, a microaggregation method for mixed-type data that integrates latent-space encoding and $\ell$-diversity constraints.  
All experiments and figures presented in the paper can be reproduced using the code provided here.

---

## Quick start

1. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

2. **Run the experiments**

- Open Jupyter Lab or Jupyter Notebook and load `MDAV.ipynb`.
- Execute the notebook cells to:
  - Prepare datasets (via `tools/utils.get_data`)
  - Configure algorithm variants and parameters
  - Run the evaluation using `run_model_evaluation` (from `model_evaluation.py`)
  - Generate and save figures under `output/MDAV/<experiment_name>/images/`

---

## Main files

| File / Folder | Description |
|----------------|-------------|
| `MDAV.ipynb` | Main notebook |
| `methods/MDAV/MDAV_LD_Mixed.py` | Core implementation of the MDAV-LD-Mixed algorithm |
| `model_evaluation.py` | Experimental pipeline providing the `run_model_evaluation` function |
| `tools/` | Auxiliary utilities for data preprocessing and evaluation |
| `output/` | Output directory for experiments |

---
