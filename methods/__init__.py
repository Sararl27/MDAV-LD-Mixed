"""methods package

Lightweight public surface for the anonymization methods package. The
repository previously contained multiple MDAV variants; this package now
exposes the remaining MDAV implementation and the preprocessing/schema
utilities used by the evaluation tooling.

Exports:
    - MDAV_LD_Mixed: primary MDAV implementation with L-diversity support
    - preprocessing.*: preprocessing utilities and models
    - AttributeSchema, QuasiIdentifiers, SensitiveAttributes: schema helpers
"""

from .MDAV import MDAV_LD_Mixed
from .preprocessing import *
from .schema import AttributeSchema, AttributeSchema, QuasiIdentifiers, SensitiveAttributes