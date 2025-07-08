from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np

@dataclass
class ActivityModel:
    features: int = 0
    new_sensors: int = 0
    new_motors: int = 0
    nexus_size: int = 0

@dataclass
class AuditResultsModel:
    nexus_growth: int
    terminus_growth: int