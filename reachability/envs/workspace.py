from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Workspace2D:
    hx_min: float
    hx_max: float
    hy_min: float
    hy_max: float

@dataclass(frozen=True)
class Workspace3D:
    hx_min: float
    hx_max: float
    hy_min: float
    hy_max: float
    hz_min: float
    hz_max: float
