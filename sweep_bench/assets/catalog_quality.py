from __future__ import annotations

from typing import TYPE_CHECKING

from sweep_bench.assets.mesh_import import _MIN_RIDER_THICKNESS

if TYPE_CHECKING:
    from collections.abc import Sequence

MERGED_SET_REASON = "merged_set"
TOO_THIN_REASON = "too_thin"


def thickness_reason(bbox: Sequence[float], role: str = "graspable_rider") -> str:
    if role != "graspable_rider":
        return ""
    thickness = 2.0 * min(bbox)
    if thickness >= _MIN_RIDER_THICKNESS:
        return ""
    return (
        f"{TOO_THIN_REASON} (smallest full dimension {1000.0 * thickness:.2f} mm below the "
        f"{1000.0 * _MIN_RIDER_THICKNESS:.1f} mm floor; the bbox-only grasp_fit check cannot see it)"
    )
