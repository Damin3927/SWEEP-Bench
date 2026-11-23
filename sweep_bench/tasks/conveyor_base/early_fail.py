from __future__ import annotations

from typing import Final

import numpy as np
import sapien
import trimesh.remesh
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

ROBOT_REACH: Final[dict[str, float]] = {"panda": 1.0, "panda_wristcam": 1.0}

RETURN_SPEED: Final[float] = 0.01

HULL_EDGE: Final[float] = 0.02
HULL_TOLERANCE: Final[float] = HULL_EDGE / 3**0.5


def hull_points(sapien_entity: sapien.Entity) -> np.ndarray:
    clouds = []
    for component in sapien_entity.components:
        if isinstance(component, sapien.physx.PhysxRigidBaseComponent):
            for mesh in get_component_meshes(component):
                vertices, _faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=HULL_EDGE)
                clouds.append(np.asarray(vertices, dtype=np.float32))
    if not clouds:
        raise ValueError(f"{sapien_entity.name!r} has no collision geometry to measure")
    return np.concatenate(clouds)
