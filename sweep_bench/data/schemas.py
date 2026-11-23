from __future__ import annotations

from dataclasses import dataclass

type FeatureSpec = dict[str, dict[str, object]]


@dataclass(frozen=True)
class BenchSchema:
    name: str
    camera_rig: str
    image_height: int
    image_width: int
    arm_dof: int
    features: FeatureSpec
    robot_type: str


def _droid_panda_features(h: int, w: int) -> FeatureSpec:
    return {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }


DROID_PANDA_SCHEMA = BenchSchema(
    name="droid_panda_v3",
    camera_rig=(
        "The three world-fixed eyes tiled into exterior_image_1_left "
        "(top row 112x112 each: side_camera_minus_y | side_camera_plus_y; "
        "bottom row full width 224x112: head_camera). "
        "side_camera_plus_y: DROID side eye at (0.60, -0.40, 0.72) rel. workspace aimed at "
        "(0, 0, 0.27), fov 1.0 rad; side_camera_minus_y is its exact mirror through y = 0. "
        "The pair covers y in [-0.64, +0.64] of the +-0.9 m belt, symmetrically, leaving both "
        "belt ends unseen. head_camera: elevated view at (0.48, 0.00, 1.40) aimed at "
        "(-0.39, 0, 0.25), fov 0.85 rad -- clears the belt deck to see the place targets. "
        "wrist_camera: gripper-mounted, fov pi/2, stored full-size in wrist_image_left and "
        "deliberately NOT in the composite. "
        "Exterior eyes jitter independently (+-0.05 m)."
    ),
    image_height=224,
    image_width=224,
    arm_dof=7,
    features=_droid_panda_features(224, 224),
    robot_type="panda",
)


SCHEMAS: dict[str, BenchSchema] = {
    DROID_PANDA_SCHEMA.name: DROID_PANDA_SCHEMA,
}


def get_schema(name: str) -> BenchSchema:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown schema {name!r}; choose from {sorted(SCHEMAS)}")
    return SCHEMAS[name]
