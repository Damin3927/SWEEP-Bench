from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from sweep_bench.data.frame_builder import PANDA_FINGER_MAX, REQUIRED_SENSOR_UIDS

if TYPE_CHECKING:
    from mani_skill.agents.controllers.base_controller import CombinedController
    from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController

    from sweep_bench.data.schemas import BenchSchema
    from sweep_bench.tasks.conveyor_base.scripted import ScriptedControlMixin


_HOLD_ACTION_EPS = 0.01

_EXTRACTOR_ARM_DOF = 7


def _arm_controller(task: ScriptedControlMixin) -> PDJointPosController:
    combined = cast("CombinedController", task.agent.controller)
    return cast("PDJointPosController", combined.controllers["arm"])


def check_schema_describes_robot(task: ScriptedControlMixin, schema: BenchSchema) -> None:
    if schema.arm_dof != _EXTRACTOR_ARM_DOF:
        raise ValueError(
            f"schema {schema.name!r} declares arm_dof={schema.arm_dof}, but `extract_dataset_action` is not "
            f"schema-driven: it writes a fixed DROID action of {_EXTRACTOR_ARM_DOF} joint targets plus one "
            "gripper value, sliced at that width from the controller's IK target. A schema of any other "
            "shape would be filled with this layout whatever it declares, so it is refused here rather "
            "than written. Teach `extract_dataset_action` the schema first."
        )

    robot = task.agent.uid
    arm_dof = len(_arm_controller(task).joints)
    if arm_dof != schema.arm_dof:
        raise ValueError(
            f"robot {robot!r} drives {arm_dof} arm joints, but schema {schema.name!r} stores "
            f"arm_dof={schema.arm_dof} in every frame's joint_position and actions (and reads the "
            f"gripper from qpos[{schema.arm_dof}] as a Panda finger position in metres). "
            f"{robot!r} needs a schema of its own -- arm_dof, gripper normalization and "
            "robot_type -- which does not exist yet; there is nothing to collect it into."
        )

    missing = [uid for uid in REQUIRED_SENSOR_UIDS if uid not in task._sensors]
    if missing:
        raise ValueError(
            f"robot {robot!r} carries no {', '.join(missing)}, and every frame of schema "
            f"{schema.name!r} needs it: the world-fixed views are tiled into exterior_image_1_left "
            f"and the wrist stream fills wrist_image_left. The env has {sorted(task._sensors)}. "
            "There is no observation path that omits a camera -- collect with a robot that has one."
        )

    joints = task.agent.robot.get_active_joints()
    if len(joints) <= schema.arm_dof:
        raise ValueError(
            f"robot {robot!r} has {len(joints)} active joints, so there is no joint at "
            f"qpos[{schema.arm_dof}] for schema {schema.name!r} to read the gripper from."
        )
    gripper = joints[schema.arm_dof]
    joint_type = gripper.type[0]
    low, high = float(gripper.limits[0, 0]), float(gripper.limits[0, 1])
    if (
        joint_type != "prismatic"
        or not math.isclose(low, 0.0, abs_tol=1e-6)
        or not math.isclose(high, PANDA_FINGER_MAX, rel_tol=1e-6)
    ):
        raise ValueError(
            f"robot {robot!r} reads its gripper from qpos[{schema.arm_dof}] = {gripper.name!r}, a "
            f"{joint_type} joint over [{low:g}, {high:g}], but schema {schema.name!r} stores that slot as a "
            f"Panda finger position: prismatic travel over [0, {PANDA_FINGER_MAX:g}] m, normalized by "
            f"dividing by {PANDA_FINGER_MAX:g}. Collecting it would write "
            f"{'radians' if 'revolute' in str(joint_type) else 'that joint'} over a finger stroke, which no "
            "downstream consumer can distinguish from a real gripper position."
        )


def extract_dataset_action(
    task: ScriptedControlMixin,
    ee_action: torch.Tensor,
    env_idx: int = 0,
) -> np.ndarray:
    arm = _arm_controller(task)
    assert arm._target_qpos is not None
    target_qpos = arm._target_qpos[env_idx, :_EXTRACTOR_ARM_DOF].detach().cpu().numpy().astype(np.float32)

    gripper_cmd = float(ee_action[env_idx, 6].item())
    gripper_droid = (gripper_cmd + 1.0) / 2.0
    gripper_droid = max(0.0, min(1.0, gripper_droid))

    out = np.empty(_EXTRACTOR_ARM_DOF + 1, dtype=np.float32)
    out[:_EXTRACTOR_ARM_DOF] = target_qpos
    out[_EXTRACTOR_ARM_DOF] = gripper_droid
    return out


def ik_fallback_detected(
    task: ScriptedControlMixin,
    ee_action: torch.Tensor,
    env_idx: int = 0,
) -> bool:
    arm = _arm_controller(task)
    assert arm._target_qpos is not None
    assert arm._start_qpos is not None

    commanded = float(torch.linalg.vector_norm(ee_action[env_idx, :6]).item())
    if commanded <= _HOLD_ACTION_EPS:
        return False

    target_row = arm._target_qpos[env_idx]
    start_row = arm._start_qpos[env_idx]
    return target_row.shape == start_row.shape and bool(torch.equal(target_row, start_row))
