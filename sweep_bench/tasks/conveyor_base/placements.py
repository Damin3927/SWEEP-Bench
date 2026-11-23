from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sweep_bench.tasks.conveyor_base.config import SceneStyle

PLACEMENTS_PATH = Path(__file__).resolve().parents[3] / "data" / "background_placements.json"

PlacedStyle = Literal["replicacad", "ai2thor", "robocasa"]
type Background = tuple[SceneStyle, int]


class BackgroundPlacement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    scene_style: PlacedStyle
    scene_idx: int = Field(ge=0)
    scene: str = Field(min_length=1)
    workspace_offset: tuple[float, float, float] | None
    unplaced: str | None = None
    passes_check: bool | None = None

    @model_validator(mode="after")
    def _offset_or_reason(self) -> BackgroundPlacement:
        if (self.workspace_offset is None) == (self.unplaced is None):
            raise ValueError("a row needs exactly one of workspace_offset and unplaced")
        if (self.workspace_offset is None) != (self.passes_check is None):
            raise ValueError("a placed row needs passes_check, and an unplaced row has none")
        return self


class PlacementTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    placements: tuple[BackgroundPlacement, ...]

    @model_validator(mode="after")
    def _one_entry_per_background(self) -> PlacementTable:
        keys = [(p.scene_style, p.scene_idx) for p in self.placements]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ValueError(f"backgrounds listed more than once: {repeated}")
        return self


def load_placement_table(path: Path = PLACEMENTS_PATH) -> PlacementTable:
    return PlacementTable.model_validate_json(path.read_bytes())


@functools.cache
def _placements() -> dict[tuple[str, int], BackgroundPlacement]:
    return {(p.scene_style, p.scene_idx): p for p in load_placement_table().placements}


def recorded_placement(scene_style: PlacedStyle, scene_idx: int) -> BackgroundPlacement:
    placement = _placements().get((scene_style, scene_idx))
    if placement is None:
        raise ValueError(
            f"{PLACEMENTS_PATH} records no workspace placement for {scene_style} index {scene_idx}; "
            "pass workspace_offset explicitly or add the background to the table"
        )
    return placement


def workspace_offset_for(scene_style: PlacedStyle, scene_idx: int) -> tuple[float, float, float]:
    placement = recorded_placement(scene_style, scene_idx)
    if placement.workspace_offset is None:
        raise ValueError(
            f"{scene_style} index {scene_idx} has no workspace placement ({placement.unplaced}); "
            "pass workspace_offset explicitly to build it anyway"
        )
    return placement.workspace_offset


def builder_scene_names() -> dict[tuple[PlacedStyle, int], str]:
    from mani_skill.utils.scene_builder.ai2thor.variants import ArchitecTHORSceneBuilder
    from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
    from mani_skill.utils.scene_builder.robocasa.utils.scene_registry import LayoutType, StyleType

    replicacad = ReplicaCADSceneBuilder(None, include_staging_scenes=True).build_configs
    ai2thor = ArchitecTHORSceneBuilder(None).build_configs
    if replicacad is None or ai2thor is None:
        raise RuntimeError("a scene builder loaded no build configs")
    layouts = [v for v in LayoutType if v.value >= 0]
    styles = [v for v in StyleType if v.value >= 0]
    names: dict[tuple[PlacedStyle, int], str] = {}
    for idx, config in enumerate(replicacad):
        names["replicacad", idx] = config
    for idx, config in enumerate(ai2thor):
        names["ai2thor", idx] = config.config_file
    for layout in layouts:
        for style in styles:
            names["robocasa", layout.value * len(styles) + style.value] = f"{layout.name}/{style.name}"
    return names


def passing_backgrounds() -> tuple[Background, ...]:
    rows = load_placement_table().placements
    return (("table", 0), *((p.scene_style, p.scene_idx) for p in rows if p.passes_check))


def validate_background_pool(pool: Sequence[Background]) -> tuple[Background, ...]:
    backgrounds = tuple((style, int(idx)) for style, idx in pool)
    if not backgrounds:
        raise ValueError("a background pool needs at least one background")
    if len(set(backgrounds)) != len(backgrounds):
        raise ValueError(f"a background pool lists a background twice: {backgrounds}")
    for style, idx in backgrounds:
        if style == "table":
            if idx != 0:
                raise ValueError(f"the table style has one background, index 0; got {idx}")
        else:
            workspace_offset_for(style, idx)
    return backgrounds


_POOL_FILE = TypeAdapter(list[tuple[Literal["table", "replicacad", "ai2thor", "robocasa"], int]])


def resolve_background_pool(spec: str) -> tuple[Background, ...]:
    if spec == "passing":
        return passing_backgrounds()
    return validate_background_pool(_POOL_FILE.validate_json(Path(spec).read_bytes(), strict=True))
