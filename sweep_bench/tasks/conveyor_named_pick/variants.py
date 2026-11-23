from __future__ import annotations

NAMED_PICK_OBJECT_IDS: tuple[str, ...] = (
    "proc_box_040x040x040_red",
    "proc_box_040x040x040_green",
    "proc_box_040x040x040_blue",
    "proc_box_040x040x040_yellow",
    "proc_box_030x030x030_red",
    "proc_box_030x030x030_green",
    "proc_box_030x030x030_blue",
    "proc_box_030x030x030_yellow",
    "proc_box_050x050x050_red",
    "proc_box_050x050x050_green",
    "proc_box_050x050x050_blue",
    "proc_box_050x050x050_yellow",
    "proc_can_040x030_red",
    "proc_can_040x030_green",
    "proc_can_040x030_blue",
    "proc_can_040x030_yellow",
    "proc_can_050x040_red",
    "proc_can_050x040_green",
    "proc_can_050x040_blue",
    "proc_can_050x040_yellow",
    "proc_can_060x040_red",
    "proc_can_060x040_green",
    "proc_can_060x040_blue",
    "proc_can_060x040_yellow",
    "apple",
)

NAMED_PICK_DISTRACTOR_IDS: tuple[str, ...] = tuple(i for i in NAMED_PICK_OBJECT_IDS if i != "apple")
