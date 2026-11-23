from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_HORIZON = 300
LOOP_CAP_SLACK = 2


def loop_cap(episodes: int, num_envs: int, horizon: int) -> int:
    rounds = math.ceil(episodes / max(num_envs, 1)) + 1
    return rounds * horizon * LOOP_CAP_SLACK


def episode_quota(episodes: int, num_envs: int) -> list[int]:
    slots = max(num_envs, 1)
    base, extra = divmod(episodes, slots)
    return [base + (1 if i < extra else 0) for i in range(slots)]


@dataclass
class Cohort:
    quota: list[int]

    def wants(self, slot: int) -> bool:
        return self.quota[slot] > 0

    def take(self, slot: int) -> bool:
        if self.quota[slot] <= 0:
            return False
        self.quota[slot] -= 1
        return True

    @property
    def done(self) -> bool:
        return not any(self.quota)


SUITE_SPEEDS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
