from __future__ import annotations

import numpy as np
import numpy.typing as npt

type Size = int | tuple[int, ...]
type FloatParam = float | npt.NDArray[np.floating] | npt.NDArray[np.integer]
type IntParam = int | npt.NDArray[np.integer]

_DRAWS = ("uniform", "normal", "randint", "choice", "permutation")
_PASSTHROUGH = frozenset({"_rngs", "_b", "_inner_shape", "_row", *_DRAWS})


def _batched(value: npt.ArrayLike, b: int) -> bool:
    return np.ndim(value) >= 1 and np.shape(value)[0] == b


class PerEnvEpisodeRNG(np.random.RandomState):
    def __init__(self, rngs: list[np.random.RandomState]) -> None:
        self._rngs = rngs
        self._b = len(rngs)

    def __getattribute__(self, item: str) -> object:
        if item in _PASSTHROUGH or item.startswith("__"):
            return object.__getattribute__(self, item)
        raise AttributeError(
            f"`_episode_rng.{item}` is not available during episode init: "
            f"PerEnvEpisodeRNG implements only the per-env draws "
            f"({', '.join(_DRAWS)}). Add it to episode_rng.py with its per-env "
            f"semantics rather than reaching for the shared stream."
        )

    def _inner_shape(self, name: str, size: Size | None, params: tuple[FloatParam, ...]) -> tuple[int, ...] | None:
        if size is None:
            if not any(_batched(p, self._b) for p in params):
                raise ValueError(
                    f"_episode_rng.{name}(...) during episode init draws one "
                    f"batch-shared value. Episode draws are per-env: pass "
                    f"size=b, or move the draw to `_load_scene`/`_load_objects` "
                    f"if it really is shared (see episode_rng.py)."
                )
            return None
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        if not shape or shape[0] != self._b:
            raise ValueError(
                f"_episode_rng.{name}(size={size!r}) during episode init: the "
                f"leading axis must be the reset batch b={self._b}. Episode "
                f"draws are per-env; a batch-shared draw belongs in "
                f"`_load_scene`/`_load_objects` (see episode_rng.py)."
            )
        return shape[1:] or None

    def _row[T: (IntParam, FloatParam)](self, value: T, j: int) -> T:
        return np.asarray(value)[j] if _batched(value, self._b) else value

    def uniform(  # ty: ignore[invalid-method-override]  # per-env draws need a batch-leading size, narrower than RandomState
        self, low: FloatParam = 0.0, high: FloatParam = 1.0, size: Size | None = None
    ) -> np.ndarray:
        inner = self._inner_shape("uniform", size, (low, high))
        return np.stack([rng.uniform(self._row(low, j), self._row(high, j), inner) for j, rng in enumerate(self._rngs)])

    def normal(  # ty: ignore[invalid-method-override]  # per-env draws need a batch-leading size, narrower than RandomState
        self, loc: FloatParam = 0.0, scale: FloatParam = 1.0, size: Size | None = None
    ) -> np.ndarray:
        inner = self._inner_shape("normal", size, (loc, scale))
        return np.stack([rng.normal(self._row(loc, j), self._row(scale, j), inner) for j, rng in enumerate(self._rngs)])

    def randint(  # ty: ignore[invalid-method-override]  # per-env draws need a batch-leading size, narrower than RandomState
        self,
        low: IntParam,
        high: IntParam | None = None,
        size: Size | None = None,
    ) -> np.ndarray:
        inner = self._inner_shape("randint", size, (low, high) if high is not None else (low,))
        return np.stack(
            [
                rng.randint(self._row(low, j), None if high is None else self._row(high, j), inner)
                for j, rng in enumerate(self._rngs)
            ]
        )

    def choice(  # ty: ignore[invalid-method-override]  # per-env draws need a batch-leading size, narrower than RandomState
        self,
        a: npt.ArrayLike,
        size: Size | None = None,
        replace: bool = True,
        p: FloatParam | None = None,
    ) -> np.ndarray:
        inner = self._inner_shape("choice", size, ())
        return np.stack([rng.choice(a, inner, replace, p) for rng in self._rngs])

    def permutation(self, x: npt.ArrayLike) -> np.ndarray:
        return np.stack([rng.permutation(x) for rng in self._rngs])
