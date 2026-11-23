class ConnectionError(Exception):  # noqa: A001 — mirrors pylibmc's API
    pass


class ServerDown(Exception):
    pass


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "pylibmc is stubbed out on this host (libmemcached is not installed). "
        "Keep DATASET.PIN_MEMORY: false in the training config."
    )


Client = _unavailable
ClientPool = _unavailable
