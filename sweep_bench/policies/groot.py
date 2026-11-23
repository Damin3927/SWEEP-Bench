from sweep_bench.data.vintage import INFO_KEY
from sweep_bench.policies.ee_dom import validate_declared_camera_vintage
from sweep_bench.policies.openpi import OpenPiDroidPolicy


class GrootPolicy(OpenPiDroidPolicy):
    """Adapter for GR00T N1.7 fine-tunes served by ``policy_servers/groot/serve.py``."""

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 8) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        validate_declared_camera_vintage(self._client.get_server_metadata().get(INFO_KEY), expected=self._SCHEMA.name)
