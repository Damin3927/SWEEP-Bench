from sweep_bench.policies.ee_dom import validate_declared_camera_vintage
from sweep_bench.policies.openpi import OpenPiDroidPolicy


class VLAJEPAPolicy(OpenPiDroidPolicy):
    """Adapter for VLA-JEPA bench fine-tunes (``policy_servers/vlajepa/serve.py``)."""

    MODEL = "vla_jepa"

    def __init__(self, host: str, port: int, prompt: str = "", replan_every: int = 7) -> None:
        super().__init__(host=host, port=port, prompt=prompt, replan_every=replan_every)
        metadata = self._client.get_server_metadata()
        if metadata.get("model") != self.MODEL:
            raise ValueError(
                f"server at {host}:{port} declares model {metadata.get('model')!r}, expected {self.MODEL!r}"
            )
        validate_declared_camera_vintage(metadata.get("bench_schema"), expected=self._SCHEMA.name)
