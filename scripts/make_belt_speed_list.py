"""Write a corpus's per-episode belt speed as one JSON list indexed by episode_index."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="merged LeRobot dataset directory")
    parser.add_argument("out", type=Path, help="JSON file to write")
    args = parser.parse_args()
    root, out = args.root, args.out
    rows = [json.loads(x) for x in (root / "meta" / "bench_episode_provenance.jsonl").read_text().splitlines() if x]
    total = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
    if [row["episode_index"] for row in rows] != list(range(total)):
        raise SystemExit(f"provenance rows are not episodes 0..{total - 1} in order")
    speeds = [row["belt_speed"] for row in rows]
    if not all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in speeds):
        raise SystemExit("every belt_speed must be a positive finite float")
    text = json.dumps(speeds) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(
        f"{out}: {len(speeds)} speeds in [{min(speeds):.4f}, {max(speeds):.4f}], md5 {hashlib.md5(text.encode()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
