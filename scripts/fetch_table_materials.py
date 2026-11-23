"""Download the table materials listed in `data/table_materials.json` into the local cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from sweep_bench.assets.mesh_import import write_bytes_atomic
from sweep_bench.assets.table_materials import (
    TableMaterial,
    cache_problems,
    default_cache_dir,
    load_manifest,
    sha256_file,
)

_HEADERS = {
    "User-Agent": "sweep-bench/1.0 texture fetcher",
    "Accept": "*/*",
}
_ALLOWED_HOSTS = ("polyhaven.org", "polyhaven.com", "ambientcg.com", "struffelproductions.com")


def _check_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not any(host == h or host.endswith(f".{h}") for h in _ALLOWED_HOSTS):
        raise ValueError(f"refusing to fetch {url!r}: only https from {_ALLOWED_HOSTS}")
    return url


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201 — urllib's signature
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirectHandler())


def _get(url: str) -> bytes:
    request = urllib.request.Request(_check_url(url), headers=_HEADERS)  # noqa: S310 — scheme/host checked
    with _OPENER.open(request, timeout=180) as response:
        return response.read()


def fetch(material: TableMaterial, cache_dir: Path) -> int:
    remaining = [
        spec
        for spec in material.maps
        if not material.path(spec.role, cache_dir).is_file()
        or sha256_file(material.path(spec.role, cache_dir)) != spec.sha256
    ]
    if not remaining:
        return 0
    archive = zipfile.ZipFile(io.BytesIO(_get(material.archive))) if material.archive is not None else None
    for spec in remaining:
        if archive is not None and spec.member is not None:
            payload = archive.read(spec.member)
        elif spec.url is not None:
            payload = _get(spec.url)
        else:
            raise ValueError(f"{material.id}: map {spec.role} has neither a URL nor an archive member")
        got = hashlib.sha256(payload).hexdigest()
        if got != spec.sha256:
            raise ValueError(f"{material.id} {spec.role}: sha256 {got}, manifest says {spec.sha256}")
        write_bytes_atomic(material.path(spec.role, cache_dir), payload)
    return len(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="only verify the cache; exit 1 if anything is wrong")
    args = parser.parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    manifest = load_manifest()
    if args.check:
        problems = [p for m in manifest.materials for p in cache_problems(m, cache_dir)]
        for problem in problems:
            print(problem)
        print(f"{len(manifest.materials)} materials, {len(problems)} problems in {cache_dir}")
        sys.exit(1 if problems else 0)
    written = 0
    for material in manifest.materials:
        n = fetch(material, cache_dir)
        written += n
        print(f"{material.id}: {'fetched ' + str(n) + ' maps' if n else 'cached'}", flush=True)
    problems = [p for m in manifest.materials for p in cache_problems(m, cache_dir)]
    if problems:
        sys.exit("cache still wrong after fetching:\n  " + "\n  ".join(problems))
    print(f"{len(manifest.materials)} materials verified in {cache_dir} ({written} maps written)")


if __name__ == "__main__":
    main()
