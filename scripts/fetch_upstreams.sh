#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r path url sha; do
  if [ -e "$path/.git" ]; then continue; fi
  rm -rf "$path"
  git clone --quiet "$url" "$path"
  git -C "$path" checkout --quiet "$sha"
done < policy_servers/UPSTREAM_PINS
