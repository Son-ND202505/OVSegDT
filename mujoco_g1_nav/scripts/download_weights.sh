#!/usr/bin/env bash
# Downloads the OVSegDT checkpoint fine-tuned for the MuJoCo houses into mujoco_g1_nav/checkpoints/ (run from the repo root).
set -euo pipefail
URL="${URL:-https://github.com/Son-ND202505/OVSegDT/releases/download/v1.0/RECOMMENDED_house.pth}"
OUT=mujoco_g1_nav/checkpoints/RECOMMENDED_house.pth
SHA=adadd4d273b3d46f76e8bc033f3b3ad2cfad9a2d58456b3e598ef48a6ad49176
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || curl -L --fail -o "$OUT" "$URL"
echo "$SHA  $OUT" | sha256sum -c -
