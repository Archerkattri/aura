#!/usr/bin/env bash
# Fetch a benchmark scene into the exact path the AURA reproduction expects.
#
# Status: documents + (where a public URL exists) downloads the scene. It does
# NOT run any training/eval. Datasets are large and some require registration,
# so this script prints the exact commands and verifies the resulting layout
# rather than silently scraping gated downloads.
#
# Usage:
#   bash scripts/fetch_scene.sh truck [DEST]
#   bash scripts/fetch_scene.sh --list
#
# Default DEST for truck is data/tanks/truck (matches configs/truck_run6.json).
#
# Expected layout after fetching the truck scene:
#   data/tanks/truck/
#     images/                 # posed RGB frames (JPG/PNG)
#     sparse/0/               # COLMAP model: cameras.bin images.bin points3D.bin
#                             # (or the .txt equivalents)
#
# Once fetched, ingest + train + eval with the commands in
# configs/truck_run6.json.
set -euo pipefail

SCENE="${1:-}"

print_truck_instructions() {
  local dest="$1"
  cat <<EOF
=====================================================================
Scene: Tanks & Temples — Truck   ->   ${dest}
=====================================================================
The Truck scene ships with the official 3D Gaussian Splatting release
from INRIA (the 'tandt_db.zip' bundle), which already contains the
COLMAP sparse model AURA seeds from. Recommended path:

  mkdir -p "${dest%/truck}"
  # Download the 3DGS Tanks&Temples + Deep Blending bundle (public):
  wget -O /tmp/tandt_db.zip \\
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
  unzip -o /tmp/tandt_db.zip -d /tmp/tandt_db
  # The bundle lays out tandt/truck/{images,sparse/0}; move it into place:
  rsync -a /tmp/tandt_db/tandt/truck/ "${dest}/"

Alternatively, the original full-resolution Truck frames are available
from the Tanks & Temples project (registration required):
  1. Visit https://www.tanksandtemples.org/download/ and register.
  2. Download the 'Truck' (intermediate) image set.
  3. Run COLMAP (or reuse the provided sparse model) to produce
     ${dest}/sparse/0/{cameras,images,points3D}.{bin|txt}.
  4. Place the frames under ${dest}/images/.

After fetching, verify the layout:
  bash scripts/fetch_scene.sh truck "${dest}"   # re-run to verify

Then reproduce run6 per configs/truck_run6.json:
  python -m aura.cli colmap-to-capture-manifest "${dest}/sparse/0" \\
      --root "${dest}" --image-dir "${dest}/images" \\
      --output outputs/truck-pts129k-manifest.json --point-seeded
=====================================================================
EOF
}

verify_layout() {
  local dest="$1"
  local ok=1
  if [[ ! -d "${dest}/images" ]]; then
    echo "MISSING: ${dest}/images (posed RGB frames)"; ok=0
  else
    local count
    count="$(find "${dest}/images" -maxdepth 1 -type f | wc -l | tr -d ' ')"
    echo "OK: ${dest}/images present (${count} files)"
  fi
  if [[ -f "${dest}/sparse/0/cameras.bin" && -f "${dest}/sparse/0/images.bin" ]]; then
    echo "OK: ${dest}/sparse/0 COLMAP binary model present"
  elif [[ -f "${dest}/sparse/0/cameras.txt" && -f "${dest}/sparse/0/images.txt" ]]; then
    echo "OK: ${dest}/sparse/0 COLMAP text model present"
  else
    echo "MISSING: ${dest}/sparse/0 COLMAP model (cameras/images .bin or .txt)"; ok=0
  fi
  if [[ "${ok}" -eq 1 ]]; then
    echo "Layout verified — ready to ingest (see configs/truck_run6.json)."
    return 0
  fi
  echo "Layout incomplete — follow the download instructions above."
  return 1
}

case "${SCENE}" in
  --list|"")
    echo "Available scenes:"
    echo "  truck   Tanks & Temples Truck (used by configs/truck_run6.json)"
    echo
    echo "Usage: bash scripts/fetch_scene.sh truck [DEST]"
    exit 0
    ;;
  truck)
    DEST="${2:-data/tanks/truck}"
    print_truck_instructions "${DEST}"
    echo
    if [[ -d "${DEST}" ]]; then
      echo "Verifying existing layout at ${DEST} ..."
      verify_layout "${DEST}" || true
    else
      echo "Destination ${DEST} does not exist yet — run the download commands above."
    fi
    ;;
  *)
    echo "Unknown scene: ${SCENE}" >&2
    echo "Run: bash scripts/fetch_scene.sh --list" >&2
    exit 2
    ;;
esac
