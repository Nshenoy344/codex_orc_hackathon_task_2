#!/usr/bin/env bash
# Offline only. Fixed measured capture list, uniform0.8px candidate admission,
# unchanged observability, depth, held-out, and solution-stability checks.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"
output="${1:-data/commissioning/charuco_handeye_candidate.yaml}"
observations=(
 data/commissioning/charuco_reference.npz
 data/commissioning/board_x03_view.npz
 data/commissioning/board_x06_seq_view.npz
 data/commissioning/board_x09_seq_view.npz
 data/commissioning/board_x12_seq_view.npz
 data/commissioning/board_x15_reobserve.npz
 data/commissioning/board_x12_return_view.npz
 data/commissioning/board_x09_return_view.npz
 data/commissioning/board_x06_corrected_view.npz
 data/commissioning/board_x03_return2_view.npz
 data/commissioning/board_x00_return2_view.npz
 data/commissioning/board_y03_out2_view.npz
 data/commissioning/board_y06_out2_view.npz
 data/commissioning/board_y09_out2_view.npz
 data/commissioning/board_y12_out2_view.npz
 data/commissioning/board_y09_return2_view.npz
 data/commissioning/board_y06_return2_view.npz
)
/usr/bin/python3 - "$output" "${observations[@]}" <<'MANIFEST'
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[1]);paths=[Path(p) for p in sys.argv[2:]]
missing=[str(p) for p in paths if not p.is_file()]
report=dict(offline_only=True,expected_observation_count=len(paths),
    existing_observation_count=len(paths)-len(missing),missing=missing,
    observation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
    board_config='data/commissioning/charuco_center_inspection.json',
    max_reprojection_rms_px=.8,
    admission_budget_note='Uniform candidate image-plane residual budget; depth/normal and complete held-out solver gates remain unchanged. No physical calibration or metric-scale verification is implied.',
    calibration_accepted=False if missing else None)
manifest=output.with_suffix('.manifest.json');manifest.parent.mkdir(parents=True,exist_ok=True)
manifest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(manifest=str(manifest),existing=report['existing_observation_count'],expected=len(paths),missing=missing),indent=2))
if missing:
    print('Batch has not finished; no solver or calibration export was run.',file=sys.stderr)
    raise SystemExit(3)
MANIFEST
exec /usr/bin/python3 -m lego_sorter.charuco_handeye "${observations[@]}" \
 --board-config data/commissioning/charuco_center_inspection.json \
 --max-rms-px .8 --output "$output"
