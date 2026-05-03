#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_parallel_patching.sh \
    --patch-script PATH \
    --source DIR \
    --save-dir DIR \
    [--num-jobs N] \
    -- [patch script args...]

Example:
  run_parallel_patching.sh \
    --patch-script ./create_patches_fp.py \
    --source /path/to/slides \
    --save-dir /path/to/output \
    --num-jobs 4 \
    -- --seg --patch --target_magnification 20
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT=""
SOURCE=""
SAVE_DIR=""
NUM_JOBS="${NUM_JOBS:-4}"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --patch-script)
            PATCH_SCRIPT="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            FORWARD_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$PATCH_SCRIPT" || -z "$SOURCE" || -z "$SAVE_DIR" ]]; then
    usage >&2
    exit 1
fi

if [[ "$PATCH_SCRIPT" != /* ]]; then
    PATCH_SCRIPT="$SCRIPT_DIR/${PATCH_SCRIPT#./}"
fi

if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "Patch script not found: $PATCH_SCRIPT" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR/parallel_process_lists"

mapfile -t SHARD_LISTS < <(
    SOURCE="$SOURCE" SAVE_DIR="$SAVE_DIR" NUM_JOBS="$NUM_JOBS" python - <<'PY'
import math
import os
import pandas as pd

source = os.environ["SOURCE"]
save_dir = os.environ["SAVE_DIR"]
num_jobs = max(1, int(os.environ["NUM_JOBS"]))
slides = sorted(
    slide for slide in os.listdir(source)
    if os.path.isfile(os.path.join(source, slide))
)

if not slides:
    raise SystemExit("No slide files found in source directory")

num_jobs = min(num_jobs, len(slides))
lists_dir = os.path.join(save_dir, "parallel_process_lists")
chunk_size = int(math.ceil(len(slides) / float(num_jobs)))

for shard_idx in range(num_jobs):
    shard_slides = slides[shard_idx * chunk_size:(shard_idx + 1) * chunk_size]
    if not shard_slides:
        continue
    rel_path = os.path.join("parallel_process_lists", f"shard_{shard_idx:02d}.csv")
    abs_path = os.path.join(save_dir, rel_path)
    pd.DataFrame({"slide_id": shard_slides, "process": 1}).to_csv(abs_path, index=False)
    print(rel_path)
PY
)

if [[ ${#SHARD_LISTS[@]} -eq 0 ]]; then
    echo "No shard process lists were generated" >&2
    exit 1
fi

PIDS=()
cleanup() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for shard_csv in "${SHARD_LISTS[@]}"; do
    shard_name="$(basename "$shard_csv" .csv)"
    shard_output="parallel_process_lists/process_list_autogen_${shard_name}.csv"
    echo "Launching $shard_name"
    python "$PATCH_SCRIPT" \
        --source "$SOURCE" \
        --save_dir "$SAVE_DIR" \
        --process_list "$shard_csv" \
        --process_list_output "$shard_output" \
        "${FORWARD_ARGS[@]}" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
trap - INT TERM

if [[ $status -ne 0 ]]; then
    echo "At least one shard failed" >&2
    exit $status
fi

SAVE_DIR="$SAVE_DIR" python - <<'PY'
import glob
import os
import pandas as pd

save_dir = os.environ["SAVE_DIR"]
paths = sorted(glob.glob(os.path.join(save_dir, "parallel_process_lists", "process_list_autogen_shard_*.csv")))
if not paths:
    raise SystemExit("No shard process_list outputs were found")

dfs = [pd.read_csv(path) for path in paths]
combined = pd.concat(dfs, ignore_index=True)
if "slide_id" in combined.columns:
    combined = combined.sort_values("slide_id").reset_index(drop=True)
combined.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
PY

echo "Combined process list written to $SAVE_DIR/process_list_autogen.csv"
