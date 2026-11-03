#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=2

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate north_caucasus

for input_variant in language_tokens no_language_tokens; do
  for proportion in 125 50 90 0; do
    batch_size=8
    splits=(0 1 2 3 4)
    if [[ "$proportion" == "90" ]]; then
      batch_size=6
      splits=(0)
    elif [[ "$proportion" == "0" ]]; then
      splits=(0)
    fi

    for split in "${splits[@]}"; do
      echo "Training prop_${proportion}, split_${split}, ${input_variant}"
      variant_args=()
      if [[ "$input_variant" == "language_tokens" ]]; then
        variant_args+=(--language-tokens)
      fi

      torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_triattn.py \
        --train-proportion "$proportion" \
        --split "$split" \
        --epochs 10 \
        --batch-size "$batch_size" \
        "${variant_args[@]}"
    done
  done
done
