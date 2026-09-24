#!/bin/bash
# scripts/run_tuning_grid.sh
# 
# Tuning Grid Execution Script (Member 4 - Phase 2)
# 
# Sweeps:
# - fusion: abs-diff, signed
# - lr: 1e-3, 3e-4, 1e-4
# - encoder_freezing: True, False
# - resolution: 256, 512

echo "Starting Tuning Grid for Phase 2..."

source .venv/bin/activate

for fusion in abs_diff signed_fusion; do
  for lr in 1e-3 3e-4 1e-4; do
    for freeze in True False; do
      for size in 256 512; do
        echo "Running config: fusion=${fusion}, lr=${lr}, freeze=${freeze}, size=${size}"
        python -m cdlib.cli.train \
            model=proposed_effnet \
            model.fusion.name=$fusion \
            train.optimizer.lr=$lr \
            +model.encoder.freeze=$freeze \
            +data.img_size=$size \
            train.batch_size=8 \
            train.epochs=60
      done
    done
  done
done

echo "Grid submission complete. Check the 'results/' folder for outputs."
