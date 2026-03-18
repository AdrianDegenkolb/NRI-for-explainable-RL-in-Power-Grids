#!/bin/bash
# Minimal test script for PPO training with minimal compute requirements

echo "Running minimal test..."
PYTHONPATH=$(pwd) python training_scripts/train_ppo.py --file_path ./configs/test_minimal.yaml --workdir . --seed 42 --model-type RAGNN --experiment-name test_minimal_run

echo "Test completed!"

