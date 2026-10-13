#!/bin/bash

mkdir -p results

python src/continuous_double_auction/experiments/temperature_experiment.py \
  --seller_valuations 80 80 80 80 80 \
  --buyer_valuations 100 100 100 100 100 \
  --rounds 30 \
  --seller_models qwen2.5:14b qwen2.5:14b qwen2.5:14b qwen2.5:14b qwen2.5:14b \
  --buyer_models qwen2.5:14b qwen2.5:14b qwen2.5:14b qwen2.5:14b qwen2.5:14b \
  --temperatures 0.1 0.4 0.7 1.0 1.3 \
  --tag ""

echo "Temperature experiment completed" 