#!/bin/bash

mkdir -p results

# Modified mix: local Qwen stands in for GPT-4.1, NVIDIA Nemotron for Claude.
QWEN="${AGENT_MODEL:-qwen2.5:14b}"
NEMOTRON="${NIM_MODEL:-nvidia/nemotron-3-super-120b-a12b}"

for i in {1..5}; do

    # Qwen buyers
    ## Nemotron sellers
    echo "Running experiment with tag qwen_buyers-nemotron_sellers-${i}"
    python src/continuous_double_auction/simulation.py \
        --buyer_models "$QWEN" "$QWEN" "$QWEN" "$QWEN" "$QWEN" \
        --seller_models "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" \
        --tag "qwen_buyers-nemotron_sellers-${i}"

    ## Mixed sellers (3 Qwen, 2 Nemotron)
    echo "Running experiment with tag qwen_buyers-mixed_sellers-${i}"
    python src/continuous_double_auction/simulation.py \
        --buyer_models "$QWEN" "$QWEN" "$QWEN" "$QWEN" "$QWEN" \
        --seller_models "$QWEN" "$QWEN" "$QWEN" "$NEMOTRON" "$NEMOTRON" \
        --tag "qwen_buyers-mixed_sellers-${i}"

    # Nemotron buyers
    ## Qwen sellers
    echo "Running experiment with tag nemotron_buyers-qwen_sellers-${i}"
    python src/continuous_double_auction/simulation.py \
        --buyer_models "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" \
        --seller_models "$QWEN" "$QWEN" "$QWEN" "$QWEN" "$QWEN" \
        --tag "nemotron_buyers-qwen_sellers-${i}"

    ## Nemotron sellers
    echo "Running experiment with tag nemotron_buyers-nemotron_sellers-${i}"
    python src/continuous_double_auction/simulation.py \
        --buyer_models "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" \
        --seller_models "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" \
        --tag "nemotron_buyers-nemotron_sellers-${i}"

    ## Mixed sellers (3 Qwen, 2 Nemotron)
    echo "Running experiment with tag nemotron_buyers-mixed_sellers-${i}"
    python src/continuous_double_auction/simulation.py \
        --buyer_models "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" "$NEMOTRON" \
        --seller_models "$QWEN" "$QWEN" "$QWEN" "$NEMOTRON" "$NEMOTRON" \
        --tag "nemotron_buyers-mixed_sellers-${i}"
done

echo "Seller mix experiment (no comms) completed"
