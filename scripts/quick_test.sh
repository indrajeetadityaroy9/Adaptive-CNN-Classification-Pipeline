#!/bin/bash

set -e

echo "Running Quick Smoke Test"
echo ""
echo "This will:"
echo "  - Train MNIST for 2 epochs"
echo "  - Use 1 GPU"
echo "  - Run evaluation"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="smoke_test_${TIMESTAMP}"

echo "Experiment name: $EXPERIMENT_NAME"
echo ""

modal run modal_train.py \
    --config-path configs/mnist_test_config.yaml \
    --num-gpus 1 \
    --experiment-name "$EXPERIMENT_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "Smoke Test Passed!"
    echo ""
    echo "Experiment: $EXPERIMENT_NAME"
    echo ""
    echo "To download results:"
    echo "  modal volume get cnn-training-vol outputs/$EXPERIMENT_NAME ./results/$EXPERIMENT_NAME"
    echo ""
else
    echo ""
    echo "Smoke Test Failed"
    exit 1
fi
