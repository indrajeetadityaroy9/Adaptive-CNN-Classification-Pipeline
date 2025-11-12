#!/bin/bash

VOLUME_NAME="cnn-training-vol"

function list_experiments() {
    echo "=== Available Experiments ==="
    modal volume ls $VOLUME_NAME outputs
    echo ""
}

function list_checkpoints() {
    echo "=== Available Checkpoints ==="
    modal volume ls $VOLUME_NAME checkpoints
    echo ""
}

function download_experiment() {
    if [ -z "$1" ]; then
        echo "Usage: $0 download_experiment <experiment_name> [output_dir]"
        return 1
    fi

    experiment_name=$1
    output_dir=${2:-"./downloads"}

    mkdir -p "$output_dir"
    echo "Downloading experiment: $experiment_name"
    echo "To: $output_dir/$experiment_name"

    modal volume get $VOLUME_NAME "outputs/$experiment_name" "$output_dir/$experiment_name"

    if [ $? -eq 0 ]; then
        echo "✓ Download complete!"
        echo "Files available at: $output_dir/$experiment_name"
        ls -lh "$output_dir/$experiment_name"
    else
        echo "✗ Download failed"
        return 1
    fi
}

function download_checkpoint() {
    if [ -z "$1" ]; then
        echo "Usage: $0 download_checkpoint <checkpoint_name> [output_dir]"
        return 1
    fi

    checkpoint_name=$1
    output_dir=${2:-"./downloads/checkpoints"}

    mkdir -p "$output_dir"
    echo "Downloading checkpoint: $checkpoint_name"

    modal volume get $VOLUME_NAME "checkpoints/$checkpoint_name" "$output_dir/$checkpoint_name"

    if [ $? -eq 0 ]; then
        echo "Download complete!"
        echo "Checkpoint saved to: $output_dir/$checkpoint_name"
    else
        echo "Download failed"
        return 1
    fi
}

function volume_info() {
    echo "=== Modal Volume Information ==="
    modal volume list | grep $VOLUME_NAME
    echo ""
    echo "=== Volume Contents ==="
    echo "Data:"
    modal volume ls $VOLUME_NAME data 2>/dev/null || echo "  (empty or no access)"
    echo ""
    echo "Checkpoints:"
    modal volume ls $VOLUME_NAME checkpoints 2>/dev/null || echo "  (empty or no access)"
    echo ""
    echo "Outputs:"
    modal volume ls $VOLUME_NAME outputs 2>/dev/null || echo "  (empty or no access)"
}

case "$1" in
    list-experiments|list-exp|le)
        list_experiments
        ;;
    list-checkpoints|list-ckpt|lc)
        list_checkpoints
        ;;
    download-experiment|dl-exp|de)
        download_experiment "$2" "$3"
        ;;
    download-checkpoint|dl-ckpt|dc)
        download_checkpoint "$2" "$3"
        ;;
    info|volume-info|vi)
        volume_info
        ;;
    clean)
        clean_checkpoints
        ;;
    *)
esac
