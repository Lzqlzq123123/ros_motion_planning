#!/bin/bash
# Launch TensorBoard for monitoring PPO training

TENSORBOARD_DIR=$1

if [ -z "$TENSORBOARD_DIR" ]; then
    echo "Usage: $0 <tensorboard_log_directory>"
    exit 1
fi

echo "Launching TensorBoard with log directory: $TENSORBOARD_DIR"
echo "TensorBoard will be available at: http://localhost:6006"

# Create directory if it doesn't exist
mkdir -p $TENSORBOARD_DIR

# Launch TensorBoard
tensorboard --logdir=$TENSORBOARD_DIR --port=6006 --host=0.0.0.0