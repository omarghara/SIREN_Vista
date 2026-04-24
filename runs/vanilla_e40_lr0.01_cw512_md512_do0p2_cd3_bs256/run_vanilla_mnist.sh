#!/bin/bash

set -euo pipefail


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/../data"

# README MNIST clean-classifier defaults:
# --lr 0.01 --cwidth 512 --mod-dim 512 --dropout 0.2 --cdepth 3 --batch-size 256 --num-epochs 40
LR="${LR:-0.01}"
CWIDTH="${CWIDTH:-512}"
MOD_DIM="${MOD_DIM:-512}"
DROPOUT="${DROPOUT:-0.2}"
CDEPTH="${CDEPTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"

SIREN_CKPT="$REPO_ROOT/model_mnist/modSiren.pth"
FUNCTASET_TRAIN="$REPO_ROOT/functaset/mnist_train.pkl"
FUNCTASET_TEST="$REPO_ROOT/functaset/mnist_test.pkl"
RAW_CLASSIFIER_CKPT="$REPO_ROOT/mnist_classifier/best_classifier.pth"

OUTPUT_BASE_DIR="/workspace/SIREN_Vista/model_mnist"
RUN_NAME="vanilla_e${NUM_EPOCHS}_lr${LR}_cw${CWIDTH}_md${MOD_DIM}_do${DROPOUT//./p}_cd${CDEPTH}_bs${BATCH_SIZE}"
RUN_OUTPUT_DIR="$OUTPUT_BASE_DIR/$RUN_NAME"
CLASSIFIER_CKPT="$OUTPUT_BASE_DIR/${RUN_NAME}.pth"

cd "$REPO_ROOT" || exit 1
mkdir -p "$RUN_OUTPUT_DIR"

echo "Starting vanilla MNIST pipeline..."
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
  python -c "import torch; print('cuda available:', True); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"
else
  DEVICE="cpu"
  echo "CUDA not available; falling back to CPU."
fi

echo "Step 1/3: Training shared SIREN"
python trainer.py --dataset mnist --data-path "$DATA_DIR" --device "$DEVICE"

echo "Step 2/3: Creating functaset"
python makeset.py --dataset mnist --data-path "$DATA_DIR" --iters 5 --checkpoint "$SIREN_CKPT" --device "$DEVICE"

echo "Step 3/3: Training vanilla classifier"
python train_classifier.py \
  --lr "$LR" \
  --cwidth "$CWIDTH" \
  --mod-dim "$MOD_DIM" \
  --dropout "$DROPOUT" \
  --cdepth "$CDEPTH" \
  --batch-size "$BATCH_SIZE" \
  --dataset mnist \
  --num-epochs "$NUM_EPOCHS" \
  --data-path "$DATA_DIR" \
  --functaset-path-train "$FUNCTASET_TRAIN" \
  --functaset-path-test "$FUNCTASET_TEST" \
  --device "$DEVICE"

if [ -f "$RAW_CLASSIFIER_CKPT" ]; then
  cp "$RAW_CLASSIFIER_CKPT" "$CLASSIFIER_CKPT"
  cp "$RAW_CLASSIFIER_CKPT" "$RUN_OUTPUT_DIR/best_classifier.pth"
  [ -f "$REPO_ROOT/mnist_classifier/classifier_loss.npy" ] && cp "$REPO_ROOT/mnist_classifier/classifier_loss.npy" "$RUN_OUTPUT_DIR/"
  [ -f "$REPO_ROOT/mnist_classifier/classifier_acc.npy" ] && cp "$REPO_ROOT/mnist_classifier/classifier_acc.npy" "$RUN_OUTPUT_DIR/"
else
  echo "Warning: expected classifier checkpoint not found at $RAW_CLASSIFIER_CKPT"
fi

echo "Done."
echo "SIREN checkpoint: $SIREN_CKPT"
echo "Functaset: $FUNCTASET_TRAIN $FUNCTASET_TEST"
echo "Classifier checkpoint: $CLASSIFIER_CKPT"
echo "Run artifacts directory: $RUN_OUTPUT_DIR"