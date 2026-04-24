#!/bin/bash

# Full soft-Lipschitz MNIST pipeline:
#   1. meta-train the SIREN backbone with the soft-Lipschitz penalty,
#   2. fit per-sample modulations into a functaset (matching variant flags),
#   3. train the downstream parameter-space classifier on that functaset,
#   4. evaluate reconstruction quality (MSE / PSNR / SSIM) vs inner-loop iter count,
#   5. run Full-PGD adversarial attack on the classifier (signal-domain epsilon).
#
# Output paths are namespaced by the variant slug so parallel runs with
# different hyperparameters do not collide with each other or with the
# vanilla pipeline. See context/CHANGES.md Section 7 for the meaning of L.

set -euo pipefail

# ---- hyperparameters (edit these) --------------------------------------------
L=0.5                 # per-layer Lipschitz budget L.
                      # Sine layers get sigma-cap L/freq, readout gets L.
                      # L=1 is tight at freq=30; L=30 ~= pre-change defaults.
LAM=1              # penalty weight lambda.
APPLY_TO=all    # sine_only | sine_and_readout | all
SKIP_FIRST=1          # 1 = exclude sine.0 from the penalty (sigma(W_0)
                      # does not enter the phi->output Lipschitz bound
                      # and its large coord-input init otherwise dominates
                      # the penalty budget). 0 = include all sine layers.
EPOCHS=6              # outer meta-training epochs.
MAKESET_ITERS=5       # per-sample inner-loop iters when fitting modulations.
CUDA_GPU=1            # which GPU (sets CUDA_VISIBLE_DEVICES).
LOG_SIGMAS_EVERY=50   # emit per-layer spectral-norm line every N outer batches.
                      # 0 = off. Costs ~12 * 30 small matmuls per emission.
EVAL_ITERS="5,20,50,100,200"   # recon-eval inner-loop snapshots. Leftmost
                      # matches MAKESET_ITERS (apples-to-apples with classifier);
                      # rightmost = saturated backbone capacity.
EVAL_MAX_SAMPLES=2000 # per-split cap for the eval step. Empty = full split.
                      # Batched eval (--eval-batch-size) makes full-split
                      # runs cheap: ~30 samples/s * batch amortization on
                      # RTX 2080 Ti for MNIST at iters<=200.
EVAL_BATCH_SIZE=128   # inner-loop batch size for evaluate_reconstruction.
                      # 32-128 saturates a 10 GB GPU for MNIST; 1 = strict
                      # per-image (~10x slower, needed for LBFGS parity).
PGD_EPSILON=16        # signal-domain l_inf budget: eps/255 (de-facto standard
                      # MNIST attack is 16/255; use 8/255 for a harder budget).
PGD_STEPS=100         # outer PGD iterations per sample.
PGD_MOD_STEPS=10      # inner modulation-refit steps per PGD iteration
                      # (unrolled through higher for differentiable attack).
PGD_LR=0.01           # PGD step size (Adam lr on the perturbation tensor).
PGD_INNER_LR=0.01     # inner-loop modulation lr (should match MAKESET_ITERS lr
                      # so clean / attacked paths use the same fitter).
# -----------------------------------------------------------------------------

# Slug must mirror variants/soft_lipschitz.py :: SoftLipschitz.slug()
# Python uses "{L:g}" (strips trailing zeros) and "{lam:.0e}" (zero-padded exp).
L_TAG=$(printf "%g" "$L")
LAM_TAG=$(printf "%.0e" "$LAM")
SLUG="softlip_L${L_TAG}_lam${LAM_TAG}_${APPLY_TO}"
if [[ "${SKIP_FIRST}" -eq 1 ]]; then
    SLUG="${SLUG}_skip0"
fi

# Collect variant flags once so every stage (trainer / makeset / eval) sees
# the identical variant configuration.
VARIANT_FLAGS=(
    --variant soft_lipschitz
    --soft-lip-cap "${L}"
    --soft-lip-lambda "${LAM}"
    --soft-lip-apply-to "${APPLY_TO}"
)
if [[ "${SKIP_FIRST}" -eq 1 ]]; then
    VARIANT_FLAGS+=(--soft-lip-skip-first)
fi

MODEL_DIR="model_mnist/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "== soft-Lipschitz MNIST pipeline =="
echo "   L            = ${L}   (sine sigma-cap = L/freq, readout sigma-cap = L)"
echo "   lambda       = ${LAM}"
echo "   apply_to     = ${APPLY_TO}"
echo "   skip_first   = ${SKIP_FIRST}   (1 = exclude sine.0 from penalty)"
echo "   slug         = ${SLUG}"
echo "   CUDA device  = ${CUDA_VISIBLE_DEVICES}"
echo "   checkpoint   = ${CHECKPOINT}"
echo "   run root     = ${RUN_ROOT}"
echo "   PGD: eps=${PGD_EPSILON}/255, steps=${PGD_STEPS}, mod_steps=${PGD_MOD_STEPS}, lr=${PGD_LR}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training soft-Lipschitz SIREN backbone"
echo "          -> ${CHECKPOINT}"
python trainer.py \
    --dataset mnist \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
    "${VARIANT_FLAGS[@]}" \
    --log-sigmas-every "${LOG_SIGMAS_EVERY}"

echo
echo "Step 2/5: Creating functaset"
echo "          -> ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"
# makeset's split() calls os.makedirs without exist_ok, so wipe any partial run.
rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"
python makeset.py \
    --dataset mnist \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 3/5: Training downstream classifier"
echo "          -> ${RUN_ROOT}/mnist_classifier/best_classifier.pth"
# Run from RUN_ROOT so train_classifier's default mnist_classifier/ lands
# inside the variant's run dir instead of at SIREN_Vista root.
pushd "${RUN_ROOT}" > /dev/null
python ~/SIREN_Vista/train_classifier.py \
    --lr 0.01 \
    --cwidth 512 \
    --mod-dim 512 \
    --dropout 0.2 \
    --cdepth 3 \
    --batch-size 256 \
    --dataset mnist \
    --num-epochs 40 \
    --data-path ~/data \
    --functaset-path-train ./functaset/mnist_train.pkl \
    --functaset-path-test ./functaset/mnist_test.pkl \
    --device cuda
popd > /dev/null

echo
echo "Step 4/5: Evaluating reconstruction quality (MSE / PSNR / SSIM)"
echo "          -> ${RUN_ROOT}/reconstruction_eval.json"
EVAL_CAP_ARGS=()
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    EVAL_CAP_ARGS=(--max-samples "${EVAL_MAX_SAMPLES}")
fi
python evaluate_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset mnist \
    --data-path ../data \
    --device cuda \
    "${VARIANT_FLAGS[@]}" \
    --iter-checkpoints "${EVAL_ITERS}" \
    --split both \
    --inner-lr 0.01 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --output "${RUN_ROOT}/reconstruction_eval.json" \
    "${EVAL_CAP_ARGS[@]}"

echo
echo "Step 5/5: Running Full-PGD adversarial attack"
echo "          -> ${RUN_ROOT}/pgd_attack.log"
# full_pgd.py has no variant flags (it loads only 'state_dict' from the
# checkpoint; soft-Lipschitz makes no architectural change so the raw
# ModulatedSIREN ctor in the attack script is sufficient). Classifier
# checkpoint is the best_classifier.pth produced by Step 3.
CLASSIFIER_CKPT="${RUN_ROOT}/mnist_classifier/best_classifier.pth"
PGD_LOG="${RUN_ROOT}/pgd_attack.log"
# Run unbuffered (-u) so tqdm lines appear live in the tee'd log.
python -u attacks/full_pgd.py \
    --dataset mnist \
    --data-path ../data \
    --siren-checkpoint "${CHECKPOINT}" \
    --classifier-checkpoint "${CLASSIFIER_CKPT}" \
    --epsilon "${PGD_EPSILON}" \
    --pgd-steps "${PGD_STEPS}" \
    --mod-steps "${PGD_MOD_STEPS}" \
    --ext-lr "${PGD_LR}" \
    --inner-lr "${PGD_INNER_LR}" \
    --cwidth 512 \
    --cdepth 3 \
    --mod-dim 512 \
    --hidden-dim 256 \
    --depth 10 \
    --device cuda 2>&1 | tee "${PGD_LOG}"

echo
echo "Done."
echo "SIREN checkpoint : ${CHECKPOINT}"
echo "Functaset        : ${RUN_ROOT}/functaset/mnist_{train,val,test}.pkl"
echo "Classifier       : ${RUN_ROOT}/mnist_classifier/best_classifier.pth"
echo "Recon eval JSON  : ${RUN_ROOT}/reconstruction_eval.json"
echo "PGD attack log   : ${PGD_LOG}"
