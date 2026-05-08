#!/bin/bash

# Functa-like Fourier + Learnable Spectral Activation CIFAR-10 pipeline:
#   1. meta-train a ModulatedFourierLSA backbone,
#   2. create a CIFAR-10 functaset,
#   3. train downstream parameter-space classifier,
#   4. evaluate reconstruction quality,
#   5. skip PGD for now.

set -euo pipefail

# ---- architecture ------------------------------------------------------------

HIDDEN_DIM=512
MOD_DIM=1024
DEPTH=15

INR_TYPE=fourier_lsa
COORD_TAG="norm01"

FOURIER_NUM_FREQS=64
FOURIER_SIGMA=5.0
FOURIER_INCLUDE_INPUT=1

LSA_NUM_HARMONICS=8
LSA_INIT_SCALE=1e-3
LSA_NO_LINEAR=0

# ---- meta-training hyperparameters ------------------------------------------

EPOCHS=10

# Important: for this experiment, use Adam inner-loop.
# This should better match your diagnostic where Adam phi fitting worked.
INT_LR=0.003
INNER_STEPS=10
INNER_OPTIM=adam

EXT_LR=1e-5
TRAIN_BATCH_SIZE=32

CUDA_GPU=1
LOG_SIGMAS_EVERY=50

# ---- functaset fitting -------------------------------------------------------

MAKESET_ITERS=50
MAKESET_LR=0.003
MAKESET_INNER_OPTIM=adam

# ---- reconstruction evaluation ----------------------------------------------

EVAL_ITERS="5,10,20,50,100,200,500,1000"
EVAL_MAX_SAMPLES=2000
EVAL_BATCH_SIZE=64
EVAL_LR=0.003
EVAL_INNER_OPTIM=adam

RUN_PGD=0

# ---- classifier hyperparameters ---------------------------------------------

CLF_LR=0.01
CLF_WIDTH=1024
CLF_DEPTH=4
CLF_DROPOUT=0.3
CLF_BATCH_SIZE=256
CLF_EPOCHS=80

# -----------------------------------------------------------------------------

EXT_LR_TAG=$(printf "%.0e" "${EXT_LR}")
SIGMA_TAG=$(printf "%g" "${FOURIER_SIGMA}" | tr '.' 'p')
LSA_INIT_TAG=$(printf "%.0e" "${LSA_INIT_SCALE}")

if [[ "${FOURIER_INCLUDE_INPUT}" -eq 1 ]]; then
    FOURIER_INPUT_TAG="rawxy"
else
    FOURIER_INPUT_TAG="norawxy"
fi

if [[ "${LSA_NO_LINEAR}" -eq 1 ]]; then
    LSA_LINEAR_TAG="nolinear"
else
    LSA_LINEAR_TAG="linear"
fi

SLUG="functa_like_cifar10_fourier_lsa_h${HIDDEN_DIM}_md${MOD_DIM}_d${DEPTH}_nf${FOURIER_NUM_FREQS}_sig${SIGMA_TAG}_${FOURIER_INPUT_TAG}_K${LSA_NUM_HARMONICS}_init${LSA_INIT_TAG}_${LSA_LINEAR_TAG}_${COORD_TAG}_inner${INNER_STEPS}_${INNER_OPTIM}_ilr${INT_LR}_extlr${EXT_LR_TAG}_e${EPOCHS}_make${MAKESET_ITERS}_${MAKESET_INNER_OPTIM}"

VARIANT_FLAGS=(
    --variant vanilla
)

INR_FLAGS=(
    --inr-type "${INR_TYPE}"
    --fourier-num-freqs "${FOURIER_NUM_FREQS}"
    --fourier-sigma "${FOURIER_SIGMA}"
    --lsa-num-harmonics "${LSA_NUM_HARMONICS}"
    --lsa-init-scale "${LSA_INIT_SCALE}"
)

if [[ "${FOURIER_INCLUDE_INPUT}" -eq 1 ]]; then
    INR_FLAGS+=(--fourier-include-input)
fi

if [[ "${LSA_NO_LINEAR}" -eq 1 ]]; then
    INR_FLAGS+=(--lsa-no-linear)
fi

MODEL_DIR="model_cifar10/${SLUG}"
CHECKPOINT="${MODEL_DIR}/modSiren.pth"
RUN_ROOT="runs/${SLUG}"

source /home/omarg/miniforge3/etc/profile.d/conda.sh
conda activate pss

export CUDA_VISIBLE_DEVICES="${CUDA_GPU}"

cd ~/SIREN_Vista || exit 1

echo "== Functa-like Fourier-LSA CIFAR-10 pipeline =="
echo "   dataset              = cifar10"
echo "   inr_type             = ${INR_TYPE}"
echo "   hidden_dim           = ${HIDDEN_DIM}"
echo "   mod_dim              = ${MOD_DIM}"
echo "   depth                = ${DEPTH}"
echo "   coord norm           = ${COORD_TAG} pixel centers"
echo "   fourier num freqs    = ${FOURIER_NUM_FREQS}"
echo "   fourier sigma        = ${FOURIER_SIGMA}"
echo "   fourier raw xy       = ${FOURIER_INCLUDE_INPUT}"
echo "   LSA harmonics K      = ${LSA_NUM_HARMONICS}"
echo "   LSA init scale       = ${LSA_INIT_SCALE}"
echo "   LSA no linear        = ${LSA_NO_LINEAR}"
echo "   meta inner lr        = ${INT_LR}"
echo "   meta inner steps     = ${INNER_STEPS}"
echo "   meta inner optim     = ${INNER_OPTIM}"
echo "   outer lr             = ${EXT_LR}"
echo "   epochs               = ${EPOCHS}"
echo "   train batch          = ${TRAIN_BATCH_SIZE}"
echo "   make iters           = ${MAKESET_ITERS}"
echo "   make lr              = ${MAKESET_LR}"
echo "   make optim           = ${MAKESET_INNER_OPTIM}"
echo "   slug                 = ${SLUG}"
echo "   checkpoint           = ${CHECKPOINT}"
echo "   run root             = ${RUN_ROOT}"
echo

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print('gpu name:', torch.cuda.get_device_name(0))"

echo
echo "Step 1/5: Training Fourier-LSA CIFAR-10 backbone"
echo "          -> ${CHECKPOINT}"

python trainer.py \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    --num-epochs "${EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --int-lr "${INT_LR}" \
    --inner-steps "${INNER_STEPS}" \
    --inner-optim "${INNER_OPTIM}" \
    --ext-lr "${EXT_LR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --mod-dim "${MOD_DIM}" \
    --depth "${DEPTH}" \
    "${INR_FLAGS[@]}" \
    --model-name "${SLUG}" \
    "${VARIANT_FLAGS[@]}" \
    --log-sigmas-every "${LOG_SIGMAS_EVERY}"

echo
echo "Verifying saved checkpoint"
echo "          -> ${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: expected checkpoint was not created: ${CHECKPOINT}" >&2
    exit 1
fi

python - <<PYCKPT
import sys
import torch

ckpt_path = "${CHECKPOINT}"
expected_model_name = "${SLUG}"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("checkpoint metadata:")
print("  epoch:", ckpt.get("epoch"))
print("  loss:", ckpt.get("loss"))
print("  variant:", ckpt.get("variant"))
print("  model_name:", ckpt.get("model_name"))
print("  model_args:", ckpt.get("model_args"))

if ckpt.get("model_name") != expected_model_name:
    print(
        f"ERROR: checkpoint model_name={ckpt.get('model_name')!r} "
        f"does not match expected {expected_model_name!r}",
        file=sys.stderr,
    )
    sys.exit(1)

model_args = ckpt.get("model_args", {})

expected = {
    "inr_type": "fourier_lsa",
    "hidden_dim": ${HIDDEN_DIM},
    "mod_dim": ${MOD_DIM},
    "depth": ${DEPTH},
    "fourier_num_freqs": ${FOURIER_NUM_FREQS},
    "lsa_num_harmonics": ${LSA_NUM_HARMONICS},
}

for k, v in expected.items():
    if model_args.get(k) != v:
        print(f"ERROR: model_args[{k!r}]={model_args.get(k)!r}, expected {v!r}", file=sys.stderr)
        sys.exit(1)

if abs(float(model_args.get("fourier_sigma")) - float(${FOURIER_SIGMA})) > 1e-9:
    print("ERROR: wrong fourier_sigma:", model_args.get("fourier_sigma"), file=sys.stderr)
    sys.exit(1)

print("checkpoint verification OK")
PYCKPT

echo
echo "Step 2/5: Creating CIFAR-10 functaset"
echo "          -> ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"

rm -rf "${RUN_ROOT}/functaset"
mkdir -p "${RUN_ROOT}"

python makeset.py \
    --dataset cifar10 \
    --data-path ../data \
    --iters "${MAKESET_ITERS}" \
    --lr "${MAKESET_LR}" \
    --inner-optim "${MAKESET_INNER_OPTIM}" \
    --checkpoint "${CHECKPOINT}" \
    --saveroot "${RUN_ROOT}" \
    --device cuda \
    --functaset-stem "${SLUG}" \
    "${INR_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}"

echo
echo "Step 3/5: Training downstream CIFAR-10 classifier"
echo "          -> ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"

pushd "${RUN_ROOT}" > /dev/null

python ~/SIREN_Vista/train_classifier.py \
    --lr "${CLF_LR}" \
    --cwidth "${CLF_WIDTH}" \
    --mod-dim "${MOD_DIM}" \
    --dropout "${CLF_DROPOUT}" \
    --cdepth "${CLF_DEPTH}" \
    --batch-size "${CLF_BATCH_SIZE}" \
    --dataset cifar10 \
    --num-epochs "${CLF_EPOCHS}" \
    --data-path ~/data \
    --functaset-path-train "./functaset/${SLUG}_train.pkl" \
    --functaset-path-test "./functaset/${SLUG}_test.pkl" \
    --device cuda

popd > /dev/null

echo
echo "Step 4/5: Evaluating CIFAR-10 reconstruction quality"
echo "          -> ${RUN_ROOT}/reconstruction_eval.json"

EVAL_CAP_ARGS=()
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    EVAL_CAP_ARGS=(--max-samples "${EVAL_MAX_SAMPLES}")
fi

python evaluate_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset cifar10 \
    --data-path ../data \
    --device cuda \
    "${INR_FLAGS[@]}" \
    "${VARIANT_FLAGS[@]}" \
    --iter-checkpoints "${EVAL_ITERS}" \
    --split both \
    --inner-lr "${EVAL_LR}" \
    --inner-optim "${EVAL_INNER_OPTIM}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --output "${RUN_ROOT}/reconstruction_eval.json" \
    "${EVAL_CAP_ARGS[@]}"

echo
echo "Step 5/5: Full-PGD adversarial attack"
echo "          skipped because RUN_PGD=0"

echo
echo "Done."
echo "Fourier-LSA checkpoint : ${CHECKPOINT}"
echo "Functaset              : ${RUN_ROOT}/functaset/${SLUG}_{train,val,test}.pkl"
echo "Classifier             : ${RUN_ROOT}/cifar10_classifier/best_classifier.pth"
echo "Recon eval JSON        : ${RUN_ROOT}/reconstruction_eval.json"