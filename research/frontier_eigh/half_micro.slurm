#!/bin/bash
# From repository root: mkdir -p output/custom-eigh
# module reset; module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
# bash research/frontier_eigh/half_micro_build.sh
# sbatch research/frontier_eigh/half_micro.slurm   # human approval only
# Select a frozen executable with HALF_MICRO_BIN=build/custom-eigh/half_micro-baseline.
# Optional backend selection trace (not competitive timing):
# ROCBLAS_LAYER=4 sbatch --export=ALL research/frontier_eigh/half_micro.slurm
# Hypotheses: TN back1 doubles already-long k in the correction launch and
# changes lda/ldb; NN back2 and NT panel are small-k/output-bandwidth sensitive,
# so a second full FP32 C read/write + launch may erase MFMA throughput gains.
# Compare gemm vs pack+gemm before pursuing cached packing or fused correction.
# Pooled buffers omit allocation/communication and cannot predict whole-solver speed.
#SBATCH -A gen150
#SBATCH -J half-micro
#SBATCH -p batch
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH -t 00:08:00
#SBATCH -o output/custom-eigh/half-micro-%j.out
#SBATCH -e output/custom-eigh/half-micro-%j.err
set -euo pipefail
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HALF_MICRO_BIN=${HALF_MICRO_BIN:-build/custom-eigh/half_micro}
echo "JOB $SLURM_JOB_ID $(date -Is) nodes=$SLURM_JOB_NODELIST"
module list 2>&1
git rev-parse HEAD
sha256sum research/frontier_eigh/half_micro.cpp research/frontier_eigh/half_fused.hpp "$HALF_MICRO_BIN"
hipcc --version
ldd "$HALF_MICRO_BIN"
while read -r lib; do
    realpath "$lib"; sha256sum "$lib"
done < <(ldd "$HALF_MICRO_BIN" | awk '/librocblas|libamdhip64|libhipblaslt/ {print $3}')
env | sort | grep -E '^(ROCBLAS|TENSILE|HIP|ROCR|ROCM|HSA|SLURM|LD_LIBRARY_PATH)' || true
set -x
srun -N1 -n1 --ntasks-per-node=1 -c7 --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 \
    "$HALF_MICRO_BIN" --warmup 3 --repeat 10 "$@"
