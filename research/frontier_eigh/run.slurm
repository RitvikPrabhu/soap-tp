#!/bin/bash
#SBATCH -A gen150
#SBATCH -J custom-eigh
#SBATCH -p batch
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -o output/custom-eigh/job-%j.out
#SBATCH -e output/custom-eigh/job-%j.err
set -euo pipefail
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/build/elpa-install/rocm/lib:$PWD/build/math-install/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
echo "JOB $SLURM_JOB_ID $(date -Is) nodes=$SLURM_JOB_NODELIST suite=${1:-smoke}"
run() {
    local ranks=$1; shift
    echo "COMMAND ranks=$ranks $*"
    srun -N1 -n "$ranks" --ntasks-per-node="$ranks" -c7 --gpus-per-task=1 \
        --gpu-bind=closest --kill-on-bad-exit=1 "${EIGH_BIN:-build/custom-eigh/eigh}" "$@"
}
if [[ ${1:-smoke} == smoke ]]; then
    for grid in '1 1' '1 2' '2 1' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run "$((pr*pc))" --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision"
        done
    done
    for matrix in diagonal clustered lowrank; do
        run 8 --n 97 --block 8 --pr 2 --pc 4 --matrix "$matrix"
    done
elif [[ $1 == baseline ]]; then
    for grid in '1 1' '1 2' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run "$((pr*pc))" --n 2048 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --repeat 2
        done
    done
    for solver in elpa1 elpa2; do
        for precision in fp64 fp32; do
            run 8 --n 2048 --block 64 --pr 2 --pc 4 --solver "$solver" --precision "$precision" --repeat 2
        done
    done
    for block in 32 128 256; do
        run 8 --n 2048 --block "$block" --pr 2 --pc 4 --repeat 2
    done
    run 8 --n 2048 --block 64 --pr 2 --pc 4 --gpu-mpi 1 --repeat 2 --reference 1
elif [[ $1 == case ]]; then
    shift
    ranks=$1; shift
    run "$ranks" "$@"
else
    echo "unknown suite" >&2; exit 2
fi
echo "FINISHED $(date -Is)"
