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
    sha256sum "${EIGH_BIN:-build/custom-eigh/eigh}"
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
elif [[ $1 == precision ]]; then
    for matrix in random lowrank indefinite clustered; do
        run 8 --n 257 --block 16 --pr 2 --pc 4 --matrix "$matrix" --precision fp64
        for quant in none fp16 bf16; do
            run 8 --n 257 --block 16 --pr 2 --pc 4 --matrix "$matrix" --precision fp32 --quant "$quant"
        done
    done
elif [[ $1 == active ]]; then
    for n in 2048 4096; do
        for grid in '2 4' '4 2'; do
            read -r pr pc <<< "$grid"
            for precision in fp64 fp32; do
                EIGH_BIN=build/custom-eigh/baseline-eigh run 8 --n "$n" --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --gpu-mpi 1 --repeat 2
                run 8 --n "$n" --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --gpu-mpi 1 --repeat 2
            done
        done
    done
elif [[ $1 == wy ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --gpu-mpi 1
        done
    done
    for matrix in random lowrank indefinite clustered; do
        run 8 --n 257 --block 16 --pr 2 --pc 4 --matrix "$matrix" --precision fp64 --back-block 32 --gpu-mpi 1
        for quant in none fp16 bf16; do
            run 8 --n 257 --block 16 --pr 2 --pc 4 --matrix "$matrix" --precision fp32 --quant "$quant" --back-block 32 --gpu-mpi 1
        done
    done
    for precision in fp64 fp32; do
        for back in 1 16 32 64 128; do
            run 8 --n 4096 --block 64 --pr 2 --pc 4 --precision "$precision" --back-block "$back" --gpu-mpi 1 --repeat 3
        done
        run 8 --n 4096 --block 64 --pr 2 --pc 4 --precision "$precision" --solver elpa2 --repeat 3
    done
    run 8 --n 4096 --block 64 --pr 2 --pc 4 --back-block 32 --gpu-mpi 0 --repeat 3
elif [[ $1 == panel ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --panel 7 --gpu-mpi 1
        done
    done
    for precision in fp64 fp32; do
        for panel in 1 8 16 32 64; do
            run 8 --n 4096 --block 64 --pr 2 --pc 4 --precision "$precision" --back-block 32 --panel "$panel" --gpu-mpi 1 --repeat 3
        done
    done
    for panel in 1 32; do
        run 8 --n 8192 --block 64 --pr 2 --pc 4 --back-block 128 --panel "$panel" --gpu-mpi 1 --repeat 2 --reference 1
    done
elif [[ $1 == components ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --back-block 13 --tridiag elpa
    done
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --back-block 13 --gpu-mpi 1 --rccl 1
    done
    for tridiag in scalapack elpa; do
        for transport in '0 0' '1 0' '1 1'; do
            read -r gpu rccl <<< "$transport"
            run 8 --n 4096 --block 64 --pr 2 --pc 4 --back-block 128 --tridiag "$tridiag" --gpu-mpi "$gpu" --rccl "$rccl" --repeat 3
        done
    done
    for gpu in 0 1; do
        run 8 --n 8192 --block 64 --pr 2 --pc 4 --back-block 128 --tridiag elpa --gpu-mpi "$gpu" --repeat 2 --reference 1
    done
    for matrix in random lowrank; do
        for panel in 1 16; do
            if run 8 --n 257 --block 16 --pr 2 --pc 4 --back-block 32 --precision fp32 --half-bulk 1 --matrix "$matrix" --panel "$panel"; then
                echo "HALF_BULK_CHECK passed matrix=$matrix panel=$panel"
            else
                echo "HALF_BULK_CHECK FAILED matrix=$matrix panel=$panel; inspect numerical metrics above"
            fi
        done
    done
elif [[ $1 == norm ]]; then
    for n in 4096 8192; do
        for binary in half-eigh norm-eigh; do
            EIGH_BIN=build/custom-eigh/$binary run 8 --n "$n" --block 64 --pr 2 --pc 4 --back-block 128 --tridiag elpa --repeat 3
        done
    done
    for matrix in random lowrank indefinite; do
        for half in 0 2; do
            EIGH_BIN=build/custom-eigh/split-eigh run 8 --n 257 --block 16 --pr 4 --pc 2 --back-block 32 --precision fp32 --half-bulk "$half" --matrix "$matrix"
        done
    done
    for half in 0 2; do
        EIGH_BIN=build/custom-eigh/split-eigh run 8 --n 8192 --block 64 --pr 2 --pc 4 --back-block 128 --tridiag elpa --precision fp32 --half-bulk "$half" --repeat 3 --reference 1
    done
elif [[ $1 == case ]]; then
    shift
    ranks=$1; shift
    run "$ranks" "$@"
else
    echo "unknown suite" >&2; exit 2
fi
echo "FINISHED $(date -Is)"
