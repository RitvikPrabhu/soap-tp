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
    echo "COMMAND ranks=$ranks OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS $*"
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
elif [[ $1 == transport ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --tridiag elpa --gpu-mpi 1 --rccl 1
        done
    done
    for binary in split-eigh async-eigh gather-eigh; do
        for transport in '0 0' '1 0' '1 1'; do
            read -r gpu rccl <<< "$transport"
            EIGH_BIN=build/custom-eigh/$binary run 8 --n 8192 --block 64 --pr 2 --pc 4 --back-block 128 --tridiag elpa --gpu-mpi "$gpu" --rccl "$rccl" --repeat 3
        done
    done
elif [[ $1 == tune ]]; then
    for block in 32 64 128 256; do
        run 8 --n 8192 --block "$block" --pr 2 --pc 4 --back-block 128 --tridiag elpa --gpu-mpi 1 --rccl 1 --repeat 3
    done
    for precision in fp64 fp32; do
        for back in 32 64 128 256; do
            [[ $precision == fp64 && $back == 128 ]] && continue
            run 8 --n 8192 --block 64 --pr 2 --pc 4 --back-block "$back" --tridiag elpa --gpu-mpi 1 --rccl 1 --precision "$precision" --repeat 3
        done
    done
    for matrix in random lowrank; do
        run 8 --n 257 --block 16 --pr 2 --pc 4 --back-block 32 --matrix "$matrix"
        for quant in none fp16 bf16; do
            run 8 --n 257 --block 16 --pr 2 --pc 4 --back-block 32 --matrix "$matrix" --precision fp32 --quant "$quant"
        done
    done
    for solver in elpa1 elpa2; do
        run 8 --n 8192 --block 64 --pr 2 --pc 4 --solver "$solver" --elpa-times 1 --repeat 2
    done
elif [[ $1 == band ]]; then
    for grid in '1 1' '1 2' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run "$((pr*pc))" --n 97 --block 8 --band 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --gpu-mpi 1 --rccl 1
        done
    done
    for matrix in diagonal clustered lowrank indefinite; do
        run 8 --n 97 --block 8 --band 8 --pr 2 --pc 4 --matrix "$matrix" --back-block 13
    done
    for n in 4096 8192; do
        for precision in fp64 fp32; do
            run 8 --n "$n" --block 64 --pr 2 --pc 4 --precision "$precision" --back-block 128 --tridiag elpa --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
            for band in 32 64 128; do
                run 8 --n "$n" --block "$band" --band "$band" --pr 2 --pc 4 --precision "$precision" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
            done
        done
    done
    if run 8 --n 8192 --block 64 --band 64 --pr 2 --pc 4 --precision fp32 --back-block 128 --gpu-mpi 1 --rccl 1 --half-bulk 2 --repeat 3 --reference 1; then
        echo 'SPLIT_BAND_CHECK passed'
    else
        echo 'SPLIT_BAND_CHECK FAILED; inspect metrics'
    fi
elif [[ $1 == blas-panel ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --band 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --gpu-mpi 1 --rccl 1
        done
    done
    for n in 4096 8192; do
        for precision in fp64 fp32; do
            for binary in band-eigh blas-panel-eigh; do
                EIGH_BIN=build/custom-eigh/$binary run 8 --n "$n" --block 32 --band 32 --pr 2 --pc 4 --precision "$precision" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
            done
        done
    done
    for band in 16 64; do
        for precision in fp64 fp32; do
            run 8 --n 8192 --block "$band" --band "$band" --pr 2 --pc 4 --precision "$precision" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
        done
    done
    for solver in elpa1 elpa2; do
        for precision in fp64 fp32; do
            run 8 --n 8192 --block 64 --pr 2 --pc 4 --solver "$solver" --precision "$precision" --repeat 3
        done
    done
    if run 8 --n 8192 --block 32 --band 32 --pr 2 --pc 4 --precision fp32 --back-block 128 --gpu-mpi 1 --rccl 1 --half-bulk 2 --repeat 3 --reference 1; then
        echo 'SPLIT_BLAS_BAND_CHECK passed'
    else
        echo 'SPLIT_BLAS_BAND_CHECK FAILED; inspect metrics'
    fi
elif [[ $1 == scaling ]]; then
    for grid in '1 1' '1 2' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for matrix in random lowrank; do
            run "$((pr*pc))" --n 97 --block 8 --pr "$pr" --pc "$pc" --precision fp32 --matrix "$matrix" --tridiag elpa32 --back-block 13 --gpu-mpi 1 --rccl 1
        done
    done
    for grid in '1 1' '1 2' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tridiag=elpa
            [[ $precision == fp32 ]] && tridiag=elpa32
            reference=0
            [[ $grid == '2 4' || $grid == '4 2' ]] && reference=1
            run "$((pr*pc))" --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference "$reference"
        done
    done
    run 8 --n 8192 --block 64 --pr 2 --pc 4 --precision fp32 --tridiag elpa --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
    run 8 --n 8192 --block 64 --pr 2 --pc 4 --precision fp32 --tridiag elpa32 --back-block 128 --gpu-mpi 1 --rccl 1 --half-bulk 2 --repeat 3 --reference 1
    for precision in fp64 fp32; do
        run 1 --n 8192 --block 64 --pr 1 --pc 1 --precision "$precision" --solver rocsolver --repeat 3
    done
elif [[ $1 == resident ]]; then
    for grid in '1 1' '1 2' '2 1' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run "$((pr*pc))" --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --tridiag elpa
        done
    done
    for matrix in diagonal clustered lowrank indefinite; do
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr 2 --pc 4 --precision "$precision" --matrix "$matrix" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --tridiag elpa
        done
    done
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tridiag=elpa
            [[ $precision == fp32 ]] && tridiag=elpa32
            EIGH_BIN=build/custom-eigh/tri32-eigh run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3
            run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
        done
    done
    EIGH_BIN=build/custom-eigh/device-vectors-eigh run 8 --n 8192 --block 64 --pr 2 --pc 4 --tridiag elpa --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        EIGH_BIN=build/custom-eigh/tri32-eigh run 8 --n 16384 --block 64 --pr 4 --pc 2 --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3
        run 8 --n 16384 --block 64 --pr 4 --pc 2 --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
        for solver in elpa1 elpa2; do
            run 8 --n 16384 --block 64 --pr 4 --pc 2 --precision "$precision" --solver "$solver" --repeat 3
        done
        run 8 --n 16384 --block 64 --band 64 --pr 4 --pc 2 --precision "$precision" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
    done
elif [[ $1 == indexed ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tridiag=elpa
            [[ $precision == fp32 ]] && tridiag=elpa32
            for vectors in 0 1; do
                run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --tri-pr 8 --back-block 13 --device-vectors "$vectors" --gpu-mpi 1 --rccl 1
            done
        done
    done
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        for binary in device-active-eigh device-indexed-eigh; do
            EIGH_BIN=build/custom-eigh/$binary run 8 --n 8192 --block 64 --pr 2 --pc 4 --precision "$precision" --tridiag "$tridiag" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
        done
        for grid in '1 8' '2 4' '4 2' '8 1'; do
            read -r pr pc <<< "$grid"
            if [[ $pr != 2 ]]; then
                run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
            fi
            run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --solver elpa1 --repeat 3
        done
        EIGH_BIN=build/custom-eigh/tri32-eigh run 8 --n 8192 --block 64 --pr 1 --pc 8 --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3
        for vectors in 0 1; do
            run 8 --n 8192 --block 64 --pr 1 --pc 8 --precision "$precision" --tridiag "$tridiag" --tri-pr 8 --back-block 128 --device-vectors "$vectors" --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
        done
    done
elif [[ $1 == packed ]]; then
    for grid in '1 1' '1 2' '2 2' '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run "$((pr*pc))" --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --gpu-mpi 1 --rccl 1
        done
    done
    for transport in '0 0' '1 0' '1 1'; do
        read -r gpu rccl <<< "$transport"
        for back in 1 13; do
            run 8 --n 97 --block 8 --pr 4 --pc 2 --back-block "$back" --gpu-mpi "$gpu" --rccl "$rccl"
        done
    done
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        for grid in '1 8' '2 4' '4 2'; do
            read -r pr pc <<< "$grid"
            for binary in retile-eigh packed-eigh; do
                EIGH_BIN=build/custom-eigh/$binary run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --precision "$precision" --tridiag "$tridiag" --tri-pr 8 --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
            done
        done
    done
    for vectors in 0 1; do
        echo "PROFILE rank=0 only vectors=$vectors; instrumentation is not a competitive timing"
        export PROFILE_OUT="$PWD/output/custom-eigh/profile-$SLURM_JOB_ID-v$vectors"
        mkdir -p "$PROFILE_OUT"
        echo "COMMAND ranks=8 instrumented=rocprofv3-rank0 --n 8192 --block 64 --pr 4 --pc 2 --tridiag elpa --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors $vectors"
        sha256sum "${EIGH_BIN:-build/custom-eigh/eigh}"
        srun -N1 -n8 --ntasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 \
            bash -c 'if [[ $SLURM_PROCID == 0 ]]; then exec rocprofv3 --kernel-trace --memory-copy-trace --stats -d "$PROFILE_OUT" -o rank0 -- "$@"; else exec "$@"; fi' \
            _ "${EIGH_BIN:-build/custom-eigh/eigh}" --n 8192 --block 64 --pr 4 --pc 2 --tridiag elpa --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors "$vectors"
    done
elif [[ $1 == twod ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            for vectors in 0 1; do
                run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --device-vectors "$vectors" --gpu-mpi 1 --rccl 1
            done
        done
    done
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        for vectors in 0 1; do
            pr=4; pc=2; tri_pr=4
            if [[ $vectors == 1 ]]; then pr=1; pc=8; tri_pr=8; fi
            for binary in basis-eigh twod-eigh; do
                EIGH_BIN=build/custom-eigh/$binary run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --tri-pr "$tri_pr" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors "$vectors" --repeat 3 --reference 1
            done
            run 8 --n 16384 --block 64 --pr "$pr" --pc "$pc" --tri-pr "$tri_pr" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors "$vectors" --repeat 3 --reference 1
        done
    done
elif [[ $1 == soap ]]; then
    # Independent singleton MPI solves, restricted to tiny correctness examples on one GCD.
    export ROCR_VISIBLE_DEVICES=0
    destination="output/custom-eigh/soap-$SLURM_JOB_ID"
    mkdir -p "$destination"
    for seed in 42 43 44; do
        for condition in 100 1000000 1000000000; do
            for n in 96 64; do
                stem="s$seed-c$condition-n$n"
                for variant in fp64 fp32 fp16_input bf16_input split_half; do
                    precision=fp32; tridiag=elpa32; extra=()
                    [[ $variant == fp64 ]] && precision=fp64 && tridiag=elpa
                    [[ $variant == fp16_input || $variant == bf16_input ]] && extra=(--quant "${variant%_input}")
                    [[ $variant == split_half ]] && extra=(--half-bulk 2)
                    echo "COMMAND ranks=1 basis_case=$stem variant=$variant --n $n --block 16 --back-block 32 --input output/custom-eigh/soap-input/$stem.bin --export-basis $destination/$stem-$variant.bin --precision $precision --tridiag $tridiag ${extra[*]}"
                    sha256sum "${EIGH_BIN:-build/custom-eigh/eigh}"
                    "${EIGH_BIN:-build/custom-eigh/eigh}" --n "$n" --block 16 --back-block 32 \
                        --input "output/custom-eigh/soap-input/$stem.bin" --export-basis "$destination/$stem-$variant.bin" \
                        --matrix supplied --precision "$precision" --tridiag "$tridiag" "${extra[@]}"
                done
            done
        done
    done
elif [[ $1 == case ]]; then
    shift
    ranks=$1; shift
    run "$ranks" "$@"
else
    echo "unknown suite" >&2; exit 2
fi
echo "FINISHED $(date -Is)"
