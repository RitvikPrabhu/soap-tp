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
elif [[ $1 == fused || $1 == vector-fusion || $1 == direct || $1 == merged ]]; then
    binaries=(twod-eigh fused-eigh)
    [[ $1 == vector-fusion ]] && binaries=(fused-eigh vector-fusion-eigh)
    [[ $1 == direct ]] && binaries=(fused-eigh vector-fusion-eigh direct-eigh)
    [[ $1 == merged ]] && binaries=(direct-eigh checked-eigh)
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1
        done
    done
    for matrix in diagonal lowrank indefinite clustered; do
        run 8 --n 97 --block 8 --pr 2 --pc 4 --matrix "$matrix" --precision fp64 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1
    done
    if [[ $1 == merged ]]; then
        run 8 --n 97 --block 8 --pr 2 --pc 4 --matrix zero --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1
        run 8 --n 97 --block 8 --pr 4 --pc 2 --matrix zero --precision fp32 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1
        for precision in fp64 fp32; do
            run 8 --n 1025 --block 32 --pr 2 --pc 4 --matrix lowrank --precision "$precision" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --reference 1
            run 8 --n 1025 --block 32 --pr 4 --pc 2 --matrix indefinite --precision "$precision" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --reference 1
        done
    fi
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        for grid in '1 8' '2 4' '4 2'; do
            read -r pr pc <<< "$grid"
            tri_pr=$pr
            [[ $pr == 1 ]] && tri_pr=8
            for binary in "${binaries[@]}"; do
                EIGH_BIN=build/custom-eigh/$binary run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --tri-pr "$tri_pr" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
            done
        done
        run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
    done
    if [[ $1 == merged ]]; then
        for precision in fp64 fp32; do
            tridiag=elpa
            [[ $precision == fp32 ]] && tridiag=elpa32
            for ranks in 1 2 4; do
                run "$ranks" --n 8192 --block 64 --pr 1 --pc "$ranks" --tri-pr "$ranks" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
            done
            for n in 8192 16384; do
                run 8 --n "$n" --block 64 --pr 4 --pc 2 --solver elpa1 --precision "$precision" --repeat 3 --reference 1
            done
        done
    fi
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
elif [[ $1 == finish ]]; then
    for precision in fp64 fp32; do
        tridiag=elpa
        [[ $precision == fp32 ]] && tridiag=elpa32
        run 8 --n 97 --block 8 --pr 4 --pc 2 --matrix zero --precision "$precision" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1
        run 8 --n 1025 --block 32 --pr 2 --pc 4 --matrix lowrank --precision "$precision" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --reference 1
        for ranks in 1 2 4 8; do
            run "$ranks" --n 8192 --block 64 --pr 1 --pc "$ranks" --tri-pr "$ranks" --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
        done
        for panel in 1 8 32; do
            run 8 --n 16384 --block 64 --pr 4 --pc 2 --precision "$precision" --panel "$panel" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
        done
        run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision "$precision" --tridiag "$tridiag" --back-block 128 --gpu-mpi 1 --rccl 1 --device-vectors 1 --repeat 3 --reference 1
        for n in 8192 16384; do
            run 8 --n "$n" --block 64 --pr 4 --pc 2 --solver elpa1 --precision "$precision" --repeat 3 --reference 1
        done
    done
elif [[ $1 == low-compute ]]; then
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for matrix in random lowrank indefinite; do
            for mode in 1 2; do
                if run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --matrix "$matrix" --precision fp32 --tridiag elpa32 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --bf16-bulk "$mode"; then
                    echo "BF16_CHECK PASSED mode=$mode grid=$grid matrix=$matrix"
                else
                    echo "BF16_CHECK FAILED mode=$mode grid=$grid matrix=$matrix"
                fi
            done
        done
    done
    run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --tridiag elpa --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
    for mode in fp32 split-half split-bf16; do
        extra=()
        [[ $mode == split-half ]] && extra=(--half-bulk 2)
        [[ $mode == split-bf16 ]] && extra=(--bf16-bulk 2)
        if run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1 "${extra[@]}"; then
            echo "COMPUTE_CHECK PASSED variant=$mode"
        else
            echo "COMPUTE_CHECK FAILED variant=$mode"
        fi
    done
    for mode in 0 2; do
        if run 8 --n 257 --block 16 --pr 2 --pc 4 --precision fp32 --band 16 --back-block 32 --gpu-mpi 1 --rccl 1 --bf16-bulk "$mode" --matrix lowrank; then
            echo "BAND_BF16_CHECK PASSED mode=$mode"
        else
            echo "BAND_BF16_CHECK FAILED mode=$mode"
        fi
        if run 8 --n 2048 --block 64 --pr 4 --pc 2 --precision fp32 --band 64 --back-block 128 --gpu-mpi 1 --rccl 1 --bf16-bulk "$mode" --repeat 3 --reference 1; then
            echo "BAND_BF16_CHECK PASSED mode=$mode"
        else
            echo "BAND_BF16_CHECK FAILED mode=$mode"
        fi
    done
elif [[ $1 == pool ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
    done
    for mode in fp64 fp32 split-half split-bf16; do
        precision=fp32; tridiag=elpa32; extra=()
        [[ $mode == fp64 ]] && precision=fp64 && tridiag=elpa
        [[ $mode == split-half ]] && extra=(--half-bulk 2)
        [[ $mode == split-bf16 ]] && extra=(--bf16-bulk 2)
        for pool in 0 1; do
            if run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision "$precision" --tridiag "$tridiag" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc "$pool" --repeat 3 --reference 1 "${extra[@]}"; then
                echo "POOL_CHECK PASSED variant=$mode pool=$pool"
            else
                echo "POOL_CHECK FAILED variant=$mode pool=$pool"
            fi
        done
    done
    for mode in fp64 fp32 split-half; do
        precision=fp32; tridiag=elpa32; extra=()
        [[ $mode == fp64 ]] && precision=fp64 && tridiag=elpa
        [[ $mode == split-half ]] && extra=(--half-bulk 2)
        for pool in 0 1; do
            run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision "$precision" --tridiag "$tridiag" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc "$pool" --repeat 3 --reference 1 "${extra[@]}"
        done
    done
elif [[ $1 == device-panel ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            for panel in 3 8; do
                run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --panel "$panel" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
            done
        done
    done
    for panel in 1 8 16 32; do
        for grid in '1 8' '4 2'; do
            read -r pr pc <<< "$grid"
            run 8 --n 8192 --block 64 --pr "$pr" --pc "$pc" --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1
        done
    done
elif [[ $1 == direct-layout ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --panel 3 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
    done
    for binary in device-panel-eigh local-collectives-eigh direct-layout-eigh; do
        EIGH_BIN="build/custom-eigh/$binary" run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1
    done
    for panel in 1 8 32; do
        run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1
    done
    for panel in 1 8; do
        if run 8 --n 8192 --block 64 --pr 4 --pc 2 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --half-bulk 2 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1; then
            echo "PANEL_HALF_CHECK PASSED panel=$panel"
        else
            echo "PANEL_HALF_CHECK FAILED panel=$panel"
        fi
    done
elif [[ $1 == jacobi-smoke ]]; then
    passed=1
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            for matrix in random lowrank indefinite; do
                if run 8 --solver jacobi --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --matrix "$matrix" --gpu-mpi 1 --rccl 1 --async-alloc 1; then
                    echo "JACOBI_CHECK PASSED"
                else
                    echo "JACOBI_CHECK FAILED"; passed=0
                fi
            done
        done
    done
    for matrix in zero diagonal clustered; do
        if run 8 --solver jacobi --n 97 --block 8 --pr 2 --pc 4 --precision fp32 --matrix "$matrix" --gpu-mpi 1 --rccl 1 --async-alloc 1; then
            echo "JACOBI_CHECK PASSED"
        else
            echo "JACOBI_CHECK FAILED"; passed=0
        fi
    done
    if [[ $passed == 1 ]]; then
        for half in 0 2; do
            if run 8 --solver jacobi --n 2048 --block 64 --pr 1 --pc 8 --precision fp32 --half-bulk "$half" --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1; then
                echo "JACOBI_CHECK PASSED"
            else
                echo "JACOBI_CHECK FAILED"
            fi
        done
    fi
elif [[ $1 == lower ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --matrix indefinite --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --lower-triangle 1
        done
    done
    for lower in 0 1; do
        run 8 --n 4096 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --lower-triangle "$lower" --repeat 3 --reference 1
    done
elif [[ $1 == components-v2 ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision fp32 --matrix indefinite --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --lower-triangle 1
    done
    for lower in 0 1; do
        run 8 --n 4096 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --lower-triangle "$lower" --repeat 3 --reference 1
    done
    for tri in mrrr mrrr32; do
        passed=1
        for grid in '2 4' '4 2'; do
            read -r pr pc <<< "$grid"
            for matrix in random lowrank zero clustered; do
                if run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision fp32 --matrix "$matrix" --tridiag "$tri" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1; then
                    echo "MRRR_CHECK PASSED backend=$tri"
                else
                    echo "MRRR_CHECK FAILED backend=$tri"; passed=0
                fi
            done
        done
        if [[ $passed == 1 ]]; then
            if run 8 --n 8192 --block 64 --pr 1 --pc 8 --precision fp32 --tridiag "$tri" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n8192; then
                echo "MRRR_CHECK PASSED backend=$tri"
            else
                echo "MRRR_CHECK FAILED backend=$tri"
            fi
        fi
    done
    for pivot64 in 0 1; do
        passed=1
        for matrix in random lowrank indefinite; do
            if run 8 --solver jacobi --n 97 --block 8 --pr 2 --pc 4 --precision fp32 --matrix "$matrix" --jacobi-pivot64 "$pivot64" --gpu-mpi 1 --rccl 1 --async-alloc 1; then
                echo "PIVOT_CHECK PASSED pivot64=$pivot64"
            else
                echo "PIVOT_CHECK FAILED pivot64=$pivot64"; passed=0
            fi
        done
        if [[ $passed == 1 ]]; then
            for half in 0 2; do
                if run 8 --solver jacobi --n 2048 --block 256 --pr 1 --pc 8 --precision fp32 --jacobi-pivot64 "$pivot64" --half-bulk "$half" --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1; then
                    echo "PIVOT_CHECK PASSED pivot64=$pivot64 half=$half"
                else
                    echo "PIVOT_CHECK FAILED pivot64=$pivot64 half=$half"
                fi
            done
        fi
    done
elif [[ $1 == panel-dot ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
    done
    for n in 8192 16384; do
        for binary in direct-layout-eigh panel-dot-eigh; do
            EIGH_BIN="build/custom-eigh/$binary" run 8 --n "$n" --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference "output/custom-eigh/torch-large/n$n"
        done
    done
    for panel in 8 64; do
        for half in 0 2; do
            run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --half-bulk "$half" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
        done
    done
elif [[ $1 == bulk-back ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision fp32 --matrix lowrank --panel 7 --back-block 13 --half-bulk 2 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
    done
    for n in 8192 16384; do
        panel=1; [[ $n == 16384 ]] && panel=32
        for binary in panel-dot-eigh joint-half-eigh device-t-eigh; do
            for half in 0 2; do
                EIGH_BIN="build/custom-eigh/$binary" run 8 --n "$n" --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --half-bulk "$half" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference "output/custom-eigh/torch-large/n$n"
            done
        done
    done
    for back in 256 512; do
        for half in 0 2; do
            run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --half-bulk "$half" --back-block "$back" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
        done
    done
elif [[ $1 == lds ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
    done
    for binary in joint-half-eigh lds-t-eigh; do
        for half in 0 2; do
            EIGH_BIN="build/custom-eigh/$binary" run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --half-bulk "$half" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n8192
        done
    done
    for back in 128 256; do
        for half in 0 2; do
            run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --half-bulk "$half" --back-block "$back" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
        done
    done
    passed=1
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            if EIGH_BIN=build/custom-eigh/jacobi-diagonal-eigh run 8 --solver jacobi --n 97 --block 8 --pr "$pr" --pc "$pc" --precision "$precision" --jacobi-pivot64 1 --gpu-mpi 1 --rccl 1 --async-alloc 1; then
                echo "DIAGONAL_CHECK PASSED tiny"
            else
                echo "DIAGONAL_CHECK FAILED tiny"; passed=0
            fi
        done
    done
    if [[ $passed == 1 ]]; then
        for pivot64 in 0 1; do
            for half in 0 2; do
                if EIGH_BIN=build/custom-eigh/jacobi-diagonal-eigh run 8 --solver jacobi --n 2048 --block 256 --pr 1 --pc 8 --precision fp32 --jacobi-pivot64 "$pivot64" --half-bulk "$half" --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1; then
                    echo "DIAGONAL_CHECK PASSED pivot64=$pivot64 half=$half"
                else
                    echo "DIAGONAL_CHECK FAILED pivot64=$pivot64 half=$half"
                fi
            done
        done
    fi
elif [[ $1 == native-components ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tri=elpa; [[ $precision == fp32 ]] && tri=elpa32
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision "$precision" --tridiag "$tri" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision fp32 --tridiag elpa32 --matrix lowrank --panel 7 --back-block 13 --half-bulk 2 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
    done
    for binary in joint-half-eigh native-retile-eigh shared-t-eigh; do
        for half in 0 2; do
            EIGH_BIN="build/custom-eigh/$binary" run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --half-bulk "$half" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n8192
        done
    done
    for half in 0 2; do
        EIGH_BIN=build/custom-eigh/joint-half-eigh run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --half-bulk "$half" --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
    done
    for back in 128 256 512; do
        for half in 0 2; do
            run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --half-bulk "$half" --back-block "$back" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
        done
    done
elif [[ $1 == threaded ]]; then
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tri=elpa; [[ $precision == fp32 ]] && tri=elpa32
            OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision "$precision" --tridiag "$tri" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
    done
    EIGH_BIN=build/custom-eigh/fused-av-eigh run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --back-block 256 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
    export OMP_PROC_BIND=close OMP_PLACES=cores
    for threads in 1 2 4 7; do
        export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
        run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --back-block 256 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
    done
elif [[ $1 == fused-av ]]; then
    # Rejected candidate; source is retained in commit 264ef64, not current eigh.cpp.
    EIGH_BIN=build/custom-eigh/fused-av-eigh
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tri=elpa; [[ $precision == fp32 ]] && tri=elpa32
            for matrix in random lowrank indefinite zero; do
                run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision "$precision" --tridiag "$tri" --matrix "$matrix" --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --fused-av 1
            done
        done
    done
    for n in 4096 8192 16384; do
        for fused in 0 1; do
            run 8 --n "$n" --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --fused-av "$fused" --repeat 3 --reference 1
        done
    done
elif [[ $1 == blas-t ]]; then
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for precision in fp64 fp32; do
            tri=elpa; [[ $precision == fp32 ]] && tri=elpa32
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision "$precision" --tridiag "$tri" --matrix indefinite --panel 7 --back-block 13 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
        done
        run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision fp32 --tridiag elpa32 --matrix zero --panel 7 --back-block 13 --half-bulk 2 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1
    done
    for n in 8192 16384; do
        panel=1; [[ $n == 16384 ]] && panel=32
        for binary in scaled-eigh blas-t-eigh; do
            for back in 128 256; do
                for half in 0 2; do
                    EIGH_BIN="build/custom-eigh/$binary" run 8 --n "$n" --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel "$panel" --half-bulk "$half" --back-block "$back" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference "output/custom-eigh/torch-large/n$n"
                done
            done
        done
    done
    for half in 0 2; do
        run 8 --n 16384 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 32 --half-bulk "$half" --back-block 512 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n16384
    done
elif [[ $1 == jacobi-local ]]; then
    passed=1
    for grid in '2 4' '4 2'; do
        read -r pr pc <<< "$grid"
        for matrix in random indefinite clustered lowrank; do
            if run 8 --solver jacobi --n 97 --block 8 --pr "$pr" --pc "$pc" --precision fp32 --matrix "$matrix" --jacobi-pivot64 1 --jacobi-local jacobi --gpu-mpi 1 --rccl 1 --async-alloc 1; then
                echo "LOCAL_JACOBI_CHECK PASSED"
            else
                echo "LOCAL_JACOBI_CHECK FAILED"; passed=0
            fi
        done
    done
    if [[ $passed == 1 ]]; then
        for local_solver in dc jacobi; do
            for half in 0 2; do
                if run 8 --solver jacobi --n 2048 --block 256 --pr 1 --pc 8 --precision fp32 --jacobi-pivot64 1 --jacobi-local "$local_solver" --half-bulk "$half" --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1; then
                    echo "LOCAL_JACOBI_CHECK PASSED solver=$local_solver half=$half"
                else
                    echo "LOCAL_JACOBI_CHECK FAILED solver=$local_solver half=$half"
                fi
            done
        done
    fi
elif [[ $1 == rccl-tune ]]; then
    for config in default LL LL128 Simple LL-c1 LL-c4 Simple-c1 Ring-LL-c1; do
        unset NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_ALGO
        case "$config" in
            LL|LL128|Simple) export NCCL_PROTO="$config" ;;
            LL-c1|LL-c4|Ring-LL-c1)
                export NCCL_PROTO=LL NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
                if [[ $config == LL-c4 ]]; then export NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4; fi
                if [[ $config == Ring-LL-c1 ]]; then export NCCL_ALGO=Ring; fi ;;
            Simple-c1) export NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 ;;
        esac
        echo "RCCL_CONFIG $config"
        if run 8 --n 8192 --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --back-block 128 --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --torch-reference output/custom-eigh/torch-large/n8192; then
            echo "RCCL_CHECK PASSED config=$config"
        else
            echo "RCCL_CHECK FAILED config=$config"
        fi
    done
elif [[ $1 == loop3-graph ]]; then
    EIGH_BIN=build/custom-eigh/graph-eigh
    for grid in '1 8' '2 4' '4 2' '8 1'; do
        read -r pr pc <<< "$grid"
        for mode in fp64 fp32 split; do
            precision=fp32; tri=elpa32; half=0
            if [[ $mode == fp64 ]]; then precision=fp64; tri=elpa; fi
            if [[ $mode == split ]]; then half=2; fi
            run 8 --n 97 --block 8 --pr "$pr" --pc "$pc" --tri-pr 8 --precision "$precision" --tridiag "$tri" --matrix indefinite --panel 7 --back-block 13 --half-bulk "$half" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --graph-reduction 1
        done
    done
    for n in 2048 8192; do
        for graph in 0 1; do
            for half in 0 2; do
                run 8 --n "$n" --block 64 --pr 1 --pc 8 --tri-pr 8 --precision fp32 --tridiag elpa32 --panel 1 --back-block 256 --half-bulk "$half" --device-vectors 1 --gpu-mpi 1 --rccl 1 --async-alloc 1 --graph-reduction "$graph" --repeat 3 --reference 1
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
