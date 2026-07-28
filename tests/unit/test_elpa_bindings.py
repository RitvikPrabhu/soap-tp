import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import pytest
from typing import Literal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mpi4py import MPI
from torch.utils.cpp_extension import load

from soap_tp.ops._utils import (
    allocate_2d_block_cyclic,
    block_cyclic_indices,
    block_cyclic_tile_views,
)
from soap_tp.ops.factorizations import initialize_basis_2d_block_cyclic_


SEED = 42
BLOCK_SIZE = 2
PROCESS_GRID_SHAPE = (2, 2)
GRADIENT_SHAPES = (
    # (8, 8),
    # (8, 12),
    (12, 8),
    # (9, 13),
    # (13, 9),
)


def _setup_mpi():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" in os.environ:
            port = int(os.environ["MASTER_PORT"])
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((address, 0))
                port = sock.getsockname()[1]
        rendezvous = address, port
    else:
        rendezvous = None

    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size


# Make a tensor of shape (m, n) using normal distribution, mean 0 and std 2
def _make_gradient(m, n, dtype=torch.float32, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.normal(mean=0, std=2, size=(m, n), generator=generator, dtype=dtype)


# Make the left preconditioner matrix of shape (m, m) and is GG^T
def _make_left_preconditioner(gradient):
    return torch.mm(gradient, gradient.T)


# Make the right preconditioner matrix of shape (n, n) and is G^TG
def _make_right_preconditioner(gradient):
    return torch.mm(gradient.T, gradient)


# Convert the full PyTorch tensor into local 2D block cyclic storage
def _pack_full_matrix_2d_block_cyclic(
    full_matrix,
    local_matrix,
    block_size,
    process_grid_shape,
    rank,
):
    views = block_cyclic_tile_views(
        local_matrix,
        tuple(full_matrix.shape),
        block_size,
        process_grid_shape,
        rank,
        mode="full",
    )

    for (block_row, block_column), local_view in views.items():
        global_row_start = block_row * block_size
        global_row_end = min(global_row_start + block_size, full_matrix.size(0))
        global_column_start = block_column * block_size
        global_column_end = min(global_column_start + block_size, full_matrix.size(1))

        local_view.copy_(
            full_matrix[
                global_row_start:global_row_end, global_column_start:global_column_end
            ]
        )


# Convert the local 2D block cyclic storage into a full PyTorch tensor
def _unpack_matrix_2d_block_cyclic(
    local_matrix,
    global_shape,
    block_size,
    process_grid_shape,
    rank,
):
    full_matrix = local_matrix.new_zeros(global_shape)
    for (block_row, block_column), local_view in block_cyclic_tile_views(
        local_matrix,
        global_shape,
        block_size,
        process_grid_shape,
        rank,
        mode="full",
    ).items():
        global_row_start = block_row * block_size
        global_column_start = block_column * block_size
        full_matrix[
            global_row_start : global_row_start + local_view.size(0),
            global_column_start : global_column_start + local_view.size(1),
        ].copy_(local_view)

    dist.all_reduce(full_matrix)
    return full_matrix


# Run the eigenvalue decomposition of a preconditioner matrix using torch.linalg.eigh
def _run_linalg_eigh(preconditioner):
    eigenvalues, eigenvectors = torch.linalg.eigh(preconditioner)

    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    return eigenvalues, eigenvectors


# Run the eigenvalue decomposition of a preconditioner matrix using ELPA
def _run_elpa_eigh(preconditioner, block_size, process_grid_shape):
    rank = dist.get_rank()
    size = preconditioner.size(0)

    local_preconditioner = allocate_2d_block_cyclic(
        tuple(preconditioner.shape),
        block_size,
        process_grid_shape,
        dtype=preconditioner.dtype,
        device=preconditioner.device,
    )
    local_eigenvectors = allocate_2d_block_cyclic(
        tuple(preconditioner.shape),
        block_size,
        process_grid_shape,
        dtype=preconditioner.dtype,
        device=preconditioner.device,
    )
    work = allocate_2d_block_cyclic(
        tuple(preconditioner.shape),
        block_size,
        process_grid_shape,
        dtype=preconditioner.dtype,
        device=preconditioner.device,
    )
    eigenvalues = preconditioner.new_empty(size)

    _pack_full_matrix_2d_block_cyclic(
        preconditioner,
        local_preconditioner,
        block_size,
        process_grid_shape,
        rank,
    )
    initialize_basis_2d_block_cyclic_(
        local_preconditioner,
        local_eigenvectors,
        work,
        eigenvalues,
        size,
        block_size,
        process_grid_shape,
    )

    eigenvectors = _unpack_matrix_2d_block_cyclic(
        local_eigenvectors,
        tuple(preconditioner.shape),
        block_size,
        process_grid_shape,
        rank,
    )
    return eigenvalues, eigenvectors


# Check that the eigenvectors from non-zero eigenvalues from PyTorch and ELPA are equivalent up to sign
def _check_eigenvectors_of_nonzero_eigenvalues_equivalent(
    expected_eigenvalues,
    expected_eigenvectors,
    actual_eigenvalues,
    actual_eigenvectors,
):
    assert expected_eigenvalues.shape == actual_eigenvalues.shape, (
        "Eigenvalues shapes do not match."
    )
    assert expected_eigenvectors.shape == actual_eigenvectors.shape, (
        "Eigenvectors shapes do not match."
    )

    # Put each set of eigenvalues on the diagonal of a matrix.
    expected_diagonal_matrix = torch.diag(expected_eigenvalues)
    actual_diagonal_matrix = torch.diag(actual_eigenvalues)

    # Let PyTorch decide how many eigenvalues are numerically nonzero.
    expected_rank = torch.linalg.matrix_rank(
        expected_diagonal_matrix,
        hermitian=True,
    ).item()

    actual_rank = torch.linalg.matrix_rank(
        actual_diagonal_matrix,
        hermitian=True,
    ).item()

    assert expected_rank == actual_rank, (
        "PyTorch and ELPA disagree on the number of nonzero eigenvalues.\n"
        f"PyTorch rank: {expected_rank}\n"
        f"ELPA rank: {actual_rank}\n"
        f"PyTorch eigenvalues: {expected_eigenvalues}\n"
        f"ELPA eigenvalues: {actual_eigenvalues}"
    )

    # Eigenvalues are descending, so the first `rank` columns correspond
    # to the numerically nonzero eigenvalues.
    for index in range(expected_rank):
        expected_vector = expected_eigenvectors[:, index]
        actual_vector = actual_eigenvectors[:, index]

        similarity = torch.cosine_similarity(
            expected_vector,
            actual_vector,
            dim=0,
        ).abs()

        torch.testing.assert_close(
            similarity,
            torch.ones_like(similarity),
            msg=(
                f"Eigenvector {index} does not match "
                "between PyTorch and ELPA up to sign."
            ),
        )

    expected = expected_eigenvectors[:, :expected_rank]
    merged = actual_eigenvectors[:, :actual_rank]
    signs = torch.where(
        (merged * expected).sum(dim=0) < 0,
        -1.0,
        1.0,
    )
    merged = merged * signs
    difference = (merged - expected).norm()
    original_size = expected.norm()
    error = difference / original_size
    if dist.get_rank() == 0:
        print(
            "Nonzero-eigenvector relative error: "
            f"difference={difference.item():.6e}, "
            f"original_size={original_size.item():.6e}, "
            f"error={error.item():.6e}"
        )


# Check that the zero eigenspaces from PyTorch and ELPA are equivalent
def _check_zero_eigenspaces_equivalent(
    pytorch_eigenvalues,
    pytorch_eigenvectors,
    elpa_eigenvalues,
    elpa_eigenvectors,
):
    """
    Same order, possible sign difference:
    overlap =
    [ 1  0]
    [ 0 -1]

    different order:
    overlap =
    [0 1] ----> ELPA vector 0 matches PyTorch vector 1, sign = +1
    [1 0]   ----> ELPA vector 1 matches PyTorch vector 0, sign = +1

    Mixed or rotated within the same zero eigenspace:
    overlap =
    [ 0.707  0.707]
    [ 0.707 -0.707]
    """
    # Convert the eigenvalue lists into diagonal matrices so that
    # PyTorch can determine their numerical ranks.
    pytorch_eigenvalue_matrix = torch.diag(pytorch_eigenvalues)
    elpa_eigenvalue_matrix = torch.diag(elpa_eigenvalues)

    pytorch_rank = torch.linalg.matrix_rank(
        pytorch_eigenvalue_matrix,
        hermitian=True,
    ).item()

    elpa_rank = torch.linalg.matrix_rank(
        elpa_eigenvalue_matrix,
        hermitian=True,
    ).item()

    # Both solvers should agree on how many eigenvalues are nonzero.
    assert pytorch_rank == elpa_rank, (
        "PyTorch and ELPA disagree on the number of zero eigenvalues.\n"
        f"PyTorch rank: {pytorch_rank}\n"
        f"ELPA rank: {elpa_rank}\n"
        f"PyTorch eigenvalues: {pytorch_eigenvalues}\n"
        f"ELPA eigenvalues: {elpa_eigenvalues}"
    )

    matrix_size = pytorch_eigenvalues.numel()
    number_of_zero_eigenvalues = matrix_size - pytorch_rank

    # There is nothing more to check when the matrix has full rank.
    if number_of_zero_eigenvalues == 0:
        return

    # Eigenvalues are descending, so zero-eigenvalue eigenvectors
    # occur in the final columns.
    pytorch_zero_vectors = pytorch_eigenvectors[:, pytorch_rank:]
    elpa_zero_vectors = elpa_eigenvectors[:, elpa_rank:]

    # Build the matrix that projects onto each zero eigenspace.
    pytorch_zero_projector = pytorch_zero_vectors @ pytorch_zero_vectors.T
    elpa_zero_projector = elpa_zero_vectors @ elpa_zero_vectors.T

    overlap = pytorch_zero_vectors.T @ elpa_zero_vectors
    absolute_overlap = overlap.abs()

    if dist.get_rank() == 0:
        print("Zero-eigenvector overlap:")
        print(overlap)

    # The individual vectors may differ, but their spanned space
    # should be the same.
    torch.testing.assert_close(
        elpa_zero_projector,
        pytorch_zero_projector,
        rtol=2e-4,
        atol=2e-4,
    )

    number_of_zero_vectors = absolute_overlap.shape[0]

    # Find which PyTorch vector best matches each ELPA vector.
    matches = absolute_overlap.argmax(dim=0)

    # Build the matrix expected for a pure reordering.
    permutation = torch.zeros_like(absolute_overlap)
    elpa_indices = torch.arange(
        number_of_zero_vectors,
        device=overlap.device,
    )
    permutation[matches, elpa_indices] = 1

    if torch.allclose(
        absolute_overlap,
        torch.eye(
            number_of_zero_vectors,
            device=overlap.device,
            dtype=overlap.dtype,
        ),
    ):
        print("The vectors have the same order and may differ by sign.")

    elif torch.unique(matches).numel() == number_of_zero_vectors and torch.allclose(
        absolute_overlap, permutation
    ):
        print("The vectors differ only by order and/or sign.")

        for elpa_index, pytorch_index in enumerate(matches.tolist()):
            sign = overlap[pytorch_index, elpa_index].item()

            print(
                f"ELPA vector {elpa_index} matches "
                f"PyTorch vector {pytorch_index}, "
                f"sign = {sign:+.0f}"
            )

    else:
        print("The vectors are mixed or rotated within the same zero eigenspace.")


def _check_elpa_matches_eigh(preconditioner):
    expected_eigenvalues, expected_eigenvectors = _run_linalg_eigh(preconditioner)
    actual_eigenvalues, actual_eigenvectors = _run_elpa_eigh(
        preconditioner,
        BLOCK_SIZE,
        PROCESS_GRID_SHAPE,
    )

    if dist.get_rank() == 0:
        print("PyTorch eigenvalues:")
        print(expected_eigenvalues)

        print("ELPA eigenvalues:")
        print(actual_eigenvalues)

        print("Difference:")
        print(actual_eigenvalues - expected_eigenvalues)

    torch.testing.assert_close(
        actual_eigenvalues,
        expected_eigenvalues,
        rtol=2e-4,
        atol=2e-4,
    )

    identity = torch.eye(
        actual_eigenvectors.shape[1],
        dtype=actual_eigenvectors.dtype,
        device=actual_eigenvectors.device,
    )
    torch.testing.assert_close(
        actual_eigenvectors.T @ actual_eigenvectors,
        identity,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        preconditioner @ actual_eigenvectors,
        actual_eigenvectors * actual_eigenvalues.unsqueeze(0),
        rtol=2e-4,
        atol=2e-4,
    )

    _check_eigenvectors_of_nonzero_eigenvalues_equivalent(
        expected_eigenvalues,
        expected_eigenvectors,
        actual_eigenvalues,
        actual_eigenvectors,
    )
    _check_zero_eigenspaces_equivalent(
        expected_eigenvalues,
        expected_eigenvectors,
        actual_eigenvalues,
        actual_eigenvectors,
    )


@pytest.fixture(scope="module")
def mpi_world():
    rank, world_size = _setup_mpi()
    try:
        if world_size != 4:
            pytest.skip(
                "ELPA binding tests require four MPI ranks; run with mpiexec -n 4."
            )
        yield rank, world_size
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_left_preconditioner_eigendecomposition(shape, mpi_world):
    gradient = _make_gradient(*shape)
    preconditioner = _make_left_preconditioner(gradient)
    _check_elpa_matches_eigh(preconditioner)


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_right_preconditioner_eigendecomposition(shape, mpi_world):
    gradient = _make_gradient(*shape)
    preconditioner = _make_right_preconditioner(gradient)
    _check_elpa_matches_eigh(preconditioner)
