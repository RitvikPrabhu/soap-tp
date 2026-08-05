import math

import pytest
import torch

from soap_tp.ops.optimizer import adam_update


SEED = 42
RELATIVE_L2_RTOL = 1e-5
MATRIX_SHAPES = (
    (8, 8),
    (8, 12),
    (12, 8),
    (9, 13),
    (13, 9),
)


def _make_matrix(rows, columns, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.normal(
        mean=0,
        std=2,
        size=(rows, columns),
        generator=generator,
        dtype=torch.float32,
    )


def _reference_adam_update(
    gradient,
    momentum,
    variance,
    *,
    step,
    beta1,
    beta2,
    eps,
):
    updated_momentum = beta1 * momentum + (1.0 - beta1) * gradient
    updated_variance = (
        beta2 * variance + (1.0 - beta2) * gradient.square()
    )
    update = (updated_momentum / (1.0 - beta1**step)) / (
        updated_variance.sqrt() / math.sqrt(1.0 - beta2**step) + eps
    )
    return update, updated_momentum, updated_variance


def _relative_l2_error(actual, reference):
    difference_norm = torch.linalg.vector_norm(
        actual.to(torch.float64) - reference.to(torch.float64)
    ).item()
    reference_norm = torch.linalg.vector_norm(
        reference.to(torch.float64)
    ).item()
    if reference_norm == 0.0:
        return 0.0 if difference_norm == 0.0 else math.inf
    return difference_norm / reference_norm


def _test_adam_update(shape):
    rows, columns = shape
    gradient = _make_matrix(rows, columns, seed=SEED)
    momentum = _make_matrix(rows, columns, seed=SEED + 1)
    variance = _make_matrix(rows, columns, seed=SEED + 2).square()

    step = 3
    beta1 = 0.8
    beta2 = 0.9
    eps = 1e-6
    expected_update, expected_momentum, expected_variance = (
        _reference_adam_update(
            gradient,
            momentum,
            variance,
            step=step,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
    )

    update = adam_update(
        gradient,
        momentum,
        variance,
        step=step,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    relative_l2_error = _relative_l2_error(update, expected_update)
    if relative_l2_error > RELATIVE_L2_RTOL:
        raise AssertionError(
            f"relative L2 error {relative_l2_error:.6e} exceeds "
            f"{RELATIVE_L2_RTOL:.6e}"
        )

    torch.testing.assert_close(momentum, expected_momentum)
    torch.testing.assert_close(variance, expected_variance)
    torch.testing.assert_close(update, expected_update)
    return relative_l2_error


@pytest.fixture(scope="module")
def adam_l2_log(output_folder):
    if output_folder is None:
        return None

    path = output_folder / "adam_l2.log"
    path.write_text("matrix_shape relative_l2_error\n")
    return path


@pytest.mark.parametrize("shape", MATRIX_SHAPES)
def test_adam_update(shape, adam_l2_log):
    relative_l2_error = _test_adam_update(shape)
    if adam_l2_log is not None:
        rows, columns = shape
        with adam_l2_log.open("a") as stream:
            stream.write(
                f"{rows}x{columns} {relative_l2_error:.6e}\n"
            )
