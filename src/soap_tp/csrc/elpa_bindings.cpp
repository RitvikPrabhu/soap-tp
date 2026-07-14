#include <pybind11/pybind11.h>
#include <elpa/elpa.h>
#include <elpa/elpa_configured_options.h>

#include <cstdint>

namespace py = pybind11;

#ifdef TORCH_EXTENSION_NAME
#define SOAP_TP_EXTENSION_NAME TORCH_EXTENSION_NAME
#else
#define SOAP_TP_EXTENSION_NAME elpa_bindings
#endif

namespace
{

    const char *compiled_gpu_backend()
    {
#if ELPA_WITH_NVIDIA_GPU_VERSION == 1
        return "cuda";
#elif ELPA_WITH_AMD_GPU_VERSION == 1
        return "rocm";
#else
        return "none";
#endif
    }

    int eigenvectors_float(
        std::uintptr_t handle,
        std::uintptr_t a,
        std::uintptr_t ev,
        std::uintptr_t q)
    {
        int error = ELPA_OK;
        elpa_eigenvectors_float(
            reinterpret_cast<elpa_t>(handle),
            reinterpret_cast<float *>(a),
            reinterpret_cast<float *>(ev),
            reinterpret_cast<float *>(q),
            &error);
        return error;
    }

} // namespace

PYBIND11_MODULE(SOAP_TP_EXTENSION_NAME, handle)
{
    handle.doc() = "Python bindings for ELPA library";
    handle.def(
        "compiled_gpu_backend",
        &compiled_gpu_backend,
        "Return the GPU backend compiled into ELPA");
    handle.def(
        "elpa_error_string",
        [](int error)
        { return elpa_strerr(error); },
        py::arg("error"),
        "Return ELPA's message for an error code");
    handle.def(
        "elpa_eigenvectors_float",
        &eigenvectors_float,
        py::arg("handle"),
        py::arg("a"),
        py::arg("ev"),
        py::arg("q"),
        py::call_guard<py::gil_scoped_release>(),
        R"doc(Compute eigenpairs of a real symmetric matrix in single precision.

The matrix and output buffers may reside in host memory or, when supported by
the compiled ELPA backend and handle configuration, device memory. This
binding accepts integer addresses so callers can pass ``Tensor.data_ptr()``
without copying data through Python.

Args:
    handle: Address of an initialized and configured ``elpa_t``. Matrix
        dimensions, distribution, block size, communicator, solver, and
        requested eigenvector count must be set before this call.
    a: Address of the local block-cyclic part of the input matrix, stored as
        column-major ``float32`` values with shape
        ``[local_nrows, local_ncols]``. ELPA may overwrite this buffer.
    ev: Address of space for ``na`` contiguous ``float32`` eigenvalues.
    q: Address of column-major ``float32`` storage for the local part of the
        eigenvectors, with shape ``[local_nrows, local_ncols]``.

Returns:
    The ELPA error code. ``ELPA_OK`` indicates success; pass another result to
    ``elpa_error_string`` for its message.

Raises:
    TypeError: If an argument cannot be converted to an integer address.

Contract:
    Preconditions:
        - All addresses are non-null, live for the duration of the call, and
          point to buffers of the configured sizes and memory location.
        - The global input matrix is real symmetric. ELPA does not validate
          symmetry.
        - Every rank in the handle's communicator enters this collective in
          the same order with a compatible matrix distribution.

    Guarantees:
        - On success, ``ev`` contains eigenvalues in ascending order and the
          requested eigenvectors are stored as columns of ``q``.
        - The Python GIL is released while ELPA performs the computation.
        - Native ELPA failures are returned as error codes rather than raised
          as Python exceptions.

    Invariants:
        - For a successful full eigendecomposition, the result satisfies
          ``A @ Q == Q @ diag(ev)`` up to single-precision numerical error.
        - Eigenvector signs are arbitrary and may differ from other solvers.

    Unchecked assumptions:
        - Dtype, shape, strides, device placement, pointer ownership, and
          buffer sizes are not validated. Invalid addresses can crash the
          process or corrupt memory.
)doc");
}
