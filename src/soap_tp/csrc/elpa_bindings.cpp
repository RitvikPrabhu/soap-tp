#include <pybind11/pybind11.h>
#include <mpi.h>
#include <elpa/elpa.h>
#include <elpa/elpa_configured_options.h>
#include <elpa/elpa_version.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

    void get_mpi_world_info(int &rank, int &size)
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error(
                "MPI must be initialized before using ELPA");
        }

        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) {
            throw std::runtime_error("MPI has already been finalized");
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    void require_elpa_ok(int error, const char *operation)
    {
        if (error != ELPA_OK) {
            throw std::runtime_error(
                std::string(operation) + ": " + elpa_strerr(error));
        }
    }

    void eigenvectors_2d_block_cyclic_float(
        std::uintptr_t a_address,
        std::uintptr_t eigenvalues_address,
        std::uintptr_t q_address,
        std::int64_t n,
        std::int64_t local_rows,
        std::int64_t local_columns,
        std::int64_t block_size,
        int process_rows,
        int process_columns)
    {
        int rank = 0;
        int world_size = 0;
        get_mpi_world_info(rank, world_size);
        if (process_rows <= 0 || process_columns <= 0 ||
            process_rows * process_columns != world_size) {
            throw std::invalid_argument(
                "process grid must be positive and contain every MPI rank");
        }
        if (a_address == 0 || eigenvalues_address == 0 ||
            q_address == 0) {
            throw std::invalid_argument("ELPA buffer addresses must be nonzero");
        }
        if (n <= 0 || local_rows <= 0 || local_columns <= 0 ||
            block_size <= 0) {
            throw std::invalid_argument(
                "matrix dimensions and block size must be positive");
        }

        const int process_row = rank / process_columns;
        const int process_column = rank % process_columns;
        const int elpa_rank =
            process_row + process_column * process_rows;
        MPI_Comm communicator = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, 0, elpa_rank, &communicator);

        bool elpa_initialized = false;
        elpa_t handle = nullptr;
        int error = ELPA_OK;
        try {
            require_elpa_ok(elpa_init(ELPA_API_VERSION), "elpa_init");
            elpa_initialized = true;
            handle = elpa_allocate(&error);
            require_elpa_ok(error, "elpa_allocate");
            if (handle == nullptr) {
                throw std::runtime_error("elpa_allocate returned null");
            }

            auto set_integer = [&](const char *name, int value) {
                elpa_set_integer(handle, name, value, &error);
                require_elpa_ok(error, name);
            };
            set_integer("na", static_cast<int>(n));
            set_integer("nev", static_cast<int>(n));
            set_integer("local_nrows", static_cast<int>(local_rows));
            set_integer("local_ncols", static_cast<int>(local_columns));
            set_integer("nblk", static_cast<int>(block_size));
            set_integer(
                "mpi_comm_parent",
                static_cast<int>(MPI_Comm_c2f(communicator)));
            set_integer("process_row", process_row);
            set_integer("process_col", process_column);
            require_elpa_ok(elpa_setup(handle), "elpa_setup");

#if ELPA_WITH_NVIDIA_GPU_VERSION == 1
            set_integer("nvidia-gpu", 1);
#elif ELPA_WITH_AMD_GPU_VERSION == 1
            set_integer("amd-gpu", 1);
#endif

            elpa_eigenvectors_float(
                handle,
                reinterpret_cast<float *>(a_address),
                reinterpret_cast<float *>(eigenvalues_address),
                reinterpret_cast<float *>(q_address),
                &error);
            require_elpa_ok(error, "elpa_eigenvectors_float");
        }
        catch (...) {
            if (handle != nullptr) {
                elpa_deallocate(handle, &error);
            }
            if (elpa_initialized) {
                elpa_uninit(&error);
            }
            MPI_Comm_free(&communicator);
            throw;
        }

        int deallocate_error = ELPA_OK;
        int uninit_error = ELPA_OK;
        elpa_deallocate(handle, &deallocate_error);
        elpa_uninit(&uninit_error);
        MPI_Comm_free(&communicator);
        require_elpa_ok(deallocate_error, "elpa_deallocate");
        require_elpa_ok(uninit_error, "elpa_uninit");
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
        "mpi_world_rank_and_size",
        []() {
            int rank = 0;
            int size = 0;
            get_mpi_world_info(rank, size);
            return py::make_tuple(rank, size);
        },
        "Return the initialized MPI world rank and size.");
    handle.def(
        "elpa_error_string",
        [](int error)
        { return elpa_strerr(error); },
        py::arg("error"),
        "Return ELPA's message for an error code");
    handle.def(
        "elpa_eigenvectors_2d_block_cyclic_float",
        &eigenvectors_2d_block_cyclic_float,
        py::arg("a"),
        py::arg("eigenvalues"),
        py::arg("q"),
        py::arg("n"),
        py::arg("local_rows"),
        py::arg("local_columns"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_columns"),
        py::call_guard<py::gil_scoped_release>(),
        R"doc(Compute a full eigendecomposition using MPI_COMM_WORLD.

The binding owns ELPA handle setup and remaps row-major process-grid ranks to
ELPA's column-major communicator ordering. The input matrix is overwritten.
)doc");
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
