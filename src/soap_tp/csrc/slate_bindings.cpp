#include <pybind11/pybind11.h>

#include <mpi.h>
#include <slate/slate.hh>

#include <cstdint>

namespace py = pybind11;

#ifdef TORCH_EXTENSION_NAME
#define SOAP_TP_EXTENSION_NAME TORCH_EXTENSION_NAME
#else
#define SOAP_TP_EXTENSION_NAME slate_bindings
#endif

#if defined(SOAP_TP_SLATE_WITH_CUDA)
constexpr char gpu_backend[] = "cuda";
constexpr slate::Target target = slate::Target::Devices;
#elif defined(SOAP_TP_SLATE_WITH_ROCM)
constexpr char gpu_backend[] = "rocm";
constexpr slate::Target target = slate::Target::Devices;
#else
constexpr char gpu_backend[] = "none";
constexpr slate::Target target = slate::Target::HostTask;
#endif

void slate_power_iteration_qr_float(
    std::uintptr_t a_address,
    std::uintptr_t q_address,
    std::uintptr_t work_address,
    std::int64_t n,
    std::int64_t lda,
    std::int64_t block_size,
    int process_rows,
    int process_cols)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int process_row = rank / process_cols;
    const int process_col = rank % process_cols;
    const int slate_rank = process_row + process_col * process_rows;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, slate_rank, &communicator);

    float *a = reinterpret_cast<float *>(a_address);
    float *q = reinterpret_cast<float *>(q_address);
    float *work = reinterpret_cast<float *>(work_address);
    // Destroy every SLATE object before freeing its communicator.
    {
        slate::Options options = {{slate::Option::Target, target}};

#if defined(SOAP_TP_SLATE_WITH_CUDA) || defined(SOAP_TP_SLATE_WITH_ROCM)
        float *a_devices[] = {a};
        float *q_devices[] = {q};
        float *work_devices[] = {work};
        auto A = slate::SymmetricMatrix<float>::fromDevices(
            slate::Uplo::Lower, n, a_devices, 1, lda, block_size,
            process_rows, process_cols, communicator);
        auto Q = slate::Matrix<float>::fromDevices(
            n, n, q_devices, 1, lda, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Y = slate::Matrix<float>::fromDevices(
            n, n, work_devices, 1, lda, block_size, block_size,
            process_rows, process_cols, communicator);
#else
        auto A = slate::SymmetricMatrix<float>::fromScaLAPACK(
            slate::Uplo::Lower, n, a, lda, block_size,
            process_rows, process_cols, communicator);
        auto Q = slate::Matrix<float>::fromScaLAPACK(
            n, n, q, lda, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Y = slate::Matrix<float>::fromScaLAPACK(
            n, n, work, lda, block_size, block_size,
            process_rows, process_cols, communicator);
#endif

        slate::symm(slate::Side::Left, 1.0f, A, Q, 0.0f, Y, options);
        slate::TriangularFactors<float> factors;
        slate::geqrf(Y, factors, options);
        slate::set(0.0f, 1.0f, Q, options);
        slate::unmqr(
            slate::Side::Left,
            slate::Op::NoTrans,
            Y,
            factors,
            Q,
            options);
    }

    MPI_Comm_free(&communicator);
}

PYBIND11_MODULE(SOAP_TP_EXTENSION_NAME, module)
{
    module.doc() = "SLATE bindings for distributed power-iteration QR.";
    module.def(
        "compiled_gpu_backend",
        []() { return gpu_backend; },
        "Return the GPU backend selected when the binding was compiled.");
    module.def(
        "slate_power_iteration_qr_float",
        &slate_power_iteration_qr_float,
        py::arg("a"),
        py::arg("q"),
        py::arg("work"),
        py::arg("n"),
        py::arg("lda"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_cols"),
        py::call_guard<py::gil_scoped_release>(),
        R"doc(Perform one distributed power iteration followed by QR.

Computes ``Y = A @ Q`` and overwrites ``Q`` with the explicit orthogonal
factor from ``torch.linalg.qr(Y)`` up to per-column signs. The caller must
perform any estimated-eigenvalue sorting before passing ``Q``. ``A`` is
symmetric and only its lower triangle is read. ``work`` is scratch storage.

``a``, ``q``, and ``work`` are integer addresses of distinct column-major
``float32`` local buffers. Their global matrices use a row-major 2D
block-cyclic process grid over ``MPI_COMM_WORLD``. ``lda`` is the local leading
dimension, and ``process_rows * process_cols`` must equal the MPI world size.

The compiled SLATE profile determines memory placement: CPU builds require
host pointers, while CUDA and ROCm builds require device pointers. The binding
does not move matrix data between ranks or devices.
)doc");
}
