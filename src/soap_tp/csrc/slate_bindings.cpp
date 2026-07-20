#include <pybind11/pybind11.h>

#include <mpi.h>
#include <slate/slate.hh>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#ifdef TORCH_EXTENSION_NAME
#define SOAP_TP_EXTENSION_NAME TORCH_EXTENSION_NAME
#else
#define SOAP_TP_EXTENSION_NAME slate_bindings
#endif

void get_mpi_world_info(int &rank, int &size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("MPI must be initialized before using SLATE");
    }

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        throw std::runtime_error("MPI has already been finalized");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

void validate_process_grid(int process_rows, int process_cols, int world_size)
{
    if (process_rows <= 0 || process_cols <= 0 ||
        process_rows * process_cols != world_size) {
        throw std::invalid_argument(
            "process grid must be positive and contain every MPI world rank");
    }
}

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

void slate_symmetric_multiply_float(
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
    int world_size = 0;
    get_mpi_world_info(rank, world_size);
    validate_process_grid(process_rows, process_cols, world_size);
    if (a_address == 0 || q_address == 0 || work_address == 0) {
        throw std::invalid_argument("SLATE buffer addresses must be nonzero");
    }
    if (n <= 0 || lda <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "matrix size, leading dimension, and block size must be positive");
    }

    const int process_row = rank / process_cols;
    const int process_col = rank % process_cols;
    const int slate_rank = process_row + process_col * process_rows;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, slate_rank, &communicator);
    float *a = reinterpret_cast<float *>(a_address);
    float *q = reinterpret_cast<float *>(q_address);
    float *work = reinterpret_cast<float *>(work_address);
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
    }

    MPI_Comm_free(&communicator);
}

void slate_qr_float(
    std::uintptr_t matrix_address,
    std::uintptr_t q_address,
    std::int64_t n,
    std::int64_t ldmatrix,
    std::int64_t ldq,
    std::int64_t block_size,
    int process_rows,
    int process_cols)
{
    int rank = 0;
    int world_size = 0;
    get_mpi_world_info(rank, world_size);
    validate_process_grid(process_rows, process_cols, world_size);
    if (matrix_address == 0 || q_address == 0) {
        throw std::invalid_argument("SLATE buffer addresses must be nonzero");
    }
    if (n <= 0 || ldmatrix <= 0 || ldq <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "matrix size, leading dimensions, and block size must be positive");
    }

    const int process_row = rank / process_cols;
    const int process_col = rank % process_cols;
    const int slate_rank = process_row + process_col * process_rows;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, slate_rank, &communicator);

    float *matrix = reinterpret_cast<float *>(matrix_address);
    float *q = reinterpret_cast<float *>(q_address);
    {
        slate::Options options = {{slate::Option::Target, target}};

#if defined(SOAP_TP_SLATE_WITH_CUDA) || defined(SOAP_TP_SLATE_WITH_ROCM)
        float *matrix_devices[] = {matrix};
        float *q_devices[] = {q};
        auto Y = slate::Matrix<float>::fromDevices(
            n, n, matrix_devices, 1, ldmatrix, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Q = slate::Matrix<float>::fromDevices(
            n, n, q_devices, 1, ldq, block_size, block_size,
            process_rows, process_cols, communicator);
#else
        auto Y = slate::Matrix<float>::fromScaLAPACK(
            n, n, matrix, ldmatrix, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Q = slate::Matrix<float>::fromScaLAPACK(
            n, n, q, ldq, block_size, block_size,
            process_rows, process_cols, communicator);
#endif

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

void slate_rotation_float(
    std::uintptr_t q_left_address,
    std::uintptr_t matrix_address,
    std::uintptr_t q_right_address,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t ldq_left,
    std::int64_t ldmatrix,
    std::int64_t ldq_right,
    std::int64_t block_size,
    int process_rows,
    int process_cols,
    bool forward)
{
    int rank = 0;
    int world_size = 0;
    get_mpi_world_info(rank, world_size);
    validate_process_grid(process_rows, process_cols, world_size);
    if (q_left_address == 0 || matrix_address == 0 ||
        q_right_address == 0) {
        throw std::invalid_argument("SLATE buffer addresses must be nonzero");
    }
    if (rows <= 0 || columns <= 0 || ldq_left <= 0 || ldmatrix <= 0 ||
        ldq_right <= 0 || block_size <= 0) {
        throw std::invalid_argument(
            "matrix dimensions, leading dimensions, and block size must be positive");
    }

    const int process_row = rank / process_cols;
    const int process_col = rank % process_cols;
    const int slate_rank = process_row + process_col * process_rows;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, slate_rank, &communicator);

    float *q_left = reinterpret_cast<float *>(q_left_address);
    float *matrix = reinterpret_cast<float *>(matrix_address);
    float *q_right = reinterpret_cast<float *>(q_right_address);

    {
        slate::Options options = {{slate::Option::Target, target}};

#if defined(SOAP_TP_SLATE_WITH_CUDA) || defined(SOAP_TP_SLATE_WITH_ROCM)
        float *q_left_devices[] = {q_left};
        float *matrix_devices[] = {matrix};
        float *q_right_devices[] = {q_right};
        auto Q_left = slate::Matrix<float>::fromDevices(
            rows, rows, q_left_devices, 1, ldq_left, block_size, block_size,
            process_rows, process_cols, communicator);
        auto X = slate::Matrix<float>::fromDevices(
            rows, columns, matrix_devices, 1, ldmatrix, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Q_right = slate::Matrix<float>::fromDevices(
            columns, columns, q_right_devices, 1, ldq_right,
            block_size, block_size, process_rows, process_cols, communicator);
#else
        auto Q_left = slate::Matrix<float>::fromScaLAPACK(
            rows, rows, q_left, ldq_left, block_size, block_size,
            process_rows, process_cols, communicator);
        auto X = slate::Matrix<float>::fromScaLAPACK(
            rows, columns, matrix, ldmatrix, block_size, block_size,
            process_rows, process_cols, communicator);
        auto Q_right = slate::Matrix<float>::fromScaLAPACK(
            columns, columns, q_right, ldq_right, block_size, block_size,
            process_rows, process_cols, communicator);
#endif

        auto H = X.emptyLike();
        H.insertLocalTiles(target);
        if (forward) {
            auto Q_left_transposed = slate::transpose(Q_left);
            slate::gemm(1.0f, Q_left_transposed, X, 0.0f, H, options);
            slate::gemm(1.0f, H, Q_right, 0.0f, X, options);
        }
        else {
            slate::gemm(1.0f, Q_left, X, 0.0f, H, options);
            auto Q_right_transposed = slate::transpose(Q_right);
            slate::gemm(1.0f, H, Q_right_transposed, 0.0f, X, options);
        }
    }

    MPI_Comm_free(&communicator);
}

void slate_forward_rotation_float(
    std::uintptr_t q_left,
    std::uintptr_t matrix,
    std::uintptr_t q_right,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t ldq_left,
    std::int64_t ldmatrix,
    std::int64_t ldq_right,
    std::int64_t block_size,
    int process_rows,
    int process_cols)
{
    slate_rotation_float(
        q_left, matrix, q_right, rows, columns,
        ldq_left, ldmatrix, ldq_right, block_size,
        process_rows, process_cols, true);
}

void slate_backward_rotation_float(
    std::uintptr_t q_left,
    std::uintptr_t matrix,
    std::uintptr_t q_right,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t ldq_left,
    std::int64_t ldmatrix,
    std::int64_t ldq_right,
    std::int64_t block_size,
    int process_rows,
    int process_cols)
{
    slate_rotation_float(
        q_left, matrix, q_right, rows, columns,
        ldq_left, ldmatrix, ldq_right, block_size,
        process_rows, process_cols, false);
}

PYBIND11_MODULE(SOAP_TP_EXTENSION_NAME, module)
{
    module.doc() = "SLATE bindings for distributed SOAP matrix operations.";
    module.def(
        "compiled_gpu_backend",
        []() { return gpu_backend; },
        "Return the GPU backend selected when the binding was compiled.");
    module.def(
        "mpi_world_rank_and_size",
        []() {
            int rank = 0;
            int size = 0;
            get_mpi_world_info(rank, size);
            return py::make_tuple(rank, size);
        },
        "Return the initialized MPI world rank and size.");
    module.def(
        "slate_symmetric_multiply_float",
        &slate_symmetric_multiply_float,
        py::arg("a"),
        py::arg("q"),
        py::arg("work"),
        py::arg("n"),
        py::arg("lda"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_cols"),
        py::call_guard<py::gil_scoped_release>(),
        "Compute work = A @ Q for a distributed symmetric A.");
    module.def(
        "slate_qr_float",
        &slate_qr_float,
        py::arg("matrix"),
        py::arg("q"),
        py::arg("n"),
        py::arg("ldmatrix"),
        py::arg("ldq"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_cols"),
        py::call_guard<py::gil_scoped_release>(),
        "Overwrite q with the distributed QR factor of matrix.");
    module.def(
        "slate_forward_rotation_float",
        &slate_forward_rotation_float,
        py::arg("q_left"),
        py::arg("matrix"),
        py::arg("q_right"),
        py::arg("rows"),
        py::arg("columns"),
        py::arg("ldq_left"),
        py::arg("ldmatrix"),
        py::arg("ldq_right"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_cols"),
        py::call_guard<py::gil_scoped_release>());
    module.def(
        "slate_backward_rotation_float",
        &slate_backward_rotation_float,
        py::arg("q_left"),
        py::arg("matrix"),
        py::arg("q_right"),
        py::arg("rows"),
        py::arg("columns"),
        py::arg("ldq_left"),
        py::arg("ldmatrix"),
        py::arg("ldq_right"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_cols"),
        py::call_guard<py::gil_scoped_release>());
}
