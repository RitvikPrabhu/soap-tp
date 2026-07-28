#include <elpa/elpa.h>
#include <elpa/elpa_version.h>
#include <mpi.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

#ifdef TORCH_EXTENSION_NAME
#define SOAP_TP_EXTENSION_NAME TORCH_EXTENSION_NAME
#else
#define SOAP_TP_EXTENSION_NAME elpa_bindings
#endif

namespace
{

void require_elpa_ok(int error, const char *operation)
{
    if (error != ELPA_OK)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + elpa_strerr(error));
    }
}

void mpi_world_rank_and_size(int &rank, int &size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        throw std::runtime_error("MPI must be initialized before using ELPA");
    }

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized)
    {
        throw std::runtime_error("MPI has already been finalized");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

void initialize_basis_2d_block_cyclic_(
    std::uintptr_t a_address,
    std::uintptr_t eigenvalues_address,
    std::uintptr_t q_address,
    int n,
    int local_rows,
    int local_columns,
    int block_size,
    int process_rows,
    int process_columns)
{
    int rank = 0;
    int world_size = 0;
    mpi_world_rank_and_size(rank, world_size);

    if (a_address == 0 || eigenvalues_address == 0 || q_address == 0)
    {
        throw std::invalid_argument("ELPA buffer addresses must be nonzero");
    }
    if (n <= 0 || local_rows <= 0 || local_columns <= 0 || block_size <= 0)
    {
        throw std::invalid_argument(
            "matrix dimensions and block size must be positive");
    }
    if (process_rows <= 0 || process_columns <= 0 ||
        process_rows * process_columns != world_size)
    {
        throw std::invalid_argument(
            "the process grid must be positive and contain every MPI rank");
    }

    const int process_row = rank / process_columns;
    const int process_column = rank % process_columns;

    const auto *a = reinterpret_cast<const float *>(a_address);
    auto *eigenvalues = reinterpret_cast<float *>(eigenvalues_address);
    auto *q = reinterpret_cast<float *>(q_address);
    const auto local_size =
        static_cast<std::vector<double>::size_type>(local_rows) *
        static_cast<std::vector<double>::size_type>(local_columns);

    std::vector<double> a_double(local_size);
    std::vector<double> eigenvalues_double(n);
    std::vector<double> q_double(local_size);

    for (std::vector<double>::size_type index = 0;
         index < local_size;
         ++index)
    {
        a_double[index] = static_cast<double>(a[index]);
    }

    bool elpa_initialized = false;
    elpa_t handle = nullptr;

    try
    {
        require_elpa_ok(elpa_init(ELPA_API_VERSION), "elpa_init");
        elpa_initialized = true;

        int error = ELPA_OK;
        handle = elpa_allocate(&error);
        require_elpa_ok(error, "elpa_allocate");
        if (handle == nullptr)
        {
            throw std::runtime_error("elpa_allocate returned a null handle");
        }

        const auto set_integer = [&](const char *name, int value)
        {
            int set_error = ELPA_OK;
            elpa_set_integer(handle, name, value, &set_error);
            require_elpa_ok(set_error, name);
        };

        set_integer("na", n);
        set_integer("nev", n);
        set_integer("local_nrows", local_rows);
        set_integer("local_ncols", local_columns);
        set_integer("nblk", block_size);
        set_integer(
            "mpi_comm_parent",
            static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)));
        set_integer("process_row", process_row);
        set_integer("process_col", process_column);

        require_elpa_ok(elpa_setup(handle), "elpa_setup");

        error = ELPA_OK;
        elpa_eigenvectors_double(
            handle,
            a_double.data(),
            eigenvalues_double.data(),
            q_double.data(),
            &error);
        require_elpa_ok(error, "elpa_eigenvectors_double");

        for (std::vector<double>::size_type index = 0;
             index < local_size;
             ++index)
        {
            q[index] = static_cast<float>(q_double[index]);
        }
        for (int index = 0; index < n; ++index)
        {
            eigenvalues[index] =
                static_cast<float>(eigenvalues_double[index]);
        }
    }
    catch (...)
    {
        if (handle != nullptr)
        {
            int error = ELPA_OK;
            elpa_deallocate(handle, &error);
        }
        if (elpa_initialized)
        {
            int error = ELPA_OK;
            elpa_uninit(&error);
        }
        throw;
    }

    int deallocate_error = ELPA_OK;
    elpa_deallocate(handle, &deallocate_error);

    int uninit_error = ELPA_OK;
    elpa_uninit(&uninit_error);

    require_elpa_ok(deallocate_error, "elpa_deallocate");
    require_elpa_ok(uninit_error, "elpa_uninit");
}

} // namespace

PYBIND11_MODULE(SOAP_TP_EXTENSION_NAME, module)
{
    module.doc() = "Minimal CPU bindings for ELPA";

    module.def(
        "compiled_gpu_backend",
        []()
        { return "none"; });

    module.def(
        "mpi_world_rank_and_size",
        []()
        {
            int rank = 0;
            int size = 0;
            mpi_world_rank_and_size(rank, size);
            return py::make_tuple(rank, size);
        });

    module.def(
        "initialize_basis_2d_block_cyclic_",
        &initialize_basis_2d_block_cyclic_,
        py::arg("a"),
        py::arg("eigenvalues"),
        py::arg("q"),
        py::arg("n"),
        py::arg("local_rows"),
        py::arg("local_columns"),
        py::arg("block_size"),
        py::arg("process_rows"),
        py::arg("process_columns"),
        py::call_guard<py::gil_scoped_release>());

    // Compatibility name used by soap_tp.ops.factorizations.
    module.attr("elpa_eigenvectors_2d_block_cyclic_float") =
        module.attr("initialize_basis_2d_block_cyclic_");
}
