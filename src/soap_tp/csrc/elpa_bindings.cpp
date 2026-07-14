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

namespace {

const char* compiled_gpu_backend()
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
        reinterpret_cast<float*>(a),
        reinterpret_cast<float*>(ev),
        reinterpret_cast<float*>(q),
        &error);
    return error;
}

}  // namespace

PYBIND11_MODULE(SOAP_TP_EXTENSION_NAME, handle)
{
    handle.doc() = "Python bindings for ELPA library";
    handle.def(
        "compiled_gpu_backend",
        &compiled_gpu_backend,
        "Return the GPU backend compiled into ELPA");
    handle.def(
        "elpa_error_string",
        [](int error) { return elpa_strerr(error); },
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
        "Compute single-precision eigenvectors using an ELPA handle and raw "
        "array addresses. Returns the ELPA error code.");
}
