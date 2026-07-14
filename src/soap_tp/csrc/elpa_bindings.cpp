#include <pybind11/pybind11.h>
#include <elpa/elpa.h>
#include <elpa/elpa_configured_options.h>

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
#elif ELPA_WITH_SYCL_GPU_VERSION == 1
    return "sycl";
#else
    return "none";
#endif
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
}
