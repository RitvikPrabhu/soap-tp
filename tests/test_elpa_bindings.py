"""Litmus test: can PyTorch compile and load the ELPA binding?"""

import os
from pathlib import Path
import sys
import unittest

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
PROFILE = os.environ.get("ELPA_PROFILE", "cpu")
PREFIX = ROOT / "build" / "elpa-install" / PROFILE


class TestElpaBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the extension against the ELPA profile built by build_elpa.sh.
        include_dir = next((PREFIX / "include").glob("elpa-*"))
        library_dir = PREFIX / "lib"
        os.environ["CXX"] = "mpicxx"
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "build" / "torch-extensions")

        cls.binding = load(
            name=f"_soap_tp_elpa_{PROFILE}_test",
            sources=[str(ROOT / "src/soap_tp/csrc/elpa_bindings.cpp")],
            extra_include_paths=[str(include_dir)],
            extra_cflags=["-O0"],
            extra_ldflags=[
                f"-L{library_dir}",
                "-lelpa",
                f"-Wl,-rpath,{library_dir}",
            ],
        )

    def test_binding_loads(self):
        # If this call works, the C++ extension compiled, linked, and imported.
        self.assertIn(
            self.binding.compiled_gpu_backend(),
            {"none", "cuda", "rocm", "sycl"},
        )


if __name__ == "__main__":
    unittest.main()
