mpiexec -n 4 python -m pytest test_slate_bindings.py --output-folder ./logs
mpiexec -n 4 python -m pytest test_elpa_bindings.py --output-folder ./logs