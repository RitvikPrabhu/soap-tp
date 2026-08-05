from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-folder",
        action="store",
        default=None,
        help="Write numerical comparison logs to this folder.",
    )


@pytest.fixture(scope="session")
def output_folder(request):
    value = request.config.getoption("--output-folder")
    if value is None:
        return None

    folder = Path(value).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder
