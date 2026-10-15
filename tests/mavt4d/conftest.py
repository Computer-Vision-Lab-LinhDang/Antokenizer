import os

import pytest

# src/ is put on sys.path by tests/conftest.py.


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MAVT4D_SERVER") == "1":
        return
    skip = pytest.mark.skip(reason="set MAVT4D_SERVER=1 to run tests that need nuScenes or DINOv3")
    for item in items:
        if item.get_closest_marker("server"):
            item.add_marker(skip)
