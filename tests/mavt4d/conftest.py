import os

import pytest
import torch

# src/ is put on sys.path by tests/conftest.py.

# Shared headnodes expose a few cores but torch sizes its pool to the whole machine; oversubscription
# made the graph tests ~8x slower there.
torch.set_num_threads(min(4, torch.get_num_threads()))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MAVT4D_SERVER") == "1":
        return
    skip = pytest.mark.skip(reason="set MAVT4D_SERVER=1 to run tests that need nuScenes or DINOv3")
    for item in items:
        if item.get_closest_marker("server"):
            item.add_marker(skip)
