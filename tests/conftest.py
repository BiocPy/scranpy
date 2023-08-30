"""Dummy conftest.py for scranpy.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import data.mock_data as mocks
import pytest


@pytest.fixture
def mock_data():
    return mocks
