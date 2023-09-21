"""Unit tests for transport_performance/population/vectorpop.py.

At this stage, VectorPop has not been implemented, so the extent of this test
suite is to ensure the NotImplementedError is raised.

TODO: add unit test for methods once VectorPop has been implemented.
"""

import pytest

from transport_performance.population.vectorpop import VectorPop


class TestVectorPop:
    """A class to test VectorPop methods."""

    def test_vectorpop_not_implemented(self) -> None:
        """Test VectorPop raises a not implemented error."""
        with pytest.raises(
            NotImplementedError,
            match="This class has not yet been implemented",
        ):
            VectorPop()
