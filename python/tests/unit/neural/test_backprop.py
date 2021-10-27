import platform
import pytest
import dartpy as dart


def test_empty_world():
    world = dart.simulation.World('my world')
    assert world.getNumSkeletons() is 0
    assert world.getNumSimpleFrames() is 0


if __name__ == "__main__":
    pytest.main()
