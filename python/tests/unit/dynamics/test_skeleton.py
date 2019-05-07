import platform
import pytest
from dartpy.dynamics import Skeleton


def test_empty_skeleton():
    skel = Skeleton.create()
    assert skel.getNumBodyNodes() is 0
    assert skel.getNumJoints() is 0


def test_skeleton_creation():
    skel = Skeleton.create()
    body1 = skel.createBodyNode(None, 'Revolute')
    body2 = skel.createBodyNode(body1, 'Revolute')
    body3 = body2.createChildBodyNode('Revolute')
    body4 = body2.createChildBodyNode('Revolute')
    assert skel.getNumBodyNodes() is 4
    assert skel.getNumJoints() is 4
    assert body1.getNumChildBodyNodes() is 1
    assert body2.getNumChildBodyNodes() is 2
    assert body3.getNumChildBodyNodes() is 0
    assert body4.getNumChildBodyNodes() is 0

    body5 = body2.createChildBodyNode('InvalidJointType')
    assert skel.getNumBodyNodes() is 4
    assert skel.getNumJoints() is 4
    assert body2.getNumChildBodyNodes() is 2


if __name__ == "__main__":
    pytest.main()
