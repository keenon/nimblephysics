import dartpy as dart
from dartpy.dynamics import Skeleton


def create_robot():
    skel = Skeleton.create()
    body1 = skel.createBodyNode(None, 'Revolute')
    body2 = skel.createBodyNode(body1, 'Revolute')
    body3 = body2.createChildBodyNode('Revolute')
    body4 = body2.createChildBodyNode('Revolute')

    return skel


def main():
    world = dart.simulation.World.create()

    robot = create_robot()
    world.addSkeleton(robot)

    node = dart.gui.osg.RealTimeWorldNode(world)

    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.run()


if __name__ == "__main__":
    main()
