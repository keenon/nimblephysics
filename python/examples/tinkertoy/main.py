import math
import numpy as np
import dartpy as dart


class TinkertoyWorldNode(dart.gui.osg.RealTimeWorldNode):
    DefaultBlockLength = 0.5
    DefaultBlockWidth = 0.075
    DefaultJointRadius = 1.5 * DefaultBlockWidth / 2.0
    BalsaWoodDensity = 0.16 * 10e3  # kg/m^3
    DefaultBlockMass = BalsaWoodDensity * DefaultBlockLength * DefaultBlockWidth**2
    DefaultDamping = 0.4

    DefaultSimulationColor = [0.5, 0.5, 1, 1]
    DefaultPausedColor = np.array([0xEE, 0xC9, 0x00, 0.0]) / 255.0 + [0, 0, 0, 1]
    DefaultSelectedColor = [1, 0, 0, 1]
    DefaultForceBodyColor = [1, 0, 0.5, 1]
    DefaultForceLineColor = [1, 0.63, 0, 1]

    MaxForce = 200.0
    DefaultForceCoeff = 100.0
    MaxForceCoeff = 1000.0
    MinForceCoeff = 10.0
    ForceIncrement = 10.0

    def __init__(self, world):
        super(InputHandler, self).__init__(world)
        self.force_coeff = DefaultForceCoeff
        self.was_simulating = False

        self.weld_joint_shape = None
        self.revolute_joint_shape = None
        self.ball_joint_shape = None
        self.block_shape = None
        self.block_offset = dart.math.Isometry3()

        self.picked_node = None
        self.picked_point = np.zeros(3)

        self.force_line = None

        self.target = dart.gui.osg.InteractiveFrame(dart.dynamics.Frame.World())
        self.getWorld().addSimpleFrame(self.target)

        self.createShapes()
        self.createInitialToy1()
        self.createInitialToy2()
        self.createForceLine()

        def setAllBodyColors(self, color):
            for i in range(getWorld().getNumSkeletons()):
                skel = getWorld().getSkeleton(i)
                for j in range(skel.getNumBodyNodes()):
                    bn = skel.getBodyNode(j)
                    for k in range(bn.getNumShapeNodes()):
                        bn.getShapeNode(k).getVisualAspect().setColor(color)

        def setPickedNodeColor(self, color):
          if self.picked_node is None:
            return

          for i in range(self.picked_node.getNumShapeNodes()):
            picked_node.getShapeNode(i).getVisualAspect().setColor(color)

        def resetForceLine(self):
          if self.picked_node is not None:
            force_line.setVertex(0, picked_node.getWorldTransform() * picked_point)
            force_line.setVertex(1, target.getWorldTransform().translation())
          else:
            force_line.setVertex(0, np.zeros(3))
            force_line.setVertex(1, np.zeros(3))

        def customPreRefresh(self):
          if self.isSimulating():
            self.setAllBodyColors(self.DefaultSimulationColor)
            self.setPickedNodeColor(self.DefaultForceBodyColor)
          else:
            self.setAllBodyColors(self.DefaultPausedColor)
            self.setPickedNodeColor(self.DefaultSelectedColor)

          self.resetForceLine()

        def customPreStep(self):
          if self.picked_node:
            F = force_coeff * (self.target.getWorldTransform().translation() - self.picked_node.getWorldTransform() * picked_point)

            F_norm = F.norm()
            if F_norm > self.MaxForce:
              F = self.MaxForce * F / F_norm

            picked_node.addExtForce(F, picked_point)

        def handlePick(self, pick):
          bn = pick.frame.getParentFrame()

          if isinstance(bn, dart.dynamics.BodyNode):
            return

          picked_node = bn
          picked_point = bn.getWorldTransform().inverse() * pick.position

          tf = bn.getWorldTransform()
          pos = pick.position + pick.normal.normalized() * DefaultBlockWidth / 2.0
          tf.set_translation(pos)

          target.setTransform(tf)

        def clearPick(self):
          self.picked_node = None
          self.target.setTransform(dart.math.Isometry3.Identity())

        def deletePick(self):
          if picked_node is None:
            return

          if self.isSimulating():
            print(' -- Please pause simulation [using the Spacebar] before attempting to delete blocks.')
            return

          temporary = self.picked_node.remove()
          for i in range(temporary.getNumBodyNodes()):
            self.viewer.disableDragAndDrop(self.viewer.enableDragAndDrop(temporary.getBodyNode(i)))

          getWorld().getConstraintSolver().getCollisionGroup().removeShapeFramesOf(temporary)

          self.clearPick()

        def createShapes(self):
          self.createWeldJointShape()
          self.createRevoluteJointShape()
          self.createBallJointShape()
          self.createBlockShape()

        def createWeldJointShape():
          self.weld_joint_shape
              = dart.dynamics.BoxShape([2.0 * self.DefaultJointRadius, self.DefaultBlockWidth, self.DefaultBlockWidth])

          self.weld_joint_shape.addDataVariance(dart.dynamics.Shape.DYNAMIC_COLOR)

        def createRevoluteJointShape():
          self.revolute_joint_shape = dart.dynamics.CylinderShape(self.DefaultJointRadius, 1.5 * self.DefaultBlockWidth)
          self.revolute_joint_shape.addDataVariance(dart.dynamics.Shape.DYNAMIC_COLOR)

        def createBallJointShape(self):
          self.ball_joint_shape = dart.dynamics.SphereShape(self.DefaultJointRadius)
          self.ball_joint_shape.addDataVariance(dart.dynamics.Shape.DYNAMIC_COLOR)

        def createBlockShape(self)
          self.block_shape = dart.dynamics.BoxShape([self.DefaultBlockLength, self.DefaultBlockWidth, self.DefaultBlockWidth])
          self.block_shape.addDataVariance(dart.dynamics.Shape.DYNAMIC_COLOR)

          self.block_offset = dart.math.Isometry3()
          pos = self.block_offset.translation()
          pos[0] = self.DefaultBlockLength / 2.0
          self.block_offset.set_translation(pos)

        def addBlock(self, parent, rel_tf, joint_shape):
          if self.isSimulating():
            print(' -- Please pause simulation [using the Spacebar] before attempting to add new bodies')
            return None, None

          dart.dynamics.SkeletonPtr skel
          if self.parent)
          {
            skel = parent.getSkeleton()
          }
          else
          {
            skel = dart.dynamics.Skeleton.create(
                "toy_#" + std.to_string(getWorld().getNumSkeletons() + 1))
            getWorld().addSkeleton(skel)
          }

          auto pair = skel.createJointAndBodyNodePair<JointType>(parent)
          JointType* joint = pair.first
          dart.dynamics.BodyNode* bn = pair.second
          bn.setName("block_#" + std.to_string(skel.getNumBodyNodes()))
          joint.setName("joint_#" + std.to_string(skel.getNumJoints()))

          joint.setTransformFromParentBodyNode(relTf)
          for (size_t i = 0 i < joint.getNumDofs() ++i)
            joint.getDof(i).setDampingCoefficient(DefaultDamping)

          bn.createShapeNodeWith<
              dart.dynamics.VisualAspect,
              dart.dynamics.CollisionAspect,
              dart.dynamics.DynamicsAspect>(jointShape)

          dart.dynamics.ShapeNode* block = bn.createShapeNodeWith<
              dart.dynamics.VisualAspect,
              dart.dynamics.CollisionAspect,
              dart.dynamics.DynamicsAspect>(block_shape)
          block.setRelativeTransform(block_offset)

          dart.dynamics.Inertia inertia = bn.getInertia()
          inertia.setMass(DefaultBlockMass)
          inertia.setMoment(block_shape.computeInertia(DefaultBlockMass))
          inertia.setLocalCOM(DefaultBlockLength / 2.0 * Eigen.Vector3d.UnitX())
          bn.setInertia(inertia)

          getWorld().getConstraintSolver().getCollisionGroup().addShapeFramesOf(
              bn)

          clearPick()

          return std.make_pair(joint, bn)

#        Eigen.Isometry3d getRelTf() const
#        {
#          return picked_node ? target.getTransform(picked_node)
#                             : target.getWorldTransform()
#        }

#        void addWeldJointBlock()
#        {
#          addWeldJointBlock(picked_node, getRelTf())
#        }

#        dart.dynamics.BodyNode* addWeldJointBlock(
#            dart.dynamics.BodyNode* parent, const Eigen.Isometry3d& relTf)
#        {
#          return addBlock<dart.dynamics.WeldJoint>(parent, relTf, weld_joint_shape)
#              .second
#        }

#        void addRevoluteJointBlock()
#        {
#          addRevoluteJointBlock(picked_node, getRelTf())
#        }

#        dart.dynamics.BodyNode* addRevoluteJointBlock(
#            dart.dynamics.BodyNode* parent, const Eigen.Isometry3d& relTf)
#        {
#          auto pair = addBlock<dart.dynamics.RevoluteJoint>(
#              parent, relTf, revolute_joint_shape)

#          if self.pair.first)
#            pair.first.setAxis(Eigen.Vector3d.UnitZ())

#          return pair.second
#        }

#        void addBallJointBlock()
#        {
#          addBallJointBlock(picked_node, getRelTf())
#        }

#        dart.dynamics.BodyNode* addBallJointBlock(
#            dart.dynamics.BodyNode* parent, const Eigen.Isometry3d& relTf)
#        {
#          return addBlock<dart.dynamics.BallJoint>(parent, relTf, ball_joint_shape)
#              .second
#        }

#        void createInitialToy1()
#        {
#          Eigen.Isometry3d tf(Eigen.Isometry3d.Identity())
#          tf.rotate(Eigen.AngleAxisd(45.0 * M_PI / 180.0, Eigen.Vector3d.UnitY()))
#          dart.dynamics.BodyNode* bn = addBallJointBlock(nullptr, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.translation()[0] = DefaultBlockLength
#          tf.linear() = Eigen.Matrix3d.Identity()
#          tf.prerotate(
#              Eigen.AngleAxisd(90.0 * M_PI / 180.0, Eigen.Vector3d.UnitX()))
#          bn = addRevoluteJointBlock(bn, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.rotate(Eigen.AngleAxisd(90.0 * M_PI / 180.0, Eigen.Vector3d.UnitZ()))
#          bn = addWeldJointBlock(bn, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.translation()[0] = DefaultBlockLength / 2.0
#          tf.translation()[2] = DefaultBlockWidth
#          tf.rotate(
#              Eigen.AngleAxisd(-30.0 * M_PI / 180.0, Eigen.Vector3d.UnitZ()))
#          bn = addBallJointBlock(bn, tf)
#        }

#        void createInitialToy2()
#        {
#          Eigen.Isometry3d tf(Eigen.Isometry3d.Identity())
#          tf.rotate(Eigen.AngleAxisd(90.0 * M_PI / 180.0, Eigen.Vector3d.UnitY()))
#          tf.pretranslate(-1.0 * Eigen.Vector3d.UnitX())
#          dart.dynamics.BodyNode* bn = addBallJointBlock(nullptr, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.translation()[0] = DefaultBlockLength
#          tf.translation()[2] = DefaultBlockLength / 2.0
#          tf.rotate(Eigen.AngleAxisd(90.0 * M_PI / 180.0, Eigen.Vector3d.UnitY()))
#          bn = addWeldJointBlock(bn, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.rotate(
#              Eigen.AngleAxisd(-90.0 * M_PI / 180.0, Eigen.Vector3d.UnitX()))
#          tf.rotate(
#              Eigen.AngleAxisd(-90.0 * M_PI / 180.0, Eigen.Vector3d.UnitZ()))
#          tf.translation()[2] = DefaultBlockWidth / 2.0
#          addRevoluteJointBlock(bn, tf)

#          tf.translation()[0] = DefaultBlockLength
#          bn = addRevoluteJointBlock(bn, tf)

#          tf = Eigen.Isometry3d.Identity()
#          tf.translation()[0] = DefaultBlockLength
#          addBallJointBlock(bn, tf)
#        }

#        void createForceLine()
#        {
#          dart.dynamics.SimpleFramePtr lineFrame
#              = std.make_shared<dart.dynamics.SimpleFrame>(
#                  dart.dynamics.Frame.World())

#          force_line = std.make_shared<dart.dynamics.LineSegmentShape>(
#              np.zeros(3), np.zeros(3), 3.0)
#          force_line.addDataVariance(dart.dynamics.Shape.DYNAMIC_VERTICES)

#          lineFrame.setShape(force_line)
#          lineFrame.createVisualAspect()
#          lineFrame.getVisualAspect().setColor(DefaultForceLineColor)

#          getWorld().addSimpleFrame(lineFrame)
#        }

#        void setForceCoeff(double coeff)
#        {
#          force_coeff = coeff

#          if self.force_coeff > MaxForceCoeff)
#            force_coeff = MaxForceCoeff
#          else if self.force_coeff < MinForceCoeff)
#            force_coeff = MinForceCoeff
#        }

#        double getForceCoeff() const
#        {
#          return force_coeff
#        }

#        void incrementForceCoeff()
#        {
#          force_coeff += ForceIncrement
#          if self.force_coeff > MaxForceCoeff)
#            force_coeff = MaxForceCoeff

#          std.cout << "[Force Coefficient: " << force_coeff << "]" << std.endl
#        }

#        void decrementForceCoeff()
#        {
#          force_coeff -= ForceIncrement
#          if self.force_coeff < MinForceCoeff)
#            force_coeff = MinForceCoeff

#          std.cout << "[Force Coefficient: " << force_coeff << "]" << std.endl
#        }

#        void reorientTarget()
#        {
#          Eigen.Isometry3d tf = target.getWorldTransform()
#          tf.linear() = Eigen.Matrix3d.Identity()
#          target.setTransform(tf)
#        }

#  float force_coeff

#protected:
#  void setupViewer() override
#  {
#    viewer.enableDragAndDrop(target.get())
#  }
#}


class TinkertoyInputHandler(dart.gui.osg.GUIEventHandler):
    def __init__(self, viewer, node):
        super(InputHandler, self).__init__()
        self.viewer = viewer
        self.node = node

    def handle(self, ea, aa):
        if ea.getEventType() == dart.gui.osg.GUIEventAdapter.KEYDOWN:
            if ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_Tab:
                self.viewer.home()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_1:
                self.node.addWeldJointBlock()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_2:
                self.node.addRevoluteJointBlock()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_3:
                self.node.addBallJointBlock()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_BackSpace:
                self.node.clearPick()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_Delete:
                self.node.deletePick()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_Up:
                self.node.incrementForceCoeff()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_Down:
                self.node.decrementForceCoeff()
                return True
            elif ea.getKey() == '`':
                self.node.reorientTarget()
                return True
            elif ea.getKey() == dart.gui.osg.GUIEventAdapter.KEY_Return:
                if not self.viewer.isRecording():
                    self.viewer.record('/screencap')
                else:
                    self.viewer.pauseRecording()
                return True
        return False


class TinkertoyMouseHandler(dart.gui.osg.MouseEventHandler):
    def __init__(self, input_handler):
        super(InputHandler, self).__init__()
        self.input_handler = input_handler

    def update(self):
        viewer = self.input_handler.viewer
        node = self.input_handler.node

        event = viewer.getDefaultEventHandler().getButtonEvent(dart.gui.osg.LEFT_MOUSE)

        if event == dart.gui.osg.BUTTON_PUSH:
            picks = viewer.getDefaultEventHandler().getButtonPicks(dart.gui.osg.LEFT_MOUSE, dart.gui.osg.BUTTON_PUSH)

            if len(picks) == 0:
                return

            node.handlePick(picks[0])


class MyWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, chain):
        super(MyWorldNode, self).__init__(world)
        self.chain = chain

    def customPreStep(self):
        pass


def main():
    world = dart.simulation.World()

    coordinates = dart.gui.osg.InteractiveFrame(dart.dynamics.Frame.World(), 'coordinates', dart.math.Isometry3(), 0.2)

#    for i in range(3):
#        for j in range(3):
#            coordinates.getTool(...)

    world.addSimpleFrame(coordinates)

    node = TinkertoyWorldNode(world, chain)

    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    # Grid settings
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.ZX)
    grid.setOffset([0, -0.55, 0])
    viewer.addAttachment(grid)

    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0.6, 0.3, 0.6], [0, -0.2, 0], [0, 1, 0])
    viewer.run()


if __name__ == "__main__":
    main()
