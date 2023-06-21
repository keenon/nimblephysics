import nimblephysics as nimble

skeleton = nimble.RajagopalHumanBodyModel().skeleton

nodes = skeleton.getBodyNodes()
shapeNode = nodes[0].getShapeNode(0)
shapeNode.getName()
