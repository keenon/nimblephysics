---
title: "Getting Started"
date: 2020-10-01T11:00:57-07:00
draft: false
menu:
  main:
    parent: "tutorials"
    name: "Getting Started"
    weight: 100
---

# Getting started with **DiffDART**

## Installation

Before you install DiffDART, you'll need Python 3.8 installed. **In alpha, we currently only support Python 3.8 on Linux and Mac**.

On Ubuntu, that's: {{< shell >}}sudo apt-get install -y python3 python3-pip{{< /shell >}}

On Mac: {{< shell >}}brew install python3{{< /shell >}}

Then, you can install DiffDART from PyPI with {{< shell >}}pip3 install diffdart{{< /shell >}}

{{< warning >}}

#### Pre-release alpha warning:

This is alpha quality software! Strap on your hardhat, and expect plenty of bugs.

{{< /warning >}}

## Introduction

DiffDART, at its heart, is just a fork of DART that allows you to pass gradients backwards through it.

You can use DiffDART in a few different ways:

- For **Deep learning**: Physics as a PyTorch layer
- For **Robotics**: High-performance trajectory optimization (and coming soon - system characterization and state estimation)
- For **Research**: a native interface for getting raw Jacobians

All of these ways require first creating a world to simulate, which is where we'll start:

## Creating a World

**TODO(keenon): Write our own docs about this.**

{{< warning >}}

#### Pre-release alpha warning:

Currently, we only support **Box colliders**! Mesh support is coming, but requires that I fork FCL and add features, which is a big headache.

{{< /warning >}}

In the end, you'll have a world like this:

{{< code python >}}

```
import diffdart as dart

world = dart.simulation.World()
world.setGravity([0, -9.81, 0])

# Box

boxSkeleton = dart.dynamics.Skeleton()

rootJoint, root = boxSkeleton.createTranslationalJoint2DAndBodyNodePair()
rootJoint.setXYPlane()
rootShape = root.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
rootVisual = rootShape.createVisualAspect()
rootShape.createCollisionAspect()
rootVisual.setColor([0.7, 0.7, 0.7])

world.addSkeleton(boxSkeleton) # Don't forget this step!

# Floor

floor = dart.dynamics.Skeleton()
floor.setName('floor')  # the name "floor" is important for rendering shadows

floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
floorOffset = dart.math.Isometry3()
floorOffset.set_translation([0, -0.7, 0])
floorJoint.setTransformFromParentBodyNode(floorOffset)
floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, .5]))
floorVisual = floorShape.createVisualAspect()
floorVisual.setColor([0.5, 0.5, 0.5])
floorShape.createCollisionAspect()

world.addSkeleton(floor) # Don't forget this step!
```

{{< /code >}}

## **Deep learning**: Physics as a PyTorch layer

This use case was the original motivation for building this library. Imagine all the hybrid neural-physical models we can train when we've got differentiable physics timestep as a non-linearity!

The data flow through your PyTorch graph for a single physics timestep will look like this:

![Image](/assets/images/data-flow-fwd.svg)

If you want to run a simulation, simply chain together as many timesteps as you want:

![Simulation](/images/Physics_Simulation.png)

If you want to train a simple (single-shooting) trajectory, just make the forces at each timestep trainable Tensors:

![Simulation](/images/simple_example.png)

If you want to do something more complex, feel free to get as complicated as you like!

![Simulation](/images/larger_graph_example.png)

Using physics as a layer in your neural network is pretty straightforward. It requires two steps:

1. Create a world
2. Apply a layer to the world to compute a timestep

Once you've created the world, it's straightforward to give it a (differentiable) time step:

{{< code python >}}

```
import diffdart as dart
import torch

### Create your world, see previous section

# You need three PyTorch Tensors as input, refer to the
# data flow diagram above

pos = torch.zeros([world.getNumDofs()],
                  dtype=torch.float64,
                  requires_grad=True)
vel = torch.zeros([world.getNumDofs()],
                  dtype=torch.float64,
                  requires_grad=True)
forces = torch.zeros([world.getNumDofs()],
                     dtype=torch.float64,
                     requires_grad=True)

# This call runs a forward timestep, and saves all
# the info inside itself required for a backwards pass.

new_pos, new_vel = dart.dart_layer(world,
                                   pos,
                                   vel,
                                   forces)

# `new_pos` and `new_vel` are both PyTorch tensors you
# can now use however you'd like

```

{{< /code >}}

## **Robotics**: Trajectory Optimization

DiffDART comes with a built-in C++ trajectory optimization framework, for solving optimization problems with constraints. By contrast to just using PyTorch, when you use the optimization framework we transcribe your problem into [IPOPT](https://coin-or.github.io/Ipopt/) and can enforce arbitrary constraints on the solution, in addition to doing ordinary gradient descent.

To use DiffDART's built-in trajectory optimizer, you'll want to:

1. Set up your world
2. Define a set of mappings
3. Create a loss function
4. Create and run your optimization problem
5. Display the results

**TODO(keenon): Describe how this works. For now, just look at the demos**

## **Research**: Getting Raw Jacobians

If you want to get raw Jacobians relating two timesteps, you're going to use a lower level interface. This is what the PyTorch bindings are built on top of. Once you've constructed your world, you can use the low-level API as follows:

{{< code python >}}

```
import diffdart as dart

### Create your world, see previous section

# This call runs a forward timestep, and returns
# a handle to a BackpropSnapshot object that can
# give you any Jacobians you want

backprop_snapshot: dart.neural.BackpropSnapshot = dart.neural.forwardPass(world)

# To use the BackpropSnapshot, call its methods.

backprop_snapshot.getPosVelJacobian(world)

```

{{< /code >}}

The structure of the `BackpropSnapshot` Jacobians works like this:

![API Graph](/images/low_level_API.svg)
