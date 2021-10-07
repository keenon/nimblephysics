import nimblephysics as nimble
import torch


class ApplyWrenchLayer(torch.autograd.Function):
  """
  Input a 6-dof wrench force in local body space to any BodyNode, and get
  the resulting generalized forces out.
  """

  @staticmethod
  def forward(ctx, world, node, wrench, state):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    node: nimble.dynamics.BodyNode
    wrench: torch.Tensor
    -> torch.Tensor
    """
    world.setState(state.detach().numpy())

    skel = node.getSkeleton()
    skel_dof_offset = world.getSkeletonDofOffset(skel)

    ctx.world = world
    ctx.node = node
    ctx.jac = skel.getJacobian(node)
    ctx.skel_pos = skel.getPositions()
    ctx.wrench = wrench.detach().numpy()

    jac = torch.tensor(ctx.jac, dtype=torch.float64)
    # torques = Jac^T(pos) * wrench
    skel_tau = torch.matmul(torch.transpose(jac, 0, 1), wrench)

    # insert resulting torques for this skel only into world torques vector
    tau = torch.zeros((ctx.world.getNumDofs()), dtype=torch.float64)
    tau[skel_dof_offset:skel_dof_offset + skel.getNumDofs()] = skel_tau
    return tau

  @staticmethod
  def backward(ctx, grad):
    jac = torch.tensor(ctx.jac, dtype=torch.float64)
    wrench = torch.tensor(ctx.wrench, dtype=torch.float64)
    skel = ctx.node.getSkeleton()
    skel_dof_offset = ctx.world.getSkeletonDofOffset(skel)

    grad_skel = grad[skel_dof_offset:skel_dof_offset + skel.getNumDofs()]
    original_pos = skel.getPositions()
    skel.setPositions(ctx.skel_pos)
    # d(Jac * dLoss/dTorques)/dPos
    jacDeriv = skel.getJacobianDerivativeWrtJoints(ctx.node, grad_skel.detach().numpy())
    jacDeriv = torch.tensor(jacDeriv, dtype=torch.float64)
    skel.setPositions(original_pos)

    # dLoss/dWrench = [dTorques/dWrench == Jac] * dLoss/dTorques
    grad_wrench = torch.matmul(jac, grad_skel)
    # dLoss/dPos = d(Jac * dLoss/dTorques)/dPos^T * wrench
    grad_pos = torch.matmul(torch.transpose(jacDeriv, 0, 1), wrench)
    grad_state = torch.zeros((2 * ctx.world.getNumDofs()))
    grad_state[:ctx.world.getNumDofs()] = grad_pos
    return (
        None,
        None,
        grad_wrench,
        grad_state,
    )


def apply_wrench(
        world: nimble.simulation.World, node: nimble.dynamics.BodyNode, wrench: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
  """
  This converts the external wrench to generalized coordinates,
  storing necessary info in order to do a backwards pass.
  """
  return ApplyWrenchLayer.apply(world, node, wrench, state)  # type: ignore
