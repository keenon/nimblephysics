#include "dart/math/GraphFlowDiscretizer.hpp"

#include <deque>
#include <iostream>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

GraphFlowDiscretizer::GraphFlowDiscretizer(
    int numNodes,
    std::vector<std::pair<int, int>> arcs,
    std::vector<bool> nodeAttachedToSink)
  : mNumNodes(numNodes), mArcs(arcs), mNodeAttachedToSink(nodeAttachedToSink)
{
}

/// This will find the least-squares closest rates of transfer across the arcs
/// to end up with the energy levels at each node we got over time. The idea
/// here is that arc rates may not perfectly reflect the observed changes in
/// energy levels.
Eigen::MatrixXs GraphFlowDiscretizer::cleanUpArcRates(
    Eigen::MatrixXs energyLevels, Eigen::MatrixXs arcRates)
{
  // Figure out which nodes are detached from the energy sink - these nodes will
  // not be allowed to sum their energy to anything except zero.
  std::vector<int> detachedNodes;
  for (int i = 0; i < mNumNodes; i++)
  {
    if (!mNodeAttachedToSink[i])
    {
      detachedNodes.push_back(i);
    }
  }

  // If everything is attached to the sink, there are no constraints on the arc
  // rates, because differences can always be made up from the sink, so return
  // immediately.
  if (detachedNodes.size() == 0)
  {
    return arcRates;
  }

  // Otherwise, there's a set of linear equations that must be satisfied at
  // every timestep. Write that as a matrix:
  Eigen::MatrixXs constraints
      = Eigen::MatrixXs::Zero(detachedNodes.size(), mArcs.size());
  for (int i = 0; i < mArcs.size(); i++)
  {
    int from = mArcs[i].first;
    int to = mArcs[i].second;

    int fromIndexInDetachedNodes
        = std::find(detachedNodes.begin(), detachedNodes.end(), from)
          - detachedNodes.begin();
    if (fromIndexInDetachedNodes < detachedNodes.size())
    {
      constraints(fromIndexInDetachedNodes, i) = -1.0;
    }
    int toIndexInDetachedNodes
        = std::find(detachedNodes.begin(), detachedNodes.end(), to)
          - detachedNodes.begin();
    if (toIndexInDetachedNodes < detachedNodes.size())
    {
      constraints(toIndexInDetachedNodes, i) = 1.0;
    }
  }

  for (int t = 1; t < energyLevels.cols(); t++)
  {
    Eigen::VectorXs nodeChanges = energyLevels.col(t) - energyLevels.col(t - 1);
    Eigen::VectorXs detachedNodeChanges
        = Eigen::VectorXs::Zero(detachedNodes.size());
    for (int i = 0; i < detachedNodes.size(); i++)
    {
      detachedNodeChanges(i) = nodeChanges(detachedNodes[i]);
    }

    Eigen::VectorXs originalArcs = arcRates.col(t - 1);
    // We want this to be 0
    // constraints * (originalArcs + diff) = detachedNodeChanges
    // constraints * originalArcs + constraints * diff = detachedNodeChanges
    // constraints * diff = detachedNodeChanges - constraints * originalArcs
    // diff = constraints^-1 * (detachedNodeChanges - constraints *
    // originalArcs)
    Eigen::VectorXs arcChanges
        = constraints.completeOrthogonalDecomposition().solve(
            detachedNodeChanges - constraints * originalArcs);
    // Add the changes back in
    arcRates.col(t - 1) = originalArcs + arcChanges;
  }

  return arcRates;
}

/// This will attempt to create a set of ParticlePath objects that map the
/// recorded graph node levels and flows as closely as possible. The particles
/// can be created and destroyed within the arcs.
std::vector<ParticlePath> GraphFlowDiscretizer::discretize(
    int maxSimultaneousParticles,
    Eigen::MatrixXs energyLevels,
    Eigen::MatrixXs arcRates)
{
  std::vector<ParticlePath> result;

  ////////////////////////
  // 0. Validate the graph that was passed to us

  Eigen::MatrixXs nodeNetEnergy
      = Eigen::MatrixXs::Zero(energyLevels.rows(), energyLevels.cols());

  bool hasErrors = false;
  for (int t = 1; t < energyLevels.cols(); t++)
  {
    Eigen::VectorXs nodeChanges = energyLevels.col(t) - energyLevels.col(t - 1);
    Eigen::VectorXs arcContributions = arcRates.col(t - 1);

    // We now go through and "explain away" all node changes due to arc
    // transfers All arcs are in <from, to> format
    for (int i = 0; i < mArcs.size(); i++)
    {
      auto& pair = mArcs[i];
      int from = pair.first;
      int to = pair.second;

      // An arc will take from (from), and give to (to), so to reverse its
      // changes (to see what changes are _not_ explained by the arcs) we need
      // to do the opposite.
      nodeChanges(from) += arcContributions(i);
      nodeChanges(to) -= arcContributions(i);
    }

    for (int i = 0; i < mNumNodes; i++)
    {
      // If this node created energy on this timestep, but isn't allowed to
      if (std::abs(nodeChanges(i)) > 1e-8 && !mNodeAttachedToSink[i])
      {
        std::cout << "GraphFlowDiscretizer error! Node " << i
                  << " is not allowed to create energy, but had "
                  << nodeChanges(i) << " energy created on timestep " << t
                  << std::endl;
        hasErrors = true;
      }
    }
    nodeNetEnergy.col(t - 1) = nodeChanges;
  }
  if (hasErrors)
  {
    std::cout << "GraphFlowDiscretizer had errors. Returning an empty set of "
                 "particles"
              << std::endl;
    return result;
  }

  ////////////////////////
  // 1. Find the discrete energy value of a particle

  // Find the maximum level that the graph reaches at any timestep
  s_t maxTotalLevel = 0.0;
  for (int t = 0; t < energyLevels.cols(); t++)
  {
    s_t level = energyLevels.col(t).sum();
    if (level > maxTotalLevel)
      maxTotalLevel = level;
  }

  // Compute how much each particle is worth
  s_t particleUnit = maxTotalLevel / maxSimultaneousParticles;

  ////////////////////////
  // 2. Use a simple "accumulated error" heuristic to find the best times to
  // create/delete particles, as well as the best times to transfer particles
  // across arcs.

  // 2.1. We create a stack for each node, where we can store the indices of the
  // particles currently at that node.
  std::vector<std::vector<int>> particlesAtNode;
  for (int i = 0; i < mNumNodes; i++)
  {
    // Every node has a set of particles located at that note
    particlesAtNode.emplace_back();
  }

  // 2.2. Now we can go timestep by timestep, and accumulate continuos net
  // energy, and then when quantum jumps happen, we can transfer particles
  // across arcs.
  Eigen::VectorXs netEnergy = energyLevels.col(0);
  Eigen::VectorXs netArcEnergy = Eigen::VectorXs::Zero(mArcs.size());
  for (int t = 0; t < energyLevels.cols(); t++)
  {
#ifndef NDEBUG
    // 2.2.0. First we go through and check that our datastructures are all
    // consistent with each other, cause we'd really like those to not get
    // inconsistent...
    for (int i = 0; i < mNumNodes; i++)
    {
      int numParticlesInThisNodeNow = 0;
      for (int p = 0; p < result.size(); p++)
      {
        if (result[p].startTime == t - result[p].nodeHistory.size()
            && result[p].nodeHistory.back() == i)
        {
          numParticlesInThisNodeNow++;
        }
      }
      assert(particlesAtNode[i].size() == numParticlesInThisNodeNow);
    }
#endif

    // 2.2.1. We'd like to do all available transfers on this frame, before we
    // create or destroy any particles.
    for (int i = 0; i < mArcs.size(); i++)
    {
      s_t netEnergyAtArc = netArcEnergy(i);
      int transferParticles = std::floor(netEnergyAtArc / particleUnit);
      int from = mArcs[i].first;
      int to = mArcs[i].second;
      s_t netEnergyDirection = 1.0;

      if (transferParticles < 0)
      {
        transferParticles = -transferParticles;
        // Flip the arc order
        from = mArcs[i].second;
        to = mArcs[i].first;
        netEnergyDirection = -1.0;
      }

      // Check how many particles we can transfer, which are particles that have
      // not already been transferred
      int availableToTransfer = 0;
      for (int j = 0; j < particlesAtNode[from].size(); j++)
      {
        int particleIndex = particlesAtNode[from][j];
        if (!result[particleIndex].alreadyTransferred)
        {
          availableToTransfer++;
        }
      }

      // We want to transfer as many particles as we can, but we can't transfer
      // more than we have
      if (transferParticles > availableToTransfer)
      {
        transferParticles = availableToTransfer;
      }
      netArcEnergy(i)
          -= (transferParticles * particleUnit * netEnergyDirection);

      // Transfer the particles
      for (int p = 0; p < transferParticles; p++)
      {
        // In an ideal world, we want to get the particle that most recently
        // transitted through the `to` node
        int bestParticleStackIndex = -1;
        int bestParticleTime = t + 1;
        for (int j = 0; j < particlesAtNode[from].size(); j++)
        {
          int particleIndex = particlesAtNode[from][j];
          for (int k = result[particleIndex].nodeHistory.size() - 1; k >= 0;
               k--)
          {
            if (result[particleIndex].alreadyTransferred)
              continue;
            if (result[particleIndex].nodeHistory[k] == to)
            {
              int visitTime = result[particleIndex].startTime + k;
              if (visitTime < bestParticleTime)
              {
                bestParticleStackIndex = j;
                bestParticleTime = visitTime;
              }
              break;
            }
          }
        }

        // Then we'll just look for a particle that hasn't been transferred yet
        for (int j = 0; j < particlesAtNode[from].size(); j++)
        {
          int particleIndex = particlesAtNode[from][j];
          if (!result[particleIndex].alreadyTransferred)
          {
            bestParticleStackIndex = j;
            break;
          }
        }

        // Barring that, we'll just take the particle that's been at the node
        // longest
        if (bestParticleStackIndex == -1)
        {
          std::cout
              << "GraphFlowDiscretizer INTERNAL ERROR: no particles "
                 "(that were not already "
                 "transferred) available to transfer on arc "
              << i << " at timestep " << t
              << ". Defaulting to an already transferred particle, but "
                 "this will break assumptions about graph paths only "
                 "traversing along single arcs, because this particle is doing "
                 "at least a double-hop this frame."
              << std::endl;
          bestParticleStackIndex = 0;
        }

        int particleIndex = particlesAtNode[from][bestParticleStackIndex];
        result[particleIndex].alreadyTransferred = true;
        // Add the particle to the queue for the sink
        particlesAtNode[to].push_back(particleIndex);
        // Remove the particle from the queue for the source
        particlesAtNode[from].erase(
            particlesAtNode[from].begin() + bestParticleStackIndex);
      }
    }

    // 2.2.2. Having done all available transfers, we now want to create or
    // destroy particles at each node to match the net energy
    for (int i = 0; i < mNumNodes; i++)
    {
      s_t netEnergyAtNode = netEnergy(i);
      int createParticles = std::floor(netEnergyAtNode / particleUnit);
      assert(mNodeAttachedToSink[i] || createParticles == 0 || t == 0);
      if (createParticles < 0)
      {
        // This can happen if we have a node that needs to destroy particles.
        // First, only destroy at most as many particles as we have.
        int destroyParticles = -createParticles;
        if (destroyParticles > particlesAtNode[i].size())
        {
          destroyParticles = particlesAtNode[i].size();
        }
        // Add the leftover energy to the "netEnergy" figure
        netEnergy(i) += (destroyParticles * particleUnit);

        // For the particles that are about to be destroyed, if they were
        // already transferred this frame, we need to mark them as having ended
        // their lives on this node. This only happens if we're not already at
        // the last frame.
        if (t < energyLevels.cols() - 1)
        {
          for (int p = 0; p < destroyParticles; p++)
          {
            int particleIndex = particlesAtNode[i][p];
            if (result[particleIndex].alreadyTransferred)
            {
              result[particleIndex].nodeHistory.push_back(i);
            }
          }
        }

        // Destroy the oldest particles in this body
        particlesAtNode[i].erase(
            particlesAtNode[i].begin(),
            particlesAtNode[i].begin() + destroyParticles);
      }
      else if (createParticles > 0)
      {
        // Add the leftover energy to the "netEnergy" figure
        netEnergy(i) -= (createParticles * particleUnit);
        for (int p = 0; p < createParticles; p++)
        {
          // Create a new particle
          int newParticleIndex = result.size();
          result.emplace_back();
          result[newParticleIndex].startTime = t;
          result[newParticleIndex].alreadyTransferred = false;
          result[newParticleIndex].energyValue = particleUnit;
          particlesAtNode[i].push_back(newParticleIndex);
        }
      }

#ifndef NDEBUG
      // Check that we've dealt with all available particles
      createParticles = std::floor(netEnergy(i) / particleUnit);
      assert(
          createParticles == 0
          || (createParticles < 0 && particlesAtNode[i].size() == 0));
#endif
    }

    // 2.3. Now we can add the history to the particles that are each node
    for (int i = 0; i < mNumNodes; i++)
    {
      for (int j = 0; j < particlesAtNode[i].size(); j++)
      {
        int particleIndex = particlesAtNode[i][j];
        result[particleIndex].nodeHistory.push_back(i);
        result[particleIndex].alreadyTransferred = false;
      }
    }

    // 2.4. Finally, we can update the net energy and net arc energy for the
    // next timestep
    netEnergy += nodeNetEnergy.col(t);
    netArcEnergy += arcRates.col(t);

#ifndef NDEBUG
    for (int i = 0; i < mNumNodes; i++)
    {
      assert(mNodeAttachedToSink[i] || netEnergy(i) == 0);
    }
#endif
  }

  return result;
}

} // namespace math
} // namespace dart