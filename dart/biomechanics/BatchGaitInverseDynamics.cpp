#include "dart/biomechanics/BatchGaitInverseDynamics.hpp"

namespace dart {

namespace biomechanics {

ContactRegimeSection::ContactRegimeSection(
    std::vector<const dynamics::BodyNode*> groundContactBodies,
    int startTime,
    int endTime)
  : groundContactBodies(groundContactBodies),
    startTime(startTime),
    endTime(endTime){};

BatchGaitInverseDynamics::BatchGaitInverseDynamics(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    Eigen::MatrixXs poses,
    std::vector<const dynamics::BodyNode*> groundContactBodies,
    Eigen::Vector3s groundNormal,
    s_t tileSize,
    int maxSectionLength,
    s_t smoothingWeight,
    s_t minTorqueWeight,
    s_t prevContactWeight,
    s_t blendWeight,
    s_t blendSteepness)
  : mSkeleton(skeleton),
    mPoses(poses),
    mBodies(groundContactBodies),
    mGroundNormal(groundNormal),
    mTileSize(tileSize),
    mLilypad(skeleton, groundContactBodies, groundNormal, tileSize)
{
  Eigen::VectorXs originalPos = mSkeleton->getPositions();
  Eigen::VectorXs originalVel = mSkeleton->getVelocities();
  Eigen::VectorXs originalControlForces = mSkeleton->getControlForces();

  // First process everything into the lilypad ground detection algorithm
  mLilypad.process(mPoses);

  // Now run through all the timesteps and figure out where contact modes switch
  mSkeleton->setPositions(poses.col(0));
  std::vector<const dynamics::BodyNode*> currentContact
      = mLilypad.getContactBodies();
  int startTime = 0;
  for (int i = 0; i < poses.cols(); i++)
  {
    mSkeleton->setPositions(poses.col(i));
    mSkeleton->setVelocities(
        mSkeleton->getPositionDifferences(poses.col(i + 1), poses.col(i))
        / mSkeleton->getTimeStep());
    std::vector<const dynamics::BodyNode*> newContactSet
        = mLilypad.getContactBodies();
    // When we switch a contact regime, or when we max out the length a single
    // section is allowed to be, we record this section and move on
    if (i - startTime > maxSectionLength || currentContact != newContactSet)
    {
      mContactRegimeSections.emplace_back(currentContact, startTime, i - 1);
      // Start a new contact regime
      startTime = i;
      currentContact = newContactSet;
    }
  }

  // Figure out the blending weights for each contact. We want to smoothly
  // transition between regimes, so we always begin and end with 0s
  std::vector<Eigen::VectorXs> bodyPenalties;
  for (int i = 0; i < groundContactBodies.size(); i++)
  {
    const dynamics::BodyNode* body = groundContactBodies[i];
    Eigen::VectorXs bodyPenalty = Eigen::VectorXs::Zero(poses.cols());

    bool lastInSection = false;
    int startTime = 0;
    for (int j = 0; j < mContactRegimeSections.size(); j++)
    {
      ContactRegimeSection& section = mContactRegimeSections[j];
      bool inSection = std::find(
                           section.groundContactBodies.begin(),
                           section.groundContactBodies.end(),
                           body)
                       != section.groundContactBodies.end();
      if (!inSection && lastInSection)
      {
        // Exit on startTime-1
        int endTime = section.startTime - 1;
        // Now we need to fill in the section between `startTime` and `endTime`
        // For lack of a better idea, let's do a pair of sigmoids, ramping in to
        // the halfway point, and then out

        int totalTime = endTime - startTime;
        for (int t = 0; t < totalTime; t++)
        {
          double percentage = (double)t / totalTime;
          if (percentage < 0.5)
          {
            double sigmoidInput
                = ((percentage * 2) - 0.5); // get onto a [-1, 1] range
            s_t e = exp(-sigmoidInput * blendSteepness);
            bodyPenalty(startTime + t) = e * blendWeight;
            /*
            s_t sigmoid = 1.0 / (1.0 + exp(-sigmoidInput * blendSteepness));
            std::cout << "Timestep " << t << "/" << totalTime << " ("
                      << percentage << "%): " << sigmoid << " [e=" << e << "]"
                      << std::endl;
            */
          }
          else
          {
            double sigmoidInput
                = (((percentage - 0.5) * 2) - 0.5); // get onto a [-1, 1] range
            s_t e = exp(sigmoidInput * blendSteepness);
            bodyPenalty(startTime + t) = e * blendWeight;
            /*
            s_t sigmoid = 1.0 / (1.0 + exp(sigmoidInput * blendSteepness));
            std::cout << "Timestep " << t << "/" << totalTime << " ("
                      << percentage << "%): " << sigmoid << " [e=" << e << "]"
                      << std::endl;
            */
          }
        }
      }
      if (inSection && !lastInSection)
      {
        // Enter on startTime
        startTime = section.startTime;
      }
      lastInSection = inSection;
    }

    bodyPenalties.push_back(bodyPenalty);
  }

  // Now solve each contact region one at a time
  for (int i = 0; i < mContactRegimeSections.size(); i++)
  {
    std::cout << "Solving section: " << i << "/"
              << mContactRegimeSections.size() << std::endl;
    std::cout << "Section band: " << mContactRegimeSections[i].startTime << "-"
              << mContactRegimeSections[i].endTime << ": "
              << mContactRegimeSections[i].groundContactBodies.size()
              << " contacts" << std::endl;

    ContactRegimeSection& section = mContactRegimeSections[i];

    Eigen::MatrixXs bodyPenaltyWeights = Eigen::MatrixXs::Zero(
        section.groundContactBodies.size(),
        (section.endTime - section.startTime) + 1);
    for (int j = 0; j < section.groundContactBodies.size(); j++)
    {
      const dynamics::BodyNode* bodyNode = section.groundContactBodies[j];
      int index = std::distance(
          groundContactBodies.begin(),
          std::find(
              groundContactBodies.begin(),
              groundContactBodies.end(),
              bodyNode));
      bodyPenaltyWeights.row(j) = bodyPenalties[index].segment(
          section.startTime, section.endTime + 1 - section.startTime);
    }

    std::vector<Eigen::Vector6s> prevContactForces;
    std::vector<const dynamics::BodyNode*> lastBodyNodes;

    if (i > 0)
    {
      lastBodyNodes = mContactRegimeSections[i - 1].groundContactBodies;
      std::vector<Eigen::Vector6s> lastContactForces
          = mContactRegimeSections[i - 1]
                .wrenches[mContactRegimeSections[i - 1].wrenches.size() - 1];

      for (const dynamics::BodyNode* bodyNode :
           mContactRegimeSections[i].groundContactBodies)
      {
        auto iterator
            = find(lastBodyNodes.begin(), lastBodyNodes.end(), bodyNode);
        // If the body node didn't exist in the last timestep, default to zero
        // force
        if (iterator == lastBodyNodes.end())
        {
          prevContactForces.push_back(Eigen::Vector6s::Zero());
        }
        else
        {
          // calculating the index
          // of K
          int index = iterator - lastBodyNodes.begin();
          prevContactForces.push_back(lastContactForces[index]);
        }
      }
    }
    else
    {
      prevContactWeight = 0.0;
      lastBodyNodes = section.groundContactBodies;
    }

    int blockWidth = (section.endTime - section.startTime)
                     + 2; // we add 2 columns to the end, because those are
                          // trimmed off by the acceleration computations
    if (section.startTime + blockWidth >= poses.cols())
    {
      blockWidth = (poses.cols() - 1) - section.startTime;
    }
    if (blockWidth < 3)
    {
      for (int i = 0; i < blockWidth; i++)
      {
        // TODO
        /*
        section.wrenches.push_back(mSkeleton->getMultipleContactInverseDynamics(
            nextVel, section.groundContactBodies));
        */
        std::vector<Eigen::Vector6s> zeros;
        for (int j = 0; j < mBodies.size(); j++)
          zeros.push_back(Eigen::Vector6s::Zero());
        section.wrenches.push_back(zeros);
      }
    }
    else
    {
      section.wrenches
          = mSkeleton
                ->getMultipleContactInverseDynamicsOverTime(
                    poses.block(
                        0, section.startTime, poses.rows(), 1 + blockWidth),
                    section.groundContactBodies,
                    smoothingWeight,
                    minTorqueWeight,
                    [](s_t /* vel */) {
                      return 0.0; // No velocity penalty
                    },
                    prevContactForces,
                    prevContactWeight,
                    bodyPenaltyWeights)
                .contactWrenches;
    }
  }
  std::cout << "Done all sections!" << std::endl;

  // Reset back to the original state

  mSkeleton->setPositions(originalPos);
  mSkeleton->setVelocities(originalVel);
  mSkeleton->setControlForces(originalControlForces);
}

int BatchGaitInverseDynamics::numTimesteps()
{
  return mPoses.cols() - 2;
}

ContactRegimeSection& BatchGaitInverseDynamics::getSectionForTimestep(
    int timestep)
{
  assert(timestep >= 0 && timestep < numTimesteps());
  assert(timestep < numTimesteps());
  for (ContactRegimeSection& section : mContactRegimeSections)
  {
    if (section.startTime <= timestep && section.endTime >= timestep)
      return section;
  }
  throw "getSectionForTimestep() called with out of bounds timestep";
}

std::vector<const dynamics::BodyNode*>
BatchGaitInverseDynamics::getContactBodiesAtTimestep(int timestep)
{
  ContactRegimeSection& section = getSectionForTimestep(timestep);
  return section.groundContactBodies;
}

std::vector<Eigen::Vector6s>
BatchGaitInverseDynamics::getContactWrenchesAtTimestep(int timestep)
{
  ContactRegimeSection& section = getSectionForTimestep(timestep);
  int offset = timestep - section.startTime;
  return section.wrenches[offset];
}

/// This will debug all the processed data over to our GUI, so we can see the
/// contact forces and positions animated
void BatchGaitInverseDynamics::debugLilypadToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server)
{
  mLilypad.debugToGUI(server);
}

/// This will debug all the processed data over to our GUI, so we can see the
/// contact forces and positions animated
void BatchGaitInverseDynamics::debugTimestepToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server, int timestep)
{
  mSkeleton->setPositions(mPoses.col(timestep));

  bool oldAutoflush = server->getAutoflush();
  server->setAutoflush(false);

  server->renderSkeleton(mSkeleton, "world");

  for (const dynamics::BodyNode* body : mBodies)
  {
    server->clearBodyWrench(body);
  }

  std::vector<const dynamics::BodyNode*> bodies
      = getContactBodiesAtTimestep(timestep);
  std::vector<Eigen::Vector6s> wrenches
      = getContactWrenchesAtTimestep(timestep);

  for (int i = 0; i < bodies.size(); i++)
  {
    server->renderBodyWrench(bodies[i], wrenches[i], 0.01);
  }

  server->flush();
  server->setAutoflush(oldAutoflush);
}

} // namespace biomechanics
} // namespace dart