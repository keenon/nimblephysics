#include "dartc/skeleton.h"

#include <unordered_map>

#include <dart/dynamics/Skeleton.hpp>

#include "SkeletonManager.hpp"

#define CAST_SKELETON(name) static_cast<Skeleton*>(name)

#ifdef __cplusplus
extern "C" {
#endif

using namespace dart::dynamics;

//==============================================================================
SkeletonId dart_skeleton_create(const char* name)
{
  return SkeletonManager::create(name);
}

//==============================================================================
void dart_skeleton_destroy(SkeletonId skel)
{
  SkeletonManager::destroy(skel);
}

//==============================================================================
const char* dart_skeleton_get_name(SkeletonId skel)
{
  assert(skel);
  return CAST_SKELETON(skel)->getName().c_str();
}

//==============================================================================
int dart_skeleton_get_num_dofs(SkeletonId skel)
{
  return static_cast<int>(CAST_SKELETON(skel)->getNumDofs());
}

//==============================================================================
void dart_skeleton_set_positions(
    SkeletonId skel, s_t* positions, int num_poisitions)
{
  const Eigen::Map<Eigen::VectorXs> eigPositions(positions, num_poisitions);
  CAST_SKELETON(skel)->setPositions(eigPositions);
}

//==============================================================================
s_t* dart_skeleton_get_positions(SkeletonId skel)
{
  return CAST_SKELETON(skel)->getPositions().data();
}

//==============================================================================
void dart_skeleton_set_velocities(
    SkeletonId skel, s_t* velocities, int num_velocities)
{
  const Eigen::Map<Eigen::VectorXs> eigVelocities(velocities, num_velocities);
  CAST_SKELETON(skel)->setVelocities(eigVelocities);
}

//==============================================================================
s_t* dart_skeleton_get_velocities(SkeletonId skel)
{
  return CAST_SKELETON(skel)->getVelocities().data();
}

//==============================================================================
void dart_skeleton_set_accelerations(
    SkeletonId skel, s_t* accelerations, int num_accelerations)
{
  const Eigen::Map<Eigen::VectorXs> eigAccelerations(
      accelerations, num_accelerations);
  CAST_SKELETON(skel)->setAccelerations(eigAccelerations);
}

//==============================================================================
s_t* dart_skeleton_get_accelerations(SkeletonId skel)
{
  return CAST_SKELETON(skel)->getAccelerations().data();
}

#ifdef __cplusplus
}
#endif
