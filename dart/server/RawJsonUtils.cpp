#include "dart/server/RawJsonUtils.hpp"

namespace dart {

//==============================================================================
/// Small helper utility
void vec3ToJson(std::stringstream& json, const Eigen::Vector3d& vec)
{
  json << "[" << vec(0) << "," << vec(1) << "," << vec(2) << "]";
}

//==============================================================================
/// Small helper utility
void vecXToJson(std::stringstream& json, const Eigen::VectorXd& vec)
{
  json << "[";
  for (int i = 0; i < vec.size(); i++)
  {
    json << vec(i);
    if (i < vec.size() - 1)
      json << ",";
  }
  json << "]";
}

}