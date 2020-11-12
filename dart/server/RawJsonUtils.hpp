#ifndef DART_JSON_UTILS
#define DART_JSON_UTILS

#include <sstream>

#include <Eigen/Dense>

namespace dart {

void vec3ToJson(std::stringstream& json, const Eigen::Vector3d& vec);
void vecXToJson(std::stringstream& json, const Eigen::VectorXd& vec);

} // namespace dart

#endif