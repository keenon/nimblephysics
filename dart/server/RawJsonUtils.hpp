#ifndef DART_JSON_UTILS
#define DART_JSON_UTILS

#include <sstream>
#include <vector>

#include <Eigen/Dense>

namespace dart {

std::string escapeJson(const std::string& str);
void vec2ToJson(std::stringstream& json, const Eigen::Vector2i& vec);
void vec3ToJson(std::stringstream& json, const Eigen::Vector3d& vec);
void vecXToJson(std::stringstream& json, const Eigen::VectorXd& vec);
void vecToJson(std::stringstream& json, const std::vector<double>& vec);

} // namespace dart

#endif