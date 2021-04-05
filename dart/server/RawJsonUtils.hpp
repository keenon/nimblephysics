#ifndef DART_JSON_UTILS
#define DART_JSON_UTILS

#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"
namespace dart {

std::string escapeJson(const std::string& str);
void vec2iToJson(std::stringstream& json, const Eigen::Vector2i& vec);
void vec2dToJson(std::stringstream& json, const Eigen::Vector2s& vec);
void vec3ToJson(std::stringstream& json, const Eigen::Vector3s& vec);
void vec3iToJson(std::stringstream& json, const Eigen::Vector3i& vec);
void vecXToJson(std::stringstream& json, const Eigen::VectorXs& vec);
void vecToJson(std::stringstream& json, const std::vector<s_t>& vec);

} // namespace dart

#endif