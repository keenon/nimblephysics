#include "dart/server/RawJsonUtils.hpp"

namespace dart {

//==============================================================================
std::string escapeJson(const std::string& str)
{
  // TODO
  return str;
}

//==============================================================================
std::string numberToJson(s_t number)
{
  if (isfinite(number))
  {
    return std::to_string(number);
  }
  else
  {
    return "0";
  }
}

//==============================================================================
void vec2iToJson(std::stringstream& json, const Eigen::Vector2i& vec)
{
  json << "[" << numberToJson(vec(0)) << "," << numberToJson(vec(1)) << "]";
}

//==============================================================================
void vec2dToJson(std::stringstream& json, const Eigen::Vector2s& vec)
{
  json << "[" << numberToJson(vec(0)) << "," << numberToJson(vec(1)) << "]";
}

//==============================================================================
void vec3ToJson(std::stringstream& json, const Eigen::Vector3s& vec)
{
  json << "[" << numberToJson(vec(0)) << "," << numberToJson(vec(1)) << ","
       << numberToJson(vec(2)) << "]";
}

//==============================================================================
void vec4ToJson(std::stringstream& json, const Eigen::Vector4s& vec)
{
  json << "[" << numberToJson(vec(0)) << "," << numberToJson(vec(1)) << ","
       << numberToJson(vec(2)) << "," << numberToJson(vec(3)) << "]";
}

//==============================================================================
void vec3iToJson(std::stringstream& json, const Eigen::Vector3i& vec)
{
  json << "[" << numberToJson(vec(0)) << "," << numberToJson(vec(1)) << ","
       << numberToJson(vec(2)) << "]";
}

//==============================================================================
void vecXToJson(std::stringstream& json, const Eigen::VectorXs& vec)
{
  json << "[";
  for (int i = 0; i < vec.size(); i++)
  {
    json << numberToJson(vec(i));
    if (i < vec.size() - 1)
      json << ",";
  }
  json << "]";
}

//==============================================================================
void vecToJson(std::stringstream& json, const std::vector<s_t>& vec)
{
  json << "[";
  for (int i = 0; i < vec.size(); i++)
  {
    json << numberToJson(vec[i]);
    if (i < vec.size() - 1)
      json << ",";
  }
  json << "]";
}

} // namespace dart