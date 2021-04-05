#ifndef AMC_SKELETON_HPP
#define AMC_SKELETON_HPP

#include <iostream>
#include <string>
#include <vector>

#include "dart/utils/amc/Quat.hpp"
#include "dart/utils/amc/Vector.hpp"

namespace Library {

using std::cerr;
using std::endl;
using std::string;
using std::vector;

// collection of joint angles, offset matrices, and so on.

class Bone
{
public:
  string name;
  int parent;
  Vector3s direction;
  Vector3s axis_offset; // in degrees
  string offset_order;

  Quatd global_to_local;

  double radius;
  double density;
  double length;

  vector<Vector2s> torque_limits;

  string dof; // list of degrees of freedom -> 'xyz' -> rotation around 'x' 'y'
              // 'z'. 'a' -> angle-axis rotation [3 values]

  Vector3s color;

  int frame_offset; // where in lists of per-frame data this fellow's data
                    // rests.

  // filled, but not yet taken advantage of
  vector<Vector3s> euler_axes; // for v-file. Actual axes of joint rotation

  //-------------------------
  // These are set every frame to reflect the bone's state.
  // Vector3s current_base_position;
  // Vector3s current_end_position;
  // Quatd current_rotation;
  //-------------------------

  inline void pre_parse()
  {
    name = "";
    parent = 0;
    direction.x = direction.y = direction.z = 0;
    axis_offset.x = axis_offset.y = axis_offset.z = 0;
    offset_order = "xyz";
    radius = density = length = -1;
    dof = "";
    torque_limits.clear();
    color.r = rand() / double(RAND_MAX);
    color.g = rand() / double(RAND_MAX);
    color.b = rand() / double(RAND_MAX);
  }

  inline bool post_parse()
  {
    if (name == "")
    {
      cerr << "Bone lacks name." << endl;
      return false;
    }
    if (length < 0)
    {
      cerr << "Bone lacks length. Picking 1." << endl;
      length = 1;
    }
    if (radius < 0)
    {
      // cerr << "Bone lacks radius. Picking length / 10." << endl;
      radius = length / 10.0;
    }
    if (density < 0)
    {
      // cerr << "Bone lacks density. Picking 1000." << endl;
      density = 1000;
    }
    if (torque_limits.size() > dof.size())
    {
      cerr << "There are extra torque limits on " << name
           << " that will be ignored." << endl;
    }
    if (torque_limits.size() < dof.size())
    {
      // cerr << "Not all dof of " << name << " have limits. Padding with 100s."
      // << endl;
      while (torque_limits.size() < dof.size())
      {
        torque_limits.push_back(make_vector(-100.0, 100.0));
      }
    }
    return true;
  }
};

ostream& operator<<(std::ostream& os, const Bone& b);

class Skeleton
{
public:
  vector<Bone> bones;

  // little letters are rotations, big letters are translations.
  // (think 'xXyzYZ' -- rotate x, trans x, rot y, rot z, trans Y, trans Z)
  string order; // order of transformation for root bone

  Vector3s position;    // position for root bone
  string offset_order;  // order for axis offset rotation values
  Vector3s axis_offset; // rotation for root bone (degrees)

  // These track the present position/rotation info.
  // Vector3s current_position;
  // Quatd current_rotation;

  // parser info:
  inline void init_parse()
  {
    in_bone = false;
    bones.clear();
    mass = 1.0;
    length = 1.0;
    ang_is_deg = true;
    offset_order = "xyz";
    order = "XYZxyz";
    timestep = 1.0 / 120.0;
    rot_is_glob = false;
    z_is_up = false;
  }

  // large, has setup for bones. Stuff like that.
  bool check_parse();

  int get_bone_by_name(string name) const;

  // returns a succinct string describing a dof
  string get_dof_description(unsigned int dof) const;

  bool in_bone;
  double mass, length, timestep;
  bool ang_is_deg;  // true -> degrees, false -> radians
  bool rot_is_glob; // true -> global rotations (v-file), false -> local (amc)
  bool z_is_up;     // If set, will rotate z-up to y-up.

  int frame_size; // how may dof per frame.

  string filename; // what file this was loaded from.
};

} // namespace Library

#endif // SKELETON_HPP
