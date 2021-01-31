#include "dart/utils/amc/Skeleton.hpp"

#include <algorithm> // std::sort
#include <iostream>

#include <assert.h>

using std::cout;
using std::endl;

namespace Library {

void put_dof_rot(
    string const& dof, Quatd const& rot, double* info, int start_pos);

namespace {

void unroll(vector<vector<int> > const& list, vector<int>& target, int row = 0)
{
  assert(row >= 0);
  target.push_back(row);
  if (row < (signed)list.size())
  {
    for (unsigned int i = 0; i < list[row].size(); ++i)
    {
      unroll(list, target, list[row][i] + 1);
    }
  }
}

Quatd ordered_rotation(string const& order, Vector3d const& rotation);

} // namespace

ostream& operator<<(std::ostream& os, const Bone& b)
{
  os << "Bone: " << b.name << endl;
  os << " parent " << b.parent << endl;
  os << " dof " << b.dof << endl;
  os << " direction " << b.direction << endl;
  os << " frame_offset " << b.frame_offset << endl;
  os << " axis_offset " << b.axis_offset << endl;
  os << " offset_order " << b.offset_order << endl;
  os << " global_to_local " << b.global_to_local << endl;
  os << "(radius,density,length) (" << b.radius << ", " << b.density << ", "
     << b.length << ")" << endl;
  return os;
}

namespace {
class NameCompare
{
public:
  NameCompare(Skeleton const* _skel) : skel(_skel)
  {
  }
  bool operator()(int a, int b) const
  {
    assert(skel);
    string name_a = "";
    string name_b = "";
    if ((unsigned)a < skel->bones.size())
      name_a = skel->bones[a].name;
    if ((unsigned)b < skel->bones.size())
      name_b = skel->bones[b].name;
    return name_a < name_b;
  }
  Skeleton const* skel;
};
} // namespace

bool Skeleton::check_parse()
{
  if (in_bone)
  {
    cerr << "Bone without closing tag." << endl;
    return false;
  }
  // might as well also set bones up for painless traversal.
  vector<vector<int> > children;
  for (unsigned int b = 0; b < bones.size(); ++b)
  {
    int list = bones[b].parent + 1; // to make root zero.
    if (list < 0)
    {
      cerr << "Bone without a valid parent." << endl;
      return false;
    }
    while (children.size() <= (unsigned)list)
    {
      children.push_back(vector<int>());
    }
    assert(list >= 0 && list < (signed)children.size());
    children[list].push_back(b);
  }

  // make sure we have the lexicographically least child ordering.
  for (unsigned int c = 0; c < children.size(); ++c)
  {
    std::sort(children[c].begin(), children[c].end(), NameCompare(this));
  }

  vector<int> new_inds;
  unroll(children, new_inds);

  vector<int> old_to_new_inds;
  old_to_new_inds.resize(new_inds.size(), -2);
  assert(new_inds.size() == bones.size() + 1);

  old_to_new_inds[0] = -1;

  frame_size = 6; // i.e. just root info.
  vector<Bone> new_bones;
  for (unsigned int i = 1; i < new_inds.size(); ++i)
  {
    int b = new_inds[i] - 1; // bone for new first index.
    assert(b >= 0 && b < (signed)bones.size());
    old_to_new_inds[b + 1] = i - 1;
    new_bones.push_back(bones[b]);
    new_bones.back().frame_offset = frame_size;
    frame_size += new_bones.back().dof.size();

    int op = new_bones.back().parent + 1;
    assert(op >= 0 && op < (signed)old_to_new_inds.size());
    int p = old_to_new_inds[op]; // look up parent.
    assert(p >= -1 && p < (signed)bones.size());
    new_bones.back().parent = p;
  }

  // bones now ordered for easy traversal.
  bones = new_bones;

  for (unsigned int b = 0; b < bones.size(); ++b)
  {
    bones[b].global_to_local
        = ordered_rotation(bones[b].offset_order, bones[b].axis_offset);
  }

  return true;
}

namespace {
Quatd ordered_rotation(string const& order, Vector3d const& rot)
{
  Quatd ret;
  ret.clear();
  for (unsigned int i = 0; i < order.size(); ++i)
  {
    switch (order[i])
    {
      case 'x':
        ret = multiply(
            rotation(rot.x * M_PI / 180.0, make_vector(1.0, 0.0, 0.0)), ret);
        break;
      case 'y':
        ret = multiply(
            rotation(rot.y * M_PI / 180.0, make_vector(0.0, 1.0, 0.0)), ret);
        break;
      case 'z':
        ret = multiply(
            rotation(rot.z * M_PI / 180.0, make_vector(0.0, 0.0, 1.0)), ret);
        break;
      case 'X':
      case 'Y':
      case 'Z':
      case 'l':
        break;
      default:
        cerr << "Unexpected offset_order character '" << order[i] << "' in '"
             << order << "'." << endl;
        assert(0);
    }
  }
  return normalize(ret);
}

/*
Vector3d get_dof_trans(string const& dof, double const* info, int start_pos)
{
  Vector3d trans;
  trans.x = trans.y = trans.z = 0;
  info += start_pos;
  for (unsigned int i = 0; i != dof.size(); ++i)
  {
    double d = *info;
    switch (dof[i])
    {
      case 'x':
      case 'y':
      case 'z':
        break;
      case 'a':
        info += 2;
        break;
      case 'X':
        trans.x = d;
        break;
      case 'Y':
        trans.y = d;
        break;
      case 'Z':
        trans.z = d;
        break;
      case 'l':
        cerr << "Currently, we're lazy about length." << endl;
        break;
      default:
        cerr << "Enountered dof '" << dof[i] << "' we don't know about in '"
             << dof << "'." << endl;
        assert(0);
    }
    ++info;
  }
  return trans;
}

void put_dof_trans(
    string const& dof, Vector3d const& trans, double* info, int start_pos)
{
  info += start_pos;
  for (unsigned int i = 0; i != dof.size(); ++i)
  {
    switch (dof[i])
    {
      case 'x':
      case 'y':
      case 'z':
        break;
      case 'a':
        info += 2;
        break;
      case 'X':
        *info = trans.x;
        break;
      case 'Y':
        *info = trans.y;
        break;
      case 'Z':
        *info = trans.z;
        break;
      case 'l':
        cerr << "Currently, we're lazy about length." << endl;
        break;
      default:
        cerr << "Enountered dof '" << dof[i] << "' we don't know about in '"
             << dof << "'." << endl;
        assert(0);
    }
    ++info;
  }
}
*/

} // namespace
Quatd get_dof_rot(string const& dof, double const* info, int start_pos)
{
  Quatd ret;
  ret.clear();
  info += start_pos;
  for (unsigned int i = 0; i != dof.size(); ++i)
  {
    double d = *info;
    switch (dof[i])
    {
      case 'x':
        ret = multiply(
            rotation(d * M_PI / 180.0, make_vector(1.0, 0.0, 0.0)), ret);
        break;
      case 'y':
        ret = multiply(
            rotation(d * M_PI / 180.0, make_vector(0.0, 1.0, 0.0)), ret);
        break;
      case 'z':
        ret = multiply(
            rotation(d * M_PI / 180.0, make_vector(0.0, 0.0, 1.0)), ret);
        break;
      case 'a': {
        Vector3d axis = *(Vector3d*)info;
        info += 2;
        ret = multiply(rotation(length(axis), normalize(axis)), ret);
        break;
      }

      case 'X':
      case 'Y':
      case 'Z':
        break;
      case 'l':
        cerr << "Currently, we're lazy about length." << endl;
        break;
      default:
        cerr << "Enountered dof '" << dof[i] << "' we don't know about in '"
             << dof << "'." << endl;
        assert(0);
    }
    ++info;
  }
  return normalize(ret);
}

namespace {
// axis & probe should be orthonormal.
double get_rotation(Vector3d axis, Vector3d probe, Quatd const& rot)
{
  Vector3d perp = cross_product(axis, probe);
  Vector3d rotated = rotate(probe, rot);
  rotated = normalize(rotated - axis * (axis * rotated));
  return atan2(rotated * perp, rotated * probe);
}

} // namespace

void put_dof_rot(
    string const& dof, Quatd const& rot, double* info, int start_pos)
{
  unsigned int ind[3];
  Vector3d vec[3];
  Vector3d perp[3];
  // double q[4];
  // q[0] = rot.w;
  unsigned int count = 0;
  for (unsigned int i = 0; i != dof.size(); ++i)
  {
    switch (dof[i])
    {
      case 'x':
        vec[count] = make_vector(1.0, 0.0, 0.0);
        perp[count] = make_vector(0.0, 1.0, 0.0);
        ind[count] = i;
        // q[count+1] = rot.x;
        ++count;
        break;
      case 'y':
        vec[count] = make_vector(0.0, 1.0, 0.0);
        perp[count] = make_vector(0.0, 0.0, 1.0);
        ind[count] = i;
        // q[count+1] = rot.y;
        ++count;
        break;
      case 'z':
        vec[count] = make_vector(0.0, 0.0, 1.0);
        perp[count] = make_vector(1.0, 0.0, 0.0);
        ind[count] = i;
        // q[count+1] = rot.z;
        ++count;
        break;
      case 'a':
        // Something special here, I guess.
        {
          // If axis angle isn't the only DOF, you're probably sodded.
          Vector3d& vec = *(Vector3d*)(&info[i + start_pos]);
          double xyzl = length(rot.xyz);
          double theta;
          if (xyzl != 0.0 && rot.w != 0)
          {
            theta = 2 * atan(xyzl / rot.w);
            vec = rot.xyz * theta / xyzl;
          }
          else
          {
            vec = rot.xyz;
          }
          return;
        }
        break;
      case 'X':
      case 'Y':
      case 'Z':
        break;
      case 'l':
        cerr << "Currently, we're lazy about length." << endl;
        break;
      default:
        cerr << "Enountered dof '" << dof[i] << "' we don't know about in '"
             << dof << "'." << endl;
        assert(0);
    }
  }
  if (count == 0)
  {
    return; // not much to do.
  }
  else if (count == 1)
  {
    // should be 1-d rotation, so map it down, sucker!
    info[start_pos + ind[0]]
        = 180.0 / M_PI * get_rotation(vec[0], perp[0], rot);
  }
  else if (count == 2)
  {
    // a 2-d rotation, I reckon.
    double ang1 = get_rotation(vec[1], vec[0], rot);
    info[start_pos + ind[1]] = ang1 * 180.0 / M_PI;
    info[start_pos + ind[0]]
        = 180.0 / M_PI
          * get_rotation(
              vec[0],
              perp[0],
              multiply(conjugate(rotation(ang1, vec[1])), rot));
  }
  else if (count == 3)
  {
    // a 3-d rotation == "problem case"
    // rot is a quaternion
    // vec[0], vec[1], vec[2] are axes, orthonormal
    // create Euler angles in info around these axes
    Vector3d new0 = rotate(vec[0], rot);
    Vector3d new1 = rotate(vec[1], rot);
    Vector3d new2 = rotate(vec[2], rot);
    double ang0 = atan2(new1 * vec[2], new2 * vec[2]);
    double ang1 = -atan2(
        new0 * vec[2], sqrt(pow(new0 * vec[0], 2) + pow(new0 * vec[1], 2)));
    double ang2 = atan2(new0 * vec[1], new0 * vec[0]);
    info[start_pos + ind[0]] = ang0 * 180.0 / M_PI;
    info[start_pos + ind[1]] = ang1 * 180.0 / M_PI;
    info[start_pos + ind[2]] = ang2 * 180.0 / M_PI;

    // info[start_pos + ind[0]] = 180.0 / M_PI * atan2(2*(q[0]*q[1]+q[2]*q[3]),
    // 1 - 2*(q[1]*q[1] + q[2]*q[2])); info[start_pos + ind[1]] = 180.0 / M_PI *
    // asin(2*(q[0]*q[2] - q[3]*q[1])); info[start_pos + ind[2]] = 180.0 / M_PI
    // * atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]));
  }
  else
  {
    assert(0);
  }
}

// basically, just pull the 'ol root positions from each frame and
// extract the x,z, and yaw delta.
/*void Skeleton::build_delta(int frame_from, int frame_to, vector< double >
const &positions, Character::StateDelta &delta) const { delta.clear(); if
(frame_from < 0 || frame_from * frame_size >= (signed)positions.size()
    || frame_to < 0 || frame_to * frame_size >= (signed)positions.size() ) {
    return;
  }
  Quatd orientation_from, orientation_to;
  orientation_from = orientation_to = ordered_rotation(offset_order,
axis_offset); Vector3d position_from, position_to; position_from = position_to =
position; position_from += get_dof_trans(order, &(positions[0]), frame_size *
frame_from); position_to += get_dof_trans(order, &(positions[0]), frame_size *
frame_to);

  orientation_from = multiply(orientation_from, get_dof_rot(order,
&(positions[0]), frame_size * frame_from)); orientation_to =
multiply(orientation_to, get_dof_rot(order, &(positions[0]), frame_size *
frame_to));

  //position delta is given in the starting orientation frame.
  delta.position = rotate(position_to - position_from,
conjugate(orientation_from)); delta.orientation =
get_yaw_angle(normalize(multiply(orientation_to, conjugate(orientation_from))));

}*/

int Skeleton::get_bone_by_name(string name) const
{
  for (unsigned int b = 0; b < bones.size(); ++b)
  {
    if (name == bones[b].name)
    {
      return b;
    }
  }
  return -1;
}

string Skeleton::get_dof_description(unsigned int dof) const
{
  string ret = "unknown";
  if (dof < order.size())
  {
    ret = "root";
    ret += order[dof];
  }
  else
  {
    for (unsigned int b = 0; b < bones.size(); ++b)
    {
      if ((signed)dof >= bones[b].frame_offset
          && (unsigned)(dof - bones[b].frame_offset) < bones[b].dof.size())
      {
        ret = bones[b].name + bones[b].dof[dof - bones[b].frame_offset];
      }
    }
  }
  return ret;
}

} // namespace Library
