#ifndef AMC_QUAT_HPP
#define AMC_QUAT_HPP

#include <iostream>

#include <assert.h>

#include "Vector.hpp"

using std::ostream;

template <typename NUM>
class Quat
{
public:
  union
  {
    NUM c[4];
    struct
    {
      NUM x, y, z, w;
    };
    struct
    {
      Vector<NUM, 3> xyz;
      NUM _padding;
    };
    Vector<NUM, 4> xyzw;
  };
  NUM& operator[](int i)
  {
    return c[i];
  }
  void clear()
  {
    x = NUM(0);
    y = NUM(0);
    z = NUM(0);
    w = NUM(1);
  }
  template <typename NUM2>
  Quat<NUM>& operator=(Quat<NUM2> const& b)
  {
    for (unsigned int i = 0; i < 4; ++i)
    {
      c[i] = (NUM)b.c[i];
    }
    return *this;
  }
};

template <typename NUM>
Quat<NUM> rotation(NUM theta, Vector<NUM, 3> axis)
{
  Quat<NUM> ret;
  ret.w = (NUM)cos(theta / 2.0);
  NUM s = (NUM)sin(theta / 2.0);
  ret.x = axis.x * s;
  ret.y = axis.y * s;
  ret.z = axis.z * s;
  return ret;
}

template <typename NUM>
Quat<NUM> rotation(Vector<NUM, 3> const& from, Vector<NUM, 3> const& to)
{
  Vector<NUM, 3> perp = cross_product(from, to);
  NUM len = length(perp);
  Quat<NUM> ret;
  if (len == 0)
  {
    ret.clear();
    return ret;
  }
  else
  {
    perp /= len;
    NUM c = from * to;
    NUM theta = atan2(len, c);

    ret.w = (NUM)cos(theta / 2.0);
    NUM s = (NUM)sin(theta / 2.0);
    ret.x = perp.x * s;
    ret.y = perp.y * s;
    ret.z = perp.z * s;

    Vector<NUM, 3> test = rotate(from, ret);
    if (length(test - to) > 0.001)
    {
      std::cout << "ERROR" << std::endl;
      std::cout << " From: " << from << std::endl;
      std::cout << "   To: " << to << std::endl;
      std::cout << " Test: " << test << std::endl;
    }
  }
  return ret;
}

template <typename NUM>
Quat<NUM> multiply(Quat<NUM> const& a, Quat<NUM> const& b)
{
  Quat<NUM> ret;
  // w is a.w * b.w - a.xyz * b.xyz
  ret.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;

  // xyz is a.w*b.xyz+b.w*a.xyz+cross_product(a.xyz,b.xyz)
  ret.x = a.y * b.z - b.y * a.z + a.w * b.x + b.w * a.x;
  ret.y = a.z * b.x - b.z * a.x + a.w * b.y + b.w * a.y;
  ret.z = a.x * b.y - b.x * a.y + a.w * b.z + b.w * a.z;
  return ret;
}

template <typename NUM>
Quat<NUM> normalize(Quat<NUM> const& a)
{
  NUM len = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
  Quat<NUM> ret;
  if (len == 0)
  {
    ret.x = 0;
    ret.y = 0;
    ret.z = 0;
    ret.w = 1;
  }
  else
  {
    len = sqrt(len);
    assert(len != 0);
    ret.x = a.x / len;
    ret.y = a.y / len;
    ret.z = a.z / len;
    ret.w = a.w / len;
  }
  return ret;
}

template <typename NUM>
Quat<NUM> conjugate(Quat<NUM> const& a)
{
  Quat<NUM> ret;
  ret.x = -a.x;
  ret.y = -a.y;
  ret.z = -a.z;
  ret.w = a.w;
  return ret;
}

template <typename NUM>
inline Quat<NUM> operator-(Quat<NUM> const& a)
{
  return conjugate(a);
}

template <typename NUM>
Vector<NUM, 3> rotate(Vector<NUM, 3> const& v, Quat<NUM> const& q)
{
  Quat<NUM> temp;
  temp.w = 0;
  temp.x = v.x;
  temp.y = v.y;
  temp.z = v.z;
  temp = multiply(q, multiply(temp, conjugate(q)));
  return make_vector(temp.x, temp.y, temp.z);
}

template <typename NUM>
Quat<NUM> lerp(Quat<NUM> a, Quat<NUM> const& b, NUM const& amt)
{
  a.w = a.w + (b.w - a.w) * amt;
  a.x = a.x + (b.x - a.x) * amt;
  a.y = a.y + (b.y - a.y) * amt;
  a.z = a.z + (b.z - a.z) * amt;
  return a;
}

template <typename NUM>
inline Quat<NUM> abs(Quat<NUM> a)
{
  if (a.w < 0)
  {
    a.x = -1 * a.x;
    a.y = -1 * a.y;
    a.z = -1 * a.z;
    a.w = -1 * a.w;
  }
  return a;
}

template <typename NUM>
inline Quat<NUM> operator+(Quat<NUM> a, Quat<NUM> b)
{
  a = abs(a);
  b = abs(b);
  Quat<NUM> ret;
  ret.w = a.w + b.w;
  ret.x = a.x + b.x;
  ret.y = a.y + b.y;
  ret.z = a.z + b.z;
  return ret;
}

template <typename NUM>
ostream& operator<<(ostream& o, Quat<NUM> const& q)
{
  o << '[' << q.x << ", " << q.y << ", " << q.z << ", " << q.w << ']';
  return o;
}

template <typename NUM>
NUM get_yaw_angle(Quat<NUM> const& axis)
{
  Vector<NUM, 3> test = rotate(make_vector<NUM>(1.0f, 0.0f, 0.0f), axis);
  return atan2(-test.z, test.x);
}

typedef Quat<double> Quatd;
typedef Quat<float> Quatf;

#endif
