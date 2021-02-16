/*

Revision 1 - Steve Lin (CMU), Jan 14, 2002
Revision 2 - Alla Safonova and Kiran Bhat (CMU), Jan 18, 2002
Revision 3 - Jernej Barbic and Yili Zhao (USC), Feb, 2012

*/
#include <cmath>
#include <cstdio>
#include "dart/utils/amc/transform.h"
#include "dart/utils/amc/types.h"
#include "dart/utils/amc/vector.h"

vector operator-( vector const& a, vector const& b )
{
  vector c;

  c.p[0] = a.p[0] - b.p[0];
  c.p[1] = a.p[1] - b.p[1];
  c.p[2] = a.p[2] - b.p[2];

  return c;
}

vector operator+( vector const& a, vector const& b )
{
  vector c;

  c.p[0] = a.p[0] + b.p[0];
  c.p[1] = a.p[1] + b.p[1];
  c.p[2] = a.p[2] + b.p[2];

  return c;
}

vector operator/( vector const& a, double b )
{
  vector c;

  c.p[0] = a.p[0] / b;
  c.p[1] = a.p[1] / b;
  c.p[2] = a.p[2] / b;

  return c;
}

//multiply
vector operator*( vector const& a, double b )
{
  vector c;

  c.p[0] = a.p[0] * b;
  c.p[1] = a.p[1] * b;
  c.p[2] = a.p[2] * b;

  return c;
}

//cross product
vector operator*( vector const& a, vector const& b )
{
  vector c;

  c.p[0] = a.p[1]*b.p[2] - a.p[2]*b.p[1];
  c.p[1] = a.p[2]*b.p[0] - a.p[0]*b.p[2];
  c.p[2] = a.p[0]*b.p[1] - a.p[1]*b.p[0];

  return c;
}

//dot product
double operator%( vector const& a, vector const& b )
{
  return ( a.p[0]*b.p[0] + a.p[1]*b.p[1] + a.p[2]*b.p[2] );
}

double len( vector const& v )
{
  return sqrt( v.p[0]*v.p[0] + v.p[1]*v.p[1] + v.p[2]*v.p[2] );
}

double vector::length() const
{
  return sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2] );
}

double angle( vector const& a, vector const& b )
{
  return acos( (a%b)/(len(a)*len(b)) );
}


