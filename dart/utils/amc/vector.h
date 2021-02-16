/*
skeleton.h

Definition of the skeleton. 

Written by Jehee Lee

Revision 1 - Steve Lin (CMU), Jan 14, 2002
Revision 2 - Alla Safonova and Kiran Bhat (CMU), Jan 18, 2002
Revision 3 - Jernej Barbic and Yili Zhao (USC), Feb, 2012

*/

#ifndef _VECTOR_H
#define _VECTOR_H


class vector
{
  // negation
  friend vector    operator-( vector const& );

  // addtion
  friend vector    operator+( vector const&, vector const& );

  // subtraction
  friend vector    operator-( vector const&, vector const& );

  // dot product
  friend double    operator%( vector const&, vector const& );

  // cross product
  friend vector    operator*( vector const&, vector const& );

  // scalar Multiplication
  friend vector    operator*( vector const&, double );

  // scalar Division
  friend vector    operator/( vector const&, double );


  friend double    len( vector const& );
  friend vector	normalize( vector const& );

  friend double       angle( vector const&, vector const& );

  // member functions
public:
  // constructors
  vector() {}
  vector( double x, double y, double z ) { p[0]=x; p[1]=y; p[2]=z; }
  vector( double a[3] ) { p[0]=a[0]; p[1]=a[1]; p[2]=a[2]; }
  ~vector() {};

  // inquiry functions
  double& operator[](int i) { return p[i];}

  double x() const { return p[0]; };
  double y() const { return p[1]; };
  double z() const { return p[2]; };
  void   getValue( double d[3] ) { d[0]=p[0]; d[1]=p[1]; d[2]=p[2]; }
  void   setValue( double d[3] ) { p[0]=d[0]; p[1]=d[1]; p[2]=d[2]; }

  double getValue( int n ) const { return p[n]; }
  vector setValue( double x, double y, double z )
  { p[0]=x, p[1]=y, p[2]=z; return *this; }
  double setValue( int n, double x )
  { return p[n]=x; }

  double length() const;

  // change functions
  void set_x( double x ) { p[0]=x; };
  void set_y( double x ) { p[1]=x; };
  void set_z( double x ) { p[2]=x; };

  //data members
  double p[3]; //X, Y, Z components of the vector
};

#endif

