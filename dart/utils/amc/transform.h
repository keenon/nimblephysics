/*
transform.h

Revision 1 - Steve Lin (CMU), Jan 14, 2002
Revision 2 - Alla Safonova and Kiran Bhat (CMU), Jan 18, 2002
Revision 3 - Jernej Barbic and Yili Zhao (USC), Feb, 2012

*/

#ifndef _TRANSFORM_H
#define _TRANSFORM_H

class Matrix
{

};

void matrix_transpose(double a[4][4], double b[4][4]);
void matrix_print(char *str, double a[4][4]);
void matrix_transform_affine(double m[4][4], double x, double y, double z, double pt[3]);
void matrix_mult(double a[][4], double b[][4], double c[][4]);

void v3_cross(double a[3], double b[3], double c[3]);
double v3_mag(double a[3]);
double v3_dot(double a[3], double b[3]);

//Rotate vector v around axis X by angle a, around axis Y by angle b and around axis Z by angle c
void vector_rotationXYZ(double *v, double a, double b, double c);

//Create Rotation matrix, that rotates around axis X by angle a
void rotationX(double r[][4], double a);
//Create Rotation matrix, that rotates around axis Y by angle a
void rotationY(double r[][4], double a);
//Create Rotation matrix, that rotates around axis Z by angle a
void rotationZ(double r[][4], double a);

//Return the angle between vectors v1 and v2 around the given axis 
double GetAngle(double* v1, double* v2, double* axis);

#endif
