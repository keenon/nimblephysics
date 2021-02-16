/*

Revision 1 - Steve Lin (CMU), Jan 14, 2002
Revision 2 - Alla Safonova and Kiran Bhat (CMU), Jan 18, 2002
Revision 3 - Jernej Barbic and Yili Zhao (USC), Feb, 2012

*/
#ifndef _TYPES_H
#define _TYPES_H

// Use this parameter to adjust the size of the skeleton. The default value is 0.06.
// This creates a human skeleton of 1.7 m in height (approximately)
//#define MOCAP_SCALE 0.06
#define MOCAP_SCALE 0.06
//static const int	NUM_BONES_IN_ASF_FILE	= 31;
#define MAX_BONES_IN_ASF_FILE 256
#define MAX_CHAR 1024
#define MAX_SKELS 16

#define PM_MAX_FRAMES 60000

#ifndef M_PI
#define M_PI 3.14159265
#endif

enum ErrorType
{
  NO_ERROR_SET = 0, BAD_OFFSET_FILE, NOT_SUPPORTED_INTERP_TYPE, BAD_INPUT_FILE
};


#endif
