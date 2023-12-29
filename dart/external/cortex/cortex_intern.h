#ifndef CORTEX_UTILS_H
#define CORTEX_UTILS_H

#include "cortex.h"
#include <stdarg.h> // for vararg of LogMessage()

void LogMessage(int iLevel, const char *szMsg, ...);

// standard definitions

//#define LOCAL static
#define LOCAL
#define GLOBAL

#define OK 0
#define ERRFLAG -1

#define ABS(x) (((x)>0)?(x):-(x))


// This is the NEW set of commands that Cortex is listening for on port 1504

// Generic handshake either way

#define PKT2_HELLO_WORLD            0  // Broadcast
#define PKT2_ARE_YOU_THERE          1  // Directed query
#define PKT2_HERE_I_AM              2
#define PKT2_COMMENT                3

#define PKT2_REQUEST_BODYDEFS      10
#define PKT2_BODYDEFS              11

#define PKT2_REQUEST_FRAME         12
#define PKT2_FRAME_OF_DATA         13

#define PKT2_GENERAL_REQUEST       14
#define PKT2_GENERAL_REPLY         15
#define PKT2_UNRECOGNIZED_REQUEST  16

#define PKT2_PUSH_BASEPOSITION     17


#define PKT2_UNRECOGNIZED_COMMAND 0x5678


// To Client

#define MESSAGE_BEGIN_RECORDING    33
#define MESSAGE_END_RECORDING      34


// From Client

#define REQUEST_START_RECORDING       0x0180
#define REQUEST_STOP_RECORDING        0x0181

#define REQUEST_FRAME                 0x0200 // This is for polling and testing



typedef struct
{
    char          szName[128];
    unsigned char Version[4];

} sMe;

typedef struct
{
    unsigned short iCommand;
    unsigned short nBytes;
    union
    {
        sMe           Me;
        char          String[256];
        unsigned char ucData[0x10000];
        char          cData[0x10000];
    } Data;

} sPacket;

#endif
