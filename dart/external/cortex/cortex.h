/*=========================================================
//
// File: Cortex.h  v200
//
// Created by Ned Phipps, Oct-2004
//
// This file defines the interface to Cortex ethernet communication.
//
//----------------------------------------------------------
Modification History:

Date      By          Comment
------------------------------------------------------------
Oct 2004  np          First version

=============================================================================*/

/*! \file Cortex.h
This file defines the structures and the API for ethernet communication of data
between Cortex and multiple client programs.
*/

#ifndef Cortex_H
#define Cortex_H


/** Return codes
*/
typedef enum maReturnCode
{
    RC_Okay=0,             //!< Okay
    RC_GeneralError,       //!< General Error
    RC_ApiError,           //!< Invalid use of the API
    RC_NetworkError,       //!< Network Error
    RC_TimeOut,            //!< No response from Cortex
    RC_MemoryError,        //!< Memory allocation failed
    RC_Unrecognized        //!< Request string not recognized
}
maReturnCode;


/** Verbosity setting for internal messages
*/
typedef enum maVerbosityLevel
{
    VL_None=0,   //!< No Messages
    VL_Error,    //!< Error Message
    VL_Warning,  //!< Warning Message [DEFAULT VALUE]
    VL_Info,     //!< Informational Message
    VL_Debug,    //!< Debug Message
}
maVerbosityLevel;


// Array dimensions

#define MAX_N_BODIES      100


#define XEMPTY 9999999.0f


/** Data for one segment
*/
typedef double tSegmentData[7]; //!<  X,Y,Z, aX,aY,aZ, Length

/** Data for one marker
*/
typedef float  tMarkerData[3];  //!<  X,Y,Z

/** Data for one forceplate
*/
typedef float  tForceData[7];   //!<  X,Y,Z, fX,fY,fZ, mZ

/** Data for one degree of freedom
*/
typedef double tDofData;        //!<  Usually an angle value in degrees




//==================================================================
//==================================================================
//
//   S T R U C T U R E S
//
//==================================================================
//==================================================================


//==================================================================

//! The description of the connection to Cortex.
/*!
This contains information about the host machine, the host program, and the connection status.
*/
typedef struct sHostInfo
{
    int           bFoundHost;              //!< True = have talked to Cortex
    int           LatestConfirmationTime;  //!< Time of last receipt from Cortex
    char          szHostMachineName[128];  //!< Name of machine Cortex is running on
    unsigned char HostMachineAddress[4];   //!< IP Address of that machine 
    char          szHostProgramName[128];  //!< Name of module communicating with
    unsigned char HostProgramVersion[4];   //!< Version number of that module

} sHostInfo;


//==================================================================

//! The rudimentary description of a skeleton's bones and hierarchy.
/*!
This description is defined by szSegmentNames[iSegment], and iParent[iSegment]
*/
typedef struct sHierarchy
{
    int            nSegments;         //!< Number of segments
    char**         szSegmentNames;    //!< Array of segment names
    int*           iParents;          //!< Array of segment parents (defines the hierarchy)

} sHierarchy;


//==================================================================

//! The description of a single tracking object that will have streaming data.
/*!
This description includes the object's name, the marker names, the skeleton hierarchy, and the DOF names.
*/
typedef struct sBodyDef
{
    char*          szName;            //!< Name of the object

    int            nMarkers;          //!< Number of markers
    char**         szMarkerNames;     //!< Array of marker names

    sHierarchy     Hierarchy;         //!< The Skeleton description for HTR data

    int            nDofs;             //!< Number of degrees of freedom
    char**         szDofNames;        //!< Array of degrees of freedom names

} sBodyDef;


//==================================================================

//! The description of all the data that will stream from Cortex.
/*!
This description includes all the body descriptions, the analog channels,
and the number of forceplates.
*/
typedef struct sBodyDefs
{
    int            nBodyDefs;               //!< Number of bodies being tracked
    sBodyDef       BodyDefs[MAX_N_BODIES];  //!< The definition of each body

    int            nAnalogChannels;         //!< The number of active analog channels
    char**         szAnalogChannelNames;    //!< The names given to each channel

    int            nForcePlates;            //!< The number of active forceplates

    void*          AllocatedSpace;          //!< Private space (DON'T TOUCH)

} sBodyDefs;


//==================================================================

//! A structure containing ALL the data to drive one markerset.
/*!
This contains the markerset's name, the marker positions, the segment positions relative to each segment's parent, and the DOFs.
*/
typedef struct sBodyData
{
    char           szName[128];          //!< For dynamic matching of objects.

    int            nMarkers;             //!< Number of markers defined
    tMarkerData*   Markers;              //!< [nMarkers][3] array.  Markers[iMarker][0] == XEMPTY means no data.
    float          fAvgMarkerResidual;   //!< Average residual of the marker triangulations

    int            nSegments;            //!< Number of segments
    tSegmentData*  Segments;             //!< [nSegments][7] array

    int            nDofs;                //!< Number of degrees-of-freedom
    tDofData*      Dofs;                 //!< Array of degree-of-freedom angles
    float          fAvgDofResidual;      //!< Average residual from the solve
    int            nIterations;          //!< Number of iterations to solve

    int            ZoomEncoderValue;     //!< Zoom value from the Camera Tracker Encoder
    int            FocusEncoderValue;    //!< Focus value from the Camera Tracker Encoder

} sBodyData;


//==================================================================

//! All the analog data for one frame's worth of time.
/*!
This includes the raw analog samples, processed forces, and also angle encoder values (if available).
*/
typedef struct sAnalogData
{
    int            nAnalogChannels;  //!< Total number of active channels
    int            nAnalogSamples;   //!< The number of samples in the current frame
    short*         AnalogSamples;    //!< The data: nChannels * nSamples of these

    int            nForcePlates;     //!< Total number of active forceplates
    int            nForceSamples;    //!< The number of samples in the current frame
    tForceData*    Forces;           //!< The forces: nForcePlates * nForceSamples of these

    int            nAngleEncoders;      //!< Number of encoders
    int            nAngleEncoderSamples;//!< Number of samples per encoder
    double*        AngleEncoderSamples; //!< The angles: nEncoders*nEncoderSamples of these

} sAnalogData;


//==================================================================

//! The recording status tells us the frame numbers and capture filename.
typedef struct sRecordingStatus
{
    int            bRecording;   //!< 0=Not Recording, anything else=Recording
    int            iFirstFrame;  //!< The frame number of the first data frame to be recorded from Cortex Live Mode
    int            iLastFrame;   //!< The frame number of the last data frame to be recorded from Cortex Live Mode
    char           szFilename[256]; //!< The full capture filename 

} sRecordingStatus;


//==================================================================

//! ALL the data for one frame streamed from Cortex.
/*!
This include the two items that describe the frame. The first is the frame number.
The second is the time delay measuring the delay between the real world action and the host sending this frame.
The actual data for the frame includes the data for each body, the unidentified markers, and data that is 
associated with the analog captures.
*/
typedef struct sFrameOfData
{
    int            iFrame;                  //!< Cortex's frame number
    float          fDelay;                  //!< Total time (seconds) from Camera to the Host sending the data

    int            nBodies;                 //!< The bodies should match the descriptions
    sBodyData      BodyData[MAX_N_BODIES];  //!< The data for each body

    int            nUnidentifiedMarkers;    //!< Number of unrecognized markers
    tMarkerData*   UnidentifiedMarkers;     //!< The unrecognized markers

    sAnalogData    AnalogData;              //!< The analog data packaged

    sRecordingStatus RecordingStatus;       //!< Info about name and frames being recorded

} sFrameOfData;



#ifdef  __cplusplus
extern "C" {
#endif


//==================================================================

/** This function returns a 4-byte version number.
 *
 * \param Version - An array of four bytes: ModuleID, Major, Minor, Bugfix
 *
 * \return RC_Okay
*/
int Cortex_GetSdkVersion(unsigned char Version[4]);

//==================================================================

/** This function sets the filter level of the LogMessages.
 *
 *  The default verbosity level is VL_Warning.
 *
 * \param iLevel - one of the maVerbosityLevel enum values.
 *
 * \return RC_Okay
*/
int Cortex_SetVerbosityLevel(int iLevel);

//==================================================================

/**   The user supplied function handles text messages posted from within the SDK.
 *
 *    Logging messages is done as a utility to help code and/or run using the SDK.
 *    Various messages get posted for help with error conditions or events that happen.
 *    Each message has a Log-Level assigned to it so the user can.
 *  \sa Cortex_SetVerbosityLevel
 *
 *
 *  \param  MyFunction - This user defined function handles messages from the SDK.
 *
 *  \return maReturnCode - RC_Okay
*/
int Cortex_SetErrorMsgHandlerFunc(void (*MyFunction)(int iLogLevel, const char* szLogMessage));

//==================================================================

/**   The user supplied function will be called whenever a frame of data arrives.
 *
 *    The ethernet servicing is done via a thread created
 *    when the connection to Cortex is made.  This function is
 *    called from that thread.  Some tasks are not sharable
 *    directly across threads.  Window redrawing, for example,
 *    should be done via events or messages.
 *
 *  \param MyFunction - This user supply callback function handles the streaming data
 *
 *  \return maReturnCode - RC_Okay
 *
 *    Notes: The data parameter points to "hot" data. That frame of data
 *           will be overwritten with the next call to the callback function.
*/
int Cortex_SetDataHandlerFunc(void (*MyFunction)(sFrameOfData* pFrameOfData));

//==================================================================

/**   This function defines the connection routes to talk to Cortex.
 *
 *    Machines can have more than one ethernet interface.  This function
 *    is used to either set the ethernet interface to use, or to let
 *    the SDK auto-select the local interface, and/or the Cortex host.
 *    This function should only be called once at startup.
 *
 *  \param szMyNicCardAddress - "a.b.c.d" or HostName.  "" and NULL mean AutoSelect
 *
 *  \param szCortexNicCardAddress - "a.b.c.d" or HostName.  "" and NULL mean AutoSelect
 *
 *  \return maReturnCode - RC_Okay, RC_ApiError, RC_NetworkError, RC_GeneralError
*/
int Cortex_Initialize(const char* szMyNicCardAddress, const char* szCortexNicCardAddress);  // Hostname or IP Address or NULL (auto find)

//==================================================================

/** This function gets information about the connection to Cortex
 *
 *  This function returns IP-Address information and Cortex version information.
 *  The version info can be used to handle incompatible changes in either our code
 *  or your code.
 *
 * \param pHostInfo - Structure containing connection information
 *
 * \return RC_Okay, RC_NetworkError
*/
int Cortex_GetHostInfo(sHostInfo *pHostInfo);

//==================================================================

/** This function stops all activity of the SDK.
 *
 *  This function should be called once before exiting.
*/
int Cortex_Exit();

//==================================================================

/**   This function sends commands to Cortex and returns a response.
 *
 *    This function is an extendable interface between the Client programs
 *    and the Host (Cortex) program.  The commands are sent as readable text strings.
 *    The response is returned unaltered.
 *
 * \param szCommand - The request to send the Cortex
 * \param ppResponse - The reply
 * \param pnBytes - The number of bytes in the response
 *
 \verbatim
Example:
    void *pResponse=NULL;
    Cortex_Request("GetContextFrameRate", &pResponse, sizeof(void*));
    fFrameRate = *(float*)pResponse;
\endverbatim 
 *
 * \return RC_Okay, RC_TimeOut, RC_NotRecognized, RC_GeneralError
*/
int Cortex_Request(const char* szCommand, void** ppResponse, int *pnBytes);  // Friendly extendable command function.

//==================================================================

/**   This function queries Cortex for its set of tracking objects.
 *
 *  \return sBodyDefs* - This is a pointer to the internal storage of
 *                       the results of the latest call to this function.
 *
 *  \sa Cortex_FreeBodyDefs
*/
sBodyDefs*     Cortex_GetBodyDefs();      // The description of what we are tracking.

//==================================================================

/** This function frees the memory allocated by Cortex_GetBodyDefs
 *
 *  The data within the structure is freed and also the structure itself.

 * \param pBodyDefs - The item to free.
 *
 * \return RC_Okay
*/
int Cortex_FreeBodyDefs(sBodyDefs* pBodyDefs);

//==================================================================

/** This function polls Cortex for the current frame
 *
 *  The SDK user has the streaming data available via the callback function.
 *  In addition, this function is available to get a frame directly.
 *
 *  Note: Cortex considers the current frame to be the latest LiveMode frame completed or,
 *        if not in LiveMode, the current frame is the one that is displayed on the screen.
 *
 * \return sFrameOfData
*/
sFrameOfData*  Cortex_GetCurrentFrame();  // Can POLL for the current frame.

//==================================================================

/** This function copies a frame of data.
 *
 *  The Destination frame should start initialized to all zeros.  The CopyFrame
 *  and FreeFrame functions will handle the memory allocations necessary to fill
 *  out the data.
 *
 * \param pSrc - The frame to copy FROM.
 * \param pDst - The frame to copy TO
 *
 * \return RC_Okay, RC_MemoryError
*/
int Cortex_CopyFrame(const sFrameOfData* pSrc, sFrameOfData* pDst);  // Allocates or reallocates pointers

//==================================================================

/** This function frees memory within the structure.
 *
 *  The sFrameOfData structure includes pointers to various pieces of data.
 *  That data is dynamically allocated or reallocated to be consistent with
 *  the data that has arrived from Cortex.  To properly use the sFrameOfData
 *  structure, you should use the utility functions supplied.  It is possible
 *  to reuse sFrameOfData variables without ever freeing them.  The SDK will
 *  reallocate the components for you.
 *
 * \param pFrame - The frame of data to free.
 *
 * \return RC_Okay
*/
int Cortex_FreeFrame(sFrameOfData* pFrame);

//==================================================================

/** This function pushes a skeleton definition to Cortex.
 *
 *  A skeleton, defined in an animation package can be used to start
 *  a skeleton model definition in Cortex.  The hierarchy and starting
 *  pose can come from the animation package.  The rest of the details
 *  of the skeleton get filled out in the Cortex interface.  The parameters
 *  to this function match the parameters defining the HTR data that
 *  normally gets sent through the SDK2.
 *
 * \param pHierarchy - The number of segments, their names and parent child
                       relationships.
 * \param pFrame - One frame of HTR data dimensioned according to the number
 *                 of segments defined in the pHierarchy parameter.
 *
 * \return - RC_Okay, RC_NetworkError
*/
int Cortex_SendHtr(sHierarchy *pHierarchy, tSegmentData *pFrame);    // Push a skeleton definition to Cortex

int Cortex_SetMetered(bool bActive, float fFixedLatency);

//==================================================================
// Euler angle utility functions
//==================================================================

#define ZYX_ORDER 1
#define XYZ_ORDER 2
#define YXZ_ORDER 3
#define YZX_ORDER 4
#define ZXY_ORDER 5
#define XZY_ORDER 6

// Special rotation orders
#define XYX_ORDER 7
#define XZX_ORDER 8
#define YZY_ORDER 9
#define YXY_ORDER 10
#define ZXZ_ORDER 11
#define ZYZ_ORDER 12


//==================================================================

/** This function constructs a rotation matrix from three Euler angles.
 *
 *  This function and its inverse are utility functions for processing
 *  the HTR rotations we send in each frame of data. We send Euler angles
 *  in ZYX format (some interpretations would call it XYZ). Using these
 *  conversion utilities should protect against any misinterpretations.
 *
 * \param matrix - 3x3 rotation matrix.
 * \param iRotationOrder - one of:
 *
 *        ZYX_ORDER
 *        XYZ_ORDER
 *        YXZ_ORDER
 *        YZX_ORDER
 *        ZXY_ORDER
 *        XZY_ORDER
 *
 * \param angles - the angles in degrees.
 *
 */
void  Cortex_ConstructRotationMatrix(
        double angles[3],
        int iRotationOrder,
        double matrix[3][3]);

//==================================================================

/** This function decodes a rotation matrix into three Euler angles.
 *
 *  This function and its inverse are utility functions for processing
 *  the HTR rotations we send in each frame of data. We send Euler angles
 *  in ZYX format (some interpretations would call it XYZ). Using these
 *  conversion utilities should protect against any misinterpretations.
 *
 * \param matrix - 3x3 rotation matrix.
 * \param iRotationOrder - one of:
 *
 *        ZYX_ORDER
 *        XYZ_ORDER
 *        YXZ_ORDER
 *        YZX_ORDER
 *        ZXY_ORDER
 *        XZY_ORDER
 *
 * \param angles - the angles in degrees.
 *
*/
void  Cortex_ExtractEulerAngles(
        double matrix[3][3],
        int    iRotationOrder,
        double angles[3]);



#ifdef  __cplusplus
}
#endif

#endif
