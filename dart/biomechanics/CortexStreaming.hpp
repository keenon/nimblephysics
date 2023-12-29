#ifndef DART_CORTEX_STREAMING_HPP_
#define DART_CORTEX_STREAMING_HPP_

#include <future>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <netinet/in.h>

#include "dart/external/cortex/cortex_intern.h"
#include "dart/math/MathTypes.hpp"

namespace dart {

namespace biomechanics {

//////////////////////////////////////////////////////////
// Structures for configuring a streaming session

typedef struct CortexBodyDef
{
  std::string name;
  std::vector<std::string> markerNames;

  std::vector<std::string> segmentNames;
  std::vector<int> segmentParents;

  std::vector<std::string> dofNames;
} CortexBodyDef;

typedef struct CortexBodyDefs
{
  std::vector<CortexBodyDef> bodyDefs;

  std::vector<std::string> analogChannelNames;
  int numForcePlates;
} CortexBodyDefs;

typedef struct CortexBodyData
{
  // Body data
  std::string name;

  // Marker data
  std::vector<std::string> markerNames;
  std::vector<Eigen::Vector3s> markers;
  float markerTriangulationAvgResidual;

  // Segments from IK
  std::vector<Eigen::Vector7s> segments;

  // IK solutions
  Eigen::VectorXs dofs;
  float avgDofResidualFromIK;
  int numIterationsToSolveIK;

  // Camera tracker encoders
  int zoomEncoderValue;
  int focusEncoderValue;
} CortexBodyData;

typedef struct CortexAnalogData
{
  int numAnalogSamplesPerFrame;
  // Raw analog samples, where each channel gets a vector of samples taken
  // during this frame
  std::vector<Eigen::VectorXi> analogSamples;

  int numForcePlateSamplesPerFrame;
  // Processed force plate data. Each force plate gets a Nx9 matrix of samples,
  // where each row is a sample.
  std::vector<Eigen::MatrixXs> plateCopForceMoment;

  // Angle encoder values
  std::vector<Eigen::VectorXs> angleEncoderSamples;
} CortexAnalogData;

//==================================================================

//! The recording status tells us the frame numbers and capture filename.
typedef struct sRecordingStatus
{
  int bRecording;  //!< 0=Not Recording, anything else=Recording
  int iFirstFrame; //!< The frame number of the first data frame to be recorded
                   //!< from Cortex Live Mode
  int iLastFrame;  //!< The frame number of the last data frame to be recorded
                   //!< from Cortex Live Mode
  char szFilename[256]; //!< The full capture filename

} sRecordingStatus;

typedef struct CortexFrameOfData
{
  int cortexFrameNumber;
  float cameraToHostDelaySeconds;
  int cortexTag;

  std::vector<CortexBodyData> bodyData; //!< The data for each body

  std::vector<Eigen::Vector3s>
      unidentifiedMarkers; //!< The unrecognized markers

  CortexAnalogData analogData; //!< The analog data packaged

  sRecordingStatus
      RecordingStatus; //!< Info about name and frames being recorded

} CortexFrameOfData;

//////////////////////////////////////////////////////////
// The actual implementation class
class CortexStreaming
{
public:
  CortexStreaming(
      std::string cortexNicAddress,
      int cortexMulticastPort = 1001,
      int cortexRequestsPort = 1510);

  ~CortexStreaming();

  /// This is the callback that gets called when a frame of data is received
  void setFrameHandler(
      std::function<void(
          std::vector<std::string> markerNames,
          std::vector<Eigen::Vector3s> markers,
          std::vector<Eigen::MatrixXs> copTorqueForces)> handler);

  /// This is used for mocking the Cortex API server for local testing. This
  /// sets the current body defs and frame of data to send back to the client.
  void mockServerSetData(
      std::vector<std::string> markerNames,
      std::vector<Eigen::Vector3s> markers,
      std::vector<Eigen::MatrixXs> copTorqueForces);

  /// This connects to Cortex, and requests the body defs and a frame of data
  void initialize();

  /// This creates a UDP socket and starts listening for packets from Cortex
  void connect();

  /// This starts a UDP server that mimicks the Cortex API, so we can test
  /// locally without having to run Cortex. This is an alternative to connect(),
  /// and cannot run in the same process as connect().
  void startMockServer();

  /// This closes the UDP socket and stops listening for packets from Cortex
  void disconnect();

  /// This sends a UDP packet to the host machine, whatever is at
  /// mHostMachineAddress.
  void sendToCortex(std::vector<unsigned char> packet);

  /// This returns the data to send over UDP to introduce ourselves to Cortex
  std::vector<unsigned char> createHelloWorldPacket();

  /// This returns the data to send over UDP to introduce ourselves to a client
  std::vector<unsigned char> createHereIAmPacket();

  /// This returns the data to send over UDP to ask Cortex for the body defs
  std::vector<unsigned char> createRequestBodyDefsPacket();

  /// This returns the data to send over UDP to ask Cortex for a frame of data
  std::vector<unsigned char> createRequestFramePacket();

  /// This is used for mocking the Cortex API for local testing
  std::vector<unsigned char> createBodyDefsPacket(CortexBodyDefs bodyDefs);

  /// This is used for mocking the Cortex API for local testing
  std::vector<unsigned char> createFrameOfDataPacket(
      CortexFrameOfData frameOfData);

  /// This is used for the mock server to set the current body defs that it
  /// will respond to requests with.
  CortexBodyDefs getCurrentBodyDefs();

  /// This is used for the mock server to set the current frame definition that
  /// it will respond to requests with.
  CortexFrameOfData getCurrentFrameOfData();

  /// This runs on any incoming UDP packets, to decide how to parse them
  void parseCortexPacket(
      sPacket* packet, sockaddr_in fromAddress, bool isMulticast);

  /// This runs on any incoming UDP packets, to decide how to parse them. This
  /// runs inside the mock server, which is used for local testing.
  void mockServerParseCortexPacket(sPacket* packet, sockaddr_in fromAddress);

  /// This sends a UDP packet to the host machine, whatever is at
  /// mHostMachineAddress.
  void mockServerSendResponsePacket(
      std::vector<unsigned char> packet, sockaddr_in fromAddress);

  /// This sends a UDP packet out on the multicast address, to tell everyone
  /// about the current frame
  void mockServerSendFrameMulticast();

  /// This is responsible for parsing the Cortex packet describing the
  /// names of the bodies, markers, force plates, analog channels, etc.
  CortexBodyDefs parseBodyDefs(char* data, int nBytes);
  std::pair<CortexBodyDef, int> parseBodyDef(char* ptr, int nBytes);
  int parseAnalogDefs(char* ptr, int nBytes, CortexBodyDefs& bodyDefs);

  void parseAndHandleFrameOfData(char* data, int nBytes);
  /// This is responsible for parsing the Cortex packet describing the
  /// data for a single frame of mocap: marker locations, force plates,
  /// analog channels, etc.
  CortexFrameOfData parseFrameOfData(char* data, int nBytes);
  std::pair<CortexBodyData, int> parseBodyData(
      char* ptr, int nBytes, int iBody);
  std::pair<CortexAnalogData, int> parseAnalogData(char* ptr, int nBytes);

protected:
  std::function<void(
      std::vector<std::string> markerNames,
      std::vector<Eigen::Vector3s> markers,
      std::vector<Eigen::MatrixXs> copTorqueForces)>
      mFrameHandler;

  CortexBodyDefs mBodyDefs;
  CortexFrameOfData mFrameOfData;

  const unsigned char VERSION_NUMBER[4]
      = {4, 1, 12, 0}; // ProgramID, Major, Minor, Bugfix

  int mCortexMulticastPort = 1001;
  int mCortexRequestsPort = 1510;

  bool mRunningThreads = false;

  // This is info for the multicast listener socket and thread
  int mMulticastListenerSocketFd = -1;
  std::future<void> mMulticastListenerThreadFuture;

  // This is info for the cortex API listener socket and thread
  int mCortexListenerSocketFd = -1;
  std::future<void> mCortexListenerThreadFuture;

  bool mFoundHost = false;      //!< True = have talked to Cortex
  std::string mHostMachineName; //!< Name of machine Cortex is running on
  unsigned char mHostMachineAddress[4]; //!< IP Address of that machine
  std::string mHostProgramName;         //!< Name of module communicating with
  unsigned char mHostProgramVersion[4]; //!< Version number of that module
};

} // namespace biomechanics
} // namespace dart

#endif