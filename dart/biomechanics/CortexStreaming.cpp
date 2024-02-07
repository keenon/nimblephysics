#include "dart/biomechanics/CortexStreaming.hpp"

#include <cstring>
#include <future>
#include <iostream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "dart/external/cortex/cortex_intern.h"
#include "dart/math/MathTypes.hpp"

using namespace dart;
using namespace biomechanics;

#define XEMPTY 9999999.0f

//==============================================================================
CortexStreaming::CortexStreaming(
    std::string cortexNicAddress,
    int cortexMulticastPort,
    int cortexRequestsPort)
  : mFoundHost(false),
    mHostMachineName(""),
    mHostProgramName(""),
    mCortexMulticastPort(cortexMulticastPort),
    mCortexRequestsPort(cortexRequestsPort),
    mFrameHandler(nullptr)
{
  int errorCode
      = inet_pton(AF_INET, cortexNicAddress.c_str(), mHostMachineAddress);
  if (errorCode == 0)
  {
    std::cout << "inet_pton() failed: Invalid cortexNicAddress "
              << cortexNicAddress << std::endl;
    return;
  }
  else if (errorCode < 0)
  {
    std::cout << "inet_pton() failed with return code " << errorCode
              << " when attempting to parse cortexNicAddress "
              << cortexNicAddress << std::endl;
    return;
  }
  memset(mHostProgramVersion, 0, 4);
}

//==============================================================================
CortexStreaming::~CortexStreaming()
{
  if (mRunningThreads)
  {
    disconnect();
  }
}

//==============================================================================
/// This is the callback that gets called when a frame of data is received
void CortexStreaming::setFrameHandler(
    std::function<void(
        std::vector<std::string> markerNames,
        std::vector<Eigen::Vector3s> markers,
        std::vector<Eigen::MatrixXs> copTorqueForces)> handler)
{
  mFrameHandler = handler;
}

//==============================================================================
/// This is used for mocking the Cortex API server for local testing. This
/// sets the current body defs and frame of data to send back to the client.
void CortexStreaming::mockServerSetData(
    std::vector<std::string> markerNames,
    std::vector<Eigen::Vector3s> markers,
    std::vector<Eigen::MatrixXs> copTorqueForces)
{
  mBodyDefs.bodyDefs.resize(1);
  mBodyDefs.bodyDefs[0].name = "MockBody";
  mBodyDefs.bodyDefs[0].markerNames = markerNames;
  mBodyDefs.bodyDefs[0].segmentNames = std::vector<std::string>();
  mBodyDefs.bodyDefs[0].segmentParents = std::vector<int>();
  mBodyDefs.bodyDefs[0].dofNames = std::vector<std::string>();
  mBodyDefs.analogChannelNames = std::vector<std::string>();
  mBodyDefs.numForcePlates = copTorqueForces.size();

  mFrameOfData.cortexFrameNumber = 0;
  mFrameOfData.cameraToHostDelaySeconds = 0.01;
  mFrameOfData.analogData.numAnalogSamplesPerFrame = 0;
  mFrameOfData.analogData.numForcePlateSamplesPerFrame
      = copTorqueForces.size() > 0 ? copTorqueForces[0].rows() : 0;
  mFrameOfData.analogData.plateCopTorqueForce = copTorqueForces;
  mFrameOfData.analogData.analogSamples = std::vector<Eigen::VectorXi>();
  mFrameOfData.analogData.angleEncoderSamples = std::vector<Eigen::VectorXs>();
  mFrameOfData.bodyData.resize(1);
  mFrameOfData.bodyData[0].name = "MockBody";
  mFrameOfData.bodyData[0].markers = markers;
  mFrameOfData.bodyData[0].dofs = Eigen::VectorXs::Zero(0);
  mFrameOfData.unidentifiedMarkers = std::vector<Eigen::Vector3s>();
  mFrameOfData.cortexTag = 0;
}

//==============================================================================
/// This connects to Cortex, and requests the body defs and a frame of data
void CortexStreaming::initialize()
{
  connect();

  // Sleep for 100ms
  usleep(100 * 1000);

  // Send a hello world packet
  sendToCortex(createHelloWorldPacket());

  // Sleep for 100ms
  usleep(100 * 1000);

  // Send a request for body defs
  sendToCortex(createRequestBodyDefsPacket());

  // Sleep for 100ms
  usleep(100 * 1000);

  // Send a request for a frame of data
  sendToCortex(createRequestFramePacket());

  // Sleep for 100ms
  usleep(100 * 1000);
}

//==============================================================================
/// This creates a UDP socket and starts listening for packets from Cortex
void CortexStreaming::connect()
{
  /////////////////////////////////////////////////////////////////////////
  // 1. Create the broadcast UDP listener socket
  /////////////////////////////////////////////////////////////////////////

  mMulticastListenerSocketFd = socket(AF_INET, SOCK_DGRAM, 0);
  if (mMulticastListenerSocketFd < 0)
  {
    throw std::runtime_error("ERROR opening socket");
  }
  struct sockaddr_in serv_addr;
  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET; // IPv4
  serv_addr.sin_port
      = htons(mCortexMulticastPort);      // Listen to the cortex multicast port
  serv_addr.sin_addr.s_addr = INADDR_ANY; // All interfaces
  if (bind(
          mMulticastListenerSocketFd,
          (struct sockaddr*)&serv_addr,
          sizeof(serv_addr))
      < 0)
  {
    close(mMulticastListenerSocketFd);
    throw std::runtime_error("ERROR on binding");
  }

  // Find the port that was assigned to us
  socklen_t len = sizeof(serv_addr);
  if (getsockname(
          mMulticastListenerSocketFd, (struct sockaddr*)&serv_addr, &len)
      == -1)
  {
    close(mMulticastListenerSocketFd);
    throw std::runtime_error("ERROR on getsockname");
  }
  int assignedPort = ntohs(serv_addr.sin_port);
  std::cout << "Cortex UDP Multicast listener socket was assigned port: "
            << assignedPort << std::endl;

  struct ip_mreqn stMreq;

  // Join the multicast group
  // Note: This is NOT necessary for the host program.
  // Cortex sends frames to this address and associated port: 225.1.1.1:1001
  in_addr MultiCastAddress = {(225u << 24) + (1u << 16) + (1u << 8) + 1u};
  stMreq.imr_multiaddr.s_addr = htonl(MultiCastAddress.s_addr);
  stMreq.imr_address.s_addr = INADDR_ANY;
  stMreq.imr_ifindex = 0;
  if (setsockopt(
          mMulticastListenerSocketFd,
          IPPROTO_IP,
          IP_ADD_MEMBERSHIP,
          (char*)&stMreq,
          sizeof(ip_mreqn))
      < 0)
  {
    close(mMulticastListenerSocketFd);
    throw std::runtime_error("Joining the multicast group failed");
  }

  mRunningThreads = true;

  auto broadcastReceiveFunc = [this]() -> void {
    sPacket buffer;
    int bufferLength = sizeof(buffer);
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    while (mRunningThreads)
    {
      int nBytes = recvfrom(
          mMulticastListenerSocketFd,
          (char*)&buffer,
          bufferLength,
          0,
          (struct sockaddr*)&cli_addr,
          &clilen);
      if (nBytes < 0)
      {
        if (mRunningThreads)
        {
          std::cerr << "Error in recvfrom" << std::endl;
        }
        break;
      }

      // std::cout << "Client received " << nBytes << " bytes from UDP
      // multicast"
      //           << std::endl;

      parseCortexPacket(&buffer, cli_addr, true);
    }
  };

  mMulticastListenerThreadFuture
      = std::async(std::launch::async, broadcastReceiveFunc);

  /////////////////////////////////////////////////////////////////////////
  // 2. Create the Cortex API socket
  /////////////////////////////////////////////////////////////////////////

  // Create a socket
  mCortexListenerSocketFd = socket(AF_INET, SOCK_DGRAM, 0);
  if (mCortexListenerSocketFd < 0)
  {
    throw std::runtime_error("Listener socket creation failed");
  }

  // Define the server address structure
  struct sockaddr_in listener_serv_addr;
  memset(&serv_addr, 0, sizeof(listener_serv_addr));
  listener_serv_addr.sin_family = AF_INET; // IPv4
  listener_serv_addr.sin_port = 0;         // Automatically assign a port
  listener_serv_addr.sin_addr.s_addr = INADDR_ANY; // Bind to all interfaces

  // Bind the socket
  if (bind(
          mCortexListenerSocketFd,
          (struct sockaddr*)&listener_serv_addr,
          sizeof(listener_serv_addr))
      < 0)
  {
    close(mCortexListenerSocketFd);
    throw std::runtime_error("ERROR on binding");
  }

  // Retrieve the assigned port number
  socklen_t listener_len = sizeof(serv_addr);
  if (getsockname(
          mCortexListenerSocketFd, (struct sockaddr*)&serv_addr, &listener_len)
      == -1)
  {
    close(mCortexListenerSocketFd);
    throw std::runtime_error("ERROR on getsockname");
  }
  int listenerAssignedPort = ntohs(serv_addr.sin_port);

  // Output the assigned port number
  std::cout << "Cortex SDK socket bound to port: " << listenerAssignedPort
            << std::endl;

  auto cortexReceiveFunc = [this]() -> void {
    sPacket buffer;
    int bufferLength = sizeof(buffer);
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    while (mRunningThreads)
    {
      int nBytes = recvfrom(
          mCortexListenerSocketFd,
          (char*)&buffer,
          bufferLength,
          0,
          (struct sockaddr*)&cli_addr,
          &clilen);
      if (nBytes < 0)
      {
        if (mRunningThreads)
        {
          std::cerr << "Error in recvfrom" << std::endl;
        }
        break;
      }

      // std::cout << "Client received " << nBytes << " bytes from SDK server"
      //           << std::endl;

      parseCortexPacket(&buffer, cli_addr, false);
    }
  };

  mCortexListenerThreadFuture
      = std::async(std::launch::async, cortexReceiveFunc);
}

//==============================================================================
/// This starts a UDP server that mimicks the Cortex API, so we can test
/// locally without having to run Cortex. This is an alternative to connect(),
/// and cannot run in the same process as connect().
void CortexStreaming::startMockServer()
{
  /////////////////////////////////////////////////////////////////////////
  // 1. Create the broadcast UDP broadcast socket
  /////////////////////////////////////////////////////////////////////////

  mMulticastListenerSocketFd = socket(AF_INET, SOCK_DGRAM, 0);
  if (mMulticastListenerSocketFd < 0)
  {
    throw std::runtime_error("ERROR opening broadcast socket");
  }
  struct sockaddr_in serv_addr;
  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;         // IPv4
  serv_addr.sin_port = 0;                 // Auto assign
  serv_addr.sin_addr.s_addr = INADDR_ANY; // All interfaces
  if (bind(
          mMulticastListenerSocketFd,
          (struct sockaddr*)&serv_addr,
          sizeof(serv_addr))
      < 0)
  {
    close(mMulticastListenerSocketFd);
    throw std::runtime_error("ERROR on binding multicast socket");
  }

  mRunningThreads = true;

  std::promise<void> promise;
  promise.set_value();
  mMulticastListenerThreadFuture = promise.get_future();

  /////////////////////////////////////////////////////////////////////////
  // 2. Create the Cortex API socket
  /////////////////////////////////////////////////////////////////////////

  // Create a socket
  mCortexListenerSocketFd = socket(AF_INET, SOCK_DGRAM, 0);
  if (mCortexListenerSocketFd < 0)
  {
    throw std::runtime_error("Listener socket creation failed");
  }

  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;                  // IPv4
  serv_addr.sin_port = htons(mCortexRequestsPort); // Port number
  serv_addr.sin_addr.s_addr = INADDR_ANY;          // All interfaces
  if (bind(
          mCortexListenerSocketFd,
          (struct sockaddr*)&serv_addr,
          sizeof(serv_addr))
      < 0)
  {
    close(mCortexListenerSocketFd);
    throw std::runtime_error("ERROR on binding SDK listener socket");
  }

  auto cortexReceiveFunc = [this]() -> void {
    sPacket buffer;
    int bufferLength = sizeof(buffer);
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    while (mRunningThreads)
    {
      int nBytes = recvfrom(
          mCortexListenerSocketFd,
          (char*)&buffer,
          bufferLength,
          0,
          (struct sockaddr*)&cli_addr,
          &clilen);
      if (nBytes < 0)
      {
        if (mRunningThreads)
        {
          std::cerr << "Error in recvfrom" << std::endl;
        }
        break;
      }

      // std::cout << "Mock server received " << nBytes << " bytes from SDK
      // client"
      //           << std::endl;

      mockServerParseCortexPacket(&buffer, cli_addr);
    }
  };

  mCortexListenerThreadFuture
      = std::async(std::launch::async, cortexReceiveFunc);
}

//==============================================================================
/// This closes the UDP socket and stops listening for packets from Cortex
void CortexStreaming::disconnect()
{
  mRunningThreads = false;
  close(mMulticastListenerSocketFd);
  mMulticastListenerThreadFuture.wait();
  close(mCortexListenerSocketFd);
  mCortexListenerThreadFuture.wait();
}

//==============================================================================
/// This sends a UDP packet to the host machine, whatever is at
/// mHostMachineAddress.
void CortexStreaming::sendToCortex(std::vector<unsigned char> packet)
{
  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));

  // Filling server information
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(mCortexRequestsPort);
  memcpy(&servaddr.sin_addr.s_addr, mHostMachineAddress, 4);

  // Send the message
  while (sendto(
             mCortexListenerSocketFd,
             packet.data(),
             packet.size(),
             0,
             (const struct sockaddr*)&servaddr,
             sizeof(servaddr))
         != packet.size())
  {
    std::cerr << "sendto failed" << std::endl;
  }
}

//==============================================================================
/// This returns the data to send over UDP to introduce ourselves to Cortex
std::vector<unsigned char> CortexStreaming::createHelloWorldPacket()
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_HELLO_WORLD;
  packetOut.nBytes = sizeof(sMe);
  strcpy(packetOut.Data.Me.szName, "ClientTest");
  memcpy(packetOut.Data.Me.Version, VERSION_NUMBER, 4);

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This returns the data to send over UDP to introduce ourselves to a client
std::vector<unsigned char> CortexStreaming::createHereIAmPacket()
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_HERE_I_AM;
  packetOut.nBytes = sizeof(sMe);
  strcpy(packetOut.Data.Me.szName, "MockServer");
  memcpy(packetOut.Data.Me.Version, VERSION_NUMBER, 4);

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This returns the data to send over UDP to ask Cortex for the body defs
std::vector<unsigned char> CortexStreaming::createRequestBodyDefsPacket()
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_REQUEST_BODYDEFS;
  packetOut.nBytes = 0;

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This returns the data to send over UDP to ask Cortex for a frame of data
std::vector<unsigned char> CortexStreaming::createRequestFramePacket()
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_REQUEST_FRAME;
  packetOut.nBytes = 0;

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This is used for mocking the Cortex API for local testing
std::vector<unsigned char> CortexStreaming::createBodyDefsPacket(
    CortexBodyDefs bodyDefs)
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_BODYDEFS;
  packetOut.nBytes = 0;

  char* ptr = packetOut.Data.cData;

  // List the number of bodies
  int nBodies = bodyDefs.bodyDefs.size();
  memcpy(ptr, &nBodies, 4);
  ptr += 4;

  for (int body = 0; body < nBodies; body++)
  {
    // The main name
    int nameLength = bodyDefs.bodyDefs[body].name.size() + 1;
    memcpy(ptr, bodyDefs.bodyDefs[body].name.c_str(), nameLength);
    ptr += nameLength;

    // Markers
    int nMarkers = bodyDefs.bodyDefs[body].markerNames.size();
    memcpy(ptr, &nMarkers, 4);
    ptr += 4;

    for (int iMarker = 0; iMarker < nMarkers; iMarker++)
    {
      int markerNameLength
          = bodyDefs.bodyDefs[body].markerNames[iMarker].size() + 1;
      memcpy(
          ptr,
          bodyDefs.bodyDefs[body].markerNames[iMarker].c_str(),
          markerNameLength);
      ptr += markerNameLength;
    }

    // Segments
    int nSegments = bodyDefs.bodyDefs[body].segmentNames.size();
    memcpy(ptr, &nSegments, 4);
    ptr += 4;
    for (int iSegment = 0; iSegment < nSegments; iSegment++)
    {
      int segmentNameLength
          = bodyDefs.bodyDefs[body].segmentNames[iSegment].size() + 1;
      memcpy(
          ptr,
          bodyDefs.bodyDefs[body].segmentNames[iSegment].c_str(),
          segmentNameLength);
      ptr += segmentNameLength;

      int parentIndex = bodyDefs.bodyDefs[body].segmentParents[iSegment];
      memcpy(ptr, &parentIndex, 4);
      ptr += 4;
    }

    // Dofs
    int nDofs = bodyDefs.bodyDefs[body].dofNames.size();
    memcpy(ptr, &nDofs, 4);
    ptr += 4;

    for (int iDof = 0; iDof < nDofs; iDof++)
    {
      int dofNameLength = bodyDefs.bodyDefs[body].dofNames[iDof].size() + 1;
      memcpy(
          ptr, bodyDefs.bodyDefs[body].dofNames[iDof].c_str(), dofNameLength);
      ptr += dofNameLength;
    }
  }

  // Write out the analog data
  int nChannels = bodyDefs.analogChannelNames.size();
  memcpy(ptr, &nChannels, 4);
  ptr += 4;

  for (int iChannel = 0; iChannel < nChannels; iChannel++)
  {
    int channelNameLength = bodyDefs.analogChannelNames[iChannel].size() + 1;
    memcpy(
        ptr, bodyDefs.analogChannelNames[iChannel].c_str(), channelNameLength);
    ptr += channelNameLength;
  }

  int nForcePlates = bodyDefs.numForcePlates;
  memcpy(ptr, &nForcePlates, 4);
  ptr += 4;

  packetOut.nBytes = (short)(ptr - packetOut.Data.cData);

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This is used for mocking the Cortex API for local testing
std::vector<unsigned char> CortexStreaming::createFrameOfDataPacket(
    CortexFrameOfData frameOfData)
{
  sPacket packetOut;
  memset(&packetOut, 0, sizeof(sPacket));
  // Let the world know we are here.
  packetOut.iCommand = PKT2_FRAME_OF_DATA;
  packetOut.nBytes = 0;

  char* ptr = packetOut.Data.cData;

  // List the number of bodies
  memcpy(ptr, &frameOfData.cortexFrameNumber, 4);
  ptr += 4;

  int nBodies = frameOfData.bodyData.size();
  memcpy(ptr, &nBodies, 4);
  ptr += 4;

  for (int iBody = 0; iBody < nBodies; iBody++)
  {
    // Name of the object

    int nameLength = frameOfData.bodyData[iBody].name.size() + 1;
    memcpy(ptr, frameOfData.bodyData[iBody].name.c_str(), nameLength);
    ptr += nameLength;

    // The Markers

    int nMarkers = frameOfData.bodyData[iBody].markers.size();
    memcpy(ptr, &nMarkers, 4);
    ptr += 4;

    for (int iMarker = 0; iMarker < nMarkers; iMarker++)
    {
      Eigen::Vector3s marker = frameOfData.bodyData[iBody].markers[iMarker];
      if (marker.hasNaN())
      {
        marker(0) = XEMPTY;
        marker(1) = XEMPTY;
        marker(2) = XEMPTY;
      }
      for (int i = 0; i < 3; i++)
      {
        float value = marker(i);
        memcpy(ptr, &value, 4);
        ptr += 4;
      }
    }

    // The Segments

    int nSegments = 0;
    memcpy(ptr, &nSegments, 4);
    ptr += 4;

    // The Dofs

    int nDofs = frameOfData.bodyData[iBody].dofs.size();
    memcpy(ptr, &nDofs, 4);
    ptr += 4;

    memcpy(ptr, frameOfData.bodyData[iBody].dofs.data(), nDofs * 4);
    for (int i = 0; i < nDofs; i++)
    {
      float value = frameOfData.bodyData[iBody].dofs(i);
      memcpy(ptr, &value, 4);
      ptr += 4;
    }
  }

  // Unnamed markers
  int nMarkers = frameOfData.unidentifiedMarkers.size();
  memcpy(ptr, &nMarkers, 4);
  ptr += 4;

  for (int iMarker = 0; iMarker < nMarkers; iMarker++)
  {
    Eigen::Vector3s marker = frameOfData.unidentifiedMarkers[iMarker];
    if (marker.hasNaN())
    {
      marker(0) = XEMPTY;
      marker(1) = XEMPTY;
      marker(2) = XEMPTY;
    }
    for (int i = 0; i < 3; i++)
    {
      float value = marker(i);
      memcpy(ptr, &value, 4);
      ptr += 4;
    }
  }

  // Handle the analog data

  int nChannels = frameOfData.analogData.analogSamples.size();
  memcpy(ptr, &nChannels, 4);
  ptr += 4;

  memcpy(ptr, &frameOfData.analogData.numAnalogSamplesPerFrame, 4);
  ptr += 4;

  for (int iChannel = 0; iChannel < nChannels; iChannel++)
  {
    for (int iSample = 0;
         iSample < frameOfData.analogData.numAnalogSamplesPerFrame;
         iSample++)
    {
      short sample = frameOfData.analogData.analogSamples[iChannel](iSample);
      memcpy(ptr, &sample, 2);
      ptr += 2;
    }
  }

  int nForcePlates = frameOfData.analogData.plateCopTorqueForce.size();
  memcpy(ptr, &nForcePlates, 4);
  ptr += 4;

  int nForceSamples = frameOfData.analogData.numForcePlateSamplesPerFrame;
  memcpy(ptr, &nForceSamples, 4);
  ptr += 4;

  // TODO: uncertain if data is packed in sample-major or plate-major order
  for (int iForceSample = 0; iForceSample < nForceSamples; iForceSample++)
  {
    for (int iForcePlate = 0; iForcePlate < nForcePlates; iForcePlate++)
    {
      if (frameOfData.analogData.plateCopTorqueForce[iForcePlate].rows()
          < iForceSample)
      {
        std::cout
            << "Warning: force plate " << iForcePlate << " only has dimension "
            << frameOfData.analogData.plateCopTorqueForce[iForcePlate].rows()
            << "x"
            << frameOfData.analogData.plateCopTorqueForce[iForcePlate].cols()
            << ", but we're trying to read sample (= row) " << iForceSample
            << std::endl;
        throw std::runtime_error("Invalid force plate data");
      }
      if (frameOfData.analogData.plateCopTorqueForce[iForcePlate].cols() != 9)
      {
        std::cout
            << "Warning: force plate " << iForcePlate << " only has dimension "
            << frameOfData.analogData.plateCopTorqueForce[iForcePlate].rows()
            << "x"
            << frameOfData.analogData.plateCopTorqueForce[iForcePlate].cols()
            << ", but we're expecting 9 columns" << std::endl;
        throw std::runtime_error("Invalid force plate data");
      }
      // //!<  X,Y,Z, fX,fY,fZ, mZ
      Eigen::Vector9s rawPlateData
          = frameOfData.analogData.plateCopTorqueForce[iForcePlate].row(
              iForceSample);
      Eigen::Vector3s cop = rawPlateData.head<3>();
      Eigen::Vector3s moment = rawPlateData.segment<3>(3);
      Eigen::Vector3s force = rawPlateData.tail<3>();
      for (int i = 0; i < 3; i++)
      {
        float value = cop(i);
        memcpy(ptr, &value, 4);
        ptr += 4;
      }
      for (int i = 0; i < 3; i++)
      {
        float value = force(i);
        memcpy(ptr, &value, 4);
        ptr += 4;
      }
      float value = moment(2);
      memcpy(ptr, &value, 4);
      ptr += 4;
    }
  }

  memcpy(ptr, &frameOfData.cortexTag, 4);
  ptr += 4;

  // Floating point delay value.
  memcpy(ptr, &frameOfData.cameraToHostDelaySeconds, 4);
  ptr += 4;

  packetOut.nBytes = (short)(ptr - packetOut.Data.cData);

  std::vector<unsigned char> result;
  result.resize(packetOut.nBytes + 4);
  memcpy(result.data(), &packetOut, packetOut.nBytes + 4);
  return result;
}

//==============================================================================
/// This is used for the mock server to set the current body defs that it
/// will respond to requests with.
CortexBodyDefs CortexStreaming::getCurrentBodyDefs()
{
  return mBodyDefs;
}

//==============================================================================
/// This is used for the mock server to set the current frame definition that it
/// will respond to requests with.
CortexFrameOfData CortexStreaming::getCurrentFrameOfData()
{
  return mFrameOfData;
}

//==============================================================================
void CortexStreaming::parseCortexPacket(
    sPacket* packet, sockaddr_in fromAddress, bool isMulticast)
{
  std::string name;
  unsigned char fromAddressBytes[4];

  if (isMulticast)
  {
    // Broadcasts only listen for BodyDefs and Frames
    switch (packet->iCommand)
    {
      case PKT2_BODYDEFS:
        mBodyDefs = parseBodyDefs(packet->Data.cData, packet->nBytes);
        break;
      case PKT2_FRAME_OF_DATA:
        parseAndHandleFrameOfData(packet->Data.cData, packet->nBytes);
        break;
      default:
        break;
    }
  }
  else
  {
    // Direct UDP packets listen for everything
    unsigned long hostAddressLong = 0;
    memcpy(&hostAddressLong, mHostMachineAddress, 4);
    switch (packet->iCommand)
    {
      case PKT2_HELLO_WORLD:
        name = packet->Data.Me.szName;
        std::cout << "Cortex HELLO_WORLD: " << name << ", Version "
                  << packet->Data.Me.Version[1] << "."
                  << packet->Data.Me.Version[1] << "."
                  << packet->Data.Me.Version[2] << std::endl;
        break;

      case PKT2_HERE_I_AM:
        if (hostAddressLong != 0 && hostAddressLong != 0xFFFFFFFF
            && hostAddressLong != fromAddress.sin_addr.s_addr)
        {
          std::cout << "Ignoring HERE_I_AM message from another machine."
                    << std::endl;
          std::cout << "We have our host machine address set to "
                    << (int)mHostMachineAddress[0] << "."
                    << (int)mHostMachineAddress[1] << "."
                    << (int)mHostMachineAddress[2] << "."
                    << (int)mHostMachineAddress[3] << std::endl;
          memcpy(fromAddressBytes, &fromAddress.sin_addr.s_addr, 4);
          std::cout << "Message came from " << (int)fromAddressBytes[0] << "."
                    << (int)fromAddressBytes[1] << "."
                    << (int)fromAddressBytes[2] << "."
                    << (int)fromAddressBytes[3] << std::endl;
          break;
        }
        std::cout << "HERE_I_AM message" << std::endl;

        mHostProgramName = packet->Data.Me.szName;
        memcpy(mHostProgramVersion, packet->Data.Me.Version, 4);
        memcpy(mHostMachineAddress, &fromAddress.sin_addr.s_addr, 4);

        if (!mFoundHost)
        {
          mFoundHost = 1;
          std::cout << "AutoConnected to: " << mHostProgramName << ", Version "
                    << (int)mHostProgramVersion[1] << "."
                    << (int)mHostProgramVersion[2] << "."
                    << (int)mHostProgramVersion[3] << " at "
                    << (int)mHostMachineAddress[0] << "."
                    << (int)mHostMachineAddress[1] << "."
                    << (int)mHostMachineAddress[2] << "."
                    << (int)mHostMachineAddress[3] << " (" << mHostMachineName
                    << ")" << std::endl;
        }
        break;

      case PKT2_BODYDEFS:
        mBodyDefs = parseBodyDefs(packet->Data.cData, packet->nBytes);
        // sem_post(&EH_CommandConfirmed);
        break;

      case PKT2_FRAME_OF_DATA:
        parseAndHandleFrameOfData(packet->Data.cData, packet->nBytes);
        // CB_DataHandler(&Polled_FrameOfData);
        // sem_post(&EH_CommandConfirmed);
        break;

      case PKT2_GENERAL_REPLY:
        // sem_post(&EH_CommandConfirmed);
        break;

      case PKT2_UNRECOGNIZED_REQUEST:
        // sem_post(&EH_CommandConfirmed);
        break;

      case PKT2_UNRECOGNIZED_COMMAND:
        // sem_post(&EH_CommandConfirmed);
        break;

      case PKT2_COMMENT:
        std::cout << "COMMENT: " << packet->Data.String << std::endl;
        break;

      default:
        std::cout
            << "parseCortexPacket(), unexpected value, PacketIn.iCommand== "
            << packet->iCommand << std::endl;
        break;
    }
  }
}

//==============================================================================
/// This sends a UDP packet to the host machine, whatever is at
/// mHostMachineAddress.
void CortexStreaming::mockServerSendResponsePacket(
    std::vector<unsigned char> packet, sockaddr_in fromAddress)
{
  // Send the message
  if (sendto(
          mCortexListenerSocketFd,
          packet.data(),
          packet.size(),
          0,
          (const struct sockaddr*)&fromAddress,
          sizeof(fromAddress))
      != packet.size())
  {
    std::cerr << "sendto reply failed" << std::endl;
  }
}

//==============================================================================
/// This sends a UDP packet out on the multicast address, to tell everyone
/// about the current frame
void CortexStreaming::mockServerSendFrameMulticast()
{
  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));

  // Filling server information
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(mCortexMulticastPort);
  servaddr.sin_addr.s_addr = inet_addr("225.1.1.1");

  auto frameOfData = createFrameOfDataPacket(mFrameOfData);

  // Send the message
  if (sendto(
          mMulticastListenerSocketFd,
          frameOfData.data(),
          frameOfData.size(),
          0,
          reinterpret_cast<struct sockaddr*>(&servaddr),
          sizeof(sockaddr_in))
      != frameOfData.size())
  {
    std::cerr << "sendto multicast failed" << std::endl;
  }
}

//==============================================================================
/// This runs on any incoming UDP packets, to decide how to parse them. This
/// runs inside the mock server, which is used for local testing.
void CortexStreaming::mockServerParseCortexPacket(
    sPacket* packet, sockaddr_in fromAddress)
{
  std::string name;

  unsigned long hostAddressLong = 0;
  memcpy(&hostAddressLong, mHostMachineAddress, 4);
  switch (packet->iCommand)
  {
    case PKT2_HELLO_WORLD:
      name = packet->Data.Me.szName;
      std::cout << "Cortex HELLO_WORLD: " << name << ", Version "
                << packet->Data.Me.Version[1] << "."
                << packet->Data.Me.Version[1] << "."
                << packet->Data.Me.Version[2] << std::endl;
      std::cout << "Sending HERE_I_AM" << std::endl;
      mockServerSendResponsePacket(createHereIAmPacket(), fromAddress);
      break;

    case PKT2_HERE_I_AM:
      std::cout << "Ignoring HERE_I_AM message on mock server." << std::endl;
      break;

    case PKT2_REQUEST_BODYDEFS:
      std::cout << "Sending BODYDEFS" << std::endl;
      mockServerSendResponsePacket(
          createBodyDefsPacket(mBodyDefs), fromAddress);
      break;

    case PKT2_REQUEST_FRAME:
      std::cout << "Sending FRAME_OF_DATA" << std::endl;
      mockServerSendResponsePacket(
          createFrameOfDataPacket(mFrameOfData), fromAddress);
      break;

    case PKT2_BODYDEFS:
      std::cout << "Ignoring BODYDEFS message on mock server." << std::endl;
      break;

    case PKT2_FRAME_OF_DATA:
      std::cout << "Ignoring FRAME_OF_DATA message on mock server."
                << std::endl;
      break;

    case PKT2_GENERAL_REPLY:
      // sem_post(&EH_CommandConfirmed);
      break;

    case PKT2_UNRECOGNIZED_REQUEST:
      // sem_post(&EH_CommandConfirmed);
      break;

    case PKT2_UNRECOGNIZED_COMMAND:
      // sem_post(&EH_CommandConfirmed);
      break;

    case PKT2_COMMENT:
      std::cout << "COMMENT: " << packet->Data.String << std::endl;
      break;

    default:
      std::cout << "mockServerParseCortexPacket(), unexpected value, "
                   "PacketIn.iCommand== "
                << packet->iCommand << std::endl;
      break;
  }
}

//==============================================================================
CortexBodyDefs CortexStreaming::parseBodyDefs(char* data, int nBytes)
{
  CortexBodyDefs result;

  char* ptr = data;

  // Read the number of bodies, which is the first 4 bytes of the packet
  int nBodies;
  memcpy(&nBodies, ptr, 4);
  ptr += 4;
  nBytes -= 4;

  for (int i = 0; i < nBodies; i++)
  {
    if (nBytes <= 0)
    {
      break;
    }
    std::pair<CortexBodyDef, int> pair = parseBodyDef(ptr, nBytes);
    result.bodyDefs.push_back(pair.first);

    int consumedBytes = nBytes - pair.second;
    ptr += consumedBytes;
    nBytes -= consumedBytes;
  }

  int consumedBytes = parseAnalogDefs(ptr, nBytes, result);
  nBytes -= consumedBytes;
  // if (nBytes != 0)
  // {
  //   std::cout << "ERROR in parseBodyDefs(), nBytes remaining != 0" <<
  //   std::endl;
  // }

  return result;
}

//==============================================================================
std::pair<CortexBodyDef, int> CortexStreaming::parseBodyDef(
    char* ptr, int nBytes)
{
  CortexBodyDef result;

  // The main name
  int nameLength = (int)strlen(ptr) + 1;
  if (nameLength > nBytes)
  {
    std::cout << "ERROR in parseBodyDef(), nameLength > nBytes" << std::endl;
    return std::make_pair(result, nBytes);
  }
  result.name = std::string(ptr);
  ptr += nameLength;
  nBytes -= nameLength;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  // Markers
  int nMarkers;
  memcpy(&nMarkers, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  for (int iMarker = 0; iMarker < nMarkers; iMarker++)
  {
    int markerNameLength = (int)strlen(ptr) + 1;
    if (markerNameLength > nBytes)
    {
      std::cout << "ERROR in parseBodyDef(), markerNameLength > nBytes"
                << std::endl;
      return std::make_pair(result, nBytes);
    }
    result.markerNames.emplace_back(ptr);
    ptr += markerNameLength;
    nBytes -= markerNameLength;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
  }

  // Segments
  int nSegments;
  memcpy(&nSegments, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }
  for (int iSegment = 0; iSegment < nSegments; iSegment++)
  {
    int segmentNameLength = (int)strlen(ptr) + 1;
    if (segmentNameLength > nBytes)
    {
      std::cout << "ERROR in parseBodyDef(), segmentNameLength > nBytes"
                << std::endl;
      return std::make_pair(result, nBytes);
    }
    result.segmentNames.emplace_back(ptr);
    ptr += segmentNameLength;
    nBytes -= segmentNameLength;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }

    int parentIndex;
    memcpy(&parentIndex, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
    result.segmentParents.push_back(parentIndex);
  }

  // Dofs
  int nDofs;
  memcpy(&nDofs, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  for (int iDof = 0; iDof < nDofs; iDof++)
  {
    int dofNameLength = strlen(ptr) + 1;
    if (dofNameLength > nBytes)
    {
      std::cout << "ERROR in parseBodyDef(), dofNameLength > nBytes"
                << std::endl;
      return std::make_pair(result, nBytes);
    }
    result.dofNames.emplace_back(ptr);
    ptr += dofNameLength;
    nBytes -= dofNameLength;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
  }

  return std::make_pair(result, nBytes);
}

//==============================================================================
int CortexStreaming::parseAnalogDefs(
    char* ptr, int nBytes, CortexBodyDefs& bodyDefs)
{
  int nChannels;
  memcpy(&nChannels, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return nBytes;
  }

  for (int iChannel = 0; iChannel < nChannels; iChannel++)
  {
    int channelNameLength = strlen(ptr) + 1;
    if (channelNameLength > nBytes)
    {
      std::cout << "ERROR in parseAnalogDefs(), channelNameLength > nBytes"
                << std::endl;
      return nBytes;
    }
    bodyDefs.analogChannelNames.emplace_back(ptr);
    ptr += channelNameLength;
    nBytes -= channelNameLength;
    if (nBytes <= 0)
    {
      return nBytes;
    }
  }

  int nForcePlates;
  memcpy(&nForcePlates, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return nBytes;
  }
  bodyDefs.numForcePlates = nForcePlates;

  return nBytes;
}

//==============================================================================
void CortexStreaming::parseAndHandleFrameOfData(char* data, int nBytes)
{
  auto result = parseFrameOfData(data, nBytes);

  std::vector<std::string> markerNames;
  std::vector<Eigen::Vector3s> markers;
  for (auto& body : result.bodyData)
  {
    for (int i = 0; i < body.markers.size(); i++)
    {
      if (!body.markers[i].hasNaN())
      {
        markerNames.push_back(body.markerNames[i]);
        Eigen::Vector3s markerTransformed = Eigen::Vector3s(
            body.markers[i](0) * 0.001,
            body.markers[i](2) * 0.001,
            body.markers[i](1) * 0.001);
        markers.push_back(markerTransformed);
      }
    }
  }
  for (int i = 0; i < result.unidentifiedMarkers.size(); i++)
  {
    if (!result.unidentifiedMarkers[i].hasNaN())
    {
      markerNames.push_back("UNIDENTIFIED_" + std::to_string(i));
      Eigen::Vector3s markerTransformed = Eigen::Vector3s(
          result.unidentifiedMarkers[i](0) * 0.001,
          result.unidentifiedMarkers[i](2) * 0.001,
          result.unidentifiedMarkers[i](1) * 0.001);
      markers.push_back(markerTransformed);
    }
  }

  if (mFrameHandler != nullptr)
  {
    mFrameHandler(markerNames, markers, result.analogData.plateCopTorqueForce);
  }
}

//==============================================================================
CortexFrameOfData CortexStreaming::parseFrameOfData(char* data, int nBytes)
{
  CortexFrameOfData result;
  if (nBytes <= 0)
  {
    return result;
  }

  char* ptr = data;
  memcpy(&result.cortexFrameNumber, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return result;
  }

  int nBodies = 0;
  memcpy(&nBodies, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return result;
  }

  if (nBodies < 0 || nBodies > MAX_N_BODIES)
  {
    std::cout << "nBodies parameter is out of range" << std::endl;
    return result;
  }

  for (int iBody = 0; iBody < nBodies; iBody++)
  {
    auto pair = parseBodyData(ptr, nBytes, iBody);
    result.bodyData.push_back(pair.first);
    int consumedBytes = nBytes - pair.second;
    ptr += consumedBytes;
    nBytes = pair.second;
  }

  // Unnamed markers
  int nMarkers;
  memcpy(&nMarkers, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return result;
  }

  for (int iMarker = 0; iMarker < nMarkers; iMarker++)
  {
    float x, y, z;
    memcpy(&x, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return result;
    }
    memcpy(&y, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return result;
    }
    memcpy(&z, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return result;
    }

    Eigen::Vector3s marker(x, y, z);
    if (marker(0) == XEMPTY)
    {
      marker(0) = std::nan("");
      marker(1) = std::nan("");
      marker(2) = std::nan("");
    }
    result.unidentifiedMarkers.push_back(marker);
  }

  auto analogPair = parseAnalogData(ptr, nBytes);
  result.analogData = analogPair.first;
  int consumedBytes = nBytes - analogPair.second;
  ptr += consumedBytes;
  nBytes = analogPair.second;
  if (nBytes <= 0)
  {
    return result;
  }

  // Grab the tag, even though we don't use it
  memcpy(&result.cortexTag, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return result;
  }

  // Floating point delay value.
  memcpy(&result.cameraToHostDelaySeconds, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return result;
  }

  return result;

  /*

  // These items are optional

  while (nTotalBytes > nBytes + 4)
  {
    int Tag;
    memcpy(&Tag, ptr, 4);
    ptr += 4;
    nBytes += 4;

    int nTheseBytes;
    memcpy(&nTheseBytes, ptr, 4);
    ptr += 4;
    nBytes += 4;

    switch (Tag)
    {
      case TAG_ENCODER_ANGLES:
        Unpack_EncoderAngles(ptr, nTheseBytes, &Frame->AnalogData);
        break;

      case TAG_RECORDING_STATUS:
        Unpack_RecordingStatus(ptr, nTheseBytes, &Frame->RecordingStatus);
        break;

      case TAG_ZOOM_FOCUS_ENCODER_DATA:
        Unpack_ZoomFocusEncoderData(ptr, nTheseBytes, Frame);
        break;
    }

    ptr += nTheseBytes;
    nBytes += nTheseBytes;
  }

  return result;
  */
}

//==============================================================================
std::pair<CortexBodyData, int> CortexStreaming::parseBodyData(
    char* ptr, int nBytes, int iBody)
{
  CortexBodyData result;

  // Name of the object

  int nameLength = strlen(ptr) + 1;
  result.name = std::string(ptr);
  ptr += nameLength;
  nBytes -= nameLength;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  // The Markers

  int nMarkers;
  memcpy(&nMarkers, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  for (int iMarker = 0; iMarker < nMarkers; iMarker++)
  {
    float x, y, z;
    memcpy(&x, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
    memcpy(&y, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
    memcpy(&z, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }

    Eigen::Vector3s marker(x, y, z);
    if (marker(0) == XEMPTY)
    {
      marker(0) = std::nan("");
      marker(1) = std::nan("");
      marker(2) = std::nan("");
    }

    std::string markerName = "MKR_" + std::to_string(iMarker);
    if (iBody < mBodyDefs.bodyDefs.size())
    {
      if (mBodyDefs.bodyDefs[iBody].markerNames.size() > iMarker)
      {
        markerName = mBodyDefs.bodyDefs[iBody].markerNames[iMarker];
      }
    }
    result.markerNames.push_back(markerName);
    result.markers.push_back(marker);
  }

  // The Segments

  int nSegments = 0;
  memcpy(&nSegments, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  int segmentBytes = nSegments * sizeof(tSegmentData);

  // Ignore the segments

  nBytes -= segmentBytes;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  // The Dofs

  int nDofs = 0;
  memcpy(&nDofs, ptr, 4);
  ptr += 4;
  nBytes -= 4;

  result.dofs = Eigen::VectorXs(nDofs);
  for (int iDof = 0; iDof < nDofs; iDof++)
  {
    float dof;
    memcpy(&dof, ptr, 4);
    ptr += 4;
    nBytes -= 4;
    if (nBytes <= 0)
    {
      return std::make_pair(result, nBytes);
    }
    result.dofs(iDof) = dof;
  }

  return std::make_pair(result, nBytes);
}

//==============================================================================
std::pair<CortexAnalogData, int> CortexStreaming::parseAnalogData(
    char* ptr, int nBytes)
{
  CortexAnalogData result;
  (void)ptr;

  int nChannels = 0;
  memcpy(&nChannels, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  int nSamples = 0;
  memcpy(&nSamples, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }
  result.numAnalogSamplesPerFrame = nSamples;

  for (int iChannel = 0; iChannel < nChannels; iChannel++)
  {
    Eigen::VectorXi channel(nSamples);
    for (int iSample = 0; iSample < nSamples; iSample++)
    {
      short sample;
      memcpy(&sample, ptr, 2);
      ptr += 2;
      nBytes -= 2;
      if (nBytes <= 0)
      {
        return std::make_pair(result, nBytes);
      }
      channel(iSample) = sample;
    }
    result.analogSamples.push_back(channel);
  }

  int nForcePlates;
  memcpy(&nForcePlates, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }

  int nForceSamples;
  memcpy(&nForceSamples, ptr, 4);
  ptr += 4;
  nBytes -= 4;
  if (nBytes <= 0)
  {
    return std::make_pair(result, nBytes);
  }
  result.numForcePlateSamplesPerFrame = nForceSamples;

  for (int iForcePlate = 0; iForcePlate < nForcePlates; iForcePlate++)
  {
    result.plateCopTorqueForce.push_back(
        Eigen::MatrixXs::Zero(nForceSamples, 9));
  }
  // TODO: uncertain if data is packed in sample-major or plate-major order
  for (int iForceSample = 0; iForceSample < nForceSamples; iForceSample++)
  {
    for (int iForcePlate = 0; iForcePlate < nForcePlates; iForcePlate++)
    {
      // //!<  X,Y,Z, fX,fY,fZ, mZ
      Eigen::Vector7s rawPlateData = Eigen::Vector7s::Zero();
      for (int i = 0; i < 7; i++)
      {
        float force;
        memcpy(&force, ptr, 4);
        ptr += 4;
        nBytes -= 4;
        if (nBytes <= 0)
        {
          return std::make_pair(result, nBytes);
        }
        rawPlateData(i) = force;
      }
      Eigen::Vector3s cop = rawPlateData.head<3>();
      Eigen::Vector3s force = rawPlateData.segment<3>(3);
      Eigen::Vector3s moment = Eigen::Vector3s::UnitZ() * rawPlateData(6);
      result.plateCopTorqueForce[iForcePlate].block<1, 3>(iForceSample, 0)
          = cop;
      result.plateCopTorqueForce[iForcePlate].block<1, 3>(iForceSample, 3)
          = moment;
      result.plateCopTorqueForce[iForcePlate].block<1, 3>(iForceSample, 6)
          = force;
    }
  }

  return std::make_pair(result, nBytes);
}