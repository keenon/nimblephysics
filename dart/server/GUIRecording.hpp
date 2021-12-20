#ifndef DART_GUI_RECORDING
#define DART_GUI_RECORDING

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/server/GUIStateMachine.hpp"
#include "dart/server/WebsocketServer.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
}

namespace trajectory {
class TrajectoryRollout;
}

namespace server {

class GUIRecording : public GUIStateMachine
{
public:
  GUIRecording();

  ~GUIRecording();

  void saveFrame();

  int getNumFrames();

  std::string getFramesJson(int startFrame = 0);

  std::string getFrameJson(int frame);

  void writeFramesJson(const std::string& path, int startFrame = 0);

  void writeFrameJson(const std::string& path, int frame);

protected:
  std::vector<std::string> mFrames;
};

} // namespace server
} // namespace dart

#endif