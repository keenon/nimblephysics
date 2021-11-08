#include "dart/server/GUIRecording.hpp"

#include <fstream>
#include <iostream>

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

GUIRecording::GUIRecording()
{
}

GUIRecording::~GUIRecording()
{
}

void GUIRecording::saveFrame()
{
  mFrames.push_back(flushJson());
}

int GUIRecording::getNumFrames()
{
  return mFrames.size();
}

std::string GUIRecording::getFramesJson(int startFrame)
{
  std::stringstream stream;
  stream << "[";
  if (startFrame < 0)
    startFrame = 0;
  for (int i = startFrame; i < mFrames.size(); i++)
  {
    if (i > 0)
    {
      stream << ",";
    }
    stream << mFrames[i];
  }
  stream << "]";
  return stream.str();
}

std::string GUIRecording::getFrameJson(int frame)
{
  if (frame < 0 || frame >= mFrames.size())
    return "";
  return mFrames[frame];
}

void GUIRecording::writeFramesJson(const std::string& path, int startFrame)
{
  std::cout << "Saving GUI Recording to file \"" << path << "\"..."
            << std::endl;

  std::ofstream jsonFile;
  jsonFile.open(path);
  jsonFile << "[";
  if (startFrame < 0)
    startFrame = 0;
  for (int i = startFrame; i < mFrames.size(); i++)
  {
    if (i % 50 == 0)
    {
      std::cout << "> Writing frame " << i << "/" << mFrames.size()
                << std::endl;
    }
    if (i > 0)
    {
      jsonFile << ",";
    }
    jsonFile << mFrames[i];
  }
  jsonFile << "]";
  jsonFile.close();

  std::cout << "Finished saving GUI Recording to file \"" << path << "\"!"
            << std::endl;
}

void GUIRecording::writeFrameJson(const std::string& path, int frame)
{
  std::ofstream jsonFile;
  jsonFile.open(path);
  if (frame < 0 || frame >= mFrames.size())
    jsonFile << "";
  else
    jsonFile << mFrames[frame];
  jsonFile.close();
}

} // namespace server
} // namespace dart