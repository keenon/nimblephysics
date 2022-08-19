#include "dart/server/GUIRecording.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>

#include "stdio.h"
// #include <google/protobuf/io/coded_stream.h>
// #include <google/protobuf/io/zero_copy_stream_impl.h>

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
  for (int i = startFrame; i < mFrames.size(); i++)
  {
    stream << mFrames[i];
  }
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

  FILE* file = fopen(path.c_str(), "wb");
  if (file == nullptr)
  {
    std::cout << "ERROR: Could not open \"" << path << "\" for writing"
              << std::endl;
    return;
  }

  if (startFrame < 0)
    startFrame = 0;
  for (int i = startFrame; i < mFrames.size(); i++)
  {
    if (i % 50 == 0)
    {
      std::cout << "> Writing frame " << i << "/" << mFrames.size()
                << std::endl;
    }
    int size = mFrames[i].size();
    assert(sizeof(int) == 4);
    fwrite(&size, 4, 1, file);
    fwrite(mFrames[i].c_str(), mFrames[i].size(), 1, file);
  }
  fclose(file);

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