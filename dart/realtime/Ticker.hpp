#ifndef DART_TICKER
#define DART_TICKER

#include <functional>
#include <thread>
#include <vector>

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace realtime {

class Ticker
{
public:
  Ticker(s_t secondsPerTick);
  ~Ticker();
  void registerTickListener(std::function<void(long)> listener);
  /// Remove all tick listeners, without deleting the Ticker
  void clear();

  void start();
  void stop();
  void toggle();
  bool isRunning();

protected:
  void mainLoop();
  bool mRunning;

  s_t mSecondsPerTick;
  std::thread* mMainThread;
  std::vector<std::function<void(long)>> mListeners;
};

} // namespace realtime
} // namespace dart

#endif