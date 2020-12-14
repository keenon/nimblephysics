#ifndef DART_TICKER
#define DART_TICKER

#include <functional>
#include <thread>
#include <vector>

namespace dart {
namespace realtime {

class Ticker
{
public:
  Ticker(double secondsPerTick);
  ~Ticker();
  void registerTickListener(std::function<void(long)> listener);
  /// Remove all tick listeners, without deleting the Ticker
  void clear();

  void start();
  void stop();

protected:
  void mainLoop();
  bool mRunning;

  double mSecondsPerTick;
  std::thread* mMainThread;
  std::vector<std::function<void(long)>> mListeners;
};

} // namespace realtime
} // namespace dart

#endif