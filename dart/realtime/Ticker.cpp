#include "dart/realtime/Ticker.hpp"

#include "dart/realtime/Millis.hpp"

namespace dart {
namespace realtime {

Ticker::Ticker(s_t secondsPerTick)
  : mRunning(false), mSecondsPerTick(secondsPerTick)
{
}

Ticker::~Ticker()
{
  stop();
}

void Ticker::registerTickListener(std::function<void(long)> listener)
{
  mListeners.push_back(listener);
}

/// Remove all tick listeners, without deleting the Ticker
void Ticker::clear()
{
  mListeners.clear();
}

void Ticker::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mMainThread = new std::thread(&Ticker::mainLoop, this);
}

void Ticker::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mMainThread->join();
  delete mMainThread;
}

void Ticker::mainLoop()
{
  while (mRunning)
  {
    int interval = (int)round(mSecondsPerTick * 1000);
    auto x = std::chrono::steady_clock::now()
             + std::chrono::milliseconds(interval);
    long millis = timeSinceEpochMillis();

    for (auto listener : mListeners)
      listener(millis);

    std::this_thread::sleep_until(x);
  }
}

} // namespace realtime
} // namespace dart