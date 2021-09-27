#include "dart/realtime/Millis.hpp"

#include <chrono>

namespace dart {
long timeSinceEpochMillis()
{
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}
}