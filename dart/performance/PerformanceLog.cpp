#include "dart/performance/PerformanceLog.hpp"

#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef HAVE_PERF_UTILS
#include <PerfUtils/Cycles.h>
#endif

#include <assert.h>

namespace dart {
namespace performance {

std::unordered_map<std::string, int> PerformanceLog::globalPerfStringIndex;
std::deque<PerformanceLog*> PerformanceLog::globalPerfLogsList;
std::unordered_map<int, std::string>
    PerformanceLog::globalPerfStringReverseIndex;
std::mutex PerformanceLog::globalPerfLogListMutex;

//==============================================================================
void PerformanceLog::initialize()
{
  globalPerfStringIndex = std::unordered_map<std::string, int>(30);
  globalPerfLogsList = std::deque<PerformanceLog*>();
  globalPerfStringReverseIndex = std::unordered_map<int, std::string>(30);
}

//==============================================================================
int PerformanceLog::mapStringToIndex(const char* c_str)
{
  std::string str(c_str);
  // Key is not present
  auto value = PerformanceLog::globalPerfStringIndex.find(str);
  if (value == PerformanceLog::globalPerfStringIndex.end())
  {
    int newKey = PerformanceLog::globalPerfStringIndex.size();
    PerformanceLog::globalPerfStringIndex[str] = newKey;
    return newKey;
  }
  else
  {
    return value->second;
  }
}

//==============================================================================
inline uint64_t getClock()
{
#ifdef HAVE_PERF_UTILS
  return PerfUtils::Cycles::rdtsc();
#else
  std::cout << "Trying to use PerformanceLog::getClock() on an unsupported architecture!" << std::endl;
  return 0;
#endif
}

//==============================================================================
/// Default constructor
PerformanceLog::PerformanceLog(int nameIndex, int parentId)
  : mNameIndex(nameIndex),
    mStartClock(getClock()),
    mEndClock(0),
    mId(rand()),
    mParentId(parentId)
{
}

//==============================================================================
PerformanceLog* PerformanceLog::startRoot(char const* name)
{
  PerformanceLog* newRoot = new PerformanceLog(mapStringToIndex(name), -1);
  // Serialize our access to the log list
  const std::lock_guard<std::mutex> lock(globalPerfLogListMutex);
  globalPerfLogsList.push_back(newRoot);
  return newRoot;
}

//==============================================================================
/// This looks through all the PerformanceLogs in the system and builds a
/// report
std::unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
PerformanceLog::finalize()
{
  // First we need to set up the reverse index so we can rapidly look up strings
  globalPerfStringReverseIndex.clear();
  for (auto pair : globalPerfStringIndex)
  {
    globalPerfStringReverseIndex[pair.second] = pair.first;
  }

  // Next we need to look through for all the root names:
  std::unordered_set<int> rootNameIds;
  for (PerformanceLog* log : globalPerfLogsList)
  {
    if (log->mParentId == -1)
      rootNameIds.insert(log->mNameIndex);
  }

  std::unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
      rootLogs;
  for (int nameId : rootNameIds)
  {
    std::vector<int> nameIdStack;
    nameIdStack.push_back(nameId);
    // Next we look through for all the
    // This will recursively look through all the PerformanceLog's and construct
    // what we need.
    rootLogs[globalPerfStringReverseIndex[nameId]]
        = FinalizedPerformanceLog::recursivelyConstruct(nameIdStack);
  }

  return rootLogs;
}

//==============================================================================
/// This checks if a given PerformanceLog object matches a stack of nameIds
bool PerformanceLog::matches(std::vector<int> nameIdStack)
{
  if (nameIdStack[nameIdStack.size() - 1] != mNameIndex)
    return false;
  if (nameIdStack.size() == 1)
    return true;
  assert(mParentId != -1);

  std::vector<int> subStack = nameIdStack;
  subStack.pop_back();
  // Find parent and recurse
  for (PerformanceLog* log : globalPerfLogsList)
  {
    if (log->mId == mParentId)
    {
      return log->matches(subStack);
    }
  }

  return false;
}

//==============================================================================
/// This starts a sub-run within this PerformanceLog, giving it a specific
/// name. After the fact we can use these names to coalesce PerformanceLog
/// objects into something sensible.
PerformanceLog* PerformanceLog::startRun(char const* name)
{
  PerformanceLog* log = new PerformanceLog(mapStringToIndex(name), mId);
  // Serialize our access to the log list
  const std::lock_guard<std::mutex> lock(globalPerfLogListMutex);
  globalPerfLogsList.push_back(log);
  return log;
}

//==============================================================================
/// This terminates the run that we're logging with this PerformanceLog
/// object.
void PerformanceLog::end()
{
  mEndClock = getClock();
}

//==============================================================================
/// This will look through the global static lists and recursively construct
/// all the FinalizedPerformanceLogs to unify everything.
std::shared_ptr<FinalizedPerformanceLog>
FinalizedPerformanceLog::recursivelyConstruct(std::vector<int> nameIdStack)
{
  std::shared_ptr<FinalizedPerformanceLog> log
      = std::make_shared<FinalizedPerformanceLog>(
          PerformanceLog::globalPerfStringReverseIndex
              [nameIdStack[nameIdStack.size() - 1]]);

  std::unordered_set<int> selfIds;

  // Scan through once looking for instances of us

  for (int i = 0; i < PerformanceLog::globalPerfLogsList.size(); i++)
  {
    PerformanceLog* rawLog = PerformanceLog::globalPerfLogsList[i];
    if (rawLog->matches(nameIdStack))
    {
      uint64_t diff = rawLog->mEndClock - rawLog->mStartClock;
      log->registerRun(diff);
      selfIds.insert(rawLog->mId);
    }
  }

  // Scan through again looking for child names

  std::unordered_set<int> childNameIds;
  for (int i = 0; i < PerformanceLog::globalPerfLogsList.size(); i++)
  {
    PerformanceLog* rawLog = PerformanceLog::globalPerfLogsList[i];
    // Found a child!
    if (selfIds.find(rawLog->mParentId) != selfIds.end())
    {
      childNameIds.insert(rawLog->mNameIndex);
    }
  }

  // Create all the children

  for (int childNameId : childNameIds)
  {
    std::vector<int> childNameIdStack = nameIdStack;
    childNameIdStack.push_back(childNameId);
    log->setChild(
        PerformanceLog::globalPerfStringReverseIndex[childNameId],
        recursivelyConstruct(childNameIdStack));
  }

  // And we're done!

  return log;
}

//==============================================================================
FinalizedPerformanceLog::FinalizedPerformanceLog(const std::string& name)
  : mName(name)
{
}

//==============================================================================
std::shared_ptr<FinalizedPerformanceLog> FinalizedPerformanceLog::getChild(
    const std::string& name)
{
  return mChildren[name];
}

//==============================================================================
void FinalizedPerformanceLog::setChild(
    const std::string& name, std::shared_ptr<FinalizedPerformanceLog> child)
{
  mChildren[name] = child;
}

//==============================================================================
void FinalizedPerformanceLog::registerRun(uint64_t duration)
{
  mRuns.push_back(duration);
}

//==============================================================================
int FinalizedPerformanceLog::getNumRuns()
{
  return mRuns.size();
}

//==============================================================================
s_t FinalizedPerformanceLog::getMeanRuntime()
{
  s_t sum = 0.0;
  for (uint64_t run : mRuns)
    sum += run;
  return sum / mRuns.size();
}

//==============================================================================
uint64_t FinalizedPerformanceLog::getTotalRuntime()
{
  uint64_t sum = 0.0;
  for (uint64_t run : mRuns)
    sum += run;
  return sum;
}

//==============================================================================
/// This will print the results in human readable format, which we can pipe to
/// a file or to std::out
std::string FinalizedPerformanceLog::prettyPrint()
{
  std::stringstream stream;
  stream << std::setprecision(3);
  recursivePrettyPrint(0, getTotalRuntime(), static_cast<s_t>(1.0), stream);
  return stream.str();
}

//==============================================================================
/// This formats our results to JSON, which we can send to the browser to be
/// rendered.
std::string FinalizedPerformanceLog::toJson()
{
  std::stringstream stream;
  recursivePrettyPrint(0, getTotalRuntime(), static_cast<s_t>(1.0), stream);
  return stream.str();
}

//==============================================================================
/// This pretty prints to a stream
void FinalizedPerformanceLog::recursivePrettyPrint(
    int tabs,
    long parentTotalCycles,
    s_t parentPercentage,
    std::stringstream& stream)
{
  for (int i = 0; i < tabs; i++)
    stream << "  ";
  long totalCycles = getTotalRuntime();
  s_t percentage
      = (static_cast<s_t>(totalCycles) / parentTotalCycles) * parentPercentage;

  stream << (percentage * 100) << "%: " << mName << " (" << getNumRuns()
         << " runs at mean " << getMeanRuntime() << " cycles = " << totalCycles
         << " total)\n";

  for (auto pair : mChildren)
  {
    pair.second->recursivePrettyPrint(
        tabs + 1, totalCycles, percentage, stream);
  }
}

} // namespace performance
} // namespace dart