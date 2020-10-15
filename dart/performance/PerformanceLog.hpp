#ifndef DART_PERFORMANCE_LOG_HPP_
#define DART_PERFORMANCE_LOG_HPP_

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Comment this out to disable performance logging in other parts of the code
#define LOG_PERFORMANCE

namespace dart {
namespace performance {

class FinalizedPerformanceLog
{
public:
  /// This will look through the global static lists and recursively construct
  /// all the FinalizedPerformanceLogs to unify everything.
  static std::shared_ptr<FinalizedPerformanceLog> recursivelyConstruct(
      std::vector<int> nameIdStack);

  FinalizedPerformanceLog(const std::string& name);

  std::shared_ptr<FinalizedPerformanceLog> getChild(const std::string& name);

  void setChild(
      const std::string& name, std::shared_ptr<FinalizedPerformanceLog> child);

  void registerRun(uint64_t duration);

  int getNumRuns();

  double getMeanRuntime();

  /// This will print the results in human readable format, which we can pipe to
  /// a file or to std::out
  std::string prettyPrint();

  /// This formats our results to JSON, which we can send to the browser to be
  /// rendered.
  std::string toJson();

protected:
  std::string mName;
  std::unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
      mChildren;
  std::vector<uint64_t> mRuns;

  /// This pretty prints to a stream
  void recursivePrettyPrint(
      int tabs,
      long parentTotalCycles,
      double parentPercentage,
      std::stringstream& stream);

  /// This JSON prints to a stream
  void recursiveJson(std::stringstream& stream);
};

class PerformanceLog
{
  friend class FinalizedPerformanceLog;

public:
  /// Default constructor
  PerformanceLog(int nameIndex, int parentId);

  /// Disable the copy constructor
  // PerformanceLog(const PerformanceLog&) = delete;

  /// This returns a new root PerformanceLog instance, which can spawn children
  static PerformanceLog* startRoot(char const* name);

  /// This looks through all the PerformanceLogs in the system and builds a
  /// report. This is not concerned about efficiency, and we attempt to offload
  /// as much slowness from elsewhere into here as possible.
  static std::
      unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
      finalize();

  /// This checks if a given PerformanceLog object matches a stack of nameIds
  bool matches(std::vector<int> nameIdStack);

  /// This starts a sub-run within this PerformanceLog, giving it a specific
  /// name. After the fact we can use these names to coalesce PerformanceLog
  /// objects into something sensible.
  PerformanceLog* startRun(char const* name);

  /// This terminates the run that we're logging with this PerformanceLog
  /// object.
  void end();

  /// This needs to be called once at the beginning of execution, and if it's
  /// called multiple times will clear previous logs.
  static void initialize();

protected:
  /// Don't store a whole copy of the name, just a numerical key
  int mNameIndex;

  /// This is the clock at the start of our existence
  uint64_t mStartClock;

  /// This is the clock when we called end()
  uint64_t mEndClock;

  /// This is the ID which we'll use to reassemble the graph after the fact
  int mId;

  /// This is the parent's ID
  int mParentId;

  static int mapStringToIndex(const char* str);

  static std::unordered_map<std::string, int> globalPerfStringIndex;
  static std::vector<PerformanceLog*> globalPerfLogsList;
  static std::unordered_map<int, std::string> globalPerfStringReverseIndex;
};

} // namespace performance
} // namespace dart

#endif