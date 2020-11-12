#ifndef DART_NEURAL_IPOPT_OPTIMIZER_HPP_
#define DART_NEURAL_IPOPT_OPTIMIZER_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/trajectory/AbstractShot.hpp"
#include "dart/trajectory/OptimizationRecord.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

/*
 * IPOPT wants to own the trajectories it's trying to optimize, so we need a way
 * to create a buffer that's possible for IPOPT to own without freeing the
 * underlying trajectory when it's done.
 */
class IPOptOptimizer
{
public:
  IPOptOptimizer();

  std::shared_ptr<OptimizationRecord> optimize(
      AbstractShot* shot,
      std::shared_ptr<OptimizationRecord> reuseRecord = nullptr);

  void setIterationLimit(int iterationLimit);

  void setTolerance(double tolerance);

  void setLBFGSHistoryLength(int historyLen);

  void setCheckDerivatives(bool checkDerivatives);

  void setPrintFrequency(int frequency);

  void setRecordPerformanceLog(bool recordPerfLog);

  void setRecoverBest(bool recoverBest);

  void setRecordFullDebugInfo(bool recordFullDebugInfo);

  void setSuppressOutput(bool suppressOutput);

  void setSilenceOutput(bool silenceOutput);

  void setDisableLinesearch(bool disableLinesearch);

  void setRecordIterations(bool recordIterations);

protected:
  int mIterationLimit;
  double mTolerance;
  int mLBFGSHistoryLength;
  bool mCheckDerivatives;
  int mPrintFrequency;
  bool mRecordPerfLog;
  bool mRecoverBest;
  bool mRecordFullDebugInfo;
  bool mSuppressOutput;
  bool mSilenceOutput;
  bool mDisableLinesearch;
  bool mRecordIterations;
};

} // namespace trajectory
} // namespace dart

#endif