#ifndef MATH_GAUSSIAN_H_
#define MATH_GAUSSIAN_H_

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class MultivariateGaussian
{
public:
  MultivariateGaussian(
      std::vector<std::string> variables,
      Eigen::VectorXs mu,
      Eigen::MatrixXs cov);

  void debugToStdout();

  const Eigen::VectorXs& getMu();

  const Eigen::MatrixXs& getCov();

  s_t computeProbablity(Eigen::VectorXs x);

  s_t computeLogProbability(Eigen::VectorXs x);

  Eigen::VectorXs computeLogProbabilityGrad(Eigen::VectorXs x);

  Eigen::VectorXs finiteDifferenceLogProbabilityGrad(Eigen::VectorXs x);

  std::string getVariableNameAtIndex(int i);

  MultivariateGaussian condition(
      const std::map<std::string, s_t>& observedValues);

  std::vector<int> getObservedIndices(
      const std::map<std::string, s_t>& observedValues);

  std::vector<int> getUnobservedIndices(
      const std::map<std::string, s_t>& observedValues);

  Eigen::VectorXs getMuSubset(const std::vector<int>& indices);

  Eigen::MatrixXs getCovSubset(
      const std::vector<int>& rowIndices, const std::vector<int>& colIndices);

  static MultivariateGaussian loadFromCSV(
      const std::string& file, std::vector<std::string> columns);

protected:
  std::vector<std::string> mVars;
  Eigen::VectorXs mMu;
  Eigen::MatrixXs mCov;
  Eigen::MatrixXs mCovInv;
  s_t mNormalizationConstant;
  s_t mLogNormalizationConstant;
};

} // namespace math
} // namespace dart

#endif