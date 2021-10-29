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

  s_t getLogNormalizationConstant();

  s_t getMean(std::string variable);

  Eigen::VectorXs convertFromMap(const std::map<std::string, s_t>& values);

  std::map<std::string, s_t> convertToMap(const Eigen::VectorXs& values);

  s_t computePDF(Eigen::VectorXs x);

  /// NOTE: there's a problem with normalizing. The normalization constants
  /// sometimes push the PDF to be way bigger than zero.
  s_t computeLogPDF(Eigen::VectorXs x, bool normalized = true);

  Eigen::VectorXs computeLogPDFGrad(Eigen::VectorXs x);

  Eigen::VectorXs finiteDifferenceLogPDFGrad(Eigen::VectorXs x);

  std::vector<std::string> getVariableNames();

  std::string getVariableNameAtIndex(int i);

  std::shared_ptr<MultivariateGaussian> condition(
      const std::map<std::string, s_t>& observedValues);

  std::vector<int> getObservedIndices(
      const std::map<std::string, s_t>& observedValues);

  std::vector<int> getUnobservedIndices(
      const std::map<std::string, s_t>& observedValues);

  Eigen::VectorXs getMuSubset(const std::vector<int>& indices);

  Eigen::MatrixXs getCovSubset(
      const std::vector<int>& rowIndices, const std::vector<int>& colIndices);

  static std::shared_ptr<MultivariateGaussian> loadFromCSV(
      const std::string& file,
      std::vector<std::string> columns,
      s_t units = 1.0);

protected:
  std::vector<std::string> mVars;
  Eigen::VectorXs mMu;
  Eigen::MatrixXs mCov;
  Eigen::LLT<Eigen::MatrixXs> mCovInv;
  s_t mNormalizationConstant;
  s_t mLogNormalizationConstant;
};

} // namespace math
} // namespace dart

#endif