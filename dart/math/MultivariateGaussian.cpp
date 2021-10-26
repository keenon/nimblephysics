#include "dart/math/MultivariateGaussian.hpp"

#include "dart/math/FiniteDifference.hpp"
#include "dart/utils/CSVParser.hpp"

namespace dart {
namespace math {

MultivariateGaussian::MultivariateGaussian(
    std::vector<std::string> variables, Eigen::VectorXs mu, Eigen::MatrixXs cov)
  : mVars(variables), mMu(mu), mCov(cov), mCovInv(cov.inverse())
{
  s_t twoPi = 2 * M_PI;
  s_t twoPiExp = pow(twoPi, ((s_t)mVars.size()) / 2);
  s_t det = mCov.determinant();
  s_t sqrtDet = sqrt(det);
  mNormalizationConstant = 1.0 / (twoPiExp * sqrtDet);
  mLogNormalizationConstant = log(mNormalizationConstant);
}

void MultivariateGaussian::debugToStdout()
{
  std::cout << "mu: " << std::endl << mMu << std::endl;
  std::cout << "cov: " << std::endl << mCov << std::endl;
  std::cout << "cov^{-1}: " << std::endl << mCovInv << std::endl;
  std::cout << "normalization constant: " << std::endl
            << mNormalizationConstant << std::endl;
  for (int i = 0; i < mVars.size(); i++)
  {
    std::cout << mVars[i] << " ~ N(" << mMu(i) << ", " << sqrt(mCov(i, i))
              << ")" << std::endl;
  }
}

const Eigen::VectorXs& MultivariateGaussian::getMu()
{
  return mMu;
}

const Eigen::MatrixXs& MultivariateGaussian::getCov()
{
  return mCov;
}

s_t MultivariateGaussian::computeProbablity(Eigen::VectorXs x)
{
  Eigen::VectorXs diff = x - mMu;
  return mNormalizationConstant * exp(-0.5 * diff.transpose() * mCovInv * diff);
}

s_t MultivariateGaussian::computeLogProbability(Eigen::VectorXs x)
{
  Eigen::VectorXs diff = x - mMu;
  return mLogNormalizationConstant + (-0.5 * diff.transpose() * mCovInv * diff);
}

Eigen::VectorXs MultivariateGaussian::computeLogProbabilityGrad(
    Eigen::VectorXs x)
{
  Eigen::VectorXs diff = x - mMu;
  return -mCovInv * diff;
}

Eigen::VectorXs MultivariateGaussian::finiteDifferenceLogProbabilityGrad(
    Eigen::VectorXs x)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(mVars.size());

  s_t eps = 1e-3;

  math::finiteDifference<Eigen::VectorXs>(
      [&](/* in*/ s_t eps,
          /* in*/ int i,
          /*out*/ s_t& out) {
        Eigen::VectorXs tweaked = x;
        tweaked(i) += eps;
        out = computeLogProbability(tweaked);
        return true;
      },
      result,
      eps,
      true);

  return result;
}

std::string MultivariateGaussian::getVariableNameAtIndex(int i)
{
  return mVars[i];
}

MultivariateGaussian MultivariateGaussian::condition(
    const std::map<std::string, s_t>& observedValues)
{
  // 1. Get indices for observed and unobserved
  std::vector<int> observedIndices = getObservedIndices(observedValues);
  std::vector<int> unobservedIndices = getUnobservedIndices(observedValues);

  // 2. Get observations as a vector
  Eigen::VectorXs observedVector
      = Eigen::VectorXs::Zero(observedIndices.size());
  for (int i = 0; i < observedIndices.size(); i++)
  {
    observedVector(i) = observedValues.at(mVars[observedIndices[i]]);
  }

  // 3. Get all the sub-blocks of the Gaussian
  Eigen::VectorXs mu_1 = getMuSubset(unobservedIndices);
  Eigen::VectorXs mu_2 = getMuSubset(observedIndices);
  Eigen::MatrixXs cov_11 = getCovSubset(unobservedIndices, unobservedIndices);
  Eigen::MatrixXs cov_12 = getCovSubset(unobservedIndices, observedIndices);
  Eigen::MatrixXs cov_21 = getCovSubset(observedIndices, unobservedIndices);
  Eigen::MatrixXs cov_22 = getCovSubset(observedIndices, observedIndices);

  Eigen::MatrixXs cov_22_Inv = cov_22.inverse();

  std::vector<std::string> subNames;
  for (int i = 0; i < unobservedIndices.size(); i++)
  {
    subNames.push_back(mVars[unobservedIndices[i]]);
  }
  Eigen::VectorXs subMu = mu_1 + cov_12 * cov_22_Inv * (observedVector - mu_2);
  Eigen::MatrixXs subCov = cov_11 - cov_12 * cov_22_Inv * cov_21;

  return MultivariateGaussian(subNames, subMu, subCov);
}

std::vector<int> MultivariateGaussian::getObservedIndices(
    const std::map<std::string, s_t>& observedValues)
{
  std::vector<int> observedIndices;
  for (int i = 0; i < mVars.size(); i++)
  {
    if (observedValues.count(mVars[i]) > 0)
    {
      observedIndices.push_back(i);
    }
  }
  return observedIndices;
}

std::vector<int> MultivariateGaussian::getUnobservedIndices(
    const std::map<std::string, s_t>& observedValues)
{
  std::vector<int> unobservedIndices;
  for (int i = 0; i < mVars.size(); i++)
  {
    if (observedValues.count(mVars[i]) == 0)
    {
      unobservedIndices.push_back(i);
    }
  }
  return unobservedIndices;
}

Eigen::VectorXs MultivariateGaussian::getMuSubset(
    const std::vector<int>& indices)
{
  Eigen::VectorXs muSubset = Eigen::VectorXs::Zero(indices.size());
  for (int i = 0; i < indices.size(); i++)
  {
    muSubset(i) = mMu(indices[i]);
  }
  return muSubset;
}

Eigen::MatrixXs MultivariateGaussian::getCovSubset(
    const std::vector<int>& rowIndices, const std::vector<int>& colIndices)
{
  Eigen::MatrixXs covSubset
      = Eigen::MatrixXs::Zero(rowIndices.size(), colIndices.size());
  for (int i = 0; i < rowIndices.size(); i++)
  {
    for (int j = 0; j < colIndices.size(); j++)
    {
      covSubset(i, j) = mCov(rowIndices[i], colIndices[j]);
    }
  }
  return covSubset;
}

MultivariateGaussian MultivariateGaussian::loadFromCSV(
    const std::string& file, std::vector<std::string> columns)
{
  int n = columns.size();

  // 1. Open the CSV file
  std::vector<std::map<std::string, std::string>> rows
      = dart::utils::CSVParser::parseFile(file);

  // 2. Read data from the CSV file
  Eigen::MatrixXs data = Eigen::MatrixXs::Zero(n, rows.size());
  for (int i = 0; i < rows.size(); i++)
  {
    std::map<std::string, std::string>& row = rows[i];
    for (int j = 0; j < columns.size(); j++)
    {
      data(j, i) = atof(row[columns[j]].c_str());
    }
  }

  // 3. Compute the mean
  Eigen::VectorXs mu = Eigen::VectorXs::Zero(n);
  for (int i = 0; i < data.cols(); i++)
  {
    mu += data.col(i);
  }
  mu /= data.cols();

  // 4. Compute covariance
  Eigen::MatrixXs cov = Eigen::MatrixXs::Zero(n, n);
  for (int i = 0; i < data.cols(); i++)
  {
    Eigen::VectorXs diff = data.col(i) - mu;
    cov += diff * diff.transpose();
  }
  cov /= data.cols();

  return MultivariateGaussian(columns, mu, cov);
}

} // namespace math
} // namespace dart