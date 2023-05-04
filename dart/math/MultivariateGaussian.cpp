#include "dart/math/MultivariateGaussian.hpp"

#include <algorithm>

#include "dart/math/FiniteDifference.hpp"
#include "dart/utils/CSVParser.hpp"

namespace dart {
namespace math {

MultivariateGaussian::MultivariateGaussian(
    std::vector<std::string> variables, Eigen::VectorXs mu, Eigen::MatrixXs cov)
  : mVars(variables), mMu(mu), mCov(cov)
{
  s_t twoPi = 2 * M_PI;
  s_t logTwoPi = log(twoPi);
  s_t logTwoPiExp = logTwoPi * (((s_t)mVars.size()) / 2);
  mCovInv = Eigen::LLT<Eigen::MatrixXs>(mCov);

  // Compute the log-determinant
  auto& U = mCovInv.matrixL();
  s_t logDet = 0.0;
  for (unsigned i = 0; i < mCov.rows(); ++i)
    logDet += log(U(i, i));
  logDet *= 2;

  s_t logSqrtDet = logDet * 0.5;

  mLogNormalizationConstant = -1 * (logSqrtDet + logTwoPiExp);

  s_t det = mCov.determinant();
  s_t twoPiExp = pow(twoPi, mVars.size());
  mNormalizationConstant = 1. / (sqrt(twoPiExp * det));
}

void MultivariateGaussian::debugToStdout()
{
  std::cout << "mu: " << std::endl << mMu << std::endl;
  std::cout << "cov: " << std::endl << mCov << std::endl;
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

s_t MultivariateGaussian::getLogNormalizationConstant()
{
  return mLogNormalizationConstant;
}

s_t MultivariateGaussian::getMean(std::string variable)
{
  for (int i = 0; i < mVars.size(); i++)
  {
    if (mVars[i] == variable)
      return mMu(i);
  }
  return 0.0;
}

s_t MultivariateGaussian::getVariance(std::string variable)
{
  for (int i = 0; i < mVars.size(); i++)
  {
    if (mVars[i] == variable)
      return sqrt(mCov(i, i));
  }
  return 0.0;
}

Eigen::VectorXs MultivariateGaussian::convertFromMap(
    const std::map<std::string, s_t>& values)
{
  Eigen::VectorXs x = Eigen::VectorXs::Zero(mVars.size());
  for (int i = 0; i < mVars.size(); i++)
  {
    if (values.count(mVars[i]))
    {
      x(i) = values.at(mVars[i]);
    }
  }
  return x;
}

std::map<std::string, s_t> MultivariateGaussian::convertToMap(
    const Eigen::VectorXs& values)
{
  std::map<std::string, s_t> result;
  for (int i = 0; i < mVars.size(); i++)
  {
    result[mVars[i]] = values(i);
  }
  return result;
}

s_t MultivariateGaussian::computePDF(Eigen::VectorXs x)
{
  return exp(computeLogPDF(x));
}

s_t MultivariateGaussian::computeLogPDF(Eigen::VectorXs x, bool normalized)
{
  Eigen::VectorXs diff = x - mMu;
  return (normalized ? mLogNormalizationConstant : 0)
         + (-0.5 * diff.transpose() * mCovInv.solve(diff));
}

Eigen::VectorXs MultivariateGaussian::computeLogPDFGrad(Eigen::VectorXs x)
{
  Eigen::VectorXs diff = x - mMu;
  return -mCovInv.solve(diff);
}

Eigen::VectorXs MultivariateGaussian::finiteDifferenceLogPDFGrad(
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
        out = computeLogPDF(tweaked);
        return true;
      },
      result,
      eps,
      true);

  return result;
}

std::vector<std::string> MultivariateGaussian::getVariableNames()
{
  return mVars;
}

std::string MultivariateGaussian::getVariableNameAtIndex(int i)
{
  return mVars[i];
}

std::shared_ptr<MultivariateGaussian> MultivariateGaussian::condition(
    const std::map<std::string, s_t>& observedValues)
{
  // 0. Check whether strings are tied to the list
  for (auto pair : observedValues)
  {
    if (std::find(mVars.begin(), mVars.end(), pair.first) == mVars.end())
    {
      std::cout << "WARNING: Attempting to condition on variable name \""
                << pair.first
                << "\", but that variable is not in this distribution. Are you "
                   "sure you spelled it right?"
                << std::endl;
    }
  }

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

  std::cout << "Coniditioning Multivariate Gaussion on:" << std::endl;
  for (int i = 0; i < observedIndices.size(); i++)
  {
    std::cout << getVariableNameAtIndex(observedIndices[i])
              << " (mu=" << mu_2(i) << "): " << observedVector(i) << std::endl;
  }

  Eigen::MatrixXs cov_22_Inv = cov_22.inverse();

  std::vector<std::string> subNames;
  for (int i = 0; i < unobservedIndices.size(); i++)
  {
    subNames.push_back(mVars[unobservedIndices[i]]);
  }
  Eigen::VectorXs subMu = mu_1 + cov_12 * cov_22_Inv * (observedVector - mu_2);
  Eigen::MatrixXs subCov = cov_11 - cov_12 * cov_22_Inv * cov_21;
  assert(!subMu.hasNaN());
  assert(!subCov.hasNaN());

  return std::make_shared<MultivariateGaussian>(subNames, subMu, subCov);
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

std::shared_ptr<MultivariateGaussian> MultivariateGaussian::loadFromCSV(
    const std::string& file, std::vector<std::string> columns, s_t units)
{
  // 1. Open the CSV file
  std::vector<std::map<std::string, std::string>> rows
      = dart::utils::CSVParser::parseFile(file);

  std::vector<std::string> processedColumns;
  if (rows.size() > 0)
  {
    std::map<std::string, std::string> firstRow = rows[0];
    for (std::string& colName : columns)
    {
      if (firstRow.count(colName) == 0)
      {
        std::cout
            << "WARNING! Trying to load a MultivariateGaussian from a "
               "CSV, but the requested column \""
            << colName
            << "\" does not appear in the CSV! This column will be ignored."
            << std::endl;
      }
      else
      {
        processedColumns.push_back(colName);
      }
    }
  }
  else
  {
    std::cout
        << "WARNING! Trying to load a MultivariateGaussian from a "
           "CSV, there are no rows in the CSV! Returning a size 0 distribution."
        << std::endl;
  }

  int n = processedColumns.size();

  // 2. Read data from the CSV file
  Eigen::MatrixXs data = Eigen::MatrixXs::Zero(n, rows.size());
  for (int i = 0; i < rows.size(); i++)
  {
    std::map<std::string, std::string>& row = rows[i];
    for (int j = 0; j < processedColumns.size(); j++)
    {
      data(j, i) = atof(row[processedColumns[j]].c_str()) * units;
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

  return std::make_shared<MultivariateGaussian>(processedColumns, mu, cov);
}

} // namespace math
} // namespace dart