#include "dart/biomechanics/IKErrorReport.hpp"

#include <fstream>
#include <ostream>
#include <string>

namespace dart {
namespace biomechanics {

IKErrorReport::IKErrorReport(
    std::shared_ptr<dynamics::Skeleton> skel,
    dynamics::MarkerMap markers,
    Eigen::MatrixXs poses,
    std::vector<std::map<std::string, Eigen::Vector3s>> observations,
    std::shared_ptr<Anthropometrics> anthropometrics)
  : averageRootMeanSquaredError(0.0),
    averageSumSquaredError(0.0),
    averageMaxError(0.0)
{
  anthroPDF = 0.0;
  if (anthropometrics)
  {
    anthroPDF = anthropometrics->getLogPDF(skel);
  }

  // Collect the names of all the observed markers on any timestep into a single
  // vector
  for (int i = 0; i < observations.size(); i++)
  {
    for (auto& pair : observations[i])
    {
      if (std::find(markerNames.begin(), markerNames.end(), pair.first)
          == markerNames.end())
      {
        markerNames.push_back(pair.first);
      }
    }
  }

  Eigen::VectorXs originalPos = skel->getPositions();

  for (std::string& name : markerNames)
  {
    rmseMarkerErrors[name] = 0;
    numMarkerObservations[name] = 0;
  }

  for (int i = 0; i < observations.size(); i++)
  {
    skel->setPositions(poses.col(i));
    std::map<std::string, Eigen::Vector3s> worldMarkers
        = skel->getMarkerMapWorldPositions(markers);

    s_t thisTotalSquaredError = 0.0;
    s_t thisMaxError = 0.0;
    std::string worstMarker = "[NONE]";
    Eigen::Vector3s worstMarkerError = Eigen::Vector3s::Zero();
    Eigen::Vector3s worstMarkerReal = Eigen::Vector3s::Zero();
    Eigen::Vector3s worstMarkerPredicted = Eigen::Vector3s::Zero();

    std::map<std::string, s_t> markerErrorTableEntry;
    for (std::string& name : markerNames)
    {
      markerErrorTableEntry[name] = 0.;
    }

    for (auto pair : observations[i])
    {
      std::string markerName = pair.first;
      if (worldMarkers.count(markerName) > 0)
      {
        Eigen::Vector3s diff
            = observations[i][markerName] - worldMarkers[markerName];
        s_t squaredError = diff.squaredNorm();
        markerErrorTableEntry[markerName] = sqrt(squaredError);
        rmseMarkerErrors[markerName] += squaredError;
        numMarkerObservations[markerName]++;
        thisTotalSquaredError += squaredError;
        thisMaxError = std::max(thisMaxError, diff.norm());
        if (diff.squaredNorm() > worstMarkerError.squaredNorm())
        {
          worstMarker = markerName;
          worstMarkerError = diff;
          worstMarkerReal = observations[i][markerName];
          worstMarkerPredicted = worldMarkers[markerName];
        }
      }
    }
    worstMarkers.push_back(worstMarker);
    worstMarkerErrors.push_back(worstMarkerError);
    worstMarkerReals.push_back(worstMarkerReal);
    worstMarkerPredicteds.push_back(worstMarkerPredicted);
    markerErrorTimesteps.push_back(markerErrorTableEntry);

    s_t thisRootMeanSquaredError
        = sqrt(thisTotalSquaredError / observations[i].size());
    this->rootMeanSquaredError.push_back(thisRootMeanSquaredError);
    this->maxError.push_back(thisMaxError);
    this->sumSquaredError.push_back(thisTotalSquaredError);

    if (isfinite(thisRootMeanSquaredError) && isfinite(thisTotalSquaredError)
        && isfinite(thisMaxError))
    {
      this->averageRootMeanSquaredError += thisRootMeanSquaredError;
      this->averageSumSquaredError += thisTotalSquaredError;
      this->averageMaxError += thisMaxError;
    }
  }
  this->averageRootMeanSquaredError /= observations.size();
  this->averageSumSquaredError /= observations.size();
  this->averageMaxError /= observations.size();

  for (std::string& name : markerNames)
  {
    if (numMarkerObservations[name] > 0)
    {
      rmseMarkerErrors[name] /= numMarkerObservations[name];
      rmseMarkerErrors[name] = sqrt(rmseMarkerErrors[name]);
    }
  }

  skel->setPositions(originalPos);
}

void IKErrorReport::printReport(int limitTimesteps)
{
  std::cout << "IK Error Report:" << std::endl;
  std::cout << "sum_squared (" << this->averageSumSquaredError
            << " avg) -- RMSE (" << this->averageRootMeanSquaredError
            << " avg) -- Max (" << this->averageMaxError << " avg) -- Anthro ("
            << this->anthroPDF << "):" << std::endl;

  int printTimesteps = this->rootMeanSquaredError.size();
  if (limitTimesteps > 0 && limitTimesteps < printTimesteps)
  {
    printTimesteps = limitTimesteps;
  }
  Eigen::MatrixXs together = Eigen::MatrixXs::Zero(printTimesteps, 3);
  for (int i = 0; i < printTimesteps; i++)
  {
    together(i, 0) = this->sumSquaredError[i];
    together(i, 1) = this->rootMeanSquaredError[i];
    together(i, 2) = this->maxError[i];
  }
  std::cout << together << std::endl;
  for (int i = 0; i < printTimesteps; i++)
  {
    std::cout << "Worst Marker at " << i << ": " << worstMarkers[i] << " -> ";
    std::cout << "real[" << worstMarkerReals[i](0) << ", "
              << worstMarkerReals[i](1) << ", " << worstMarkerReals[i](2)
              << "]";
    std::cout << " - predicted[" << worstMarkerPredicteds[i](0) << ", "
              << worstMarkerPredicteds[i](1) << ", "
              << worstMarkerPredicteds[i](2) << "]";
    std::cout << " = error[" << worstMarkerErrors[i](0) << ", "
              << worstMarkerErrors[i](1) << ", " << worstMarkerErrors[i](2)
              << "]";
    std::cout << std::endl;
  }
}

void IKErrorReport::saveCSVMarkerErrorReport(const std::string& path)
{
  std::ofstream errorCSV;
  errorCSV.open(path);

  errorCSV << "Timestep";
  for (std::string& markerName : markerNames)
  {
    errorCSV << "," << markerName;
  }
  errorCSV << std::endl;

  errorCSV << "All Timesteps RMSE";
  for (std::string& markerName : markerNames)
  {
    errorCSV << "," << rmseMarkerErrors.at(markerName);
  }
  errorCSV << std::endl;

  for (int i = 0; i < markerErrorTimesteps.size(); i++)
  {
    errorCSV << i;
    for (std::string& markerName : markerNames)
    {
      errorCSV << "," << markerErrorTimesteps.at(i).at(markerName);
    }
    errorCSV << std::endl;
  }

  errorCSV.close();
}

std::vector<std::pair<std::string, s_t>> IKErrorReport::getSortedMarkerRMSE()
{
  std::vector<std::pair<std::string, s_t>> sortedRMSE;
  for (auto& pair : rmseMarkerErrors)
  {
    sortedRMSE.push_back(pair);
  }
  std::sort(
      sortedRMSE.begin(),
      sortedRMSE.end(),
      [](const std::pair<std::string, s_t>& s1,
         const std::pair<std::string, s_t>& s2) {
        return s1.second > s2.second;
      });
  return sortedRMSE;
}

} // namespace biomechanics
} // namespace dart