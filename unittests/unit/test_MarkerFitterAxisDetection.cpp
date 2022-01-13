#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

TEST(MarkerFitterAxisDetection, SVD)
{
  srand(42);
  Eigen::MatrixXs testPoints = Eigen::MatrixXs::Zero(50, 3);
  Eigen::Vector3s axis1 = Eigen::Vector3s::Random();
  Eigen::Vector3s axis2 = Eigen::Vector3s::Random();
  for (int i = 0; i < testPoints.rows(); i++)
  {
    double r1 = ((double)rand() / (RAND_MAX));
    double r2 = ((double)rand() / (RAND_MAX));
    testPoints.row(i) = axis1 * r1 * 10 + axis2 * r2;
  }

  Eigen::JacobiSVD<Eigen::MatrixXs> svd(
      testPoints, Eigen::ComputeThinU | Eigen::ComputeThinV);
  std::cout << "Main axis is: " << axis1 << std::endl;
  std::cout << "Secondary axis is: " << axis2 << std::endl;
  std::cout << "Singular values are: " << svd.singularValues() << std::endl;
  std::cout << "Its left singular vectors are the columns of the thin U matrix:"
            << std::endl
            << svd.matrixU() << std::endl;
  std::cout
      << "Its right singular vectors are the columns of the thin V matrix:"
      << std::endl
      << svd.matrixV() << std::endl;
}