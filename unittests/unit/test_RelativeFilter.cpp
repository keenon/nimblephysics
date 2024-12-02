#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/math/RelativeFilter.hpp"

using namespace dart;
using namespace math;

// Helper function for comparing matrices
bool matrixClose(
    const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2, double tol = 1e-6)
{
  return (mat1 - mat2).norm() < tol;
}

#define ALL_TESTS

#ifdef ALL_TESTS

TEST(RelativeFilterTest, InitialState)
{
  Eigen::Vector3d acc_std = Eigen::Vector3d::Constant(0.5);
  Eigen::Vector3d gyro_std = Eigen::Vector3d::Constant(0.05);
  Eigen::Vector3d mag_std = Eigen::Vector3d::Constant(0.3);
  RelativeFilter filter(acc_std, gyro_std, mag_std);

  EXPECT_TRUE(matrixClose(filter.get_R_pc(), Eigen::Matrix3d::Identity()))
      << "Initial q_pc is not identity.";
  EXPECT_TRUE(matrixClose(filter.P, Eigen::MatrixXd::Identity(6, 6)))
      << "Initial P is not identity.";
}

TEST(RelativeFilterTest, MeasurementJacobianAnalyticalIdentity)
{
  Eigen::Matrix3d R_wp = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_wc = Eigen::Matrix3d::Identity();
  Eigen::Vector3d acc_jc_p(0.0, 0.0, 1.0);
  Eigen::Vector3d acc_jc_c(0.0, 0.0, 1.0);
  Eigen::Vector3d mag_jc_p(1.0, 0.0, 0.0);
  Eigen::Vector3d mag_jc_c(1.0, 0.0, 0.0);

  Eigen::Vector3d acc_std = Eigen::Vector3d::Constant(0.5);
  Eigen::Vector3d gyro_std = Eigen::Vector3d::Constant(0.05);
  Eigen::Vector3d mag_std = Eigen::Vector3d::Constant(0.3);
  RelativeFilter filter(acc_std, gyro_std, mag_std);

  Eigen::MatrixXd H = filter.get_H_jacobian(
      R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);

  for (int col = 0; col < 6; ++col)
  {
    double epsilon = 1e-6;
    Eigen::VectorXd perturbation = Eigen::VectorXd::Zero(6);
    perturbation[col] = epsilon;

    Eigen::VectorXd pos_measurements = filter.get_h(
        R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, perturbation);
    Eigen::VectorXd neg_measurements = filter.get_h(
        R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, -perturbation);
    Eigen::VectorXd delta_measurements
        = (pos_measurements - neg_measurements) / (2.0 * epsilon);

    for (int row = 0; row < 6; ++row)
    {
      EXPECT_NEAR(H(row, col), delta_measurements[row], 1e-3)
          << "Jacobian element (" << row << ", " << col << ") is incorrect.";
    }
  }
}

TEST(RelativeFilterTest, MeasurementJacobianAnalyticalNonIdentity)
{
  Eigen::Matrix3d R_wp = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_wc
      = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()).toRotationMatrix()
        * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix()
        * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Matrix3d R_cp = R_wc.transpose() * R_wp;

  Eigen::Vector3d acc_jc_p(0.0, 0.0, 1.0);
  Eigen::Vector3d acc_jc_c = R_cp * acc_jc_p;
  Eigen::Vector3d mag_jc_p(1.0, 0.0, 0.0);
  Eigen::Vector3d mag_jc_c = R_cp * mag_jc_p;

  Eigen::Vector3d acc_std = Eigen::Vector3d::Constant(0.5);
  Eigen::Vector3d gyro_std = Eigen::Vector3d::Constant(0.05);
  Eigen::Vector3d mag_std = Eigen::Vector3d::Constant(0.3);
  RelativeFilter filter(acc_std, gyro_std, mag_std);

  Eigen::VectorXd baseline_error
      = filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);
  EXPECT_TRUE(matrixClose(baseline_error, Eigen::VectorXd::Zero(6)))
      << "Baseline error is not zero.";

  Eigen::MatrixXd H = filter.get_H_jacobian(
      R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);

  for (int col = 0; col < 6; ++col)
  {
    double epsilon = 1e-6;
    Eigen::VectorXd perturbation = Eigen::VectorXd::Zero(6);
    perturbation[col] = epsilon;

    Eigen::VectorXd pos_measurements = filter.get_h(
        R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, perturbation);
    Eigen::VectorXd neg_measurements = filter.get_h(
        R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, -perturbation);
    Eigen::VectorXd delta_measurements
        = (pos_measurements - neg_measurements) / (2.0 * epsilon);

    for (int row = 0; row < 6; ++row)
    {
      EXPECT_NEAR(H(row, col), delta_measurements[row], 1e-3)
          << "Jacobian element (" << row << ", " << col << ") is incorrect.";
    }
  }
}

TEST(RelativeFilterTest, TimeUpdateIdentity)
{
  Eigen::Vector3d acc_std = Eigen::Vector3d::Constant(0.5);
  Eigen::Vector3d gyro_std = Eigen::Vector3d::Constant(0.05);
  Eigen::Vector3d mag_std = Eigen::Vector3d::Constant(0.3);
  RelativeFilter filter(acc_std, gyro_std, mag_std);

  Eigen::Vector3d gyro(0.01, 0.0, 0.0);
  double dt = 1.0;

  Eigen::Quaternion<double> q_lin_wp(1.0, 0.0, 0.0, 0.0);
  Eigen::Quaternion<double> q_lin_wc(1.0, 0.0, 0.0, 0.0);

  filter.set_qs(q_lin_wp, q_lin_wc);
  for (int i = 0; i < 10000; ++i)
  {
    auto pair = filter.get_time_update(gyro, gyro, dt);
    auto updated_q_wp = pair.first;
    auto updated_q_wc = pair.second;
    filter.set_qs(updated_q_wp, updated_q_wc);

    EXPECT_TRUE(matrixClose(filter.get_R_pc(), Eigen::Matrix3d::Identity()))
        << "R_pc is not identity after time update.";
  }
}

#endif