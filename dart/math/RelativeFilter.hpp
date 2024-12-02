#ifndef RELATIVE_FILTER_HPP
#define RELATIVE_FILTER_HPP

#include <utility> // For std::pair

#include <Eigen/Dense>
#include <Eigen/Geometry> // For Eigen::Quaternion

#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

class RelativeFilter
{
public:
  // Constructor
  RelativeFilter(
      Eigen::Vector3d acc_std = Eigen::Vector3d::Constant(0.05),
      Eigen::Vector3d gyro_std = Eigen::Vector3d::Constant(0.05),
      Eigen::Vector3d mag_std = Eigen::Vector3d::Constant(0.05));

  // Getters
  Eigen::Quaternion<double> get_q_pc() const;
  Eigen::Matrix3d get_R_pc() const;

  // Update filter with new sensor data
  void update(
      const Eigen::Vector3d& gyro_p,
      const Eigen::Vector3d& gyro_c,
      const Eigen::Vector3d& acc_jc_p,
      const Eigen::Vector3d& acc_jc_c,
      const Eigen::Vector3d& mag_p,
      const Eigen::Vector3d& mag_c,
      double dt);

  // Set quaternions
  void set_qs(
      const Eigen::Quaternion<double>& q_wp,
      const Eigen::Quaternion<double>& q_wc);

  // Static utility method
  static Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v);

  // Methods for time and measurement updates
  std::pair<Eigen::Quaternion<double>, Eigen::Quaternion<double>>
  get_time_update(
      const Eigen::Vector3d& gyro_p, const Eigen::Vector3d& gyro_c, double dt);

  Eigen::Quaternion<double> get_gyro_orientation_estimate(
      const Eigen::Quaternion<double>& q,
      const Eigen::Vector3d& gyro,
      double dt) const;

  std::pair<Eigen::Quaternion<double>, Eigen::Quaternion<double>>
  get_measurement_update(
      Eigen::Quaternion<double> q_lin_wp,
      Eigen::Quaternion<double> q_lin_wc,
      Eigen::Vector3d acc_jc_p,
      Eigen::Vector3d acc_jc_c,
      Eigen::Vector3d mag_jc_p,
      Eigen::Vector3d mag_jc_c);

  // Jacobian and measurement functions
  static Eigen::MatrixXd get_H_jacobian(
      const Eigen::Matrix3d& R_wp,
      const Eigen::Matrix3d& R_wc,
      const Eigen::Vector3d& acc_jc_p,
      const Eigen::Vector3d& acc_jc_c,
      const Eigen::Vector3d& mag_jc_p,
      const Eigen::Vector3d& mag_jc_c);

  static Eigen::VectorXd get_h(
      const Eigen::Matrix3d& R_wp,
      const Eigen::Matrix3d& R_wc,
      const Eigen::Vector3d& acc_jc_p,
      const Eigen::Vector3d& acc_jc_c,
      const Eigen::Vector3d& mag_jc_p,
      const Eigen::Vector3d& mag_jc_c,
      const Eigen::Vector6d& perturbation = Eigen::Vector6d::Zero());

  Eigen::MatrixXd get_M_jacobian(
      const Eigen::Matrix3d& R_wp,
      const Eigen::Matrix3d& R_wc,
      const Eigen::Vector6d& update = Eigen::Vector6d::Zero());

  // Member variables
  Eigen::MatrixXd Q;              // Process noise covariance
  Eigen::MatrixXd R;              // Measurement noise covariance
  Eigen::MatrixXd P;              // Error covariance
  Eigen::Quaternion<double> q_wp; // Parent quaternion
  Eigen::Quaternion<double> q_wc; // Child quaternion
};

} // namespace math
} // namespace dart

#endif // RELATIVE_FILTER_HPP
