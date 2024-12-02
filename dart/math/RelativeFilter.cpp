#include "dart/math/RelativeFilter.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace dart {
namespace math {

RelativeFilter::RelativeFilter(
    Eigen::Vector3d acc_std, Eigen::Vector3d gyro_std, Eigen::Vector3d mag_std)
{
  Eigen::VectorXd gyro_diag(6);
  gyro_diag << gyro_std, gyro_std;
  Q = gyro_diag.asDiagonal();

  Eigen::VectorXd sensor_diag(12);
  sensor_diag << acc_std, acc_std, mag_std, mag_std;
  R = sensor_diag.asDiagonal();

  P = Eigen::MatrixXd::Identity(6, 6);
  q_wp = Eigen::Quaternion<double>(1.0, 0.0, 0.0, 0.0);
  q_wc = Eigen::Quaternion<double>(1.0, 0.0, 0.0, 0.0);
}

Eigen::Quaternion<double> RelativeFilter::get_q_pc() const
{
  return q_wp.conjugate() * q_wc;
}

Eigen::Matrix3d RelativeFilter::get_R_pc() const
{
  return get_q_pc().toRotationMatrix();
}

void RelativeFilter::update(
    const Eigen::Vector3d& gyro_p,
    const Eigen::Vector3d& gyro_c,
    const Eigen::Vector3d& acc_jc_p,
    const Eigen::Vector3d& acc_jc_c,
    const Eigen::Vector3d& mag_p,
    const Eigen::Vector3d& mag_c,
    double dt)
{
  auto pair = get_time_update(gyro_p, gyro_c, dt);
  auto q_lin_wp = pair.first;
  auto q_lin_wc = pair.second;
  std::tie(q_lin_wp, q_lin_wc) = get_measurement_update(
      q_lin_wp, q_lin_wc, acc_jc_p, acc_jc_c, mag_p, mag_c);

  q_wp = q_lin_wp;
  q_wc = q_lin_wc;
}

void RelativeFilter::set_qs(
    const Eigen::Quaternion<double>& new_q_wp,
    const Eigen::Quaternion<double>& new_q_wc)
{
  q_wp = new_q_wp;
  q_wc = new_q_wc;
}

Eigen::Matrix3d RelativeFilter::skew_symmetric(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d mat;
  mat << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return mat;
}

std::pair<Eigen::Quaternion<double>, Eigen::Quaternion<double>>
RelativeFilter::get_time_update(
    const Eigen::Vector3d& gyro_p, const Eigen::Vector3d& gyro_c, double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Zero(6, 6);
  F.block<3, 3>(0, 0)
      = Eigen::AngleAxis<double>(gyro_p.norm() * dt, gyro_p.normalized())
            .toRotationMatrix();
  F.block<3, 3>(3, 3)
      = Eigen::AngleAxis<double>(gyro_c.norm() * dt, gyro_c.normalized())
            .toRotationMatrix();

  Eigen::MatrixXd G = Eigen::MatrixXd::Identity(6, 6) * dt;

  auto q_lin_wp = get_gyro_orientation_estimate(q_wp, gyro_p, dt);
  auto q_lin_wc = get_gyro_orientation_estimate(q_wc, gyro_c, dt);

  P = F * P * F.transpose() + G * Q * G.transpose();

  return {q_lin_wp, q_lin_wc};
}

Eigen::Quaternion<double> RelativeFilter::get_gyro_orientation_estimate(
    const Eigen::Quaternion<double>& q,
    const Eigen::Vector3d& gyro,
    double dt) const
{
  double angle = gyro.norm() * dt;
  Eigen::Vector3d axis = gyro.normalized();
  Eigen::Quaternion<double> delta_q(Eigen::AngleAxis<double>(angle, axis));
  return q * delta_q;
}

std::pair<Eigen::Quaternion<double>, Eigen::Quaternion<double>>
RelativeFilter::get_measurement_update(
    Eigen::Quaternion<double> q_lin_wp,
    Eigen::Quaternion<double> q_lin_wc,
    Eigen::Vector3d acc_jc_p,
    Eigen::Vector3d acc_jc_c,
    Eigen::Vector3d mag_jc_p,
    Eigen::Vector3d mag_jc_c)
{
  acc_jc_p.normalize();
  acc_jc_c.normalize();

  if (mag_jc_p.norm() != 0.0 || mag_jc_c.norm() != 0.0)
  {
    mag_jc_p.normalize();
    mag_jc_c.normalize();
  }

  Eigen::Matrix3d R_wp = q_lin_wp.toRotationMatrix();
  Eigen::Matrix3d R_wc = q_lin_wc.toRotationMatrix();

  auto H = get_H_jacobian(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);
  auto e = get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);
  auto M = get_M_jacobian(R_wp, R_wc);

  Eigen::MatrixXd S = H * P * H.transpose() + M * R * M.transpose();
  Eigen::MatrixXd K = P * H.transpose() * S.inverse();

  Eigen::VectorXd n = -K * e;

  Eigen::Quaternion<double> delta_wp(
      Eigen::AngleAxis<double>(n.head<3>().norm(), n.head<3>().normalized()));
  Eigen::Quaternion<double> delta_wc(
      Eigen::AngleAxis<double>(n.tail<3>().norm(), n.tail<3>().normalized()));

  q_lin_wp = q_lin_wp * delta_wp;
  q_lin_wc = q_lin_wc * delta_wc;

  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(6, 6);
  J.block<3, 3>(0, 0) = delta_wp.toRotationMatrix();
  J.block<3, 3>(3, 3) = delta_wc.toRotationMatrix();

  P = J * (P - K * S * K.transpose()) * J.transpose();

  return {q_lin_wp, q_lin_wc};
}

Eigen::MatrixXd RelativeFilter::get_H_jacobian(
    const Eigen::Matrix3d& R_wp,
    const Eigen::Matrix3d& R_wc,
    const Eigen::Vector3d& acc_jc_p,
    const Eigen::Vector3d& acc_jc_c,
    const Eigen::Vector3d& mag_jc_p,
    const Eigen::Vector3d& mag_jc_c)
{
  Eigen::MatrixXd H(6, 6);
  H.block<3, 3>(0, 0) = R_wp * skew_symmetric(acc_jc_p).transpose();
  H.block<3, 3>(0, 3) = -R_wc * skew_symmetric(acc_jc_c).transpose();
  H.block<3, 3>(3, 0) = R_wp * skew_symmetric(mag_jc_p).transpose();
  H.block<3, 3>(3, 3) = -R_wc * skew_symmetric(mag_jc_c).transpose();
  return H;
}

Eigen::VectorXd RelativeFilter::get_h(
    const Eigen::Matrix3d& R_wp,
    const Eigen::Matrix3d& R_wc,
    const Eigen::Vector3d& acc_jc_p,
    const Eigen::Vector3d& acc_jc_c,
    const Eigen::Vector3d& mag_jc_p,
    const Eigen::Vector3d& mag_jc_c,
    const Eigen::Vector6d& perturbation)
{
  Eigen::Matrix3d R_wp_perturbed = R_wp
                                   * Eigen::AngleAxisd(
                                         perturbation.head<3>().norm(),
                                         perturbation.head<3>().normalized())
                                         .toRotationMatrix();
  Eigen::Matrix3d R_wc_perturbed = R_wc
                                   * Eigen::AngleAxisd(
                                         perturbation.tail<3>().norm(),
                                         perturbation.tail<3>().normalized())
                                         .toRotationMatrix();

  Eigen::VectorXd h(6);
  h.head<3>() = R_wp_perturbed * acc_jc_p - R_wc_perturbed * acc_jc_c;
  h.tail<3>() = R_wp_perturbed * mag_jc_p - R_wc_perturbed * mag_jc_c;

  return h;
}

Eigen::MatrixXd RelativeFilter::get_M_jacobian(
    const Eigen::Matrix3d& R_wp,
    const Eigen::Matrix3d& R_wc,
    const Eigen::Vector6d& update)
{
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 12);
  Eigen::Matrix3d I_plus_skew_wp
      = Eigen::Matrix3d::Identity() + skew_symmetric(update.head<3>());
  Eigen::Matrix3d I_plus_skew_wc
      = Eigen::Matrix3d::Identity() + skew_symmetric(update.tail<3>());

  M.block<3, 3>(0, 0) = R_wp * I_plus_skew_wp;
  M.block<3, 3>(0, 3) = -R_wc * I_plus_skew_wc;
  M.block<3, 3>(3, 6) = R_wp * I_plus_skew_wp;
  M.block<3, 3>(3, 9) = -R_wc * I_plus_skew_wc;

  return M;
}

} // namespace math
} // namespace dart
