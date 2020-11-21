#ifndef DART_PROTO_EIGEN
#define DART_PROTO_EIGEN

#include <Eigen/Dense>

#include "dart/proto/Eigen.pb.h"

namespace dart {
namespace proto {

void serializeVector(proto::VectorXd& proto, const Eigen::VectorXd& vec);
Eigen::VectorXd deserializeVector(const proto::VectorXd& proto);

void serializeMatrix(proto::MatrixXd& proto, const Eigen::MatrixXd& mat);
Eigen::MatrixXd deserializeMatrix(const proto::MatrixXd& proto);

} // namespace proto
} // namespace dart

#endif