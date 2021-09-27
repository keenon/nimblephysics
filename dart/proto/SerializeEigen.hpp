#ifndef DART_PROTO_EIGEN
#define DART_PROTO_EIGEN

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"
#include "dart/proto/Eigen.pb.h"

namespace dart {
namespace proto {

void serializeVector(proto::VectorXs& proto, const Eigen::VectorXs& vec);
Eigen::VectorXs deserializeVector(const proto::VectorXs& proto);

void serializeMatrix(proto::MatrixXs& proto, const Eigen::MatrixXs& mat);
Eigen::MatrixXs deserializeMatrix(const proto::MatrixXs& proto);

} // namespace proto
} // namespace dart

#endif