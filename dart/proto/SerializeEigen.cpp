#include "dart/proto/SerializeEigen.hpp"

namespace dart {
namespace proto {

void serializeVector(proto::VectorXs& proto, const Eigen::VectorXs& vec)
{
  proto.set_size(vec.size());
  for (int i = 0; i < vec.size(); i++)
  {
    proto.add_values(static_cast<double>(vec(i)));
  }
}

Eigen::VectorXs deserializeVector(const proto::VectorXs& proto)
{
  Eigen::VectorXs recovered = Eigen::VectorXs::Zero(proto.size());
  for (int i = 0; i < proto.size(); i++)
  {
    recovered(i) = static_cast<s_t>(proto.values(i));
  }
  return recovered;
}

void serializeMatrix(proto::MatrixXs& proto, const Eigen::MatrixXs& mat)
{
  proto.set_rows(mat.rows());
  proto.set_cols(mat.cols());
  for (int col = 0; col < mat.cols(); col++)
  {
    for (int row = 0; row < mat.rows(); row++)
    {
      proto.add_values(static_cast<double>(mat(row, col)));
    }
  }
}

Eigen::MatrixXs deserializeMatrix(const proto::MatrixXs& proto)
{
  Eigen::MatrixXs recovered = Eigen::MatrixXs::Zero(proto.rows(), proto.cols());
  int cursor = 0;
  for (int col = 0; col < proto.cols(); col++)
  {
    for (int row = 0; row < proto.rows(); row++)
    {
      recovered(row, col) = static_cast<s_t>(proto.values(cursor));
      cursor++;
    }
  }
  return recovered;
}

} // namespace proto
} // namespace dart