#include "dart/proto/SerializeEigen.hpp"

namespace dart {
namespace proto {

void serializeVector(proto::VectorXd& proto, const Eigen::VectorXd& vec)
{
  proto.set_size(vec.size());
  for (int i = 0; i < vec.size(); i++)
  {
    proto.add_values(vec(i));
  }
}

Eigen::VectorXd deserializeVector(const proto::VectorXd& proto)
{
  Eigen::VectorXd recovered = Eigen::VectorXd(proto.size());
  for (int i = 0; i < proto.size(); i++)
  {
    recovered(i) = proto.values(i);
  }
  return recovered;
}

void serializeMatrix(proto::MatrixXd& proto, const Eigen::MatrixXd& mat)
{
  proto.set_rows(mat.rows());
  proto.set_cols(mat.cols());
  for (int col = 0; col < mat.cols(); col++)
  {
    for (int row = 0; row < mat.rows(); row++)
    {
      proto.add_values(mat(row, col));
    }
  }
}

Eigen::MatrixXd deserializeMatrix(const proto::MatrixXd& proto)
{
  Eigen::MatrixXd recovered = Eigen::MatrixXd(proto.rows(), proto.cols());
  int cursor = 0;
  for (int col = 0; col < proto.cols(); col++)
  {
    for (int row = 0; row < proto.rows(); row++)
    {
      recovered(row, col) = proto.values(cursor);
      cursor++;
    }
  }
  return recovered;
}

} // namespace proto
} // namespace dart