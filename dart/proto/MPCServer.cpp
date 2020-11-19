#include "dart/proto/MPCServer.hpp"

namespace dart {
namespace proto {

::grpc::Status MPCServer::Request(
    ::grpc::ServerContext* context,
    const ::dart::proto::MPCRequest* request,
    ::dart::proto::MPCResponse* response)
{
  return ::grpc::Status::OK;
}

} // namespace proto
} // namespace dart