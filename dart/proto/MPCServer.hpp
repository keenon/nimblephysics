#ifndef DART_MPC_SERVER
#define DART_MPC_SERVER

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "dart/proto/MPC.grpc.pb.h"

namespace dart {
namespace proto {

class MPCServer final : public MPCService::Service
{
  ::grpc::Status Request(
      ::grpc::ServerContext* context,
      const ::dart::proto::MPCRequest* request,
      ::dart::proto::MPCResponse* response) override;
};

} // namespace proto
} // namespace dart

#endif