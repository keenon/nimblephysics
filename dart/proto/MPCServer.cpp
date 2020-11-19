#include "dart/proto/MPCServer.hpp"

namespace dart {
namespace proto {

void MPCServer::serve(int ports)
{
  std::string server_address("0.0.0.0:" + std::to_string(ports));

  grpc::EnableDefaultHealthCheckService(true);
  // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(this);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

::grpc::Status MPCServer::Request(
    ::grpc::ServerContext* context,
    const ::dart::proto::MPCRequest* request,
    ::dart::proto::MPCResponse* response)
{
  switch (request->message_case())
  {
    case MPCRequest::MessageCase::kStart:
      long clock = request->start().clientclock();
      std::cout << "Got clock start: " << clock << std::endl;
      break;
  }
  return ::grpc::Status::OK;
}

} // namespace proto
} // namespace dart