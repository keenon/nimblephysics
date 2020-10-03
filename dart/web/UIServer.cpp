#include "dart/web/UIServer.hpp"

#include <iostream>

#include "pistache/endpoint.h"
#include "pistache/http_header.h"
#include "pistache/http_headers.h"

using namespace dart;
using namespace Pistache;

namespace dart {
namespace web {

class HelloHandler : public Http::Handler
{
public:
  HTTP_PROTOTYPE(HelloHandler)

  void onRequest(const Http::Request& request, Http::ResponseWriter response)
  {
    response.headers()
        .add<Http::Header::ContentType>(MIME(Application, Json))
        .add<Http::Header::AccessControlAllowOrigin>("*");
    auto stream = response.stream(Http::Code::Ok);
    stream << "{ \"hello\": \"world\" }";
    stream.ends();
  }
};

//==============================================================================
UIServer::UIServer()
{
}

//==============================================================================
void UIServer::serve()
{
  std::cout << "Starting a server on port 9080 (lsof -i :9080 to find others)"
            << std::endl;
  Pistache::Address addr(Pistache::Ipv4::any(), Pistache::Port(9080));
  auto opts = Pistache::Http::Endpoint::options().threads(1);

  Http::Endpoint server(addr);
  server.init(opts);
  server.setHandler(Http::make_handler<HelloHandler>());
  server.serve();
}

} // namespace web
} // namespace dart