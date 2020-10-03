#ifndef DART_WEB_UI_SERVER_HPP_
#define DART_WEB_UI_SERVER_HPP_

#include <memory>

namespace dart {

namespace web {

class UIServer
{
public:
  UIServer();

  void serve();
};

} // namespace web
} // namespace dart

#endif