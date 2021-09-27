#include "WebsocketServer.hpp"

#include <algorithm>
#include <functional>
#include <iostream>

#include <websocketpp/logger/levels.hpp>

#include "dart/common/Console.hpp"

// The name of the special JSON field that holds the message type for messages
#define MESSAGE_FIELD "__MESSAGE__"

Json::Value WebsocketServer::parseJson(const string& json)
{
  Json::Value root;
  Json::Reader reader;
  reader.parse(json, root);
  return root;
}

string WebsocketServer::stringifyJson(const Json::Value& val)
{
  // When we transmit JSON data, we omit all whitespace
  Json::StreamWriterBuilder wbuilder;
  wbuilder["commentStyle"] = "None";
  wbuilder["indentation"] = "";

  return Json::writeString(wbuilder, val);
}

WebsocketServer::WebsocketServer() : mRunning(false)
{
  // Wire up our event handlers
  this->endpoint.set_open_handler(
      std::bind(&WebsocketServer::onOpen, this, std::placeholders::_1));
  this->endpoint.set_close_handler(
      std::bind(&WebsocketServer::onClose, this, std::placeholders::_1));
  this->endpoint.set_message_handler(std::bind(
      &WebsocketServer::onMessage,
      this,
      std::placeholders::_1,
      std::placeholders::_2));

  // Avoid errors with address already in use after ctrl+C
  this->endpoint.set_reuse_addr(true);

  // Initialise the Asio library, using our own event loop object
  this->endpoint.init_asio(&(this->eventLoop));
  this->endpoint.clear_access_channels(websocketpp::log::alevel::all);
}

bool WebsocketServer::run(int port)
{
  // Listen on the specified port number and start accepting connections
  websocketpp::lib::error_code error;
  this->endpoint.listen(port, error);
  if (error)
  {
    std::cout << "Error listening! " << error << std::endl;
    return false;
  }

  this->endpoint.start_accept(error);
  if (error)
  {
    std::cout << "Error in start accept! " << error << std::endl;
    return false;
  }

  mRunning = true;

  // Start the Asio event loop
  //
  // Do this in a loop to catch Websocket errors thrown by typical Chrome lazy
  // Websocket implementation, per:
  // https://github.com/zaphoyd/websocketpp/issues/580#issuecomment-689703724
  while (mRunning)
  {
    try
    {
      this->endpoint.run();
    }
    catch (websocketpp::exception const& e)
    {
      dterr << e.what() << std::endl;
      dterr << "Exception thrown from m_io_service->run(). Restarting "
               "m_io_service->run()"
            << std::endl;
    }
    catch (...)
    {
      dterr << "Hit critial error. Restarting m_io_service->run()" << std::endl;
    }
  }
  return true;
}

void WebsocketServer::stop()
{
  mRunning = false;
  this->endpoint.stop();
}

size_t WebsocketServer::numConnections()
{
  // Prevent concurrent access to the list of open connections from multiple
  // threads
  std::lock_guard<std::mutex> lock(this->connectionListMutex);

  return this->openConnections.size();
}

void WebsocketServer::sendJsonObject(
    ClientConnection conn,
    const string& messageType,
    const Json::Value& arguments)
{
  // Copy the argument values, and bundle the message type into the object
  Json::Value messageData = arguments;
  messageData[MESSAGE_FIELD] = messageType;

  // Send the JSON data to the client (will happen on the networking thread's
  // event loop)
  this->endpoint.send(
      conn,
      WebsocketServer::stringifyJson(messageData),
      websocketpp::frame::opcode::text);
}

// Sends a raw text message to a specific client
void WebsocketServer::send(ClientConnection conn, const string& message)
{
  // Send the message data to the client (will happen on the networking thread's
  // event loop)
  try
  {
    this->endpoint.send(conn, message, websocketpp::frame::opcode::text);
  }
  catch (websocketpp::exception const& e)
  {
    dterr << e.what() << std::endl;
    dterr << "Exception thrown from endpoint.send(). Continuing." << std::endl;
  }
  catch (...)
  {
    dterr << "Hit unknown error in endpoint.send(). Continuing." << std::endl;
  }
}

void WebsocketServer::broadcastJsonObject(
    const string& messageType, const Json::Value& arguments)
{
  // Prevent concurrent access to the list of open connections from multiple
  // threads
  std::lock_guard<std::mutex> lock(this->connectionListMutex);

  for (auto conn : this->openConnections)
  {
    this->sendJsonObject(conn, messageType, arguments);
  }
}

// Broadcast a raw text message to all clients
void WebsocketServer::broadcast(const string& message)
{
  // Prevent concurrent access to the list of open connections from multiple
  // threads
  std::lock_guard<std::mutex> lock(this->connectionListMutex);

  for (auto conn : this->openConnections)
  {
    this->send(conn, message);
  }
}

void WebsocketServer::onOpen(ClientConnection conn)
{
  {
    // Prevent concurrent access to the list of open connections from multiple
    // threads
    std::lock_guard<std::mutex> lock(this->connectionListMutex);

    // Add the connection handle to our list of open connections
    this->openConnections.push_back(conn);
  }

  // Invoke any registered handlers
  for (auto handler : this->connectHandlers)
  {
    handler(conn);
  }
}

void WebsocketServer::onClose(ClientConnection conn)
{
  {
    // Prevent concurrent access to the list of open connections from multiple
    // threads
    std::lock_guard<std::mutex> lock(this->connectionListMutex);

    // Remove the connection handle from our list of open connections
    auto connVal = conn.lock();
    auto newEnd = std::remove_if(
        this->openConnections.begin(),
        this->openConnections.end(),
        [&connVal](ClientConnection elem) {
          // If the pointer has expired, remove it from the vector
          if (elem.expired() == true)
          {
            return true;
          }

          // If the pointer is still valid, compare it to the handle for the
          // closed connection
          auto elemVal = elem.lock();
          if (elemVal.get() == connVal.get())
          {
            return true;
          }

          return false;
        });

    // Truncate the connections vector to erase the removed elements
    this->openConnections.resize(
        std::distance(openConnections.begin(), newEnd));
  }

  // Invoke any registered handlers
  for (auto handler : this->disconnectHandlers)
  {
    handler(conn);
  }
}

void WebsocketServer::onMessage(
    ClientConnection conn, WebsocketEndpoint::message_ptr msg)
{
  // Validate that the incoming message contains valid JSON
  Json::Value messageObject = WebsocketServer::parseJson(msg->get_payload());
  if (messageObject.isNull() == false)
  {
    // If any handlers are registered for the message type, invoke them
    for (auto handler : this->messageHandlers)
    {
      handler(conn, messageObject);
    }
  }
}