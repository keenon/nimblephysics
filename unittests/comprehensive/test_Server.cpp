/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <asio/io_service.hpp>
#include <gtest/gtest.h>

#include "dart/server/WebsocketServer.hpp"

// The port number the WebSocket server listens on
#define PORT_NUMBER 8080

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(SERVER, BASIC_SERVER)
{
  // Create the event loop for the main thread, and the WebSocket server
  asio::io_service mainEventLoop;
  WebsocketServer server;

  // Register our network callbacks, ensuring the logic is run on the main
  // thread's event loop
  server.connect([&mainEventLoop, &server](ClientConnection conn) {
    mainEventLoop.post([conn, &server]() {
      std::clog << "Connection opened." << std::endl;
      std::clog << "There are now " << server.numConnections()
                << " open connections." << std::endl;

      // Send a hello message to the client
      server.sendJsonObject(conn, "hello", Json::Value());
    });
  });
  server.disconnect([&mainEventLoop, &server](ClientConnection conn) {
    mainEventLoop.post([conn, &server]() {
      std::clog << "Connection closed." << std::endl;
      std::clog << "There are now " << server.numConnections()
                << " open connections." << std::endl;
    });
  });
  server.message(
      "message",
      [&mainEventLoop, &server](
          ClientConnection conn, const Json::Value& args) {
        mainEventLoop.post([conn, args, &server]() {
          std::clog << "message handler on the main thread" << std::endl;
          std::clog << "Message payload:" << std::endl;
          for (auto key : args.getMemberNames())
          {
            std::clog << "\t" << key << ": " << args[key].asString()
                      << std::endl;
          }

          // Echo the message pack to the client
          server.sendJsonObject(conn, "message", args);
        });
      });

  // Start the networking thread
  std::thread serverThread([&server]() { server.run(PORT_NUMBER); });

  // Start a keyboard input thread that reads from stdin
  std::thread inputThread([&server, &mainEventLoop]() {
    string input;
    while (1)
    {
      // Read user input from stdin
      std::getline(std::cin, input);

      // Broadcast the input to all connected clients (is sent on the network
      // thread)
      Json::Value payload;
      payload["input"] = input;
      server.broadcastJsonObject("userInput", payload);

      // Debug output on the main thread
      mainEventLoop.post([]() {
        std::clog << "User input debug output on the main thread" << std::endl;
      });
    }
  });

  // Start the event loop for the main thread
  asio::io_service::work work(mainEventLoop);
  mainEventLoop.run();
}
#endif
