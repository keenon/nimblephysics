#! /bin/bash

protoc --proto_path=../dart/proto --ts_out=src/proto ../dart/proto/GUI.proto