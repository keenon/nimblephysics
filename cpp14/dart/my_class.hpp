#pragma once

#include <memory>

class MyClass {
public:
  MyClass();
private:
  std::unique_ptr<int> m_val;
};
