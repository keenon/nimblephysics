#include "my_class.hpp"

MyClass::MyClass() : m_val(::std::make_unique<int>(10)) {
  // Do nothing
}
