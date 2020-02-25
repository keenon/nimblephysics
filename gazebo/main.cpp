#include <iostream>
#include <memory>

int main()
{
  auto val = ::std::make_unique<int>(0);
  std::cout << "int: " << *val << std::endl;
  return 0;
}
