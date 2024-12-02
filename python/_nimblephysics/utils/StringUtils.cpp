#include <memory>

#include <dart/utils/StringUtils.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void StringUtils(py::module& m)
{
  auto sm = m.def_submodule("StringUtils");
  sm.def(
      "ltrim",
      [](const std::string& s) {
        return dart::utils::ltrim(s);
      },
      ::py::arg("s"));
  sm.def(
      "rtrim",
      [](const std::string& s) {
        return dart::utils::rtrim(s);
      },
      ::py::arg("s"));
  sm.def(
      "trim",
      [](const std::string& s) {
        return dart::utils::trim(s);
      },
      ::py::arg("s"));
}

} // namespace python
} // namespace dart