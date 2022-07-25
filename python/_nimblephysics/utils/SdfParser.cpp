#include <memory>

#include <dart/utils/sdf/SdfParser.hpp>
#include <pybind11/pybind11.h>

#include "dart/dynamics/Skeleton.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void SdfParser(py::module& m)
{
  auto sm = m.def_submodule("SdfParser");
  sm.def(
      "readSkeleton",
      [](const std::string path) {
        return dart::utils::SdfParser::readSkeleton(path);
      },
      ::py::arg("path"));
  sm.def(
      "writeSkeleton",
      [](const std::string path, std::shared_ptr<dynamics::Skeleton> skel) {
        return dart::utils::SdfParser::writeSkeleton(path, skel);
      },
      ::py::arg("path"),
      ::py::arg("skeleton"));
  sm.def(
      "readWorld",
      [](const std::string path) {
        return dart::utils::SdfParser::readWorld(path);
      },
      ::py::arg("path"));
}

} // namespace python
} // namespace dart
