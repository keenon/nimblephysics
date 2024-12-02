#include <dart/utils/MJCFExporter.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MJCFExporter(py::module& m)
{
  ::py::class_<dart::utils::MJCFExporter>(m, "MJCFExporter")
      .def_static(
          "writeSkeleton",
          &dart::utils::MJCFExporter::writeSkeleton,
          ::py::arg("path"),
          ::py::arg("skel"));
}

} // namespace python
} // namespace dart
