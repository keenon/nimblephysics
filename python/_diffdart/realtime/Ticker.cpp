#include <dart/realtime/Ticker.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void Ticker(py::module& m)
{
  ::py::class_<dart::realtime::Ticker, std::shared_ptr<dart::realtime::Ticker>>(
      m, "Ticker")
      .def(::py::init<double>(), ::py::arg("secondsPerTick"))
      .def(
          "registerTickListener",
          &dart::realtime::Ticker::registerTickListener,
          ::py::arg("listener"))
      .def("start", &dart::realtime::Ticker::start)
      .def("stop", &dart::realtime::Ticker::stop)
      .def("clear", &dart::realtime::Ticker::clear);
}

} // namespace python
} // namespace dart
