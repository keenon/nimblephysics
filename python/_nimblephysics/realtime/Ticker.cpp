#include <iostream>

#include <Python.h>
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
      .def(::py::init<s_t>(), ::py::arg("secondsPerTick"))
      .def(
          "registerTickListener",
          +[](dart::realtime::Ticker* self,
              std::function<void(long)> callback) -> void {
            std::function<void(long now)> wrappedCallback
                = [callback](long now) {
                    /* Acquire GIL before calling Python code */
                    py::gil_scoped_acquire acquire;
                    try
                    {
                      callback(now);
                    }
                    catch (::py::error_already_set& e)
                    {
                      if (e.matches(PyExc_KeyboardInterrupt))
                      {
                        std::cout
                            << "Nimble caught a keyboard interrupt in a "
                               "callback from registerTickListener(). Exiting "
                               "with code 0."
                            << std::endl;
                        exit(0);
                      }
                      else
                      {
                        std::cout << "Nimble caught an exception calling "
                                     "callback from registerTickListener():"
                                  << std::endl
                                  << std::string(e.what()) << std::endl;
                      }
                    }
                  };
            self->registerTickListener(wrappedCallback);
          },
          ::py::arg("listener"))
      .def(
          "start",
          &dart::realtime::Ticker::start,
          ::py::call_guard<py::gil_scoped_release>())
      .def("stop", &dart::realtime::Ticker::stop)
      .def("clear", &dart::realtime::Ticker::clear);
}

} // namespace python
} // namespace dart
