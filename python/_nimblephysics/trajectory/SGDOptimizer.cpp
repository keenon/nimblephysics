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

#include <functional>

#include <Python.h>
#include <dart/trajectory/Problem.hpp>
#include <dart/trajectory/SGDOptimizer.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace python {

void SGDOptimizer(py::module& m)
{
  ::py::class_<
      dart::trajectory::SGDOptimizer,
      std::shared_ptr<dart::trajectory::SGDOptimizer>,
      dart::trajectory::Optimizer>(m, "SGDOptimizer")
      .def(::py::init<>())
      .def(
          "optimize",
          &dart::trajectory::SGDOptimizer::optimize,
          ::py::arg("shot"),
          ::py::arg("reuseRecord") = nullptr,
          ::py::call_guard<py::gil_scoped_release>())
      .def(
          "setIterationLimit",
          &dart::trajectory::SGDOptimizer::setIterationLimit,
          ::py::arg("iterationLimit") = 500)
      .def(
          "setTolerance",
          &dart::trajectory::SGDOptimizer::setTolerance,
          ::py::arg("tol") = 1e-7)
      .def(
          "setLearningRate",
          &dart::trajectory::SGDOptimizer::setLearningRate,
          ::py::arg("learningRate") = 0.1);
  /*
  .def(
      "registerIntermediateCallback",
      +[](dart::trajectory::IPOptOptimizer* self,
          std::function<bool(
              dart::trajectory::Problem * problem,
              int,
              s_t primal,
              s_t dual)> callback) -> void {
        std::function<bool(
            dart::trajectory::Problem * problem,
            int,
            s_t primal,
            s_t dual)>
            wrappedCallback = [callback](
                                  dart::trajectory::Problem* problem,
                                  int step,
                                  s_t primal,
                                  s_t dual) {
              // Acquire GIL before calling Python code
              py::gil_scoped_acquire acquire;
              try
              {
                return callback(problem, step, primal, dual);
              }
              catch (::py::error_already_set& e)
              {
                if (e.matches(PyExc_KeyboardInterrupt))
                {
                  std::cout << "Nimble caught a keyboard interrupt in a "
                               "callback from "
                               "registerIntermediateCallback(). Exiting "
                               "with code 0."
                            << std::endl;
                  exit(0);
                }
                else
                {
                  std::cout
                      << "Nimble caught an exception calling "
                         "callback from registerIntermediateCallback():"
                      << std::endl
                      << std::string(e.what()) << std::endl;
                  return true;
                }
              }
            };
        self->registerIntermediateCallback(wrappedCallback);
      },
      ::py::arg("callback"));
      */
}

} // namespace python
} // namespace dart
