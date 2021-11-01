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

#include <tuple>

#include <gtest/gtest.h>

#include "dart/math/MultivariateGaussian.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;

//==============================================================================
TEST(MultivariateGaussian, BASICS)
{
  std::vector<std::string> cols;
  cols.push_back("thumbtipreach");
  cols.push_back("tibialheight");
  /*
  cols.push_back("Age");
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  cols.push_back("chestheight");
  cols.push_back("handlength");
  cols.push_back("shoulderelbowlength");
  cols.push_back("shoulderlength");
  cols.push_back("kneeheightmidpatella");
  cols.push_back("footlength");
  */
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv", cols, 0.1);
  gauss->debugToStdout();

  srand(42);
  Eigen::VectorXs x = Eigen::VectorXs::Random(cols.size());

  Eigen::VectorXs grad = gauss->computeLogPDFGrad(x);
  Eigen::VectorXs grad_fd = gauss->finiteDifferenceLogPDFGrad(x);
  if (!equals(grad, grad_fd, 1e-9))
  {
    std::cout << "Error on log probability grad!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(grad.size(), 3);
    compare.col(0) = grad;
    compare.col(1) = grad_fd;
    compare.col(2) = grad - grad_fd;
    std::cout << "analytical - FD - diff" << std::endl << compare << std::endl;
  }
}

//==============================================================================
TEST(MultivariateGaussian, OBSERVE_2_TO_1)
{
  std::vector<std::string> cols;
  cols.push_back("Age");
  cols.push_back("Weightlbs");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv", cols);
  gauss->debugToStdout();

  srand(42);

  std::map<std::string, s_t> observedValues;
  observedValues["Age"] = 35.0;

  std::vector<int> observedIndices = gauss->getObservedIndices(observedValues);
  std::vector<int> unobservedIndices
      = gauss->getUnobservedIndices(observedValues);

  EXPECT_EQ(observedIndices.size(), 1);
  EXPECT_EQ(unobservedIndices.size(), 1);
  EXPECT_EQ(gauss->getVariableNameAtIndex(observedIndices[0]), "Age");
  EXPECT_EQ(gauss->getVariableNameAtIndex(unobservedIndices[0]), "Weightlbs");

  Eigen::VectorXs observedMu = gauss->getMuSubset(observedIndices);
  Eigen::VectorXs unobservedMu = gauss->getMuSubset(unobservedIndices);

  EXPECT_EQ(observedMu.size(), 1);
  EXPECT_EQ(unobservedMu.size(), 1);
  EXPECT_EQ(gauss->getMu()(observedIndices[0]), observedMu(0));
  EXPECT_EQ(gauss->getMu()(unobservedIndices[0]), unobservedMu(0));

  Eigen::MatrixXs observedCov
      = gauss->getCovSubset(observedIndices, observedIndices);
  Eigen::MatrixXs unobservedCov
      = gauss->getCovSubset(unobservedIndices, unobservedIndices);
  Eigen::MatrixXs offDiagonalCov
      = gauss->getCovSubset(observedIndices, unobservedIndices);

  EXPECT_EQ(observedCov.size(), 1);
  EXPECT_EQ(unobservedCov.size(), 1);
  EXPECT_EQ(
      gauss->getCov()(observedIndices[0], observedIndices[0]),
      observedCov(0, 0));
  EXPECT_EQ(
      gauss->getCov()(unobservedIndices[0], unobservedIndices[0]),
      unobservedCov(0, 0));
  EXPECT_EQ(
      gauss->getCov()(observedIndices[0], unobservedIndices[0]),
      offDiagonalCov(0, 0));

  std::shared_ptr<MultivariateGaussian> conditioned
      = gauss->condition(observedValues);
  EXPECT_EQ(conditioned->getMu().size(), 1);
  EXPECT_EQ(conditioned->getCov().size(), 1);

  conditioned->debugToStdout();
}

//==============================================================================
TEST(MultivariateGaussian, OBSERVE)
{
  std::vector<std::string> cols;
  cols.push_back("Age");
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  cols.push_back("chestheight");
  cols.push_back("handlength");
  cols.push_back("shoulderelbowlength");
  cols.push_back("shoulderlength");
  cols.push_back("kneeheightmidpatella");
  cols.push_back("footlength");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv", cols);
  gauss->debugToStdout();

  srand(42);

  std::map<std::string, s_t> observedValues;
  observedValues["Age"] = 35.0;
  observedValues["Weightlbs"] = 190.0;
  observedValues["footlength"] = 250.0;

  std::shared_ptr<MultivariateGaussian> conditioned
      = gauss->condition(observedValues);
  EXPECT_EQ(conditioned->getMu().size(), cols.size() - observedValues.size());
  EXPECT_EQ(conditioned->getCov().rows(), conditioned->getMu().size());

  conditioned->debugToStdout();
}