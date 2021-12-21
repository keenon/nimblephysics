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

#include <iostream>

#ifdef HAVE_PERF_UTILS

#include <PerfUtils/TimeTrace.h>
#include <gtest/gtest.h>

#include "dart/performance/PerformanceLog.hpp"

using namespace dart;
using namespace dart::performance;

TEST(PERFORMANCE, TIME_TRACE)
{
  uint64_t start = PerfUtils::Cycles::rdtsc();
  for (int i = 0; i < 100; i++)
  {
    PerfUtils::TimeTrace::record("testing");
  }
  uint64_t end = PerfUtils::Cycles::rdtsc();
  std::cout << PerfUtils::TimeTrace::getTrace() << std::endl;
  std::cout << "Cycles: " << (end - start) << std::endl;
}

TEST(PERFORMANCE, TWO_ROOTS)
{
  PerformanceLog::initialize();
  PerformanceLog* root1 = PerformanceLog::startRoot("root1");
  PerformanceLog* root2 = PerformanceLog::startRoot("root2");
  root2->end();
  root1->end();

  std::unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
      finalizedRoots = PerformanceLog::finalize();

  EXPECT_EQ(finalizedRoots.size(), 2);
  std::shared_ptr<FinalizedPerformanceLog> finalizedRoot
      = finalizedRoots["root1"];
  std::cout << finalizedRoot->prettyPrint() << std::endl;
}

TEST(PERFORMANCE, NESTED)
{
  PerformanceLog::initialize();
  PerformanceLog* root1 = PerformanceLog::startRoot("root1");
  root1->end();
  // PerformanceLog root = PerformanceLog(0, -1);
  PerformanceLog* root2 = PerformanceLog::startRoot("root2");
  for (int i = 0; i < 100; i++)
  {
    PerformanceLog* child = root2->startRun("child");
    child->end();
  }
  root2->end();

  std::unordered_map<std::string, std::shared_ptr<FinalizedPerformanceLog>>
      finalizedRoots = PerformanceLog::finalize();

  EXPECT_EQ(finalizedRoots.size(), 2);
  std::shared_ptr<FinalizedPerformanceLog> finalizedRoot
      = finalizedRoots["root2"];

  EXPECT_EQ(finalizedRoot->getNumRuns(), 1);
  EXPECT_EQ(finalizedRoot->getChild("child")->getNumRuns(), 100);

  std::cout << finalizedRoot->prettyPrint() << std::endl;
}

#endif