#include <gtest/gtest.h>

#include "dart/utils/OpenSimParser.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace utils;

TEST(OpenSimParser, RAJAGOPAL)
{
  OpenSimParser::readSkeleton("dart://sample/osim/Rajagopal_scaled.osim");
}