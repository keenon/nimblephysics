#include <algorithm> // std::sort
#include <vector>

#include "dart/include_eigen.hpp"
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;

#define ALL_TESTS

std::string getAbsolutePath(std::string uri)
{
  const utils::CompositeResourceRetrieverPtr resourceRetriever
      = std::make_shared<utils::CompositeResourceRetriever>();
  common::LocalResourceRetrieverPtr localResourceRetriever
      = std::make_shared<common::LocalResourceRetriever>();
  resourceRetriever->addSchemaRetriever("file", localResourceRetriever);
  utils::PackageResourceRetrieverPtr packageRetriever
      = std::make_shared<utils::PackageResourceRetriever>(
          localResourceRetriever);
  resourceRetriever->addSchemaRetriever("package", packageRetriever);
  resourceRetriever->addSchemaRetriever(
      "dart", utils::DartResourceRetriever::create());
  return resourceRetriever->getFilePath(uri);
}

#ifdef ALL_TESTS
TEST(C3D, LOAD)
{
  std::vector<std::vector<Eigen::Vector3s>> pointData;
  int nFrames;
  int nMarkers;
  double freq;
  std::string file
      = getAbsolutePath("dart://sample/c3d/cmu_dribble_shoot_basketball.c3d");

  bool success
      = utils::loadC3DFile(file.c_str(), pointData, &nFrames, &nMarkers, &freq);
  EXPECT_TRUE(success);

  std::cout << pointData.size() << std::endl;
}
#endif