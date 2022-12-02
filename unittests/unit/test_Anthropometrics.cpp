#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;

// #define ALL_TESTS

// #ifdef ALL_TESTS
TEST(ANTHROPOMETRICS, LOAD)
{
  std::shared_ptr<Anthropometrics> result = Anthropometrics::loadFromFile(
      "dart://sample/osim/ANSUR/ANSUR_metrics.xml");
  std::vector<std::string> cols = result->getMetricNames();
  cols.push_back("Age");
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv", cols, 0.001);

  std::cout << "Mu: " << std::endl << gauss->getMu() << std::endl;
  std::cout << "Cov: " << std::endl << gauss->getCov() << std::endl;

  result->setDistribution(gauss);

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  skel->autogroupSymmetricSuffixes();
  skel->setScaleGroupUniformScaling(skel->getBodyNode("hand_r"));

  Eigen::VectorXs mu = gauss->getMu();
  Eigen::VectorXs x = gauss->convertFromMap(result->measure(skel));
  Eigen::MatrixXs compare = Eigen::MatrixXs(mu.size(), 3);
  compare.col(0) = x;
  compare.col(1) = mu;
  compare.col(2) = x - mu;
  std::cout << "x - mu - diff" << std::endl << compare << std::endl;

  std::cout << "Initial log PDF: " << result->getLogPDF(skel) << std::endl;

  std::map<std::string, s_t> measurements = result->measure(skel);
  for (auto pair : measurements)
  {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }

  Eigen::VectorXs grad = result->getGradientOfLogPDFWrtBodyScales(skel);
  Eigen::VectorXs grad_fd
      = result->finiteDifferenceGradientOfLogPDFWrtBodyScales(skel);

  if (!equals(grad, grad_fd, 5e-9))
  {
    std::cout << "Errors on anthropometric log probability grad wrt bodies!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(grad.size(), 3);
    compare.col(0) = grad;
    compare.col(1) = grad_fd;
    compare.col(2) = grad - grad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;

    EXPECT_TRUE(equals(grad, grad_fd, 5e-9));
  }

  grad = result->getGradientOfLogPDFWrtGroupScales(skel);
  grad_fd = result->finiteDifferenceGradientOfLogPDFWrtGroupScales(skel);

  if (!equals(grad, grad_fd, 5e-9))
  {
    std::cout << "Errors on anthropometric log probability grad wrt groups!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(grad.size(), 3);
    compare.col(0) = grad;
    compare.col(1) = grad_fd;
    compare.col(2) = grad - grad_fd;
    std::cout << "Grad - FD - Diff" << std::endl << compare << std::endl;

    EXPECT_TRUE(equals(grad, grad_fd, 5e-9));
  }
}
// #endif

#ifdef BLOCKING_GUI_TEST
// #ifdef ALL_TESTS
TEST(ANTHROPOMETRICS, GUI)
{
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_metrics.xml");

  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Age");
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss
      = MultivariateGaussian::loadFromCSV(
          "dart://sample/osim/ANSUR/ANSUR_II_BOTH_Public.csv",
          cols,
          0.001); // mm -> m

  std::map<std::string, s_t> observedValues;
  observedValues["Age"] = 30 * 0.001;
  observedValues["Weightlbs"] = 190 * 0.001;
  observedValues["Heightin"] = (5 * 12 + 9) * 0.001;

  std::cout << "Old mu: " << std::endl << gauss->getMu() << std::endl;
  std::cout << "Old cov: " << std::endl << gauss->getCov() << std::endl;

  gauss = gauss->condition(observedValues);

  std::cout << "New mu: " << std::endl << gauss->getMu() << std::endl;
  std::cout << "New cov: " << std::endl << gauss->getCov() << std::endl;

  anthropometrics->setDistribution(gauss);

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/11_01_Marilyn_Bug/prod/Models/"
      "unscaled_generic.osim");
  // "dart://sample/osim/CompleteHumanModel/CompleteHumanModel.osim");
  // OpenSimFile file = OpenSimParser::parseOsim(
  //     "dart://sample/osim/LaiArnoldSubject5/"
  //     "LaiArnoldModified2017_poly_withArms_weldHand_generic.osim");
  std::shared_ptr<dynamics::Skeleton> skel = file.skeleton;
  skel->autogroupSymmetricSuffixes();
  skel->setScaleGroupUniformScaling(skel->getBodyNode("hand_r"));

  s_t logProb = anthropometrics->getLogPDF(skel);
  std::cout << "Log prob: " << logProb << std::endl;

  Eigen::VectorXs originalGroupScales = skel->getGroupScales();

  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderSkeleton(skel);
  anthropometrics->debugToGUI(server, skel);

  Eigen::VectorXs mu = gauss->getMu();
  Eigen::VectorXs x = gauss->convertFromMap(anthropometrics->measure(skel));
  Eigen::MatrixXs compare = Eigen::MatrixXs(mu.size(), 3);
  compare.col(0) = x;
  compare.col(1) = mu;
  compare.col(2) = x - mu;
  std::cout << "x - mu - diff" << std::endl << compare << std::endl;

  realtime::Ticker ticker = realtime::Ticker(0.3);

  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    skel->getBodyNode(i)->setScaleLowerBound(Eigen::Vector3s::Ones() * 0.001);
    skel->getBodyNode(i)->setScaleUpperBound(Eigen::Vector3s::Ones() * 50.0);
  }

  Eigen::VectorXs scales = originalGroupScales;
  int step = 0;
  ticker.registerTickListener([&](long) {
    step++;
    auto measurements = anthropometrics->measure(skel);
    auto means = gauss->convertToMap(gauss->getMu());
    for (auto pair : measurements)
    {
      std::cout << "   " << pair.first << ": " << pair.second
                << " (mu=" << means[pair.first] << ")" << std::endl;
    }

    scales += 1e-4 * anthropometrics->getGradientOfLogPDFWrtGroupScales(skel);
    skel->setGroupScales(scales, false);
    scales = skel->getGroupScales();

    s_t newLogProb = anthropometrics->getLogPDF(skel);
    std::cout << "Step " << step << ": " << newLogProb << std::endl;

    if (step % 10 == 0)
    {
      server->renderSkeleton(skel);
      anthropometrics->debugToGUI(server, skel);
    }

    if (step > 30)
    {
      step = 0;
      scales = originalGroupScales;
      std::map<std::string, s_t> measurements = anthropometrics->measure(skel);
      for (auto pair : measurements)
      {
        std::cout << pair.first << ": " << pair.second
                  << " (mean = " << gauss->getMean(pair.first) << ")"
                  << std::endl;
      }

      Eigen::VectorXs mu = gauss->getMu();
      Eigen::VectorXs x = gauss->convertFromMap(anthropometrics->measure(skel));
      Eigen::MatrixXs compare = Eigen::MatrixXs(mu.size(), 3);
      compare.col(0) = x;
      compare.col(1) = mu;
      compare.col(2) = x - mu;
      std::cout << "x - mu - diff" << std::endl << compare << std::endl;

      return;
    }
  });
  ticker.start();

  server->blockWhileServing();
}
// #endif
#endif