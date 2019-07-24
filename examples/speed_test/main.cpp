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

#include <chrono>
#include <numeric>

#include <dart/dart.hpp>
#include "dart/utils/utils.hpp"

#include <dart/collision/ode/OdeCollisionDetector.hpp>

#include <tinyxml2.h>
#include <dart/utils/sdf/SdfParser.hpp>

#include <dart/gui/osg/Viewer.hpp>
#include <dart/gui/osg/RealTimeWorldNode.hpp>
#include <dart/gui/osg/DefaultEventHandler.hpp>

#include <boost/filesystem.hpp>

double testForwardKinematicSpeed(
    dart::dynamics::SkeletonPtr skel,
    bool position = true,
    bool velocity = true,
    bool acceleration = true,
    std::size_t numTests = 100000)
{
  if (nullptr == skel)
    return 0;

  dart::dynamics::BodyNode* bn = skel->getBodyNode(0);
  while (bn->getNumChildBodyNodes() > 0)
    bn = bn->getChildBodyNode(0);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for (std::size_t i = 0; i < numTests; ++i)
  {
    for (std::size_t i = 0; i < skel->getNumDofs(); ++i)
    {
      dart::dynamics::DegreeOfFreedom* dof = skel->getDof(i);
      dof->setPosition(dart::math::Random::uniform(
          std::max(dof->getPositionLowerLimit(), -1.0),
          std::min(dof->getPositionUpperLimit(), 1.0)));
    }

    for (std::size_t i = 0; i < skel->getNumBodyNodes(); ++i)
    {
      if (position)
        skel->getBodyNode(i)->getWorldTransform();
      if (velocity)
      {
        skel->getBodyNode(i)->getSpatialVelocity();
        skel->getBodyNode(i)->getPartialAcceleration();
      }
      if (acceleration)
        skel->getBodyNode(i)->getSpatialAcceleration();
    }
  }

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  return elapsed_seconds.count();
}

void runKinematicsTest(
    std::vector<double>& results,
    const std::vector<dart::simulation::WorldPtr>& worlds,
    bool position,
    bool velocity,
    bool acceleration)
{
  double totalTime = 0;
  std::cout << "Testing: ";
  if (position)
    std::cout << "Position ";
  if (velocity)
    std::cout << "Velocity ";
  if (acceleration)
    std::cout << "Acceleration ";
  std::cout << "\n";

  // Test for updating the whole skeleton
  for (std::size_t i = 0; i < worlds.size(); ++i)
  {
    dart::simulation::WorldPtr world = worlds[i];
    totalTime += testForwardKinematicSpeed(
        world->getSkeleton(0), position, velocity, acceleration);
  }
  results.push_back(totalTime);
  std::cout << "Result: " << totalTime << "s" << std::endl;
}

double testDynamicsSpeed(
    dart::simulation::WorldPtr world, std::size_t numIterations = 10000)
{
  if (nullptr == world)
    return 0;

  for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
  {
    dart::dynamics::SkeletonPtr skel = world->getSkeleton(i);
    skel->resetPositions();
    skel->resetVelocities();
    skel->resetAccelerations();
  }

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for (std::size_t i = 0; i < numIterations; ++i)
  {
    world->step();
  }

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  return elapsed_seconds.count();
}

void runDynamicsTest(
    std::vector<double>& results,
    const std::vector<dart::simulation::WorldPtr>& worlds)
{
  double totalTime = 0;

  for (std::size_t i = 0; i < worlds.size(); ++i)
  {
    totalTime += testDynamicsSpeed(worlds[i]);
  }

  results.push_back(totalTime);
  std::cout << "Result: " << totalTime << "s" << std::endl;
}

void print_results(const std::vector<double>& result)
{
  double sum = std::accumulate(result.begin(), result.end(), 0.0);
  double mean = sum / result.size();
  std::cout << "Average: " << mean << "\n";
  std::vector<double> diff(result.size());
  std::transform(
      result.begin(),
      result.end(),
      diff.begin(),
      std::bind2nd(std::minus<double>(), mean));
  double stddev = std::sqrt(
      std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)
      / result.size());
  std::cout << "Std Dev: " << stddev << "\n";
}

std::vector<std::string> getSceneFiles()
{
  std::vector<std::string> scenes;
  scenes.push_back("dart://sample/skel/test/chainwhipa.skel");
  scenes.push_back("dart://sample/skel/test/single_pendulum.skel");
  scenes.push_back("dart://sample/skel/test/single_pendulum_euler_joint.skel");
  scenes.push_back("dart://sample/skel/test/single_pendulum_ball_joint.skel");
  scenes.push_back("dart://sample/skel/test/double_pendulum.skel");
  scenes.push_back("dart://sample/skel/test/double_pendulum_euler_joint.skel");
  scenes.push_back("dart://sample/skel/test/double_pendulum_ball_joint.skel");
  scenes.push_back("dart://sample/skel/test/serial_chain_revolute_joint.skel");
  scenes.push_back("dart://sample/skel/test/serial_chain_eulerxyz_joint.skel");
  scenes.push_back("dart://sample/skel/test/serial_chain_ball_joint.skel");
  scenes.push_back("dart://sample/skel/test/serial_chain_ball_joint_20.skel");
  scenes.push_back("dart://sample/skel/test/serial_chain_ball_joint_40.skel");
  scenes.push_back("dart://sample/skel/test/simple_tree_structure.skel");
  scenes.push_back(
      "dart://sample/skel/test/simple_tree_structure_euler_joint.skel");
  scenes.push_back(
      "dart://sample/skel/test/simple_tree_structure_ball_joint.skel");
  scenes.push_back("dart://sample/skel/test/tree_structure.skel");
  scenes.push_back("dart://sample/skel/test/tree_structure_euler_joint.skel");
  scenes.push_back("dart://sample/skel/test/tree_structure_ball_joint.skel");
  scenes.push_back("dart://sample/skel/fullbody1.skel");

  return scenes;
}

//std::vector<dart::simulation::WorldPtr> getWorlds()
//{
//  std::vector<std::string> sceneFiles = getSceneFiles();
//  std::vector<dart::simulation::WorldPtr> worlds;
//  for (std::size_t i = 0; i < sceneFiles.size(); ++i)
//    worlds.push_back(dart::utils::SkelParser::readWorld(sceneFiles[i]));

//  for (auto& world : worlds)
//    world->getConstraintSolver()->setCollisionDetector(
//          dart::collision::OdeCollisionDetector::create());

//  return worlds;
//}

std::string getModelName(const std::string& uri)
{
  const std::string model_uri_component = "openrobotics/models/";

  const std::size_t i =
      uri.find(model_uri_component) + model_uri_component.size();

  std::string name = uri.substr(i);
  name.erase(name.find_last_not_of(" \n\r\t")+1);

  return name;
}

std::string getModelFile(const std::string& uri)
{
  const std::string fuel_model_path =
      "/home/grey/.ignition/fuel/fuel.ignitionrobotics.org/openrobotics/models/";

  const std::string& model_name = getModelName(uri);
  if(model_name == "Rescue Randy Sitting")
    return "";

  const std::string dir = fuel_model_path + model_name;
  for(const auto& f : boost::filesystem::recursive_directory_iterator(dir))
  {
    const std::string& path = f.path().string();
    const std::string file_name = path.substr(path.find_last_of('/')+1);
    if(file_name == "model.sdf")
      return path;
  }

  std::cout << "Could not find a model file for: " << dir << std::endl;

  return "";
}

Eigen::Isometry3d getTransform(const std::string& text)
{
  double x, y, z, r, p, yaw;
  std::istringstream stream(text);
  stream >> x >> y >> z >> r >> p >> yaw;
  std::cout << "Setting pose: " << x << ", " << y << ", " << z << ", "
            << r << ", " << p << ", " << yaw << std::endl;

  Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
  tf.translation() = Eigen::Vector3d(100.0*x, 100.0*y, 100.0*z);
  tf.rotate(Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()));
  tf.rotate(Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()));
  tf.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

  return tf;
}

std::vector<dart::simulation::WorldPtr> getWorlds()
{
  dart::simulation::WorldPtr world = dart::simulation::World::create();

  tinyxml2::XMLDocument doc;
  std::cout << "Loading xml file" << std::endl;
  doc.LoadFile("/home/grey/projects/dartsim/tunnel_circuit_practice_01.sdf");
  tinyxml2::XMLElement* root = doc.RootElement();

  tinyxml2::XMLElement* world_elem = root->FirstChildElement("world");
  tinyxml2::XMLElement* include_elem = world_elem->FirstChildElement("include");

//  tinyxml2::XMLElement* include_elem = root->FirstChildElement("include");

  std::size_t count = 0;
  std::cout << "Iterating through includes" << std::endl;
  while(include_elem)
  {
    std::cout << "Count: " << count++ << std::endl;
    const std::string& uri = include_elem->FirstChildElement("uri")->GetText();
    const std::string& model_file = getModelFile(uri);
    if(model_file.empty())
    {
      std::cout << "could not find a model for: " << uri << std::endl;
      include_elem = include_elem->NextSiblingElement("include");
      continue;
    }

    const dart::dynamics::SkeletonPtr model =
        dart::utils::SdfParser::readSkeleton(model_file);

    const auto name_elem = include_elem->FirstChildElement("name");
    if(name_elem)
      model->setName(name_elem->GetText());

    const auto pose_elem = include_elem->FirstChildElement("pose");
    if(pose_elem)
      model->getJoint(0)->setTransformFromChildBodyNode(
            getTransform(pose_elem->GetText()));

    std::cout << "About to add skeleton" << std::endl;
    world->addSkeleton(model);
    include_elem = include_elem->NextSiblingElement("include");

//    if(count > 100)
//      break;
  }
  if(count == 0)
    std::cout << "NO INCLUDE ELEMENTS??" << std::endl;

  std::cout << "Done gathering the includes" << std::endl;

  return {world};
}

class InputHandler : public osgGA::GUIEventHandler
{
public:

  dart::gui::osg::Viewer* mViewer;

  InputHandler(dart::gui::osg::Viewer* viewer)
    : mViewer(viewer)
  {
    // Do nothing
  }

  bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&) override
  {
    if(ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
    {
      if(ea.getKey() == osgGA::GUIEventAdapter::KEY_Tab)
      {
        mViewer->home();
        return true;
      }
    }

    return false;
  }
};

int main(int argc, char* argv[])
{
  bool test_kinematics = false;
  for (int i = 1; i < argc; ++i)
  {
    if (std::string(argv[i]) == "-k")
      test_kinematics = true;
  }

  std::vector<dart::simulation::WorldPtr> worlds = getWorlds();
  ::osg::ref_ptr<dart::gui::osg::WorldNode> node =
      new dart::gui::osg::RealTimeWorldNode(worlds.front());
  dart::gui::osg::Viewer viewer;

  viewer.addWorldNode(node);

  osg::ref_ptr<InputHandler> input = new InputHandler(&viewer);
  viewer.addEventHandler(input);

  std::cout << "About to run" << std::endl;
  return viewer.run();


  if (test_kinematics)
  {
    std::cout << "Testing Kinematics" << std::endl;
    std::vector<double> acceleration_results;
    std::vector<double> velocity_results;
    std::vector<double> position_results;

    for (std::size_t i = 0; i < 10; ++i)
    {
      std::cout << "\nTrial #" << i + 1 << std::endl;
      runKinematicsTest(acceleration_results, worlds, true, true, true);
      runKinematicsTest(velocity_results, worlds, true, true, false);
      runKinematicsTest(position_results, worlds, true, false, false);
    }

    std::cout << "\n\n --- Final Kinematics Results --- \n\n";

    std::cout << "Position, Velocity, Acceleration\n";
    print_results(acceleration_results);

    std::cout << "\nPosition, Velocity\n";
    print_results(velocity_results);

    std::cout << "\nPosition\n";
    print_results(position_results);

    return 0;
  }

  std::cout << "Testing Dynamics" << std::endl;
  std::vector<double> dynamics_results;
  for (std::size_t i = 0; i < 10; ++i)
  {
    std::cout << "\nTrial #" << i + 1 << std::endl;
    runDynamicsTest(dynamics_results, worlds);
  }

  std::cout << "\n\n --- Final Dynamics Results --- \n\n";
  print_results(dynamics_results);
}
