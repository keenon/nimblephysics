#include "dart/biomechanics/OpenSimParser.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/TranslationalJoint2D.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/PolynomialFunction.hpp"
#include "dart/math/SimmSpline.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/XmlHelpers.hpp"

using namespace std;

// Source: https://stackoverflow.com/a/145309/13177487
#include <stdio.h> /* defines FILENAME_MAX */
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

namespace dart {
using namespace utils;
namespace biomechanics {

//==============================================================================
OpenSimFile::OpenSimFile(
    dynamics::SkeletonPtr skeleton, dynamics::MarkerMap markersMap)
  : skeleton(skeleton), markersMap(markersMap)
{
}

//==============================================================================
OpenSimFile::OpenSimFile()
{
}

//==============================================================================
common::ResourceRetrieverPtr ensureRetriever(
    const common::ResourceRetrieverPtr& _retriever)
{
  if (_retriever)
  {
    return _retriever;
  }
  else
  {
    auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
    newRetriever->addSchemaRetriever(
        "file", std::make_shared<common::LocalResourceRetriever>());
    newRetriever->addSchemaRetriever("dart", DartResourceRetriever::create());
    return newRetriever;
  }
}

//==============================================================================
Eigen::Vector2s readVec2(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector2s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  return vec;
}

Eigen::Vector3s readVec3(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector3s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  vec(2) = strtod(q, &q);
  return vec;
}

Eigen::Vector6s readVec6(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector6s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  vec(2) = strtod(q, &q);
  vec(3) = strtod(q, &q);
  vec(4) = strtod(q, &q);
  vec(5) = strtod(q, &q);
  return vec;
}

std::vector<s_t> readVecX(tinyxml2::XMLElement* elem)
{
  const char* c = elem->GetText();
  char* q = const_cast<char*>(c);
  std::vector<s_t> values;
  while (q != nullptr && *q != '\0')
  {
    s_t value = strtod(q, &q);
    values.push_back(value);
  }
  return values;
}

dynamics::EulerJoint::AxisOrder getAxisOrder(
    std::vector<Eigen::Vector3s> axisList)
{
  if (axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitZ())
  {
    return dynamics::EulerJoint::AxisOrder::XYZ;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitX())
  {
    return dynamics::EulerJoint::AxisOrder::ZYX;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return dynamics::EulerJoint::AxisOrder::ZXY;
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return dynamics::EulerJoint::AxisOrder::XZY;
  }
  assert(false);
  // don't break the build when building as prod
  return dynamics::EulerJoint::AxisOrder::XYZ;
}

Eigen::Vector3s getAxisFlips(std::vector<Eigen::Vector3s> axisList)
{
  Eigen::Vector3s vec;
  vec(0) = (axisList[0] == -Eigen::Vector3s::UnitX()
            || axisList[0] == -Eigen::Vector3s::UnitY()
            || axisList[0] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  vec(1) = (axisList[1] == -Eigen::Vector3s::UnitX()
            || axisList[1] == -Eigen::Vector3s::UnitY()
            || axisList[1] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  vec(2) = (axisList[2] == -Eigen::Vector3s::UnitX()
            || axisList[2] == -Eigen::Vector3s::UnitY()
            || axisList[2] == -Eigen::Vector3s::UnitZ())
               ? -1.0
               : 1.0;
  return vec;
}

//==============================================================================
OpenSimFile OpenSimParser::parseOsim(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument osimFile;
  try
  {
    openXMLFile(osimFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return null_file;
  }

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = osimFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (jointSet != nullptr)
  {
    // This is the older format, where JointSet specifies the joints separately
    // from the body hierarchy
    return readOsim30(uri, docElement, retriever);
  }
  else
  {
    // This is the newer format, where Joints are specified as childen of Bodies
    return readOsim40(uri, docElement, retriever);
  }
}

//==============================================================================
/// This creates an XML configuration file, which you can pass to the OpenSim
/// scaling tool to rescale a skeleton
///
/// You can use this with the command: "opensim-cmd run-tool
/// ScalingInstructions.xml" to rescale an OpenSim model
void OpenSimParser::saveOsimScalingXMLFile(
    const std::string& subjectName,
    std::shared_ptr<dynamics::Skeleton> skel,
    double massKg,
    double heightM,
    const std::string& osimInputPath,
    const std::string& osimInputMarkersPath,
    const std::string& osimOutputPath,
    const std::string& scalingInstructionsOutputPath)
{
  using namespace tinyxml2;

  // clang-format off
  /**

  Here's an example file:

  <OpenSimDocument Version="40000">
    <ScaleTool name="subject01-scaled">
      <!--Mass of the subject in kg.  Subject-specific model generated by scaling step will have this total mass.-->
      <mass>72.599999999999994</mass>
      <!--Height of the subject in mm.  For informational purposes only (not used by scaling).-->
      <height>-1</height>
      <!--Age of the subject in years.  For informational purposes only (not used by scaling).-->
      <age>-1</age>
      <!--Notes for the subject.-->
      <notes>Unassigned</notes>
      <!--Specifies the name of the unscaled model (.osim) and the marker set.-->
      <GenericModelMaker>
        <!--Model file (.osim) for the unscaled model.-->
        <model_file>Unassigned</model_file>
        <!--Set of model markers used to scale the model. Scaling is done based on distances between model markers compared to the same distances between the corresponding experimental markers.-->
        <marker_set_file>Unassigned</marker_set_file>
      </GenericModelMaker>
      <!--Specifies parameters for scaling the model.-->
      <ModelScaler>
        <!--Whether or not to use the model scaler during scale-->
        <apply>true</apply>
        <!--Specifies the scaling method and order. Valid options are 'measurements', 'manualScale', singly or both in any sequence.-->
        <scaling_order> manualScale</scaling_order>
        <!--Specifies the measurements by which body segments are to be scaled.-->
        <MeasurementSet>
          <objects />
          <groups />
        </MeasurementSet>
        <!--Scale factors to be used for manual scaling.-->
        <ScaleSet>
          <objects>
            <Scale>
              <scales> 0.6 0.7 0.8</scales>
              <segment>pelvis</segment>
              <apply>true</apply>
            </Scale>
          </objects>
          <groups />
        </ScaleSet>
        <!--TRC file (.trc) containing the marker positions used for measurement-based scaling. This is usually a static trial, but doesn't need to be.  The marker-pair distances are computed for each time step in the TRC file and averaged across the time range.-->
        <marker_file>Unassigned</marker_file>
        <!--Time range over which to average marker-pair distances in the marker file (.trc) for measurement-based scaling.-->
        <time_range> -1 -1</time_range>
        <!--Flag (true or false) indicating whether or not to preserve relative mass between segments.-->
        <preserve_mass_distribution>false</preserve_mass_distribution>
        <!--Name of OpenSim model file (.osim) to write when done scaling.-->
        <output_model_file>Unassigned</output_model_file>
        <!--Name of file to write containing the scale factors that were applied to the unscaled model (optional).-->
        <output_scale_file>Unassigned</output_scale_file>
      </ModelScaler>
      </ScaleTool>
  </OpenSimDocument>

  */
  // clang-format on

  XMLDocument xmlDoc;
  XMLElement* openSimRoot = xmlDoc.NewElement("OpenSimDocument");
  openSimRoot->SetAttribute("Version", "40000");
  xmlDoc.InsertFirstChild(openSimRoot);

  XMLElement* scaleToolRoot = xmlDoc.NewElement("ScaleTool");
  scaleToolRoot->SetAttribute("name", subjectName.c_str());
  openSimRoot->InsertEndChild(scaleToolRoot);

  XMLElement* mass = xmlDoc.NewElement("mass");
  mass->SetText(std::to_string(massKg).c_str());
  scaleToolRoot->InsertEndChild(mass);

  XMLElement* height = xmlDoc.NewElement("height");
  height->SetText(std::to_string(heightM).c_str());
  scaleToolRoot->InsertEndChild(height);

  XMLElement* age = xmlDoc.NewElement("age");
  age->SetText(std::to_string(-1).c_str());
  scaleToolRoot->InsertEndChild(age);

  XMLElement* notes = xmlDoc.NewElement("notes");
  notes->SetText("Unassigned");
  scaleToolRoot->InsertEndChild(notes);

  XMLElement* genericModelMaker = xmlDoc.NewElement("GenericModelMaker");
  scaleToolRoot->InsertEndChild(genericModelMaker);

  XMLElement* genericModelMaker_modelFile = xmlDoc.NewElement("model_file");
  genericModelMaker_modelFile->SetText(osimInputPath.c_str());
  genericModelMaker->InsertEndChild(genericModelMaker_modelFile);

  XMLElement* genericModelMaker_markerSetFile
      = xmlDoc.NewElement("marker_set_file");
  genericModelMaker_markerSetFile->SetText(osimInputMarkersPath.c_str());
  genericModelMaker->InsertEndChild(genericModelMaker_markerSetFile);

  XMLElement* modelScaler = xmlDoc.NewElement("ModelScaler");
  scaleToolRoot->InsertEndChild(modelScaler);

  XMLElement* apply = xmlDoc.NewElement("apply");
  apply->SetText("true");
  modelScaler->InsertEndChild(apply);

  XMLElement* scalingOrder = xmlDoc.NewElement("scaling_order");
  scalingOrder->SetText(" manualScale");
  modelScaler->InsertEndChild(scalingOrder);

  XMLElement* measurementSet = xmlDoc.NewElement("MeasurementSet");
  modelScaler->InsertEndChild(measurementSet);
  XMLElement* objects = xmlDoc.NewElement("objects");
  measurementSet->InsertEndChild(objects);
  XMLElement* groups = xmlDoc.NewElement("groups");
  measurementSet->InsertEndChild(groups);

  XMLElement* scaleSet = xmlDoc.NewElement("ScaleSet");
  modelScaler->InsertEndChild(scaleSet);
  XMLElement* scaleSet_objects = xmlDoc.NewElement("objects");
  scaleSet->InsertEndChild(scaleSet_objects);

  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    XMLElement* scaleBody = xmlDoc.NewElement("Scale");
    scaleSet_objects->InsertEndChild(scaleBody);

    dynamics::BodyNode* body = skel->getBodyNode(i);

    XMLElement* scales = xmlDoc.NewElement("scales");
    // This is pretty much the whole point of this XML file - how much we'd like
    // to scale each body element
    Eigen::Vector3s scale = body->getScale();
    scales->SetText((" " + std::to_string(scale(0)) + " "
                     + std::to_string(scale(1)) + " "
                     + std::to_string(scale(2)))
                        .c_str());
    scaleBody->InsertEndChild(scales);

    XMLElement* segment = xmlDoc.NewElement("segment");
    segment->SetText(body->getName().c_str());
    scaleBody->InsertEndChild(segment);

    XMLElement* apply = xmlDoc.NewElement("apply");
    apply->SetText("true");
    scaleBody->InsertEndChild(apply);
  }

  XMLElement* markerFile = xmlDoc.NewElement("marker_file");
  markerFile->SetText("Unassigned");
  modelScaler->InsertEndChild(markerFile);

  XMLElement* timeRange = xmlDoc.NewElement("time_range");
  timeRange->SetText(" -1 1");
  modelScaler->InsertEndChild(timeRange);

  XMLElement* preserveMassDistribution
      = xmlDoc.NewElement("preserve_mass_distribution");
  preserveMassDistribution->SetText("false");
  modelScaler->InsertEndChild(preserveMassDistribution);

  XMLElement* outputModelFile = xmlDoc.NewElement("output_model_file");
  // This is the name of the output Osim file the scale tool will write when
  // done scaling
  outputModelFile->SetText(osimOutputPath.c_str());
  modelScaler->InsertEndChild(outputModelFile);

  XMLElement* outputScaleFile = xmlDoc.NewElement("output_scale_file");
  outputScaleFile->SetText("Unassigned");
  modelScaler->InsertEndChild(outputScaleFile);

  xmlDoc.SaveFile(scalingInstructionsOutputPath.c_str());
}

//==============================================================================
/// This creates an XML configuration file, which you can pass to the OpenSim
/// IK tool to recreate / validate the results of IK created from this tool
void OpenSimParser::saveOsimInverseKinematicsXMLFile(
    const std::string& subjectName,
    std::vector<std::string> markerNames,
    const std::string& osimInputModelPath,
    const std::string& osimInputTrcPath,
    const std::string& osimOutputMotPath,
    const std::string& ikInstructionsOutputPath)
{
  using namespace tinyxml2;

  // clang-format off
  /**

  Here's an example file:

  <OpenSimDocument Version="30000">
    <InverseKinemtaticsTool name="subject01">
        <!--Name of the .osim file used to construct a model.-->
        <model_file>subject01_simbody.osim</model_file>
        <!--Specify which optimizer to use (ipopt or cfsqp or jacobian).-->
        <!--Task set used to specify IK weights.-->
        <IKTaskSet name="gait2354_IK">
            <objects>
                <!-- Markers -->
                <IKMarkerTask name="Sternum"> <weight>1</weight> </IKMarkerTask>
                <IKMarkerTask name="R.Acromium"> <weight>0.5</weight></IKMarkerTask>
                <IKMarkerTask name="L.Acromium"> <weight> 0.5 </weight></IKMarkerTask>
                <IKMarkerTask name="Top.Head"> <weight> 0.1 </weight> </IKMarkerTask>
                <!-- . . additional <IKMarkerTask> tags cut for brevity . . -->
 
                <!-- Coordinates -->
                <IKCoordinateTask name="subtalar_angle_r"> <value> 0 </value></IKCoordinateTask>
                <IKCoordinateTask name="mtp_angle_r"> <value> 0 </value></IKCoordinateTask>
                <IKCoordinateTask name="subtalar_angle_l"> <value> 0 </value></IKCoordinateTask>
                <IKCoordinateTask name="mtp_angle_l"> <value> 0 </value></IKCoordinateTask>
         </objects>
        </IKTaskSet>
        <!--Parameters for solving the IK problem for each trial. Each trial
        should get a seperate SimmIKTril block.-->
        <!--TRC file (.trc) containing the time history of experimental marker
                    positions.-->
        <marker_file>subject01_walk1.trc</marker_file>
        <!--Name of file containing the joint angles used to set the initial
                    configuration of the model -->
        <coordinate_file></coordinate_file>
        <!--Time range over which the IK problem is solved.-->
        <time_range>0.4 1.60</time_range>
        <!--Name of the motion file (.mot) to which the results should be written.-->
        <output_motion_file>subject01_walk1_ik.mot</output_motion_file>
        <!--A positive scalar that is used to weight the importance of satisfying 
            constraints.A weighting of 'Infinity' or if it is unassigned results in 
            the constraints being strictly enforced.-->
        <constraint_weight>20.0</constraint_weight>
        <!--The accuracy of the solution in absolute terms. I.e. the number of significant
            digits to which the solution can be trusted.-->
        <accuracy>1e-5</accuracy>
    </InverseKinemtaticsTool>
  </OpenSimDocument>

  */
  // clang-format on

  XMLDocument xmlDoc;
  XMLElement* openSimRoot = xmlDoc.NewElement("OpenSimDocument");
  openSimRoot->SetAttribute("Version", "40000");
  xmlDoc.InsertFirstChild(openSimRoot);

  XMLElement* toolRoot = xmlDoc.NewElement("InverseKinematicsTool");
  toolRoot->SetAttribute("name", subjectName.c_str());
  openSimRoot->InsertEndChild(toolRoot);

  XMLElement* modelFile = xmlDoc.NewElement("model_file");
  modelFile->SetText(osimInputModelPath.c_str());
  toolRoot->InsertEndChild(modelFile);

  XMLElement* taskSet = xmlDoc.NewElement("IKTaskSet");
  taskSet->SetAttribute("name", "IK_tasks");
  toolRoot->InsertEndChild(taskSet);
  XMLElement* taskList = xmlDoc.NewElement("objects");
  taskSet->InsertEndChild(taskList);
  for (std::string& markerName : markerNames)
  {
    XMLElement* markerNode = xmlDoc.NewElement("IKMarkerTask");
    markerNode->SetAttribute("name", markerName.c_str());
    taskList->InsertEndChild(markerNode);
    XMLElement* markerWeight = xmlDoc.NewElement("weight");
    markerWeight->SetText("1.0");
    markerNode->InsertEndChild(markerWeight);
    XMLElement* markerApply = xmlDoc.NewElement("apply");
    markerApply->SetText("true");
    markerNode->InsertEndChild(markerApply);
  }

  XMLElement* markerFile = xmlDoc.NewElement("marker_file");
  markerFile->SetText(osimInputTrcPath.c_str());
  toolRoot->InsertEndChild(markerFile);

  XMLElement* outputMotionFile = xmlDoc.NewElement("output_motion_file");
  outputMotionFile->SetText(osimOutputMotPath.c_str());
  toolRoot->InsertEndChild(outputMotionFile);

  XMLElement* accuracy = xmlDoc.NewElement("accuracy");
  accuracy->SetText("1e-5");
  toolRoot->InsertEndChild(accuracy);

  XMLElement* reportErrors = xmlDoc.NewElement("report_errors");
  reportErrors->SetText("true");
  toolRoot->InsertEndChild(reportErrors);

  xmlDoc.SaveFile(ikInstructionsOutputPath.c_str());
}

//==============================================================================
/// This creates an XML configuration file, which you can pass to the OpenSim
/// ID tool to recreate / validate the results of ID created from this tool
void OpenSimParser::saveOsimInverseDynamicsForcesXMLFile(
    const std::string& subjectName,
    std::shared_ptr<dynamics::Skeleton> skel,
    const Eigen::MatrixXs& poses,
    const std::vector<biomechanics::ForcePlate> forcePlates,
    const std::string& grfForcesPath,
    const std::string& forcesOutputPath)
{
  using namespace tinyxml2;

  // 1. First we need to figure out which bones are the feet -- default to
  // calcaneous bones

  std::vector<std::string> footBodyNames;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    std::string bodyName = skel->getBodyNode(i)->getName();
    if (bodyName.find("calcn") != std::string::npos)
    {
      footBodyNames.push_back(bodyName);
    }
  }

  // 2. Next we need to figure out which force plates measure which foot. This
  // is done pretty heuristically, by just choosing the foot that gets closest
  // to that force plate throughout the whole trajectory.

  std::vector<int> closestBodyToForcePlate;
  std::vector<s_t> closestBodyToForcePlateDistance;

  for (int j = 0; j < forcePlates.size(); j++)
  {
    closestBodyToForcePlate.push_back(0);
    closestBodyToForcePlateDistance.push_back(
        std::numeric_limits<double>::infinity());
  }

  Eigen::VectorXs originalPos = skel->getPositions();
  for (int t = 0; t < poses.cols(); t++)
  {
    skel->setPositions(poses.col(t));

    for (int i = 0; i < footBodyNames.size(); i++)
    {
      dynamics::BodyNode* body = skel->getBodyNode(footBodyNames[i]);
      Eigen::Vector3s pos = body->getWorldTransform().translation();
      for (int j = 0; j < forcePlates.size(); j++)
      {
        s_t sumDist = 0.0;
        for (Eigen::Vector3s corner : forcePlates[j].corners)
        {
          sumDist += (pos - corner).squaredNorm();
        }

        if (sumDist < closestBodyToForcePlateDistance[j])
        {
          closestBodyToForcePlateDistance[j] = sumDist;
          closestBodyToForcePlate[j] = i;
        }
      }
    }
  }
  skel->setPositions(originalPos);

  // 3. Finally, we can write out the XML file

  // clang-format off
  /**

  Here's an example file:

  <OpenSimDocument Version="40000">
    <ExternalLoads name="DJ1">
      <objects>
        <ExternalForce name="RightGRF">
          <!--Name of the body the force is applied to.-->
          <applied_to_body>calcn_r</applied_to_body>
          <!--Name of the body the force is expressed in (default is ground).-->
          <force_expressed_in_body>ground</force_expressed_in_body>
          <!--Name of the body the point is expressed in (default is ground).-->
          <point_expressed_in_body>ground</point_expressed_in_body>
          <!--Identifier (string) to locate the force to be applied in the data source.-->
          <force_identifier>R_ground_force_v</force_identifier>
          <!--Identifier (string) to locate the point to be applied in the data source.-->
          <point_identifier>R_ground_force_p</point_identifier>
          <!--Identifier (string) to locate the torque to be applied in the data source.-->
          <torque_identifier>R_ground_torque_</torque_identifier>
        </ExternalForce>
        <ExternalForce name="LeftGRF">
          <!--Name of the body the force is applied to.-->
          <applied_to_body>calcn_l</applied_to_body>
          <!--Name of the body the force is expressed in (default is ground).-->
          <force_expressed_in_body>ground</force_expressed_in_body>
          <!--Name of the body the point is expressed in (default is ground).-->
          <point_expressed_in_body>ground</point_expressed_in_body>
          <!--Identifier (string) to locate the force to be applied in the data source.-->
          <force_identifier>L_ground_force_v</force_identifier>
          <!--Identifier (string) to locate the point to be applied in the data source.-->
          <point_identifier>L_ground_force_p</point_identifier>
          <!--Identifier (string) to locate the torque to be applied in the data source.-->
          <torque_identifier>L_ground_torque_</torque_identifier>
        </ExternalForce>
      </objects>
      <groups />
      <!--Storage file (.sto) containing (3) components of force and/or torque and point of application.Note: this file overrides the data source specified by the individual external forces if specified.-->
      <datafile>DJ1_forces.mot</datafile>
    </ExternalLoads>
  </OpenSimDocument>

  */
  // clang-format on

  XMLDocument xmlDoc;
  XMLElement* openSimRoot = xmlDoc.NewElement("OpenSimDocument");
  openSimRoot->SetAttribute("Version", "40000");
  xmlDoc.InsertFirstChild(openSimRoot);

  XMLElement* toolRoot = xmlDoc.NewElement("ExternalLoads");
  toolRoot->SetAttribute("name", subjectName.c_str());
  openSimRoot->InsertEndChild(toolRoot);

  XMLElement* loadsList = xmlDoc.NewElement("objects");
  toolRoot->InsertEndChild(loadsList);

  for (int i = 0; i < forcePlates.size(); i++)
  {
    XMLElement* forceNode = xmlDoc.NewElement("ExternalForce");
    std::string plateNumber = std::to_string(i + 1);
    forceNode->SetAttribute("name", ("ForcePlate" + plateNumber).c_str());
    loadsList->InsertEndChild(forceNode);

    XMLElement* appliedToBody = xmlDoc.NewElement("applied_to_body");
    appliedToBody->SetText(footBodyNames[closestBodyToForcePlate[i]].c_str());
    forceNode->InsertEndChild(appliedToBody);

    XMLElement* forceExpressedInBody
        = xmlDoc.NewElement("force_expressed_in_body");
    forceExpressedInBody->SetText("ground");
    forceNode->InsertEndChild(forceExpressedInBody);

    XMLElement* pointExpressedInBody
        = xmlDoc.NewElement("point_expressed_in_body");
    pointExpressedInBody->SetText("ground");
    forceNode->InsertEndChild(pointExpressedInBody);

    XMLElement* forceIdentifier = xmlDoc.NewElement("force_identifier");
    forceIdentifier->SetText(("ground_force_" + plateNumber + "_v").c_str());
    forceNode->InsertEndChild(forceIdentifier);

    XMLElement* pointIdentifier = xmlDoc.NewElement("point_identifier");
    pointIdentifier->SetText(("ground_force_" + plateNumber + "_p").c_str());
    forceNode->InsertEndChild(pointIdentifier);

    XMLElement* torqueIdentifier = xmlDoc.NewElement("torque_identifier");
    torqueIdentifier->SetText(("ground_force_" + plateNumber + "_m").c_str());
    forceNode->InsertEndChild(torqueIdentifier);
  }

  XMLElement* grfFile = xmlDoc.NewElement("datafile");
  grfFile->SetText(grfForcesPath.c_str());
  toolRoot->InsertEndChild(grfFile);

  xmlDoc.SaveFile(forcesOutputPath.c_str());
}

//==============================================================================
/// This creates an XML configuration file, which you can pass to the OpenSim
/// ID tool to recreate / validate the results of ID created from this tool
void OpenSimParser::saveOsimInverseDynamicsXMLFile(
    const std::string& subjectName,
    const std::string& osimInputModelPath,
    const std::string& osimInputMotPath,
    const std::string& osimForcesXmlPath,
    const std::string& osimOutputStoPath,
    const std::string& osimOutputBodyForcesStoPath,
    const std::string& idInstructionsOutputPath)
{
  using namespace tinyxml2;

  // clang-format off
  /**

  Here's an example file:

  <OpenSimDocument Version="40000">
    <InverseDynamicsTool name="DJ1">
      <!--Name of the .osim file used to construct a model.-->
      <model_file>LaiArnoldModified2017_poly_withArms_weldHand\LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim</model_file>
      <!--List of forces by individual or grouping name (e.g. All, actuators, muscles, ...) to be excluded when computing model dynamics. 'All' also excludes external loads added via 'external_loads_file'.-->
      <forces_to_exclude> Muscles</forces_to_exclude>
      <!--XML file (.xml) containing the external loads applied to the model as a set of ExternalForce(s).-->
      <external_loads_file>Setup_EL_DJ1.xml</external_loads_file>
      <!--The name of the file containing coordinate data. Can be a motion (.mot) or a states (.sto) file.-->
      <coordinates_file>ik.mot</coordinates_file>
      <!--Low-pass cut-off frequency for filtering the coordinates_file data (currently does not apply to states_file or speeds_file). A negative value results in no filtering. The default value is -1.0, so no filtering.-->
      <lowpass_cutoff_frequency_for_coordinates>30</lowpass_cutoff_frequency_for_coordinates>
      <!--Name of the storage file (.sto) to which the generalized forces are written.-->
      <output_gen_force_file>DJ1.sto</output_gen_force_file>
      <!--List of joints (keyword All, for all joints) to report body forces acting at the joint frame expressed in ground.-->
      <joints_to_report_body_forces />
      <!--Name of the storage file (.sto) to which the body forces at specified joints are written.-->
      <output_body_forces_file>body_forces_at_joints.sto</output_body_forces_file>
    </InverseDynamicsTool>
  </OpenSimDocument>

  */
  // clang-format on

  XMLDocument xmlDoc;
  XMLElement* openSimRoot = xmlDoc.NewElement("OpenSimDocument");
  openSimRoot->SetAttribute("Version", "40000");
  xmlDoc.InsertFirstChild(openSimRoot);

  XMLElement* toolRoot = xmlDoc.NewElement("InverseDynamicsTool");
  toolRoot->SetAttribute("name", subjectName.c_str());
  openSimRoot->InsertEndChild(toolRoot);

  XMLElement* modelFile = xmlDoc.NewElement("model_file");
  modelFile->SetText(osimInputModelPath.c_str());
  toolRoot->InsertEndChild(modelFile);

  XMLElement* forcesToExclude = xmlDoc.NewElement("forces_to_exclude");
  forcesToExclude->SetText("Muscles");
  toolRoot->InsertEndChild(forcesToExclude);

  XMLElement* externalLoadsFile = xmlDoc.NewElement("external_loads_file");
  externalLoadsFile->SetText(osimForcesXmlPath.c_str());
  toolRoot->InsertEndChild(externalLoadsFile);

  XMLElement* inputMotionFile = xmlDoc.NewElement("coordinates_file");
  inputMotionFile->SetText(osimInputMotPath.c_str());
  toolRoot->InsertEndChild(inputMotionFile);

  XMLElement* outputForceFile = xmlDoc.NewElement("output_gen_force_file");
  outputForceFile->SetText(osimOutputStoPath.c_str());
  toolRoot->InsertEndChild(outputForceFile);

  XMLElement* jointsToReportBodyForces
      = xmlDoc.NewElement("joints_to_report_body_forces");
  toolRoot->InsertEndChild(jointsToReportBodyForces);

  XMLElement* outputBodyForcesFile
      = xmlDoc.NewElement("output_body_forces_file");
  outputBodyForcesFile->SetText(osimOutputBodyForcesStoPath.c_str());
  toolRoot->InsertEndChild(outputBodyForcesFile);

  xmlDoc.SaveFile(idInstructionsOutputPath.c_str());
}

/// This gets called by rationalizeCustomJoints()
template <std::size_t Dimension>
void OpenSimParser::updateCustomJointXML(
    tinyxml2::XMLElement* element, dynamics::CustomJoint<Dimension>* joint)
{
  (void)element;
  (void)joint;
  string parent_offset_frame
      = string(element->FirstChildElement("socket_parent_frame")->GetText());
  string child_offset_frame
      = string(element->FirstChildElement("socket_child_frame")->GetText());

  // 1. Update the getTransformFromParentBodyNode() and
  // getTransformFromChildBodyNode()
  tinyxml2::XMLElement* frames = element->FirstChildElement("frames");
  tinyxml2::XMLElement* framesCursor = frames->FirstChildElement();
  while (framesCursor)
  {
    string name(framesCursor->Attribute("name"));
    Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
    if (name == parent_offset_frame)
    {
      // Update from parent
      T = joint->getTransformFromParentBodyNode();
    }
    else if (name == child_offset_frame)
    {
      // Update from child
      T = joint->getTransformFromChildBodyNode();
    }
    framesCursor->FirstChildElement("translation")
        ->SetText((std::to_string(T.translation()(0)) + " "
                   + std::to_string(T.translation()(1)) + " "
                   + std::to_string(T.translation()(2)))
                      .c_str());

    framesCursor = framesCursor->NextSiblingElement();
  }

  // 2. Update the custom functions

  tinyxml2::XMLElement* spatialTransform
      = element->FirstChildElement("SpatialTransform");
  tinyxml2::XMLElement* transformAxisCursor
      = spatialTransform->FirstChildElement("TransformAxis");
  int rawIndex = 0;
  while (transformAxisCursor)
  {
    tinyxml2::XMLElement* function
        = transformAxisCursor->FirstChildElement("function");

    int index = rawIndex;
    // For the translation coordinates, we shuffle the axis order when we
    // construct the joint, so we have to reconstruct that for edits
    if (index >= 3)
    {
      Eigen::Vector3s axis
          = readVec3(transformAxisCursor->FirstChildElement("axis"));
      if (axis == Eigen::Vector3s::UnitX())
      {
        index = 3;
      }
      else if (axis == Eigen::Vector3s::UnitY())
      {
        index = 4;
      }
      else if (axis == Eigen::Vector3s::UnitZ())
      {
        index = 5;
      }
    }

    // On v3 files, there is no "function" wrapper tag
    if (function == nullptr)
    {
      function = transformAxisCursor;
    }
    tinyxml2::XMLElement* linearFunction
        = function->FirstChildElement("LinearFunction");
    tinyxml2::XMLElement* simmSpline
        = function->FirstChildElement("SimmSpline");
    tinyxml2::XMLElement* polynomialFunction
        = function->FirstChildElement("PolynomialFunction");
    // This only exists in v4 files
    tinyxml2::XMLElement* constant = function->FirstChildElement("Constant");
    // This only exists in v3 files
    tinyxml2::XMLElement* multiplier
        = function->FirstChildElement("MultiplierFunction");
    if (multiplier != nullptr)
    {
      tinyxml2::XMLElement* childFunction
          = multiplier->FirstChildElement("function");
      assert(childFunction != nullptr);
      if (childFunction != nullptr)
      {
        constant = childFunction->FirstChildElement("Constant");
        simmSpline = childFunction->FirstChildElement("SimmSpline");
        linearFunction = childFunction->FirstChildElement("LinearFunction");
        polynomialFunction
            = childFunction->FirstChildElement("PolynomialFunction");
        assert(constant || simmSpline || linearFunction || polynomialFunction);
      }
    }

    if (constant != nullptr)
    {
      constant->FirstChildElement("value")->SetText(
          std::to_string(static_cast<math::ConstantFunction*>(
                             joint->getCustomFunction(index).get())
                             ->mValue)
              .c_str());
    }
    else if (linearFunction != nullptr)
    {
      math::LinearFunction* linear = static_cast<math::LinearFunction*>(
          joint->getCustomFunction(index).get());

      linearFunction->FirstChildElement("coefficients")
          ->SetText((std::to_string(linear->mSlope) + " "
                     + std::to_string(linear->mYIntercept))
                        .c_str());
    }
    else if (polynomialFunction != nullptr)
    {
      math::PolynomialFunction* polynomial
          = static_cast<math::PolynomialFunction*>(
              joint->getCustomFunction(index).get());

      std::string coeffString = "";
      for (int i = 0; i < polynomial->mCoeffs.size(); i++)
      {
        if (i > 0)
          coeffString += " ";
        coeffString += std::to_string(polynomial->mCoeffs[i]);
      }
      polynomialFunction->FirstChildElement("coefficients")
          ->SetText(coeffString.c_str());
    }
    else if (simmSpline != nullptr)
    {
      math::SimmSpline* spline = dynamic_cast<math::SimmSpline*>(
          joint->getCustomFunction(index).get());
      assert(spline != nullptr);
      std::string xString = "";
      for (int i = 0; i < spline->_x.size(); i++)
      {
        if (i > 0)
          xString += " ";
        xString += std::to_string(spline->_x[i]);
      }
      simmSpline->FirstChildElement("x")->SetText(xString.c_str());

      std::string yString = "";
      for (int i = 0; i < spline->_y.size(); i++)
      {
        if (i > 0)
          yString += " ";
        yString += std::to_string(spline->_y[i]);
      }
      simmSpline->FirstChildElement("y")->SetText(yString.c_str());
    }
    else
    {
      assert(false && "Unrecognized function type");
    }

    rawIndex++;
    transformAxisCursor
        = transformAxisCursor->NextSiblingElement("TransformAxis");
  }
}

/// This gets called by rationalizeJoints()
void OpenSimParser::updateRootJointLimits(
    tinyxml2::XMLElement* element, dynamics::EulerFreeJoint* joint)
{
  (void)element;
  (void)joint;
  for (int i = 0; i < 3; i++)
  {
    joint->getDof(i)->setPositionLowerLimit(-M_PI);
    joint->getDof(i)->setPositionUpperLimit(M_PI);
  }

  tinyxml2::XMLElement* coordinateCursor = nullptr;
  // This is how the coordinate set is specified in OpenSim v4 files
  tinyxml2::XMLElement* coordinateSet
      = element->FirstChildElement("CoordinateSet");
  if (coordinateSet)
  {
    tinyxml2::XMLElement* objects = coordinateSet->FirstChildElement("objects");
    if (objects)
    {
      coordinateCursor = objects->FirstChildElement("Coordinate");
    }
  }
  // This is how the coordinate set is specified in OpenSim v3 files
  if (coordinateCursor == nullptr)
  {
    coordinateSet = element->FirstChildElement("coordinates");
    if (coordinateSet != nullptr)
    {
      coordinateCursor = coordinateSet->FirstChildElement("Coordinate");
    }
  }
  // Iterate through the coordinates, and update the range values
  int dofIndex = 0;
  while (coordinateCursor)
  {
    if (dofIndex < 3)
    {
      coordinateCursor->FirstChildElement("range")->SetText(
          (std::to_string(joint->getDof(dofIndex)->getPositionLowerLimit())
           + " "
           + std::to_string(joint->getDof(dofIndex)->getPositionUpperLimit()))
              .c_str());
    }
    coordinateCursor = coordinateCursor->NextSiblingElement("Coordinate");
    dofIndex++;
  }
}

/// Read an *.osim file, move any transforms saved in Custom function
/// translation elements into the joint offsets, and write it out to a new
/// *.osim file. If there are no "irrational" CustomJoints, then this will
/// just save a copy of the original skeleton.
void OpenSimParser::rationalizeJoints(
    const common::Uri& uri,
    const std::string& outputPath,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  OpenSimFile file = parseOsim(uri);
  // This is the key call, this goes through and fixes all the custom functions
  // in-memory. The rest of this method is just translating that back to XML.
  file.skeleton->zeroTranslationInCustomFunctions();

  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument originalFile;
  try
  {
    openXMLFile(originalFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return;
  }

  //--------------------------------------------------------------------------
  // Deep copy document

  tinyxml2::XMLDocument newFile;
  originalFile.DeepCopy(&newFile);

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = newFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return;
  }

  //--------------------------------------------------------------------------
  // Go through and adjust the custom joints

  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");
  // For an Opensim 30 mode, where the JointSet is a separate element
  if (jointSet != nullptr)
  {
    tinyxml2::XMLElement* jointSetList = jointSet->FirstChildElement("objects");
    tinyxml2::XMLElement* jointCursor = jointSetList->FirstChildElement();
    while (jointCursor)
    {
      std::string name(jointCursor->Attribute("name"));
      dynamics::Joint* joint = file.skeleton->getJoint(name);

      if (joint == nullptr)
      {
        jointCursor = jointCursor->NextSiblingElement();
        continue;
      }

      if (joint->getJointIndexInSkeleton() == 0
          && joint->getType() == dynamics::EulerFreeJoint::getStaticType())
      {
        updateRootJointLimits(
            jointCursor, static_cast<dynamics::EulerFreeJoint*>(joint));
      }
      else if (joint->getType() == dynamics::CustomJoint<1>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<1>*>(joint));
      }
      else if (joint->getType() == dynamics::CustomJoint<2>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<2>*>(joint));
      }
      else if (joint->getType() == dynamics::CustomJoint<3>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<3>*>(joint));
      }

      jointCursor = jointCursor->NextSiblingElement();
    }
  }
  else
  {
    tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
    tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
    tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
    while (bodyCursor)
    {
      tinyxml2::XMLElement* jointCursor
          = bodyCursor->FirstChildElement("Joint");
      std::string name(jointCursor->Attribute("name"));
      dynamics::Joint* joint = file.skeleton->getJoint(name);
      if (joint == nullptr)
      {
        bodyCursor = bodyCursor->NextSiblingElement();
        continue;
      }

      if (joint->getJointIndexInSkeleton() == 0
          && joint->getType() == dynamics::EulerFreeJoint::getStaticType())
      {
        updateRootJointLimits(
            jointCursor, static_cast<dynamics::EulerFreeJoint*>(joint));
      }
      if (joint->getType() == dynamics::CustomJoint<1>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<1>*>(joint));
      }
      else if (joint->getType() == dynamics::CustomJoint<2>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<2>*>(joint));
      }
      else if (joint->getType() == dynamics::CustomJoint<3>::getStaticType())
      {
        updateCustomJointXML(
            jointCursor, static_cast<dynamics::CustomJoint<3>*>(joint));
      }

      bodyCursor = bodyCursor->NextSiblingElement();
    }
  }

  //--------------------------------------------------------------------------
  // Save out the result
  newFile.SaveFile(outputPath.c_str());
}

//==============================================================================
/// Read an *.osim file, rescale the body, and write it out to a new *.osim
/// file
void OpenSimParser::moveOsimMarkers(
    const common::Uri& uri,
    const std::map<std::string, Eigen::Vector3s>& bodyScales,
    const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
        markerOffsets,
    const std::string& outputPath,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  (void)bodyScales;
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument originalFile;
  try
  {
    openXMLFile(originalFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return;
  }

  //--------------------------------------------------------------------------
  // Deep copy document

  tinyxml2::XMLDocument newFile;
  originalFile.DeepCopy(&newFile);

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = newFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return;
  }

  //--------------------------------------------------------------------------
  // Go through the body nodes and adjust the scaling
  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");

  if (markerSet == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <MarkerSet> as the child of the root "
             "<Model> element.\n";
    return;
  }

  tinyxml2::XMLElement* markerSetObjects
      = markerSet->FirstChildElement("objects");
  if (markerSetObjects == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <objects> as the child of the root "
             "<MarkerSet> element.\n";
    return;
  }

  tinyxml2::XMLElement* marker = markerSetObjects->FirstChildElement("Marker");
  while (marker != nullptr)
  {
    tinyxml2::XMLElement* location = marker->FirstChildElement("location");

    std::string markerName(marker->Attribute("name"));

    // If we've got this marker in our offsets set
    if (markerOffsets.count(markerName))
    {
      std::string bodyName = markerOffsets.at(markerName).first;
      Eigen::Vector3s markerOffset = markerOffsets.at(markerName).second;
      Eigen::Vector3s bodyScale = bodyScales.count(bodyName)
                                      ? bodyScales.at(bodyName)
                                      : Eigen::Vector3s::Ones();
      markerOffset = markerOffset.cwiseProduct(bodyScale);
      location->SetText((" " + std::to_string(markerOffset(0)) + " "
                         + std::to_string(markerOffset(1)) + " "
                         + std::to_string(markerOffset(2)))
                            .c_str());
    }
    else
    {
      std::cout << "WARNING: moveOsimMarkers() found a marker in the .osim "
                   "file that isn't specified in our scalings: \""
                << markerName << "\"" << std::endl;
    }

    marker = marker->NextSiblingElement("Marker");
  }

  newFile.SaveFile(outputPath.c_str());
}

//==============================================================================
/// Read an *.osim file, then save just the markers to a new *.osim file
void OpenSimParser::filterJustMarkers(
    const common::Uri& uri,
    const std::string& outputPath,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument originalFile;
  try
  {
    openXMLFile(originalFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    std::cout << "LoadFile [" << uri.toString() << "] Fails: " << e.what()
              << std::endl;
    return;
  }

  //--------------------------------------------------------------------------
  // Deep copy document

  tinyxml2::XMLDocument newFile;
  originalFile.DeepCopy(&newFile);

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = newFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return;
  }

  //--------------------------------------------------------------------------
  // Go through the body nodes and adjust the scaling
  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");
  tinyxml2::XMLNode* cursor = modelElement->FirstChild();
  while (cursor != nullptr)
  {
    if (cursor != markerSet)
    {
      tinyxml2::XMLNode* tmpCursor = cursor;
      cursor = cursor->NextSibling();
      modelElement->DeleteChild(tmpCursor);
    }
    else
    {
      cursor = cursor->NextSibling();
    }
  }
  while (markerSet->NextSibling() != nullptr)
  {
    modelElement->DeleteChild(markerSet->NextSibling());
  }

  newFile.SaveFile(outputPath.c_str());
}

//==============================================================================
/// This grabs the marker trajectories from a TRC file
OpenSimTRC OpenSimParser::loadTRC(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimTRC result;
  const std::string content = retriever->readAll(uri);
  double unitsMultiplier = 1.0;

  std::vector<std::string> markerNames;
  Eigen::Vector3s markerSwapSpace = Eigen::Vector3s::Zero();

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    std::map<std::string, Eigen::Vector3s> markerPositions;
    double timestamp = 0.0;

    int tokenNumber = 0;
    std::string whitespace = " \t";
    auto tokenStart = line.find_first_not_of(whitespace);
    while (tokenStart != std::string::npos)
    {
      auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
      std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

      /////////////////////////////////////////////////////////
      // Process the token, given tokenNumber and lineNumber

      if (lineNumber == 2)
      {
        if (tokenNumber == 0)
        { // DataRate
          // timestep = 1.0 / atof(token.c_str());
        }
        if (tokenNumber == 4)
        { // Units
          if (token == "m")
            unitsMultiplier = 1.0;
          else if (token == "mm")
            unitsMultiplier = 1.0 / 1000;
          else
          {
            std::cout << "Unknown length units in .trc file: \"" << token
                      << "\"" << std::endl;
          }
        }
      }
      else if (lineNumber == 3 && tokenNumber > 1)
      {
        markerNames.push_back(token);
      }
      else if (lineNumber > 5)
      {
        if (tokenNumber == 1)
        {
          timestamp = atof(token.c_str());
        }
        else if (tokenNumber > 1)
        {
          int offset
              = tokenNumber - 2; // first two cols are "frame #" and "time"
          int markerNumber = (int)floor((double)offset / 3);
          int axisNumber = offset - (markerNumber * 3);
          markerSwapSpace(axisNumber) = atof(token.c_str()) * unitsMultiplier;
          if (axisNumber == 2)
          {
            if (!markerSwapSpace.hasNaN())
            {
              markerPositions[markerNames[markerNumber]]
                  = Eigen::Vector3s(markerSwapSpace);
            }
          }
        }
      }

      /////////////////////////////////////////////////////////

      tokenNumber++;
      if (tokenEnd == std::string::npos)
      {
        break;
      }
      tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
    }

    if (lineNumber > 5)
    {
      result.markerTimesteps.push_back(markerPositions);
      result.timestamps.push_back(timestamp);
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
    lineNumber++;
  }

  // Translate into a "lines" format, where each marker gets a full trajectory
  for (int i = 0; i < result.markerTimesteps.size(); i++)
  {
    // TODO: this will result in a bug if some timesteps are missing marker
    // observations
    for (auto pair : result.markerTimesteps[i])
    {
      result.markerLines[pair.first].push_back(pair.second);
    }
  }

  if (result.timestamps.size() > 1)
  {
    int frames = result.timestamps.size();
    s_t elapsed = result.timestamps[result.timestamps.size() - 1]
                  - result.timestamps[0];
    result.framesPerSecond = std::round(frames / elapsed);
  }

  return result;
}

// clang-format off
/*
PathFileType	4	(X/Y/Z)	/Volumes/Michael/Balance Metric/Balance Metric Pilot/Vicon Processed Data/S01DN6/../../OpenSim/S01DN6/Marker_data/S01DN603.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
100.000000	100.000000	3681	52	mm	100.000000	0	3681
Frame#	Time	RASI			LASI			RPSI			LPSI			RLKN			RMKN			RTH1			RTH2			RTH3			LLKN			LMKN			LTH1			LTH2			LTH3			LLAK			LMAK			LSH1			LSH2			LSH3			LTOE			LMT5			LCAL			RLAK			RMAK			RSH1			RSH2			RSH3			RTOE			RMT5			RCAL			RSHL			LSHL			CLAV			C7			RLEL			RASH			RPSH			RUA1			RUA2			RUA3			RULN			RRAD			RFA			LLEL			LASH			LPSH			LUA1			LUA2			LUA3			LULN			LRAD			LFA			
		X1	Y1	Z1	X2	Y2	Z2	X3	Y3	Z3	X4	Y4	Z4	X5	Y5	Z5	X6	Y6	Z6	X7	Y7	Z7	X8	Y8	Z8	X9	Y9	Z9	X10	Y10	Z10	X11	Y11	Z11	X12	Y12	Z12	X13	Y13	Z13	X14	Y14	Z14	X15	Y15	Z15	X16	Y16	Z16	X17	Y17	Z17	X18	Y18	Z18	X19	Y19	Z19	X20	Y20	Z20	X21	Y21	Z21	X22	Y22	Z22	X23	Y23	Z23	X24	Y24	Z24	X25	Y25	Z25	X26	Y26	Z26	X27	Y27	Z27	X28	Y28	Z28	X29	Y29	Z29	X30	Y30	Z30	X31	Y31	Z31	X32	Y32	Z32	X33	Y33	Z33	X34	Y34	Z34	X35	Y35	Z35	X36	Y36	Z36	X37	Y37	Z37	X38	Y38	Z38	X39	Y39	Z39	X40	Y40	Z40	X41	Y41	Z41	X42	Y42	Z42	X43	Y43	Z43	X44	Y44	Z44	X45	Y45	Z45	X46	Y46	Z46	X47	Y47	Z47	X48	Y48	Z48	X49	Y49	Z49	X50	Y50	Z50	X51	Y51	Z51	X52	Y52	Z52	

1	0	679.8120727539062	1002.085083007813	-995.9971923828124	420.5277099609375	1007.553588867188	-1011.751403808594	605.3575439453125	1015.321166992188	-803.5368041992186	445.6817016601562	1016.545532226562	-815.5397949218749	734.9002685546875	547.1029663085938	-877.1836547851562	607.6942138671875	516.8250732421875	-874.6161499023438	734.4181518554688	777.7933349609376	-963.7550048828125	743.39208984375	781.5364990234375	-884.7509155273438	729.3284301757812	698.0943603515626	-968.7823486328125	371.3560791015625	550.8246459960938	-877.82958984375	483.412353515625	520.704833984375	-893.31591796875	351.0415649414062	784.4949340820312	-880.312255859375	362.41796875	696.0311889648439	-963.1719360351562	355.9143981933594	699.4909057617188	-878.6652221679688	353.6072692871094	78.65764617919928	-816.2015380859375	434.0369567871094	74.43429565429693	-846.0233154296875	357.4876708984375	303.4053344726563	-787.0587158203125	355.6858520507812	297.6666564941407	-872.4854736328125	358.7340393066406	218.1407775878907	-776.3829345703125	351.4863586425781	36.82650756835943	-962.2296142578125	306.8850708007812	21.91055488586431	-916.4121704101562	402.5976257324219	31.77148628234868	-757.2083740234375	760.8834838867188	77.23938751220709	-814.3300170898438	678.1585083007812	70.47503662109381	-824.1223754882812	753.8941650390625	313.1148681640626	-878.5075073242188	762.740234375	233.6301879882813	-788.2710571289062	739.5429077148438	230.6612243652344	-873.1398315429688	749.739990234375	37.7818107604981	-953.46728515625	804.1757202148438	17.91181945800787	-911.0529174804688	723.5494995117188	28.47524642944341	-743.7086181640625	715.7427978515625	1492.004638671875	-871.6303100585936	362.2188110351562	1489.388427734375	-898.5730590820311	543.41015625	1439.690063476562	-996.8657226562499	538.708740234375	1514.868896484375	-842.0878906249999	1040.762329101562	1378.688720703125	-845.5075073242186	738.3087158203125	1422.356811523438	-947.8111572265624	721.3455810546875	1442.70947265625	-802.1453247070311	852.66650390625	1427.635620117188	-901.6832885742186	940.7133178710938	1413.813110351562	-830.8558959960936	935.8841552734375	1407.561157226562	-908.6595458984374	1058.98046875	1395.446533203125	-1129.428100585938	995.7217407226562	1385.29345703125	-1127.363403320312	991.9557495117188	1394.415161132812	-1007.243103027344	55.15938186645508	1380.868286132812	-828.5310058593749	346.5594787597656	1429.554809570312	-959.1314697265624	367.15185546875	1443.79638671875	-819.9207153320311	257.5729064941406	1414.4267578125	-806.1875610351561	235.8312835693359	1444.634399414062	-884.1889038085936	174.8842315673828	1384.784912109375	-797.9143066406249	-1.636113405227661	1385.105346679688	-1108.021362304688	55.1119384765625	1366.475219726562	-1120.221923828125	73.18951416015625	1379.125854492188	-1003.793395996094	
2	0.01	679.7892456054688	1002.118530273438	-996.0073242187499	420.5177307128906	1007.567749023438	-1011.757263183594	605.3228759765625	1015.312622070312	-803.5576171874999	445.6553955078125	1016.576293945312	-815.5219116210936	734.8585815429688	547.1071166992188	-877.3190307617188	607.898193359375	516.6675415039062	-875.0960693359375	734.3909301757812	777.8029174804689	-963.7960815429688	743.3662719726562	781.5385131835938	-884.7935791015625	729.322265625	698.1297607421876	-968.8185424804688	371.35595703125	550.8648071289062	-877.8683471679688	483.401611328125	520.7337646484375	-893.4196166992188	351.0072631835938	784.5162353515625	-880.3706665039062	362.4089660644531	696.0507812500001	-963.2354125976562	355.8748779296875	699.521728515625	-878.7264404296875	353.5990600585938	78.69834136962896	-816.1893310546875	434.0178527832031	74.46508789062506	-846.0328369140625	357.4574584960938	303.4641723632813	-787.0787963867188	355.6640319824219	297.7083740234376	-872.52783203125	358.7194213867188	218.1809387207032	-776.3993530273438	351.4712829589844	36.86539077758795	-962.2379760742188	306.8740844726562	21.94730377197271	-916.4033203125	402.5932312011719	31.79903030395512	-757.2274169921875	760.8512573242188	77.26872253417974	-814.3527221679688	678.1286010742188	70.52541351318365	-824.1339721679688	753.8585205078125	313.1050415039063	-878.5496215820312	762.7291259765625	233.6574249267579	-788.3436279296875	739.5399780273438	230.6400451660157	-873.1807861328125	749.72216796875	37.81101608276373	-953.4923095703125	804.1530151367188	17.93189811706549	-911.0638427734375	723.5054321289062	28.56663322448735	-743.6469116210938	715.6266479492188	1492.030639648438	-871.6072387695311	362.1220703125	1489.401489257812	-898.5961303710936	543.3572387695312	1439.704833984375	-996.8660278320311	538.6014404296875	1514.878662109375	-842.0782470703124	1040.680786132812	1378.760131835938	-845.4357299804686	738.2379760742188	1422.38427734375	-947.7607421874999	721.2595825195312	1442.717041015625	-802.1060791015624	852.5690307617188	1427.677734375	-901.6367187499999	940.6114501953125	1413.876220703125	-830.7924194335936	935.7893676757812	1407.611083984375	-908.5748901367186	1058.856811523438	1395.48291015625	-1129.349853515625	995.5992431640625	1385.28857421875	-1127.292114257812	991.8384399414062	1394.438720703125	-1007.162231445312	55.05607604980469	1380.807373046875	-828.6679077148436	346.4953308105469	1429.560180664062	-959.1687011718749	367.0283508300781	1443.803833007812	-819.9612426757811	257.4390563964844	1414.393920898438	-806.2971191406249	235.7202606201172	1444.59912109375	-884.2778930664061	174.7428131103516	1384.960327148438	-797.8155517578124	-1.639191508293152	1384.988525390625	-1108.180541992188	55.12150192260742	1366.379272460938	-1120.3564453125	73.15041351318359	1379.052734375	-1003.909301757812	
*/
// clang-format on

//==============================================================================
/// This saves the *.trc file from a motion for the skeleton
void OpenSimParser::saveTRC(
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const std::vector<std::map<std::string, Eigen::Vector3s>>& markerTimesteps)
{
  std::vector<std::string> markerNames;
  for (auto& pair : markerTimesteps[0])
  {
    markerNames.push_back(pair.first);
  }

  std::ofstream trcFile;
  trcFile.open(outputPath);
  trcFile << "PathFileType\t4\t(X/Y/Z)\t" << outputPath << "\n";
  trcFile << "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate"
             "\tOrigDataStartFrame\tOrigNumFrames\n";
  double rate = 1.0 / (timestamps[1] - timestamps[0]);
  trcFile << rate << "\t" << rate << "\t" << markerTimesteps.size() << "\t"
          << markerTimesteps[0].size() << "\tm\t" << rate << "\t0\t"
          << markerTimesteps.size() << "\n";

  trcFile << "Frame#\tTime";
  for (std::string name : markerNames)
  {
    trcFile << "\t" << name << "\t\t";
  }
  trcFile << "\n";

  trcFile << "\t";
  for (int i = 0; i < markerNames.size(); i++)
  {
    trcFile << "\tX" << (i + 1) << "\tY" << (i + 1) << "\tZ" << (i + 1);
  }
  trcFile << "\n\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    trcFile << (t + 1) << "\t";
    trcFile << timestamps[t];
    for (int i = 0; i < markerNames.size(); i++)
    {
      // We default to meters internally
      Eigen::Vector3s p = markerTimesteps[t].count(markerNames[i]) > 0
                              ? markerTimesteps[t].at(markerNames[i])
                              : Eigen::Vector3s::Ones() * NAN;
      trcFile << "\t" << p(0) << "\t" << p(1) << "\t" << p(2);
    }
    trcFile << "\n";
  }

  trcFile.close();
}

//==============================================================================
/// This grabs the joint angles from a *.mot file
OpenSimMot OpenSimParser::loadMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const common::Uri& uri,
    Eigen::Matrix3s rotateBy,
    int downsampleByFactor,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimTRC result;
  const std::string content = retriever->readAll(uri);
  std::vector<int> columnToDof;
  std::vector<bool> rotationalDof;

  std::vector<Eigen::VectorXs> poses;
  std::vector<double> timestamps;

  bool inHeader = true;
  bool inDegrees = false;

  int downsampleClock = 0;
  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    // Trim '\r', in case this file was saved on a Windows machine
    if (line.size() > 0 && line[line.size() - 1] == '\r')
    {
      line = line.substr(0, line.size() - 1);
    }

    if (inHeader)
    {
      std::string ENDHEADER = "endheader";
      if (line.size() >= ENDHEADER.size()
          && line.substr(0, ENDHEADER.size()) == ENDHEADER)
      {
        inHeader = false;
      }
      auto tokenEnd = line.find("=");
      if (tokenEnd != std::string::npos)
      {
        std::string variable = line.substr(0, tokenEnd);
        std::string value
            = line.substr(tokenEnd + 1, line.size() - tokenEnd - 1);
        if (variable == "inDegrees")
        {
          inDegrees = (value == "yes");
        }
      }
    }
    else
    {
      int tokenNumber = 0;
      std::string whitespace = " \t";
      auto tokenStart = line.find_first_not_of(whitespace);
      Eigen::VectorXs pose = Eigen::VectorXs::Zero(skel->getNumDofs());
      double timestamp = 0.0;
      while (tokenStart != std::string::npos)
      {
        auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
        std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

        /////////////////////////////////////////////////////////
        // Process the token, given tokenNumber and lineNumber

        if (lineNumber == 0)
        {
          if (tokenNumber > 0)
          {
            // This means we're on the row defining the names of the joints
            // we're recording positions of
            dynamics::DegreeOfFreedom* dof = skel->getDof(token);
            bool isRotationalJoint = true;
            if (dof != nullptr)
            {
              columnToDof.push_back(dof->getIndexInSkeleton());
              dynamics::Joint* joint = dof->getJoint();
              if (joint->getType()
                      == dynamics::TranslationalJoint2D::getStaticType()
                  || joint->getType()
                         == dynamics::TranslationalJoint::getStaticType()
                  || joint->getType()
                         == dynamics::PrismaticJoint::getStaticType())
              {
                isRotationalJoint = false;
              }
              if (joint->getType() == dynamics::EulerFreeJoint::getStaticType()
                  && dof->getIndexInJoint() >= 3)
              {
                isRotationalJoint = false;
              }
            }
            else
            {
              columnToDof.push_back(-1);
            }
            rotationalDof.push_back(isRotationalJoint);
          }
        }
        else
        {
          double value = atof(token.c_str());
          if (tokenNumber == 0)
          {
            timestamp = value;
          }
          else
          {
            int dofIndex = columnToDof[tokenNumber - 1];
            if (dofIndex != -1)
            {
              bool isRotationalJoint = rotationalDof[tokenNumber - 1];
              if (inDegrees && isRotationalJoint)
              {
                value *= M_PI / 180.0;
              }
              pose(dofIndex) = value;
            }
          }
        }

        /////////////////////////////////////////////////////////

        tokenNumber++;
        if (tokenEnd == std::string::npos)
        {
          break;
        }
        tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
      }

      if (lineNumber > 0)
      {
        downsampleClock--;
        if (downsampleClock <= 0)
        {
          downsampleClock = downsampleByFactor;
          poses.push_back(pose);
          timestamps.push_back(timestamp);
        }
      }
      lineNumber++;
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }
  Eigen::MatrixXs posesMatrix
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), poses.size());
  for (int i = 0; i < poses.size(); i++)
  {
    if (skel->getJoint(0)->getType()
        == dynamics::EulerFreeJoint::getStaticType())
    {
      Eigen::VectorXs ballPoses = skel->convertPositionsToBallSpace(poses[i]);

      // Rotate the orientation
      Eigen::Vector3s so3 = ballPoses.segment<3>(0);
      Eigen::Matrix3s R = math::expMapRot(so3);
      ballPoses.segment<3>(0) = math::logMap(rotateBy * R);

      // Rotate the offset
      ballPoses.segment<3>(3) = rotateBy * ballPoses.segment<3>(3);

      posesMatrix.col(i) = skel->convertPositionsFromBallSpace(ballPoses);
    }
    else
    {
      posesMatrix.col(i) = poses[i];
    }
  }
  OpenSimMot mot;
  mot.poses = posesMatrix;
  mot.timestamps = timestamps;

  return mot;
}

//==============================================================================
/// This tries a number of rotations as it's loading a .mot file, and returns
/// the one with the lowest marker error, since that's likely to be the
/// correct orientation.
OpenSimMot OpenSimParser::loadMotAtLowestMarkerRMSERotation(
    OpenSimFile& osim,
    const common::Uri& uri,
    C3D& c3d,
    int downsampleByFactor,
    const common::ResourceRetrieverPtr& retriever)
{
  Eigen::Matrix3s customOsimR = Eigen::Matrix3s::Identity();
  customOsimR.col(1) = Eigen::Vector3s::UnitZ();
  customOsimR.col(2) = -1 * Eigen::Vector3s::UnitY();

  std::vector<Eigen::Matrix3s> rotationsToTry;
  rotationsToTry.push_back(customOsimR);
  rotationsToTry.push_back(c3d.dataRotation);
  rotationsToTry.push_back(c3d.dataRotation * customOsimR);

  OpenSimMot bestMot = loadMot(
      osim.skeleton,
      uri,
      Eigen::Matrix3s::Identity(),
      downsampleByFactor,
      retriever);
  s_t bestRMSE
      = IKErrorReport(
            osim.skeleton, osim.markersMap, bestMot.poses, c3d.markerTimesteps)
            .averageRootMeanSquaredError;
  for (Eigen::Matrix3s bestRotations : rotationsToTry)
  {
    OpenSimMot otherMot = loadMot(
        osim.skeleton, uri, bestRotations, downsampleByFactor, retriever);
    s_t otherRMSE = IKErrorReport(
                        osim.skeleton,
                        osim.markersMap,
                        otherMot.poses,
                        c3d.markerTimesteps)
                        .averageRootMeanSquaredError;
    if (otherRMSE < bestRMSE)
    {
      bestMot = otherMot;
      bestRMSE = otherRMSE;
    }
  }

  return bestMot;
}

//==============================================================================
/// This saves the *.mot file from a motion for the skeleton
void OpenSimParser::saveMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const Eigen::MatrixXs& poses)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "Coordinates\n";
  motFile << "version=1\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "nColumns=" << poses.rows() + 1 << "\n";
  motFile << "inDegrees=no\n";
  motFile << "\n";
  motFile << "Units are S.I. units (second, meters, Newtons, ...)\n";
  motFile
      << "If the header above contains a line with 'inDegrees', this indicates "
         "whether rotational values are in degrees (yes) or radians (no).\n";
  motFile << "\n";
  motFile << "endheader\n";

  motFile << "time";
  for (int i = 0; i < skel->getNumDofs(); i++)
  {
    motFile << "\t" << skel->getDof(i)->getName();
  }
  motFile << "\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    motFile << timestamps[t];
    for (int i = 0; i < skel->getNumDofs(); i++)
    {
      motFile << "\t" << poses(i, t);
    }
    motFile << "\n";
  }

  motFile.close();
}

//==============================================================================
double zeroIfNan(double n)
{
  if (isnan(n))
  {
    return 0.0;
  }
  return n;
}

//==============================================================================
/// This saves the *.mot file for the ground reaction forces we've read from
/// a C3D file
void OpenSimParser::saveGRFMot(
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const std::vector<biomechanics::ForcePlate> forcePlates)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "nColumns=" << (9 * forcePlates.size()) + 1 << "\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "DataType=double\n";
  motFile << "version=3\n";
  motFile << "OpenSimVersion=4.1\n";
  motFile << "endheader\n";

  motFile << "time";
  for (int i = 0; i < forcePlates.size(); i++)
  {
    std::string num = std::to_string(i + 1);
    motFile << "\t"
            << "ground_force_" + num + "_vx";
    motFile << "\t"
            << "ground_force_" + num + "_vy";
    motFile << "\t"
            << "ground_force_" + num + "_vz";
    motFile << "\t"
            << "ground_force_" + num + "_px";
    motFile << "\t"
            << "ground_force_" + num + "_py";
    motFile << "\t"
            << "ground_force_" + num + "_pz";
    motFile << "\t"
            << "ground_force_" + num + "_mx";
    motFile << "\t"
            << "ground_force_" + num + "_my";
    motFile << "\t"
            << "ground_force_" + num + "_mz";
  }
  motFile << "\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    motFile << timestamps[t];
    for (int i = 0; i < forcePlates.size(); i++)
    {
      motFile << "\t" << zeroIfNan(forcePlates[i].forces[t](0));
      motFile << "\t" << zeroIfNan(forcePlates[i].forces[t](1));
      motFile << "\t" << zeroIfNan(forcePlates[i].forces[t](2));
      motFile << "\t" << zeroIfNan(forcePlates[i].centersOfPressure[t](0));
      motFile << "\t" << zeroIfNan(forcePlates[i].centersOfPressure[t](1));
      motFile << "\t" << zeroIfNan(forcePlates[i].centersOfPressure[t](2));
      motFile << "\t" << zeroIfNan(forcePlates[i].moments[t](0));
      motFile << "\t" << zeroIfNan(forcePlates[i].moments[t](1));
      motFile << "\t" << zeroIfNan(forcePlates[i].moments[t](2));
    }
    motFile << "\n";
  }

  motFile.close();
}

//==============================================================================
/// This saves the *.mot file with 3 columns for each body. This is
/// basically only used for verifying consistency between Nimble and OpenSim.
void OpenSimParser::saveBodyLocationsMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const Eigen::MatrixXs& poses)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "Coordinates\n";
  motFile << "version=1\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "nColumns=" << (skel->getNumBodyNodes() * 3) + 1 << "\n";
  motFile << "inDegrees=no\n";
  motFile << "\n";
  motFile << "Units are S.I. units (second, meters, Newtons, ...)\n";
  motFile
      << "If the header above contains a line with 'inDegrees', this indicates "
         "whether rotational values are in degrees (yes) or radians (no).\n";
  motFile << "\n";
  motFile << "endheader\n";

  motFile << "time";

  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    std::string bodyName = skel->getBodyNode(i)->getName();

    motFile << "\t" << bodyName << "_x\t" << bodyName << "_y\t" << bodyName
            << "_z";
  }
  motFile << "\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    motFile << timestamps[t];
    skel->setPositions(poses.col(t));

    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
      Eigen::Vector3s bodyPos
          = skel->getBodyNode(i)->getWorldTransform().translation();
      motFile << "\t" << bodyPos(0) << "\t" << bodyPos(1) << "\t" << bodyPos(2);
    }
    motFile << "\n";
  }

  motFile.close();
}

//==============================================================================
/// This saves the *.mot file with 3 columns for each marker. This is
/// basically only used for verifying consistency between Nimble and OpenSim.
void OpenSimParser::saveMarkerLocationsMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const dynamics::MarkerMap& markers,
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const Eigen::MatrixXs& poses)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "Coordinates\n";
  motFile << "version=1\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "nColumns=" << (markers.size() * 3) + 1 << "\n";
  motFile << "inDegrees=no\n";
  motFile << "\n";
  motFile << "Units are S.I. units (second, meters, Newtons, ...)\n";
  motFile
      << "If the header above contains a line with 'inDegrees', this indicates "
         "whether rotational values are in degrees (yes) or radians (no).\n";
  motFile << "\n";
  motFile << "endheader\n";

  motFile << "time";

  std::vector<std::string> markerNames;
  for (auto& pair : markers)
  {
    std::string markerName = pair.first;
    markerNames.push_back(markerName);

    motFile << "\t" << markerName << "_x\t" << markerName << "_y\t"
            << markerName << "_z";
  }
  motFile << "\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    motFile << timestamps[t];
    skel->setPositions(poses.col(t));
    std::map<std::string, Eigen::Vector3s> markerPoses
        = skel->getMarkerMapWorldPositions(markers);

    for (int i = 0; i < markerNames.size(); i++)
    {
      std::string markerName = markerNames[i];
      Eigen::Vector3s markerPos = markerPoses[markerName];
      motFile << "\t" << markerPos(0) << "\t" << markerPos(1) << "\t"
              << markerPos(2);
    }
    motFile << "\n";
  }

  motFile.close();
}

//==============================================================================
/// This grabs the GRF forces from a *.mot file
std::vector<ForcePlate> OpenSimParser::loadGRF(
    const common::Uri& uri,
    int targetFramesPerSecond,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimGRF result;
  const std::string content = retriever->readAll(uri);

  bool inHeader = true;

  std::vector<std::string> colNames;
  std::vector<int> colToPlate;
  std::vector<int> colToCOP;
  std::vector<int> colToWrench;
  int numPlates = 0;

  std::vector<s_t> timestamps;
  std::vector<std::vector<Eigen::Vector3s>> copRows;
  std::vector<std::vector<Eigen::Vector6s>> wrenchRows;

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (end != std::string::npos)
  {
    std::string line = content.substr(start, end - start);

    // Trim '\r', in case this file was saved on a Windows machine
    if (line.size() > 0 && line[line.size() - 1] == '\r')
    {
      line = line.substr(0, line.size() - 1);
    }

    if (inHeader)
    {
      std::string ENDHEADER = "endheader";
      if (line.size() >= ENDHEADER.size()
          && line.substr(0, ENDHEADER.size()) == ENDHEADER)
      {
        inHeader = false;
      }
      auto tokenEnd = line.find("=");
      if (tokenEnd != std::string::npos)
      {
        std::string variable = line.substr(0, tokenEnd);
        std::string value
            = line.substr(tokenEnd + 1, line.size() - tokenEnd - 1);
        // Currently we don't read anything from the variables
        (void)variable;
        (void)value;
      }
    }
    else
    {
      int tokenNumber = 0;
      std::string whitespace = " \t";
      auto tokenStart = line.find_first_not_of(whitespace);

      double timestamp = 0.0;
      std::vector<Eigen::Vector3s> cops;
      std::vector<Eigen::Vector6s> wrenches;

      for (int i = 0; i < numPlates; i++)
      {
        wrenches.push_back(Eigen::Vector6s::Zero());
        cops.push_back(Eigen::Vector3s::Zero());
      }

      while (tokenStart != std::string::npos)
      {
        auto tokenEnd = line.find_first_of(whitespace, tokenStart + 1);
        std::string token = line.substr(tokenStart, tokenEnd - tokenStart);

        /////////////////////////////////////////////////////////
        // Process the token, given tokenNumber and lineNumber

        if (lineNumber == 0)
        {
          colNames.push_back(token);
        }
        else
        {
          double value = atof(token.c_str());
          if (tokenNumber == 0)
          {
            timestamp = value;
          }
          else
          {
            int plateIndex = colToPlate[tokenNumber];
            int copIndex = colToCOP[tokenNumber];
            int wrenchIndex = colToWrench[tokenNumber];
            if (plateIndex != -1)
            {
              if (wrenchIndex != -1)
              {
                wrenches[plateIndex](wrenchIndex) = value;
              }
              if (copIndex != -1)
              {
                cops[plateIndex](copIndex) = value;
              }
            }
          }
        }

        /////////////////////////////////////////////////////////

        tokenNumber++;
        if (tokenEnd == std::string::npos)
        {
          break;
        }
        tokenStart = line.find_first_not_of(whitespace, tokenEnd + 1);
      }

      if (lineNumber == 0)
      {
        // Find the unique prefix/suffixes
        std::map<std::string, int> prefixSuffixNumbers;

        for (int i = 0; i < colNames.size(); i++)
        {
          const std::string& token = colNames[i];

          // Compute the names of the columns
          int plate = -1;
          int cop = -1;
          int wrench = -1;

          std::string empty = "";
          std::string prefixSuffix = token;

          if (token.find("px") != std::string::npos)
          {
            prefixSuffix.replace(token.find("px"), 2, empty);
            cop = 0;
          }
          if (token.find("py") != std::string::npos)
          {
            prefixSuffix.replace(token.find("py"), 2, empty);
            cop = 1;
          }
          if (token.find("pz") != std::string::npos)
          {
            prefixSuffix.replace(token.find("pz"), 2, empty);
            cop = 2;
          }
          if (token.find("mx") != std::string::npos)
          {
            prefixSuffix.replace(token.find("mx"), 2, empty);
            wrench = 0;
          }
          if (token.find("my") != std::string::npos)
          {
            prefixSuffix.replace(token.find("my"), 2, empty);
            wrench = 1;
          }
          if (token.find("mz") != std::string::npos)
          {
            prefixSuffix.replace(token.find("mz"), 2, empty);
            wrench = 2;
          }
          if (token.find("torque_x") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("torque_x"), std::string("torque_x").size(), empty);
            wrench = 0;
          }
          if (token.find("torque_y") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("torque_y"), std::string("torque_y").size(), empty);
            wrench = 1;
          }
          if (token.find("torque_z") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("torque_z"), std::string("torque_z").size(), empty);
            wrench = 2;
          }
          if (token.find("vx") != std::string::npos)
          {
            prefixSuffix.replace(token.find("vx"), 2, empty);
            wrench = 3;
          }
          if (token.find("vy") != std::string::npos)
          {
            prefixSuffix.replace(token.find("vy"), 2, empty);
            wrench = 4;
          }
          if (token.find("vz") != std::string::npos)
          {
            prefixSuffix.replace(token.find("vz"), 2, empty);
            wrench = 5;
          }

          if (prefixSuffix.find("force") != std::string::npos)
          {
            prefixSuffix.replace(prefixSuffix.find("force"), 5, empty);
          }
          if (prefixSuffix.find("moment") != std::string::npos)
          {
            prefixSuffix.replace(prefixSuffix.find("moment"), 6, empty);
          }
          if (prefixSuffix.find("torque") != std::string::npos)
          {
            prefixSuffix.replace(prefixSuffix.find("torque"), 6, empty);
          }

          while (prefixSuffix.find("__") != std::string::npos)
          {
            prefixSuffix.replace(prefixSuffix.find("__"), 2, std::string("_"));
          }

          if (token == "time")
          {
            // Default to plate 0
            plate = 0;
          }
          else if (prefixSuffixNumbers.count(prefixSuffix) > 0)
          {
            plate = prefixSuffixNumbers.at(prefixSuffix);
          }
          else
          {
            std::cout << "Reading new GRF column prefixSuffix: " << prefixSuffix
                      << std::endl;
            plate = prefixSuffixNumbers.size();
            prefixSuffixNumbers[prefixSuffix] = plate;
          }

          colToPlate.push_back(plate);
          colToCOP.push_back(cop);
          colToWrench.push_back(wrench);
        }

        numPlates = prefixSuffixNumbers.size();
      }
      else
      {
        copRows.push_back(cops);
        wrenchRows.push_back(wrenches);
        timestamps.push_back(timestamp);
      }
      lineNumber++;
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }

  assert(timestamps.size() == copRows.size());
  assert(timestamps.size() == wrenchRows.size());

  int downsampleByFactor = 1;
  if (timestamps.size() > 1)
  {
    int frames = timestamps.size();
    s_t elapsed = timestamps[timestamps.size() - 1] - timestamps[0];
    int framesPerSecond = std::round(frames / elapsed);
    if (framesPerSecond < targetFramesPerSecond)
    {
      std::cout << "WARNING!!! OpenSimParser is trying to load "
                   "ground-reaction-force data from "
                << uri.toString()
                << ", but the requested target frames per second ("
                << targetFramesPerSecond
                << ", probably to match a corresponding .trc file) is HIGHER "
                   "than the file's native frames per second ("
                << framesPerSecond
                << "). We don't yet support up-sampling GRF data, so this will "
                   "result in mismatched data!"
                << std::endl;
    }
    else
    {
      downsampleByFactor = framesPerSecond / targetFramesPerSecond;
    }
  }

  // Process result into its final form

  std::vector<ForcePlate> forcePlates;
  for (int i = 0; i < numPlates; i++)
  {
    forcePlates.emplace_back();
    ForcePlate& forcePlate = forcePlates[forcePlates.size() - 1];

    int downsampleClock = 0;
    Eigen::Vector3s copAvg = Eigen::Vector3s::Zero();
    Eigen::Vector6s wrenchAvg = Eigen::Vector6s::Zero();
    int numAveraged = 0;
    int cursor = 0;
    for (int t = 0; t < timestamps.size(); t++)
    {
      copAvg += copRows[t][i];
      wrenchAvg += wrenchRows[t][i];
      numAveraged++;
      downsampleClock--;

      if (downsampleClock <= 0)
      {
        downsampleClock = downsampleByFactor;
        forcePlate.centersOfPressure.push_back(copAvg / numAveraged);
        forcePlate.moments.push_back(wrenchAvg.segment<3>(0) / numAveraged);
        forcePlate.forces.push_back(wrenchAvg.segment<3>(3) / numAveraged);
        cursor++;

        numAveraged = 0;
        copAvg.setZero();
        wrenchAvg.setZero();
      }
    }
  }

  return forcePlates;
}

template <std::size_t Dimension>
std::pair<dynamics::CustomJoint<Dimension>*, dynamics::BodyNode*>
createCustomJoint(
    dynamics::SkeletonPtr skel,
    std::string jointName,
    dynamics::BodyNode::Properties bodyProps,
    dynamics::BodyNode* parentBody,
    std::vector<std::shared_ptr<math::CustomFunction>> customFunctions,
    std::vector<int> drivenByDofs,
    std::vector<Eigen::Vector3s> eulerAxisOrder,
    std::vector<Eigen::Vector3s> transformAxisOrder)
{
  dynamics::CustomJoint<Dimension>* customJoint = nullptr;
  dynamics::BodyNode* childBody = nullptr;
  // Create a CustomJoint
  typename dynamics::CustomJoint<Dimension>::Properties props;
  props.mName = jointName;
  if (parentBody == nullptr)
  {
    auto pair
        = skel->createJointAndBodyNodePair<dynamics::CustomJoint<Dimension>>(
            nullptr, props, bodyProps);
    customJoint = pair.first;
    childBody = pair.second;
  }
  else
  {
    auto pair = parentBody->createChildJointAndBodyNodePair<
        dynamics::CustomJoint<Dimension>>(props, bodyProps);
    customJoint = pair.first;
    childBody = pair.second;
  }

  assert(customFunctions.size() == 6);

  dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
  Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);
  customJoint->setAxisOrder(axisOrder);
  customJoint->setFlipAxisMap(flips);

#ifndef NDEBUG
  std::vector<bool> setFunction;
  for (int i = 0; i < 6; i++)
  {
    setFunction.push_back(false);
  }
#endif

  for (int i = 0; i < customFunctions.size(); i++)
  {
    if (i < 3)
    {
      customJoint->setCustomFunction(i, customFunctions[i], drivenByDofs[i]);
#ifndef NDEBUG
      setFunction[i] = true;
#endif
    }
    else
    {
      // Map to the appropriate slot based on the axis
      Eigen::Vector3s axis = transformAxisOrder[i - 3];
      if (axis == Eigen::Vector3s::UnitX())
      {
        customJoint->setCustomFunction(3, customFunctions[i], drivenByDofs[i]);
#ifndef NDEBUG
        setFunction[3] = true;
#endif
      }
      else if (axis == Eigen::Vector3s::UnitY())
      {
        customJoint->setCustomFunction(4, customFunctions[i], drivenByDofs[i]);
#ifndef NDEBUG
        setFunction[4] = true;
#endif
      }
      else if (axis == Eigen::Vector3s::UnitZ())
      {
        customJoint->setCustomFunction(5, customFunctions[i], drivenByDofs[i]);
#ifndef NDEBUG
        setFunction[5] = true;
#endif
      }
      else
      {
        assert(false);
      }
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < 6; i++)
  {
    assert(setFunction[i]);
  }
#endif

  return std::make_pair(customJoint, childBody);
}

//==============================================================================
/// When people finish preparing their model in OpenSim, they save a *.osim
/// file with all the scales and offsets baked in. This is a utility to go
/// through and get out the scales and offsets in terms of a standard
/// skeleton, so that we can include their values in standard datasets.
OpenSimScaleAndMarkerOffsets OpenSimParser::getScaleAndMarkerOffsets(
    const OpenSimFile& standardSkeleton, const OpenSimFile& scaledSkeleton)
{
  OpenSimScaleAndMarkerOffsets config;

  // Check that the two skeletons are compatible
  for (int i = 0; i < standardSkeleton.skeleton->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* bodyNode = standardSkeleton.skeleton->getBodyNode(i);
    if (!scaledSkeleton.skeleton->getBodyNode(bodyNode->getName()))
    {
      std::cout << "OpenSimParser::getConfiguration() failed because the "
                   "skeletons were too different! The standard skeleton has a "
                   "body named \""
                << bodyNode->getName() << "\", but the scaled skeleton doesn't."
                << std::endl;
      config.success = false;
      return config;
    }
  }

  // Now go through both skeletons and try to work out what the body scales are

  config.bodyScales
      = Eigen::VectorXs::Ones(standardSkeleton.skeleton->getNumBodyNodes() * 3);
  for (int i = 0; i < standardSkeleton.skeleton->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* scaledNode = scaledSkeleton.skeleton->getBodyNode(
        standardSkeleton.skeleton->getBodyNode(i)->getName());
    if (scaledNode->getNumShapeNodes() > 0)
    {
      dynamics::Shape* shape = scaledNode->getShapeNode(0)->getShape().get();
      if (shape->getType() == dynamics::MeshShape::getStaticType())
      {
        dynamics::MeshShape* mesh = static_cast<dynamics::MeshShape*>(shape);
        Eigen::Vector3s scale = mesh->getScale();
        config.bodyScales.segment<3>(i * 3) = scale;
      }
    }
  }

  // Try to convert the markers into the standard skeleton

  for (auto pair : scaledSkeleton.markersMap)
  {
    std::string markerName = pair.first;
    dynamics::BodyNode* standardBody
        = standardSkeleton.skeleton->getBodyNode(pair.second.first->getName());
    if (standardBody != nullptr)
    {
      Eigen::Vector3s bodyScale = config.bodyScales.segment<3>(
          standardBody->getIndexInSkeleton() * 3);
      Eigen::Vector3s goldOffset = pair.second.second.cwiseQuotient(bodyScale);
      config.markers[markerName] = std::make_pair(standardBody, goldOffset);
      config.markerOffsets[markerName] = goldOffset - pair.second.second;
    }
  }

  // Return

  config.success = true;
  return config;
}

//==============================================================================
std::pair<dynamics::Joint*, dynamics::BodyNode*> createJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    tinyxml2::XMLElement* bodyCursor,
    tinyxml2::XMLElement* jointDetail,
    Eigen::Isometry3s transformFromParent,
    Eigen::Isometry3s transformFromChild,
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever)
{
  std::string bodyName(bodyCursor->Attribute("name"));
  dynamics::BodyNode::Properties bodyProps;
  bodyProps.mName = bodyName;

  dynamics::BodyNode* childBody = nullptr;
  std::string jointName(jointDetail->Attribute("name"));

  dynamics::Joint* joint = nullptr;
  std::string jointType(jointDetail->Name());

  std::vector<std::string> dofNames;

  tinyxml2::XMLElement* coordinateCursor = nullptr;
  // This is how the coordinate set is specified in OpenSim v4 files
  tinyxml2::XMLElement* coordinateSet
      = jointDetail->FirstChildElement("CoordinateSet");
  if (coordinateSet)
  {
    tinyxml2::XMLElement* objects = coordinateSet->FirstChildElement("objects");
    if (objects)
    {
      coordinateCursor = objects->FirstChildElement("Coordinate");
    }
  }
  // This is how the coordinate set is specified in OpenSim v3 files
  if (coordinateCursor == nullptr)
  {
    coordinateSet = jointDetail->FirstChildElement("coordinates");
    if (coordinateSet != nullptr)
    {
      coordinateCursor = coordinateSet->FirstChildElement("Coordinate");
    }
  }
  // Iterate through the coordinates
  while (coordinateCursor)
  {
    std::string dofName(coordinateCursor->Attribute("name"));
    dofNames.push_back(dofName);
    coordinateCursor = coordinateCursor->NextSiblingElement("Coordinate");
  }

  // Build custom joints
  bool isCustomJoint = false;
  if (jointType == "CustomJoint")
  {
    tinyxml2::XMLElement* spatialTransform
        = jointDetail->FirstChildElement("SpatialTransform");
    tinyxml2::XMLElement* transformAxisCursor
        = spatialTransform->FirstChildElement("TransformAxis");

    std::vector<std::shared_ptr<math::CustomFunction>> customFunctions;
    std::vector<int> drivenByDofs;
    std::vector<Eigen::Vector3s> eulerAxisOrder;
    std::vector<Eigen::Vector3s> transformAxisOrder;

    int numLinear = 0;
    int numConstant = 0;
    int lastLinearIndex = 0;

    int dofIndex = 0;
    /// If all linear, then we're just a EulerFreeJoint
    bool allLinear = true;
    /// If first 3 are linear, last 3 are constant, then we're just an
    /// EulerJoint
    bool first3Linear = true;
    /// If any are splines, we need a full blown CustomJoint
    bool anySpline = false;
    /// If all 6 DOFs are fixed at 0, then we're just a WeldJoint
    bool allLocked = true;
    while (transformAxisCursor)
    {
      Eigen::Vector3s axis
          = readVec3(transformAxisCursor->FirstChildElement("axis"));

      tinyxml2::XMLElement* function
          = transformAxisCursor->FirstChildElement("function");
      // On v3 files, there is no "function" wrapper tag
      if (function == nullptr)
      {
        function = transformAxisCursor;
      }

      int drivenByDof = 0;
      tinyxml2::XMLElement* coordinates
          = transformAxisCursor->FirstChildElement("coordinates");
      if (coordinates != nullptr)
      {
        const char* text_pointer = coordinates->GetText();
        if (text_pointer != nullptr)
        {
          std::string coordinateName(text_pointer);
          auto iterator
              = std::find(dofNames.begin(), dofNames.end(), coordinateName);
          if (iterator != dofNames.end())
          {
            drivenByDof = iterator - dofNames.begin();
          }
        }
      }
      drivenByDofs.push_back(drivenByDof);

      tinyxml2::XMLElement* linearFunction
          = function->FirstChildElement("LinearFunction");
      tinyxml2::XMLElement* simmSpline
          = function->FirstChildElement("SimmSpline");
      tinyxml2::XMLElement* polynomialFunction
          = function->FirstChildElement("PolynomialFunction");
      // This only exists in v4 files
      tinyxml2::XMLElement* constant = function->FirstChildElement("Constant");
      // This only exists in v3 files
      tinyxml2::XMLElement* multiplier
          = function->FirstChildElement("MultiplierFunction");
      s_t scale = 1.0;
      if (multiplier != nullptr)
      {
        tinyxml2::XMLElement* childFunction
            = multiplier->FirstChildElement("function");
        scale = atof(multiplier->FirstChildElement("scale")->GetText());
        assert(childFunction != nullptr);
        if (childFunction != nullptr)
        {
          constant = childFunction->FirstChildElement("Constant");
          simmSpline = childFunction->FirstChildElement("SimmSpline");
          linearFunction = childFunction->FirstChildElement("LinearFunction");
          polynomialFunction
              = childFunction->FirstChildElement("PolynomialFunction");
          assert(
              constant || simmSpline || linearFunction || polynomialFunction);
        }
      }

      if (constant != nullptr)
      {
        numConstant++;
        allLinear = false;
        if (dofIndex < 3)
        {
          first3Linear = false;
        }

        double value
            = atof(constant->FirstChildElement("value")->GetText()) * scale;
        if (value != 0)
        {
          allLocked = false;
        }
        customFunctions.push_back(
            std::make_shared<math::ConstantFunction>(value));
      }
      else if (linearFunction != nullptr)
      {
        numLinear++;
        lastLinearIndex = dofIndex;
        allLocked = false;
        Eigen::Vector2s coeffs
            = readVec2(linearFunction->FirstChildElement("coefficients"));
        // Bake coeff flips into the axis
        if (coeffs(0) == -1)
        {
          axis *= coeffs(0);
          coeffs(0) = 1.0;
        }
        // Example coeffs for linear: 1 0
        customFunctions.push_back(std::make_shared<math::LinearFunction>(
            coeffs(0) * scale, coeffs(1) * scale));
      }
      else if (polynomialFunction != nullptr)
      {
        anySpline = true;
        allLocked = false;
        std::vector<s_t> coeffs
            = readVecX(polynomialFunction->FirstChildElement("coefficients"));
        for (int i = 0; i < coeffs.size(); i++)
        {
          coeffs[i] *= scale;
        }
        customFunctions.push_back(
            std::make_shared<math::PolynomialFunction>(coeffs));
      }
      else if (simmSpline != nullptr)
      {
        anySpline = true;
        allLocked = false;
        if (dofIndex < 3)
        {
          first3Linear = false;
        }

        std::vector<s_t> x = readVecX(simmSpline->FirstChildElement("x"));
        std::vector<s_t> y = readVecX(simmSpline->FirstChildElement("y"));
        for (int i = 0; i < y.size(); i++)
        {
          y[i] *= scale;
        }
        customFunctions.push_back(std::make_shared<math::SimmSpline>(x, y));
      }
      else
      {
        assert(false && "Unrecognized function type");
      }

      if (dofIndex < 3)
        eulerAxisOrder.push_back(axis);
      else
        transformAxisOrder.push_back(axis);

      dofIndex++;
      transformAxisCursor
          = transformAxisCursor->NextSiblingElement("TransformAxis");
    }

    if (allLocked)
    {
      dynamics::WeldJoint::Properties props;
      props.mName = jointName;
      auto pair
          = parentBody->createChildJointAndBodyNodePair<dynamics::WeldJoint>(
              props, bodyProps);
      joint = pair.first;
      childBody = pair.second;
      std::cout << "WARNING! Creating a WeldJoint as an intermediate "
                   "(non-root) joint. This will cause the gradient "
                   "computations to run with slower algorithms. If you find a "
                   "way to remove this WeldJoint, things should run faster."
                << std::endl;
    }
    else if (anySpline)
    {
      isCustomJoint = true;
      if (dofNames.size() == 1)
      {
        auto pair = createCustomJoint<1>(
            skel,
            jointName,
            bodyProps,
            parentBody,
            customFunctions,
            drivenByDofs,
            eulerAxisOrder,
            transformAxisOrder);
        joint = pair.first;
        childBody = pair.second;
      }
      else if (dofNames.size() == 2)
      {
        auto pair = createCustomJoint<2>(
            skel,
            jointName,
            bodyProps,
            parentBody,
            customFunctions,
            drivenByDofs,
            eulerAxisOrder,
            transformAxisOrder);
        joint = pair.first;
        childBody = pair.second;
      }
      else if (dofNames.size() == 3)
      {
        auto pair = createCustomJoint<3>(
            skel,
            jointName,
            bodyProps,
            parentBody,
            customFunctions,
            drivenByDofs,
            eulerAxisOrder,
            transformAxisOrder);
        joint = pair.first;
        childBody = pair.second;
      }
      else
      {
        assert(false && "Unsupported number of DOFs in CustomJoint");
      }
    }
    else if (allLinear)
    {
      dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
      dynamics::EulerJoint::AxisOrder transOrder
          = getAxisOrder(transformAxisOrder);
      (void)transOrder;
      assert(transOrder == dynamics::EulerJoint::AxisOrder::XYZ);

      Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);

      // Create a EulerFreeJoint
      dynamics::EulerFreeJoint* eulerFreeJoint = nullptr;
      dynamics::EulerFreeJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>(
            nullptr, props, bodyProps);
        eulerFreeJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody
                  ->createChildJointAndBodyNodePair<dynamics::EulerFreeJoint>(
                      props, bodyProps);
        eulerFreeJoint = pair.first;
        childBody = pair.second;
      }
      eulerFreeJoint->setAxisOrder(axisOrder);
      eulerFreeJoint->setFlipAxisMap(flips);
      joint = eulerFreeJoint;
    }
    else if (first3Linear)
    {
      dynamics::EulerJoint::AxisOrder axisOrder = getAxisOrder(eulerAxisOrder);
      Eigen::Vector3s flips = getAxisFlips(eulerAxisOrder);
      // assert(!flips[0] && !flips[1] && !flips[2]);

      // Create an EulerJoint
      dynamics::EulerJoint* eulerJoint = nullptr;
      dynamics::EulerJoint::Properties props;
      props.mName = jointName;
      if (parentBody == nullptr)
      {
        auto pair = skel->createJointAndBodyNodePair<dynamics::EulerJoint>(
            nullptr, props, bodyProps);
        eulerJoint = pair.first;
        childBody = pair.second;
      }
      else
      {
        auto pair
            = parentBody->createChildJointAndBodyNodePair<dynamics::EulerJoint>(
                props, bodyProps);
        eulerJoint = pair.first;
        childBody = pair.second;
      }
      eulerJoint->setFlipAxisMap(flips);
      eulerJoint->setAxisOrder(axisOrder);
      joint = eulerJoint;
    }
    else if (numLinear == 1 && numConstant == 5)
    {
      if (lastLinearIndex < 3)
      {
        Eigen::Vector3s axis = eulerAxisOrder[lastLinearIndex];

        // Create a RevoluteJoint
        dynamics::RevoluteJoint* revoluteJoint = nullptr;
        dynamics::RevoluteJoint::Properties props;
        props.mName = jointName;
        if (parentBody == nullptr)
        {
          auto pair = skel->createJointAndBodyNodePair<dynamics::RevoluteJoint>(
              nullptr, props, bodyProps);
          revoluteJoint = pair.first;
          childBody = pair.second;
        }
        else
        {
          auto pair
              = parentBody
                    ->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>(
                        props, bodyProps);
          revoluteJoint = pair.first;
          childBody = pair.second;
        }
        revoluteJoint->setAxis(axis);
        joint = revoluteJoint;
      }
      else
      {
        Eigen::Vector3s axis = transformAxisOrder[lastLinearIndex - 3];

        // Create a PrismaticJoint
        dynamics::PrismaticJoint* prismaticJoint = nullptr;
        dynamics::PrismaticJoint::Properties props;
        props.mName = jointName;
        if (parentBody == nullptr)
        {
          auto pair
              = skel->createJointAndBodyNodePair<dynamics::PrismaticJoint>(
                  nullptr, props, bodyProps);
          prismaticJoint = pair.first;
          childBody = pair.second;
        }
        else
        {
          auto pair
              = parentBody
                    ->createChildJointAndBodyNodePair<dynamics::PrismaticJoint>(
                        props, bodyProps);
          prismaticJoint = pair.first;
          childBody = pair.second;
        }
        prismaticJoint->setAxis(axis);
        joint = prismaticJoint;
      }
    }
    else
    {
      assert(false);
    }
  }
  if (jointType == "WeldJoint")
  {
    dynamics::WeldJoint::Properties props;
    props.mName = jointName;
    auto pair
        = parentBody->createChildJointAndBodyNodePair<dynamics::WeldJoint>(
            props, bodyProps);
    joint = pair.first;
    childBody = pair.second;
    std::cout << "WARNING! Creating a WeldJoint as an intermediate "
                 "(non-root) joint. This will cause the gradient "
                 "computations to run with slower algorithms. If you find a "
                 "way to remove this WeldJoint, things should run faster."
              << std::endl;
  }
  if (jointType == "PinJoint")
  {
    // Create a RevoluteJoint
    dynamics::RevoluteJoint* revoluteJoint = nullptr;
    dynamics::RevoluteJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<dynamics::RevoluteJoint>(
          nullptr, props, bodyProps);
      revoluteJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair
          = parentBody
                ->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>(
                    props, bodyProps);
      revoluteJoint = pair.first;
      childBody = pair.second;
    }
    joint = revoluteJoint;
  }
  if (jointType == "UniversalJoint")
  {
    // Create a UniversalJoint
    dynamics::UniversalJoint* universalJoint = nullptr;
    dynamics::UniversalJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<dynamics::UniversalJoint>(
          nullptr, props, bodyProps);
      universalJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair
          = parentBody
                ->createChildJointAndBodyNodePair<dynamics::UniversalJoint>(
                    props, bodyProps);
      universalJoint = pair.first;
      childBody = pair.second;
    }
    joint = universalJoint;
  }
  assert(childBody != nullptr);
  joint->setName(jointName);
  // std::cout << jointName << std::endl;
  joint->setTransformFromChildBodyNode(transformFromChild);
  joint->setTransformFromParentBodyNode(transformFromParent);

  // Rename the DOFs for each joint

  coordinateCursor = nullptr;
  // This is how the coordinate set is specified in OpenSim v4 files
  coordinateSet = jointDetail->FirstChildElement("CoordinateSet");
  if (coordinateSet)
  {
    tinyxml2::XMLElement* objects = coordinateSet->FirstChildElement("objects");
    if (objects)
    {
      coordinateCursor = objects->FirstChildElement("Coordinate");
    }
  }
  // This is how the coordinate set is specified in OpenSim v3 files
  if (coordinateCursor == nullptr)
  {
    coordinateSet = jointDetail->FirstChildElement("coordinates");
    if (coordinateSet != nullptr)
    {
      coordinateCursor = coordinateSet->FirstChildElement("Coordinate");
    }
  }
  // Iterate through the coordinates
  int i = 0;
  while (coordinateCursor)
  {
    std::string dofName(coordinateCursor->Attribute("name"));
    double defaultValue
        = atof(coordinateCursor->FirstChildElement("default_value")->GetText());
    double defaultSpeedValue = atof(
        coordinateCursor->FirstChildElement("default_speed_value")->GetText());
    Eigen::Vector2s range
        = readVec2(coordinateCursor->FirstChildElement("range"));
    bool locked
        = std::string(coordinateCursor->FirstChildElement("locked")->GetText())
          == "true";
    bool clamped
        = std::string(coordinateCursor->FirstChildElement("clamped")->GetText())
          == "true";

    dynamics::DegreeOfFreedom* dof = joint->getDof(i);
    dof->setName(dofName);
    dof->setPosition(defaultValue);
    dof->setVelocity(defaultSpeedValue);
    // Always lock a custom joint
    if (clamped || true)
    {
      if (!clamped && isCustomJoint)
      {
        std::cout
            << "DOF \"" << dofName
            << "\" drives a custom function, yet is not listed as "
               "clamped. This is unsupported, so we're defaulting to clamped"
            << std::endl;
      }
      dof->setPositionLowerLimit(range(0));
      dof->setPositionUpperLimit(range(1));
    }
    if (locked)
    {
      // TODO: Just replace with a Weld joint
      /*
      dof->setVelocityUpperLimit(0);
      dof->setVelocityLowerLimit(0);
      */
      dof->setPositionLowerLimit(defaultValue);
      dof->setPositionUpperLimit(defaultValue);
    }

    i++;
    coordinateCursor = coordinateCursor->NextSiblingElement("Coordinate");
  }

  // OpenSim v4 files specify visible geometry this way
  tinyxml2::XMLElement* visibleObject
      = bodyCursor->FirstChildElement("VisibleObject");
  if (visibleObject && childBody != nullptr)
  {
    tinyxml2::XMLElement* geometrySet
        = visibleObject->FirstChildElement("GeometrySet");
    if (geometrySet)
    {
      tinyxml2::XMLElement* objects = geometrySet->FirstChildElement("objects");
      if (objects)
      {
        tinyxml2::XMLElement* displayGeometryCursor
            = objects->FirstChildElement("DisplayGeometry");
        while (displayGeometryCursor)
        {
          std::string mesh_file(
              displayGeometryCursor->FirstChildElement("geometry_file")
                  ->GetText());
          Eigen::Vector3s colors
              = readVec3(displayGeometryCursor->FirstChildElement("color"))
                * 0.7;
          Eigen::Vector6s transformVec
              = readVec6(displayGeometryCursor->FirstChildElement("transform"));
          Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
          transform.linear() = math::eulerXYZToMatrix(transformVec.head<3>());
          transform.translation() = transformVec.tail<3>();
          Eigen::Vector3s scale = readVec3(
              displayGeometryCursor->FirstChildElement("scale_factors"));
          double opacity = atof(
              displayGeometryCursor->FirstChildElement("opacity")->GetText());

          common::Uri meshUri = common::Uri::createFromRelativeUri(
              uri, "./Geometry/" + mesh_file + ".ply");
          std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr
              = dynamics::MeshShape::loadMesh(meshUri, retriever);

          if (meshPtr)
          {
            std::shared_ptr<dynamics::MeshShape> meshShape
                = std::make_shared<dynamics::MeshShape>(
                    scale, meshPtr, meshUri, retriever);

            dynamics::ShapeNode* meshShapeNode
                = childBody->createShapeNodeWith<dynamics::VisualAspect>(
                    meshShape);
            meshShapeNode->setRelativeTransform(transform);

            dynamics::VisualAspect* meshVisualAspect
                = meshShapeNode->getVisualAspect();
            meshVisualAspect->setColor(colors);
            meshVisualAspect->setAlpha(opacity);
          }

          displayGeometryCursor
              = displayGeometryCursor->NextSiblingElement("DisplayGeometry");
        }
      }
    }
  }
  // OpenSim v3 files specify visible geometry this way
  tinyxml2::XMLElement* attachedGeometry
      = bodyCursor->FirstChildElement("attached_geometry");
  if (attachedGeometry && childBody != nullptr)
  {
    tinyxml2::XMLElement* meshCursor
        = attachedGeometry->FirstChildElement("Mesh");
    while (meshCursor)
    {
      if (meshCursor->FirstChildElement("mesh_file")->GetText() == nullptr)
      {
        std::cout << "Body Node " << bodyName
                  << " has an attached <Mesh> object where <mesh_file> is "
                     "empty. Ignoring."
                  << std::endl;
        meshCursor = meshCursor->NextSiblingElement("Mesh");
        continue;
      }

      std::string mesh_file(
          meshCursor->FirstChildElement("mesh_file")->GetText());
      Eigen::Vector3s scale
          = readVec3(meshCursor->FirstChildElement("scale_factors"));

      common::Uri meshUri = common::Uri::createFromRelativeUri(
          uri, "./Geometry/" + mesh_file + ".ply");
      std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr
          = dynamics::MeshShape::loadMesh(meshUri, retriever);

      if (meshPtr)
      {
        std::shared_ptr<dynamics::MeshShape> meshShape
            = std::make_shared<dynamics::MeshShape>(
                scale, meshPtr, meshUri, retriever);

        dynamics::ShapeNode* meshShapeNode
            = childBody->createShapeNodeWith<dynamics::VisualAspect>(meshShape);

        /*
        Eigen::Vector6s transformVec
            = readVec6(displayGeometryCursor->FirstChildElement("transform"));
        Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
        transform.linear() = math::eulerXYZToMatrix(transformVec.head<3>());
        transform.translation() = transformVec.tail<3>();
        meshShapeNode->setRelativeTransform(transform);
        */

        dynamics::VisualAspect* meshVisualAspect
            = meshShapeNode->getVisualAspect();

        tinyxml2::XMLElement* appearance
            = meshCursor->FirstChildElement("Appearance");
        if (appearance != nullptr)
        {
          Eigen::Vector3s colors
              = readVec3(appearance->FirstChildElement("color")) * 0.7;
          double opacity
              = atof(appearance->FirstChildElement("opacity")->GetText());
          meshVisualAspect->setColor(colors);
          meshVisualAspect->setAlpha(opacity);
        }
      }

      meshCursor = meshCursor->NextSiblingElement("Mesh");
    }
  }

  assert(childBody != nullptr);

  return std::pair<dynamics::Joint*, dynamics::BodyNode*>(joint, childBody);
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim40(
    const common::Uri& uri,
    tinyxml2::XMLElement* docElement,
    const common::ResourceRetrieverPtr& retriever)
{
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");

  if (bodySet == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] missing <BodySet> group.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  //--------------------------------------------------------------------------
  // Build out the physical structure
  unordered_map<string, dynamics::BodyNode*> bodyLookupMap;
  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    std::string name(bodyCursor->Attribute("name"));
    // std::cout << name << std::endl;

    if (name == "ground")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }
    // Skip the kneecaps
    if (name == "patella_r" || name == "patella_l")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }

    tinyxml2::XMLElement* joint = bodyCursor->FirstChildElement("Joint");

    dynamics::BodyNode* childBody = nullptr;

    if (joint)
    {
      // WeldJoint
      if (joint->FirstChild() == nullptr)
      {
      }
      tinyxml2::XMLElement* jointDetail = nullptr;

      // CustomJoint
      tinyxml2::XMLElement* customJoint
          = joint->FirstChildElement("CustomJoint");
      if (customJoint)
      {
        jointDetail = customJoint;
      }
      // PinJoint
      tinyxml2::XMLElement* pinJoint = joint->FirstChildElement("PinJoint");
      if (pinJoint)
      {
        jointDetail = pinJoint;
      }
      // UniversalJoint
      tinyxml2::XMLElement* universalJoint
          = joint->FirstChildElement("UniversalJoint");
      if (universalJoint)
      {
        jointDetail = universalJoint;
      }

      if (jointDetail != nullptr)
      {
        std::string parentName = std::string(
            jointDetail->FirstChildElement("parent_body")->GetText());
        dynamics::BodyNode* parentBody = bodyLookupMap[parentName];
        assert(parentName == "ground" || parentBody != nullptr);
        // Get shared properties across all joint types
        Eigen::Vector3s locationInParent
            = readVec3(jointDetail->FirstChildElement("location_in_parent"));
        Eigen::Vector3s orientationInParent
            = readVec3(jointDetail->FirstChildElement("orientation_in_parent"));

        Eigen::Vector3s locationInChild
            = readVec3(jointDetail->FirstChildElement("location"));
        Eigen::Vector3s orientationInChild
            = readVec3(jointDetail->FirstChildElement("orientation"));

        Eigen::Isometry3s transformFromParent = Eigen::Isometry3s::Identity();
        transformFromParent.linear()
            = math::eulerXYZToMatrix(orientationInParent);
        transformFromParent.translation() = locationInParent;
        Eigen::Isometry3s transformFromChild = Eigen::Isometry3s::Identity();
        transformFromChild.linear()
            = math::eulerXYZToMatrix(orientationInChild);
        transformFromChild.translation() = locationInChild;

        auto pair = createJoint(
            skel,
            parentBody,
            bodyCursor,
            jointDetail,
            transformFromParent,
            transformFromChild,
            uri,
            retriever);
        childBody = pair.second;
      }
    }
    double mass = atof(bodyCursor->FirstChildElement("mass")->GetText());
    Eigen::Vector3s massCenter
        = readVec3(bodyCursor->FirstChildElement("mass_center"));
    double inertia_xx
        = atof(bodyCursor->FirstChildElement("inertia_xx")->GetText());
    double inertia_yy
        = atof(bodyCursor->FirstChildElement("inertia_yy")->GetText());
    double inertia_zz
        = atof(bodyCursor->FirstChildElement("inertia_zz")->GetText());
    double inertia_xy
        = atof(bodyCursor->FirstChildElement("inertia_xy")->GetText());
    double inertia_xz
        = atof(bodyCursor->FirstChildElement("inertia_xz")->GetText());
    double inertia_yz
        = atof(bodyCursor->FirstChildElement("inertia_yz")->GetText());
    dynamics::Inertia inertia(
        mass,
        massCenter(0),
        massCenter(1),
        massCenter(2),
        inertia_xx,
        inertia_yy,
        inertia_zz,
        inertia_xy,
        inertia_xz,
        inertia_yz);
    childBody->setInertia(inertia);

    childBody->setName(name);
    bodyLookupMap[name] = childBody;

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  /*
  std::cout << "Num dofs: " << skel->getNumDofs() << std::endl;
  std::cout << "Num bodies: " << skel->getNumBodyNodes() << std::endl;
  */

  OpenSimFile file;
  file.skeleton = skel;

  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");
  if (markerSet != nullptr)
  {
    tinyxml2::XMLElement* markerList = markerSet->FirstChildElement("objects");
    if (markerList != nullptr)
    {
      tinyxml2::XMLElement* markerCursor
          = markerList->FirstChildElement("Marker");
      while (markerCursor)
      {
        std::string name(markerCursor->Attribute("name"));
        Eigen::Vector3s offset
            = readVec3(markerCursor->FirstChildElement("location"));
        std::string bodyName
            = markerCursor->FirstChildElement("body")->GetText();
        bool fixed
            = std::string(markerCursor->FirstChildElement("fixed")->GetText())
              == "true";

        dynamics::BodyNode* body = skel->getBodyNode(bodyName);
        if (body != nullptr)
        {
          file.markersMap[name] = std::make_pair(body, offset);
          if (fixed)
          {
            file.anatomicalMarkers.push_back(name);
          }
          else
          {
            file.trackingMarkers.push_back(name);
          }
        }
        else
        {
          std::cout << "Warning: OpenSimParser attempting to read marker \""
                    << name << "\" attached to body \"" << bodyName
                    << "\" which does not exist! Marker will be ignored."
                    << std::endl;
        }

        markerCursor = markerCursor->NextSiblingElement();
      }
    }
  }

  return file;
}

struct OpenSimBodyXML;

struct OpenSimJointXML
{
  string name;
  string type;
  OpenSimBodyXML* parent;
  OpenSimBodyXML* child;
  Eigen::Isometry3s fromParent;
  Eigen::Isometry3s fromChild;
  tinyxml2::XMLElement* xml;
  dynamics::BodyNode* parentBody;
};

struct OpenSimBodyXML
{
  string name;
  OpenSimJointXML* parent;
  std::vector<OpenSimJointXML*> children;
  tinyxml2::XMLElement* xml;
};

//==============================================================================
void recursiveCreateJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    OpenSimJointXML* joint,
    const common::Uri& uri,
    const common::ResourceRetrieverPtr& retriever)
{
  (void)skel;
  (void)parentBody;
  (void)joint;

  // std::cout << "Building joint: " << joint->name << std::endl;

  tinyxml2::XMLElement* jointNode = joint->xml;
  tinyxml2::XMLElement* bodyNode = joint->child->xml;

  auto pair = createJoint(
      skel,
      parentBody,
      bodyNode,
      jointNode,
      joint->fromParent,
      joint->fromChild,
      uri,
      retriever);

  dynamics::BodyNode* childBody = pair.second;

  double mass = atof(bodyNode->FirstChildElement("mass")->GetText());
  Eigen::Vector3s massCenter
      = readVec3(bodyNode->FirstChildElement("mass_center"));
  Eigen::Vector6s inertia = readVec6(bodyNode->FirstChildElement("inertia"));
  if (mass <= 0)
  {
    std::cout << "WARNING! We're refusing to set a 0 mass for "
              << childBody->getName()
              << ", because NimblePhysics does not support 0 masses. "
                 "Defaulting to 0.0001."
              << std::endl;
    mass = 0.0001;
  }
  if (inertia.segment<3>(0).norm() == 0)
  {
    inertia.segment<3>(0).setConstant(0.0001);
  }

  dynamics::Inertia inertiaObj(
      mass,
      massCenter(0),
      massCenter(1),
      massCenter(2),
      inertia(0),
      inertia(1),
      inertia(2),
      inertia(3),
      inertia(4),
      inertia(5));
  childBody->setInertia(inertiaObj);

  // Recurse to the next layer
  for (OpenSimJointXML* grandChildJoint : joint->child->children)
  {
    recursiveCreateJoint(skel, childBody, grandChildJoint, uri, retriever);
  }
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim30(
    const common::Uri& uri,
    tinyxml2::XMLElement* docElement,
    const common::ResourceRetrieverPtr& retriever)
{
  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  (void)retriever;
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (bodySet == nullptr || jointSet == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] missing <BodySet> or <JointSet> groups.\n";
    return null_file;
  }

  //--------------------------------------------------------------------------
  // Read BodySet
  unordered_map<string, OpenSimBodyXML> bodyLookupMap;

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    std::string name(bodyCursor->Attribute("name"));
    // std::cout << name << std::endl;

    if (name == "patella_r" || name == "patella_l")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }

    bodyLookupMap["/bodyset/" + name].xml = bodyCursor;
    bodyLookupMap["/bodyset/" + name].parent = nullptr;
    bodyLookupMap["/bodyset/" + name].children.clear();
    bodyLookupMap["/bodyset/" + name].name = name;

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Read JointSet
  unordered_map<string, OpenSimJointXML> jointLookupMap;

  tinyxml2::XMLElement* jointSetList = jointSet->FirstChildElement("objects");
  tinyxml2::XMLElement* jointCursor = jointSetList->FirstChildElement();
  while (jointCursor)
  {
    std::string type(jointCursor->Name());
    std::string name(jointCursor->Attribute("name"));

    string parent_offset_frame = string(
        jointCursor->FirstChildElement("socket_parent_frame")->GetText());
    string child_offset_frame = string(
        jointCursor->FirstChildElement("socket_child_frame")->GetText());

    tinyxml2::XMLElement* frames = jointCursor->FirstChildElement("frames");
    tinyxml2::XMLElement* framesCursor = frames->FirstChildElement();

    string parentName = "";
    Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
    string childName = "";
    Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();

    while (framesCursor)
    {
      string parent_body
          = string(framesCursor->FirstChildElement("socket_parent")->GetText());
      Eigen::Vector3s translation
          = readVec3(framesCursor->FirstChildElement("translation"));
      Eigen::Vector3s rotationXYZ
          = readVec3(framesCursor->FirstChildElement("orientation"));
      Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
      T.linear() = math::eulerXYZToMatrix(rotationXYZ);
      T.translation() = translation;

      string name(framesCursor->Attribute("name"));
      if (name == parent_offset_frame)
      {
        parentName = parent_body;
        fromParent = T;
      }
      else if (name == child_offset_frame)
      {
        childName = parent_body;
        fromChild = T;
      }

      framesCursor = framesCursor->NextSiblingElement();
    }

    if (name == "patellofemoral_r" || name == "patellofemoral_l"
        || childName == "/bodyset/patella_r"
        || childName == "/bodyset/patella_l")
    {
      jointCursor = jointCursor->NextSiblingElement();
      continue;
    }

    OpenSimJointXML& joint = jointLookupMap[name];
    joint.name = name;
    joint.type = type;
    joint.fromParent = fromParent;
    joint.fromChild = fromChild;
    joint.xml = jointCursor;

    // A body can have only one parent joint
    if (bodyLookupMap.find(parentName) != bodyLookupMap.end())
    {
      bodyLookupMap[parentName].children.push_back(&joint);
      joint.parent = &bodyLookupMap[parentName];
    }
    else
    {
      joint.parent = nullptr;
    }

    if (bodyLookupMap.find(childName) == bodyLookupMap.end())
    {
      std::cout << "ERROR loading *.osim file: Joint " << name
                << " has child body \"" << childName
                << "\", yet the child was not found in the <BodySet>"
                << std::endl;
      exit(1);
    }
    assert(bodyLookupMap.find(childName) != bodyLookupMap.end());
    assert(bodyLookupMap[childName].parent == nullptr);
    joint.child = &bodyLookupMap[childName];
    bodyLookupMap[childName].parent = &joint;

    jointCursor = jointCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Check tree invarients

  OpenSimJointXML* root = nullptr;
  for (auto& pair : jointLookupMap)
  {
    if (pair.second.parent == nullptr)
    {
      assert(root == nullptr);
      root = &pair.second;
    }
  }

  assert(root != nullptr);

  //--------------------------------------------------------------------------
  // Build out the physical structure

  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();
  root->parentBody = nullptr;
  recursiveCreateJoint(skel, nullptr, root, uri, retriever);

  /*
  std::cout << "Num dofs: " << skel->getNumDofs() << std::endl;
  std::cout << "Num bodies: " << skel->getNumBodyNodes() << std::endl;
  */

  OpenSimFile file;
  file.skeleton = skel;

  tinyxml2::XMLElement* markerSet
      = modelElement->FirstChildElement("MarkerSet");
  if (markerSet != nullptr)
  {
    tinyxml2::XMLElement* markerList = markerSet->FirstChildElement("objects");
    if (markerList != nullptr)
    {
      tinyxml2::XMLElement* markerCursor
          = markerList->FirstChildElement("Marker");
      while (markerCursor)
      {
        std::string name(markerCursor->Attribute("name"));
        Eigen::Vector3s offset
            = readVec3(markerCursor->FirstChildElement("location"));
        std::string socketName
            = markerCursor->FirstChildElement("socket_parent_frame")->GetText();
        std::string bodyName = bodyLookupMap[socketName].name;

        tinyxml2::XMLElement* fixedElem
            = markerCursor->FirstChildElement("fixed");
        bool fixed = fixedElem == nullptr
                     || std::string(fixedElem->GetText()) == "true";
        dynamics::BodyNode* body = skel->getBodyNode(bodyName);

        if (body != nullptr)
        {
          file.markersMap[name]
              = std::make_pair(skel->getBodyNode(bodyName), offset);
          if (fixed)
          {
            file.anatomicalMarkers.push_back(name);
          }
          else
          {
            file.trackingMarkers.push_back(name);
          }
        }
        else
        {
          std::cout << "Warning: OpenSimParser attempting to read marker \""
                    << name << "\" attached to body \"" << bodyName
                    << "\" which does not exist! Marker will be ignored."
                    << std::endl;
        }

        markerCursor = markerCursor->NextSiblingElement();
      }
    }
  }

  return file;
}

}; // namespace biomechanics
}; // namespace dart