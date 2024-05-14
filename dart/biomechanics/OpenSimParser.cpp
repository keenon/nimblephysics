#include "dart/biomechanics/OpenSimParser.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tinyxml2.h>

#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/macros.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/ConstantCurveIncompressibleJoint.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/EllipsoidJoint.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Inertia.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ScapulathoracicJoint.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/TranslationalJoint2D.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/ConstantFunction.hpp"
#include "dart/math/CustomFunction.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/PiecewiseLinearFunction.hpp"
#include "dart/math/PolynomialFunction.hpp"
#include "dart/math/SimmSpline.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/MJCFExporter.hpp"
#include "dart/utils/StringUtils.hpp"
#include "dart/utils/XmlHelpers.hpp"
#include "dart/utils/sdf/SdfParser.hpp"

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

std::string to_string(double d)
{
  // Create an output string stream
  std::ostringstream ss;
  // Set Fixed -Point Notation
  ss << std::fixed;
  // Set precision to 10 digits
  ss << std::setprecision(10);
  // Add double to stream
  ss << d;
  // Get string from output string stream
  return ss.str();
}

// https://www.geeksforgeeks.org/round-the-given-number-to-nearest-multiple-of-10/
int roundToNearestMultiple(int n, int multiple)
{
  int a = (n / multiple) * multiple;
  int b = a + multiple;
  return (n - a > b - n) ? b : a;
}

bool endsWith(std::string const& fullString, std::string const& ending)
{
  if (fullString.length() >= ending.length())
  {
    return (
        0
        == fullString.compare(
            fullString.length() - ending.length(), ending.length(), ending));
  }
  else
  {
    return false;
  }
}

bool beginsWith(std::string const& fullString, std::string const& beginning)
{
  if (fullString.length() >= beginning.length())
  {
    return (0 == fullString.compare(0, beginning.length(), beginning));
  }
  else
  {
    return false;
  }
}

template <typename T>
int findIndex(const std::vector<T>& vec, const T& value)
{
  auto it = std::find(vec.begin(), vec.end(), value);

  if (it != vec.end())
  {
    return std::distance(vec.begin(), it);
  }
  else
  {
    return -1;
  }
}

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

std::string writeVec3(Eigen::Vector3s vec)
{
  return to_string((double)vec(0)) + " " + to_string((double)vec(1)) + " "
         + to_string((double)vec(2));
}

std::string writeVec6(Eigen::Vector6s vec)
{
  return to_string((double)vec(0)) + " " + to_string((double)vec(1)) + " "
         + to_string((double)vec(2)) + " " + to_string((double)vec(3)) + " "
         + to_string((double)vec(4)) + " " + to_string((double)vec(5));
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

std::pair<dynamics::EulerJoint::AxisOrder, bool> getAxisOrder(
    std::vector<Eigen::Vector3s> axisList)
{
  if (axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitZ())
  {
    return std::make_pair<dynamics::EulerJoint::AxisOrder, bool>(
        dynamics::EulerJoint::AxisOrder::XYZ, false);
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitY()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitX())
  {
    return std::make_pair<dynamics::EulerJoint::AxisOrder, bool>(
        dynamics::EulerJoint::AxisOrder::ZYX, false);
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return std::make_pair<dynamics::EulerJoint::AxisOrder, bool>(
        dynamics::EulerJoint::AxisOrder::ZXY, false);
  }
  else if (
      axisList[0].cwiseAbs() == Eigen::Vector3s::UnitX()
      && axisList[1].cwiseAbs() == Eigen::Vector3s::UnitZ()
      && axisList[2].cwiseAbs() == Eigen::Vector3s::UnitY())
  {
    return std::make_pair<dynamics::EulerJoint::AxisOrder, bool>(
        dynamics::EulerJoint::AxisOrder::XZY, false);
  }
  // don't break the build when building as prod
  return std::make_pair<dynamics::EulerJoint::AxisOrder, bool>(
      dynamics::EulerJoint::AxisOrder::XYZ, true);
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
    const common::Uri& uri,
    std::string geometryFolder,
    bool ignoreGeometry,
    const common::ResourceRetrieverPtr& nullOrRetriever)
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
  return parseOsim(
      osimFile,
      uri.toString(),
      geometryFolder == "" && !ignoreGeometry
          ? common::Uri::createFromRelativeUri(uri.toString(), "./Geometry/")
                .toString()
          : geometryFolder,
      ignoreGeometry,
      retriever);
}

//==============================================================================
OpenSimFile OpenSimParser::parseOsim(
    tinyxml2::XMLDocument& osimFile,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    bool ignoreGeometry,
    const common::ResourceRetrieverPtr& nullOrGeometryRetriever)
{
  const common::ResourceRetrieverPtr geometryRetriever
      = ensureRetriever(nullOrGeometryRetriever);

  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement
      = osimFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  OpenSimFile result;
  if (jointSet != nullptr)
  {
    // This is the older format, where JointSet specifies the joints separately
    // from the body hierarchy
    result = readOsim40(
        docElement,
        fileNameForErrorDisplay,
        geometryFolder,
        geometryRetriever,
        ignoreGeometry);
  }
  else
  {
    // This is the newer format, where Joints are specified as childen of Bodies
    result = readOsim30(
        docElement,
        fileNameForErrorDisplay,
        geometryFolder,
        geometryRetriever,
        ignoreGeometry);
  }

  tinyxml2::XMLElement* constraintSet
      = modelElement->FirstChildElement("ConstraintSet");
  if (constraintSet != nullptr)
  {
    tinyxml2::XMLElement* objectsList
        = constraintSet->FirstChildElement("objects");
    if (objectsList != nullptr)
    {
      constraintSet = objectsList;
    }

    tinyxml2::XMLElement* couplerConstraintCursor
        = objectsList->FirstChildElement("CoordinateCouplerConstraint");
    while (couplerConstraintCursor != nullptr)
    {
      std::string name(couplerConstraintCursor->Attribute("name"));
      tinyxml2::XMLElement* independentCoordinates
          = couplerConstraintCursor->FirstChildElement(
              "independent_coordinate_names");
      tinyxml2::XMLElement* dependentCoordinate
          = couplerConstraintCursor->FirstChildElement(
              "dependent_coordinate_name");
      if (independentCoordinates != nullptr && dependentCoordinate != nullptr)
      {
        std::string independent(independentCoordinates->GetText());
        std::string dependent(dependentCoordinate->GetText());

        // We can special case the patella, and only the patella, because it's
        // so tiny
        if ((independent == "knee_angle_r" && dependent == "knee_angle_r_beta")
            || (independent == "knee_angle_l" && dependent == "knee_angle_l"))
        {
          result.jointsDrivenBy.emplace_back(dependent, independent);
        }
        else
        {
          result.warnings.push_back(
              "Constraints are not supported by AddBiomechanics. Ignoring "
              "CoordinateCouplerConstraint \""
              + name + "\"");
        }
      }
      else
      {
        result.warnings.push_back(
            "Constraints are not supported by AddBiomechanics. Ignoring "
            "CoordinateCouplerConstraint \""
            + name + "\"");
      }
      couplerConstraintCursor = couplerConstraintCursor->NextSiblingElement(
          "CoordinateCouplerConstraint");
    }
  }

  return result;
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
      <MarkerPlacer>
        <apply>false</apply>
      </MarkerPlacer>
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
  mass->SetText(to_string(massKg).c_str());
  scaleToolRoot->InsertEndChild(mass);

  XMLElement* height = xmlDoc.NewElement("height");
  height->SetText(to_string(heightM).c_str());
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

  XMLElement* markerPlacer = xmlDoc.NewElement("MarkerPlacer");
  scaleToolRoot->InsertEndChild(markerPlacer);

  XMLElement* markerPlacerApply = xmlDoc.NewElement("apply");
  markerPlacerApply->SetText("false");
  markerPlacer->InsertEndChild(markerPlacerApply);

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
    scales->SetText((" " + to_string((double)scale(0)) + " "
                     + to_string((double)scale(1)) + " "
                     + to_string((double)scale(2)))
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
void OpenSimParser::saveOsimInverseDynamicsRawForcesXMLFile(
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
    if (bodyName.find("calcn") != std::string::npos
        || bodyName.find("Foot") != std::string::npos
        || bodyName.find("foot") != std::string::npos
        || bodyName.find("talus") != std::string::npos)
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
void OpenSimParser::saveOsimInverseDynamicsProcessedForcesXMLFile(
    const std::string& subjectName,
    const std::vector<dynamics::BodyNode*> contactBodies,
    const std::string& grfForcesPath,
    const std::string& forcesOutputPath)
{
  using namespace tinyxml2;

  // We can write out the XML file

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

  for (int i = 0; i < contactBodies.size(); i++)
  {
    XMLElement* forceNode = xmlDoc.NewElement("ExternalForce");
    std::string plateNumber = contactBodies[i]->getName();
    forceNode->SetAttribute("name", ("ForcePlate_" + plateNumber).c_str());
    loadsList->InsertEndChild(forceNode);

    XMLElement* appliedToBody = xmlDoc.NewElement("applied_to_body");
    appliedToBody->SetText(contactBodies[i]->getName().c_str());
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
    const std::string& idInstructionsOutputPath,
    const s_t startTime,
    const s_t endTime)
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
      <!--The time range to consider for inverse dynamics, in seconds-->
      <time_range> 0 1.3</time_range>
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

  XMLElement* timeRange = xmlDoc.NewElement("time_range");
  timeRange->SetText(
      (" " + std::to_string(startTime) + " " + std::to_string(endTime))
          .c_str());
  toolRoot->InsertEndChild(timeRange);

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
  string parent_offset_frame;
  if (element->FirstChildElement("socket_parent_frame"))
  {
    parent_offset_frame
        = string(element->FirstChildElement("socket_parent_frame")->GetText());
  }
  else if (element->FirstChildElement("socket_parent_frame_connectee_name"))
  {
    parent_offset_frame = string(
        element->FirstChildElement("socket_parent_frame_connectee_name")
            ->GetText());
  }
  else
  {
    std::cout << "OpenSimParser encountered an error! Joint \""
              << joint->getName()
              << "\" does not specify either a <socket_parent_frame> or a "
                 "<socket_parent_frame_connectee_name>! This may be because "
                 "the OpenSim file is in an older and unsupported version. "
                 "Try a newer format."
              << std::endl;
    return;
  }
  string child_offset_frame;
  if (element->FirstChildElement("socket_child_frame"))
  {
    child_offset_frame
        = string(element->FirstChildElement("socket_child_frame")->GetText());
  }
  else if (element->FirstChildElement("socket_child_frame_connectee_name"))
  {
    child_offset_frame
        = string(element->FirstChildElement("socket_child_frame_connectee_name")
                     ->GetText());
  }
  else
  {
    std::cout << "OpenSimParser encountered an error! Joint \""
              << joint->getName()
              << "\" does not specify either a <socket_child_frame> or a "
                 "<socket_child_frame_connectee_name>! This may be because "
                 "the OpenSim file is particularly old. Try a newer format."
              << std::endl;
    return;
  }

  // 1. Update the getTransformFromParentBodyNode() and
  // getTransformFromChildBodyNode()
  tinyxml2::XMLElement* frames = element->FirstChildElement("frames");
  if (frames)
  {
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
          ->SetText((to_string((double)T.translation()(0)) + " "
                     + to_string((double)T.translation()(1)) + " "
                     + to_string((double)T.translation()(2)))
                        .c_str());

      framesCursor = framesCursor->NextSiblingElement();
    }
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
    tinyxml2::XMLElement* piecewiseLinear
        = function->FirstChildElement("PiecewiseLinearFunction");
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
        piecewiseLinear
            = function->FirstChildElement("PiecewiseLinearFunction");
        linearFunction = childFunction->FirstChildElement("LinearFunction");
        polynomialFunction
            = childFunction->FirstChildElement("PolynomialFunction");
        assert(
            constant || simmSpline || piecewiseLinear || linearFunction
            || polynomialFunction);
      }
    }

    if (constant != nullptr)
    {
      constant->FirstChildElement("value")->SetText(
          to_string((double)(static_cast<math::ConstantFunction*>(
                                 joint->getCustomFunction(index).get())
                                 ->mValue))
              .c_str());
    }
    else if (linearFunction != nullptr)
    {
      math::LinearFunction* linear = static_cast<math::LinearFunction*>(
          joint->getCustomFunction(index).get());

      linearFunction->FirstChildElement("coefficients")
          ->SetText((to_string((double)linear->mSlope) + " "
                     + to_string((double)linear->mYIntercept))
                        .c_str());
    }
    else if (polynomialFunction != nullptr)
    {
      math::PolynomialFunction* polynomial
          = static_cast<math::PolynomialFunction*>(
              joint->getCustomFunction(index).get());

      std::string coeffString = "";
      for (int i = polynomial->mCoeffs.size() - 1; i >= 0; --i)
      {
        if (i < polynomial->mCoeffs.size() - 1)
          coeffString += " ";
        coeffString += to_string((double)polynomial->mCoeffs[i]);
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
        xString += to_string((double)spline->_x[i]);
      }
      simmSpline->FirstChildElement("x")->SetText(xString.c_str());

      std::string yString = "";
      for (int i = 0; i < spline->_y.size(); i++)
      {
        if (i > 0)
          yString += " ";
        yString += to_string((double)spline->_y[i]);
      }
      simmSpline->FirstChildElement("y")->SetText(yString.c_str());
    }
    else if (piecewiseLinear != nullptr)
    {
      math::PiecewiseLinearFunction* pl
          = dynamic_cast<math::PiecewiseLinearFunction*>(
              joint->getCustomFunction(index).get());
      assert(pl != nullptr);
      std::string xString = "";
      for (int i = 0; i < pl->_x.size(); i++)
      {
        if (i > 0)
          xString += " ";
        xString += to_string((double)pl->_x[i]);
      }
      piecewiseLinear->FirstChildElement("x")->SetText(xString.c_str());

      std::string yString = "";
      for (int i = 0; i < pl->_y.size(); i++)
      {
        if (i > 0)
          yString += " ";
        yString += to_string((double)pl->_y[i]);
      }
      piecewiseLinear->FirstChildElement("y")->SetText(yString.c_str());
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
          (to_string((double)joint->getDof(dofIndex)->getPositionLowerLimit())
           + " "
           + to_string(
               (double)joint->getDof(dofIndex)->getPositionUpperLimit()))
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

      std::string name;
      if (jointCursor->FirstChildElement() != nullptr)
      {
        name = std::string(jointCursor->FirstChildElement()->Attribute("name"));
      }

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

/// Read an *.osim file, overwrite all the markers, and write it out
/// to a new *.osim file
void OpenSimParser::replaceOsimMarkers(
    const common::Uri& uri,
    const std::map<std::string, std::pair<std::string, Eigen::Vector3s>>&
        markers,
    const std::map<std::string, bool> isAnatomical,
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
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");
  bool isOldFormat = jointSet != nullptr;

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

  markerSetObjects->DeleteChildren();

  for (auto& pair : markers)
  {
    tinyxml2::XMLElement* marker
        = markerSetObjects->InsertNewChildElement("Marker");
    /*
            <Marker name="RACR">
                <!--Body segment in the model on which the marker resides.-->
                <body>torso</body>
                <!--Location of a marker on the body segment.-->
                <location> -0.003000 0.425000 0.130000</location>
                <!--Flag (true or false) specifying whether or not a marker
       should be kept fixed in the marker placement step.  i.e. If false, the
       marker is allowed to move.--> <fixed>false</fixed>
            </Marker>
    */
    marker->SetAttribute("name", pair.first.c_str());

    tinyxml2::XMLElement* body = marker->InsertNewChildElement(
        isOldFormat ? "socket_parent_frame" : "body");
    std::string bodyName
        = isOldFormat ? "/bodyset/" + pair.second.first : pair.second.first;
    body->SetText(bodyName.c_str());

    tinyxml2::XMLElement* location = marker->InsertNewChildElement("location");
    Eigen::Vector3s markerOffset = pair.second.second;
    location->SetText((" " + to_string((double)markerOffset(0)) + " "
                       + to_string((double)markerOffset(1)) + " "
                       + to_string((double)markerOffset(2)))
                          .c_str());

    tinyxml2::XMLElement* fixed = marker->InsertNewChildElement("fixed");
    fixed->SetText(
        (isAnatomical.count(pair.first) > 0 && isAnatomical.at(pair.first))
            ? "true"
            : "false");
  }

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
      location->SetText((" " + to_string((double)markerOffset(0)) + " "
                         + to_string((double)markerOffset(1)) + " "
                         + to_string((double)markerOffset(2)))
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
/// Read an original *.osim file (which contains a marker set), and a target
/// *.osim file, and translate the markers from the original to the target,
/// and write it out to a new *.osim file
///
/// This method returns a pair of lists, (guessedMarkers, missingMarkers).
/// The guessedMarkers array contains the markers which were placed on the
/// target model using heuristics. The missingMarkers array contains the
/// markers which were not placed on the target model, because they could not
/// be matched to any body even after heuristics were applied (this can often
/// happen because the markers are on the arms, but the target model has no
/// arms, for example). These should be reviewed by a person, to verify that
/// the results look reasonable.
std::pair<std::vector<std::string>, std::vector<std::string>>
OpenSimParser::translateOsimMarkers(
    const common::Uri& originalModelPath,
    const common::Uri& targetModelPath,
    const std::string& outputPath,
    bool verbose)
{
  OpenSimFile originalModel = OpenSimParser::parseOsim(originalModelPath);
  OpenSimFile targetModel = OpenSimParser::parseOsim(targetModelPath);

  originalModel.skeleton->setPositions(
      Eigen::VectorXs::Zero(originalModel.skeleton->getNumDofs()));
  targetModel.skeleton->setPositions(
      Eigen::VectorXs::Zero(targetModel.skeleton->getNumDofs()));

  // Only scale the models to match height if they either both have a torso, or
  // both do not
  if (hasTorso(originalModel.skeleton) == hasTorso(targetModel.skeleton))
  {
    // 0. Scale the original model to match the target model height
    s_t originalHeight = originalModel.skeleton->getHeight(
        originalModel.skeleton->getPositions());
    s_t targetHeight
        = targetModel.skeleton->getHeight(targetModel.skeleton->getPositions());
    s_t scaleOriginalModel = (targetHeight / originalHeight);
    if (verbose)
    {
      std::cout << "Original model height: " << originalHeight << "m"
                << std::endl;
      std::cout << "Target model height: " << targetHeight << "m" << std::endl;
      std::cout << "Scaling original model to: " << scaleOriginalModel
                << std::endl;
    }
    for (int i = 0; i < 100; i++)
    {
      s_t error = originalModel.skeleton->getHeight(
                      originalModel.skeleton->getPositions())
                  - targetModel.skeleton->getHeight(
                      targetModel.skeleton->getPositions());
      s_t grad = 2 * error
                 * originalModel.skeleton
                       ->getGradientOfHeightWrtBodyScales(
                           originalModel.skeleton->getPositions())
                       .sum();
      if (std::abs(grad) < 1e-4)
        break;
      scaleOriginalModel -= grad * 0.05;
      // Want the original height to match the target height
      originalModel.skeleton->setBodyScales(
          Eigen::VectorXs::Ones(originalModel.skeleton->getNumBodyNodes() * 3)
          * scaleOriginalModel);
      if (verbose)
      {
        s_t newHeight = originalModel.skeleton->getHeight(
            originalModel.skeleton->getPositions());
        std::cout << "Updated original model height: " << newHeight
                  << "m at scale " << scaleOriginalModel << std::endl;
      }
    }
  }

  // 1. To prepare, we want to run through all the meshes on both the original
  // and target models, and derive an equality between meshes (even if they may
  // have different file names).
  std::map<std::string, std::vector<std::string>> equalMeshes;
  std::map<std::string, Eigen::Isometry3s> originalMeshWorldTransforms;
  std::map<std::string, Eigen::Isometry3s> targetMeshWorldTransforms;
  for (int i = 0; i < originalModel.skeleton->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* originalBody = originalModel.skeleton->getBodyNode(i);
    for (auto* originalShapeNode : originalBody->getShapeNodes())
    {
      if (originalShapeNode->getShape()->getType()
          == dynamics::MeshShape::getStaticType())
      {
        // 1.1. Get all the meshes on the original model
        dynamics::MeshShape* originalMeshShape
            = static_cast<dynamics::MeshShape*>(
                originalShapeNode->getShape().get());
        originalMeshWorldTransforms[originalMeshShape->getMeshPath()]
            = originalShapeNode->getWorldTransform();
        // 1.2. Now we need to compare this mesh with all the meshes on the
        // target model
        for (int j = 0; j < targetModel.skeleton->getNumBodyNodes(); j++)
        {
          dynamics::BodyNode* targetBody = targetModel.skeleton->getBodyNode(j);
          for (auto* targetShapeNode : targetBody->getShapeNodes())
          {
            if (targetShapeNode->getShape()->getType()
                == dynamics::MeshShape::getStaticType())
            {
              dynamics::MeshShape* targetMeshShape
                  = static_cast<dynamics::MeshShape*>(
                      targetShapeNode->getShape().get());
              targetMeshWorldTransforms[targetMeshShape->getMeshPath()]
                  = targetShapeNode->getWorldTransform();
              // 1.3. Now we need to check if the meshes are functionally
              // equivalent
              bool equal = true;
              if (originalMeshShape->getMesh()->mNumMeshes
                  != targetMeshShape->getMesh()->mNumMeshes)
              {
                equal = false;
              }
              else
              {
                for (int m = 0; m < originalMeshShape->getMesh()->mNumMeshes;
                     m++)
                {
                  if (originalMeshShape->getMesh()->mMeshes[m]->mNumVertices
                      != targetMeshShape->getMesh()->mMeshes[m]->mNumVertices)
                  {
                    equal = false;
                    break;
                  }

                  for (int v = 0;
                       v
                       < originalMeshShape->getMesh()->mMeshes[m]->mNumVertices;
                       v++)
                  {
                    aiVector3D originalVertex = originalMeshShape->getMesh()
                                                    ->mMeshes[m]
                                                    ->mVertices[v];
                    aiVector3D targetVertex
                        = targetMeshShape->getMesh()->mMeshes[m]->mVertices[v];
                    if (originalVertex.x != targetVertex.x
                        || originalVertex.y != targetVertex.y
                        || originalVertex.z != targetVertex.z)
                    {
                      equal = false;
                      break;
                    }
                  }
                  if (!equal)
                    break;
                }
              }

              if (equal)
              {
                if (verbose)
                {
                  std::cout << "Detected that mesh \""
                            << originalMeshShape->getMeshPath()
                            << "\" is equivalent to mesh \""
                            << targetMeshShape->getMeshPath() << "\""
                            << std::endl;
                }
                equalMeshes[originalMeshShape->getMeshPath()].push_back(
                    targetMeshShape->getMeshPath());
              }
            }
          }
        }
      }
    }
  }

  // 2. Next, we convert the markerset on the original model to be expressed in
  // terms of the meshes on that body
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>>
      markersOnMeshes;
  for (auto& pair : originalModel.markersMap)
  {
    std::string markerName = pair.first;
    Eigen::Vector3s markerOffset = pair.second.second;
    dynamics::BodyNode* body = pair.second.first;

    s_t bestVertexDistance = std::numeric_limits<double>::infinity();
    for (auto* shape : body->getShapeNodes())
    {
      if (shape->getShape()->getType() == dynamics::MeshShape::getStaticType())
      {
        dynamics::MeshShape* meshShape
            = static_cast<dynamics::MeshShape*>(shape->getShape().get());
        Eigen::Vector3s markerInMeshFrame
            = shape->getRelativeTransform().inverse() * markerOffset;
        std::string meshPath = meshShape->getMeshPath();

        for (int i = 0; i < meshShape->getMesh()->mNumMeshes; i++)
        {
          aiMesh* mesh = meshShape->getMesh()->mMeshes[i];
          for (int j = 0; j < mesh->mNumVertices; j++)
          {
            aiVector3D vertex = mesh->mVertices[j];
            Eigen::Vector3s vertexEigen(vertex.x, vertex.y, vertex.z);
            s_t markerToVertexDistance
                = (vertexEigen - markerInMeshFrame).norm();
            if (markerToVertexDistance < bestVertexDistance)
            {
              markersOnMeshes[markerName]
                  = std::pair<std::string, Eigen::Vector3s>(
                      meshPath, markerInMeshFrame);
              bestVertexDistance = markerToVertexDistance;
            }
          }
        }
      }
    }
  }

  if (verbose)
  {
    std::cout << "Markers on meshes: " << std::endl;
    for (auto& pair : markersOnMeshes)
    {
      std::cout << "  " << pair.first << " -> " << pair.second.first
                << std::endl;
    }
  }

  // 3. Now that we have the markerset defined on meshes, we convert the
  // markerset in meshes back into bodies on the target model
  std::map<std::string, std::pair<std::string, Eigen::Vector3s>>
      convertedMarkers;
  for (auto& pair : markersOnMeshes)
  {
    std::string markerName = pair.first;
    std::string meshPath = pair.second.first;
    Eigen::Vector3s markerOffset = pair.second.second;

    for (int i = 0; i < targetModel.skeleton->getNumBodyNodes(); i++)
    {
      dynamics::BodyNode* body = targetModel.skeleton->getBodyNode(i);
      for (auto* shapeNode : body->getShapeNodes())
      {
        if (shapeNode->getShape()->getType()
            == dynamics::MeshShape::getStaticType())
        {
          dynamics::MeshShape* targetMeshShape
              = static_cast<dynamics::MeshShape*>(shapeNode->getShape().get());
          std::string targetMeshPath = targetMeshShape->getMeshPath();
          if (targetMeshPath == meshPath
              || (equalMeshes.count(meshPath) > 0
                  && std::find(
                         equalMeshes.at(meshPath).begin(),
                         equalMeshes.at(meshPath).end(),
                         targetMeshPath)
                         != equalMeshes.at(meshPath).end()))
          {
            Eigen::Vector3s markerInBodyFrame
                = shapeNode->getRelativeTransform() * markerOffset;
            convertedMarkers[markerName]
                = std::pair<std::string, Eigen::Vector3s>(
                    body->getName(), markerInBodyFrame);
          }
        }
      }
    }
  }

  std::vector<std::string> failedMarkers;
  bool targetHasArms = hasArms(targetModel.skeleton);
  bool sourceHasArms = hasArms(originalModel.skeleton);

  for (auto& pair : originalModel.markersMap)
  {
    if (convertedMarkers.count(pair.first) == 0)
    {
      if (sourceHasArms && !targetHasArms)
      {
        if (isArmBodyHeuristic(
                originalModel.skeleton, pair.second.first->getName()))
        {
          if (verbose)
          {
            std::cout << "Ignoring marker \"" << pair.first
                      << "\" because it is on an arm, and the target model "
                         "doesn't have arms"
                      << std::endl;
          }
          continue;
        }
      }
      failedMarkers.push_back(pair.first);
      if (verbose)
      {
        std::cout << "Failed to convert marker \"" << pair.first
                  << "\" using exact mesh registration" << std::endl;
      }
    }
  }

  // 4. If we failed to register all the markers (because the meshes are
  // sufficiently different) then we want to run a more sophisticated
  // registration where we align the meshes as closely as possible on the two
  // skeletons, then try to align the markers to the body on the target skeleton
  // with the nearest vertex.
  // 4.1. We want to align the target back to the original model. We'll start
  // with an initial guess based on any meshes the are known equivalent.
  Eigen::Isometry3s targetRootTransform = Eigen::Isometry3s::Identity();
  if (failedMarkers.size() > 0)
  {
    if (verbose)
    {
      std::cout << "We will now attempt to apply heuristics to match up the "
                   "two skeletons, and see if we can guess the locations of "
                   "more markers on the target skeleton."
                << std::endl;
    }
    // 4.2. This transform will map the vertices from the target model to match
    // the original model as closely as possible. We can initialize an estimate
    // based on matching meshes, if they exist on both skeletons.
    for (auto& pair : equalMeshes)
    {
      std::string originalMeshPath = pair.first;
      std::string targetMeshPath = pair.second[0];
      assert(originalMeshWorldTransforms.count(originalMeshPath));
      Eigen::Isometry3s originalMeshWorldTransform
          = originalMeshWorldTransforms[originalMeshPath];
      assert(targetMeshWorldTransforms.count(targetMeshPath));
      Eigen::Isometry3s targetMeshWorldTransform
          = targetMeshWorldTransforms[targetMeshPath];
      // X * Left = Right
      // X = Right * Left^-1
      Eigen::Isometry3s targetRootTransformGuess
          = originalMeshWorldTransform * targetMeshWorldTransform.inverse();
      // TODO: average multiple guesses, rather than simply overwriting the
      // previous guess
      targetRootTransform = targetRootTransformGuess;
    }

    // 4.3. Given that initial guess for the target root transform, we want to
    // align all the vertices on the original model with all the vertices on the
    // target model using iterative-closest-point, to catch any small
    // differences in the meshes that are otherwise equivalent.
    std::vector<Eigen::Vector3s> originalVertices;
    std::vector<Eigen::Vector3s> targetVertices;
    std::vector<std::string> targetBodyName;
    for (int i = 0; i < originalModel.skeleton->getNumBodyNodes(); i++)
    {
      dynamics::BodyNode* originalBody = originalModel.skeleton->getBodyNode(i);
      for (auto* originalShapeNode : originalBody->getShapeNodes())
      {
        if (originalShapeNode->getShape()->getType()
            == dynamics::MeshShape::getStaticType())
        {
          // 1.1. Get all the meshes on the original model
          dynamics::MeshShape* originalMeshShape
              = static_cast<dynamics::MeshShape*>(
                  originalShapeNode->getShape().get());
          for (int m = 0; m < originalMeshShape->getMesh()->mNumMeshes; m++)
          {
            for (int v = 0;
                 v < originalMeshShape->getMesh()->mMeshes[m]->mNumVertices;
                 v++)
            {
              aiVector3D aiVertex
                  = originalMeshShape->getMesh()->mMeshes[m]->mVertices[v];
              Eigen::Vector3s rawVertex
                  = Eigen::Vector3s(aiVertex.x, aiVertex.y, aiVertex.z);
              Eigen::Vector3s vertex
                  = originalMeshShape->getScale().cwiseProduct(rawVertex);
              originalVertices.push_back(
                  originalShapeNode->getWorldTransform() * vertex);
            }
          }
        }
      }
    }
    for (int j = 0; j < targetModel.skeleton->getNumBodyNodes(); j++)
    {
      dynamics::BodyNode* targetBody = targetModel.skeleton->getBodyNode(j);
      for (auto* targetShapeNode : targetBody->getShapeNodes())
      {
        if (targetShapeNode->getShape()->getType()
            == dynamics::MeshShape::getStaticType())
        {
          dynamics::MeshShape* targetMeshShape
              = static_cast<dynamics::MeshShape*>(
                  targetShapeNode->getShape().get());
          for (int m = 0; m < targetMeshShape->getMesh()->mNumMeshes; m++)
          {
            for (int v = 0;
                 v < targetMeshShape->getMesh()->mMeshes[m]->mNumVertices;
                 v++)
            {
              aiVector3D aiVertex
                  = targetMeshShape->getMesh()->mMeshes[m]->mVertices[v];
              Eigen::Vector3s rawVertex
                  = Eigen::Vector3s(aiVertex.x, aiVertex.y, aiVertex.z);
              Eigen::Vector3s vertex
                  = targetMeshShape->getScale().cwiseProduct(rawVertex);
              targetVertices.push_back(
                  targetShapeNode->getWorldTransform() * vertex);
              targetBodyName.push_back(targetBody->getName());
            }
          }
        }
      }
    }

    // 4.3.2. Actually run ICP with our point clouds
    if (verbose)
    {
      std::cout << "Running Iterative Closest Point to align the vertices of "
                   "the meshes on both models..."
                << std::endl;
    }
    targetRootTransform = math::iterativeClosestPoint(
        targetVertices, originalVertices, targetRootTransform, verbose);
    Eigen::Vector6s rootPos = Eigen::Vector6s::Zero();
    rootPos.tail<3>() = targetRootTransform.translation();
    targetModel.skeleton->getRootJoint()->setPositions(rootPos);

    // 4.4. Motivated by slight differences in default model bone lengths, at
    // this point we want to try to scale the original skeleton to match the
    // target skeleton, so that the markers end up at the correct locations. We
    // do this by picking pairs of joints on the original and target skeletons,
    // and then running an IK+Scale step on the original skeleton to try to
    // match them up.
    Eigen::VectorXs targetJointCenters
        = targetModel.skeleton->getJointWorldPositions(
            targetModel.skeleton->getJoints());
    Eigen::VectorXs originalJointCenters
        = originalModel.skeleton->getJointWorldPositions(
            originalModel.skeleton->getJoints());

    std::vector<dynamics::Joint*> originalJointsToMove;
    std::vector<Eigen::Vector3s> jointCenterTargets;
    std::vector<int> targetJointIndices;
    for (int i = 0; i < originalModel.skeleton->getNumJoints(); i++)
    {
      Eigen::Vector3s originalModelJointCenter
          = originalJointCenters.segment<3>(i * 3);
      s_t bestDistance = std::numeric_limits<double>::infinity();
      int bestIndex = -1;
      for (int j = 0; j < targetModel.skeleton->getNumJoints(); j++)
      {
        Eigen::Vector3s targetModelJointCenter
            = targetJointCenters.segment<3>(j * 3);
        s_t dist = (originalModelJointCenter - targetModelJointCenter).norm();
        if (dist < 0.07 && dist < bestDistance)
        {
          bestDistance = dist;
          bestIndex = j;
        }
      }
      if (bestIndex != -1)
      {
        // We don't want multiple joints to move to the same target joint,
        // because that will cause bone scales to go to 0, which is bad. So we
        // tie break by saying only the closest joint center matches any given
        // target.
        auto index = std::find(
            targetJointIndices.begin(), targetJointIndices.end(), bestIndex);
        if (index != targetJointIndices.end())
        {
          // We want to check the distance that the other joint has, and if
          // we're better, we want to replace them
          int otherIndex = std::distance(targetJointIndices.begin(), index);
          Eigen::Vector3s otherJointCenter
              = originalJointCenters.segment<3>(otherIndex * 3);
          s_t otherDist
              = (jointCenterTargets[otherIndex] - otherJointCenter).norm();
          // If we're closer to this target joint, then we want to bump this
          // other joint
          if (bestDistance < otherDist)
          {
            originalJointsToMove[otherIndex]
                = originalModel.skeleton->getJoint(i);
            assert(
                jointCenterTargets[otherIndex]
                == targetJointCenters.segment<3>(bestIndex * 3));
            assert(targetJointIndices[otherIndex] == bestIndex);
          }
        }
        else
        {
          // We haven't seen this target joint before, so we can add it without
          // checks
          originalJointsToMove.push_back(originalModel.skeleton->getJoint(i));
          jointCenterTargets.push_back(
              targetJointCenters.segment<3>(bestIndex * 3));
          targetJointIndices.push_back(bestIndex);
        }
      }
    }
    Eigen::VectorXs flattenedJointCenterTargets
        = Eigen::VectorXs::Zero(jointCenterTargets.size() * 3);
    for (int i = 0; i < jointCenterTargets.size(); i++)
    {
      flattenedJointCenterTargets.segment<3>(i * 3) = jointCenterTargets[i];
    }
    if (jointCenterTargets.size() > 0)
    {
      if (verbose)
      {
        std::cout << "Running IK+scaling to try to align the nearest joint "
                     "centers of both models..."
                  << std::endl;
      }
      // Now we want to actually run the IK + Scaling step
      MarkerFitter fitter(originalModel.skeleton, originalModel.markersMap);
      auto result = MarkerFitter::scaleAndFit(
          &fitter,
          std::map<std::string, Eigen::Vector3s>(),
          originalModel.skeleton->getPositions(),
          std::map<std::string, s_t>(),
          std::map<std::string, Eigen::Vector3s>(),
          originalJointsToMove,
          flattenedJointCenterTargets,
          Eigen::VectorXs::Ones(originalJointsToMove.size()),
          Eigen::VectorXs::Zero(originalJointsToMove.size() * 6),
          Eigen::VectorXs::Zero(originalJointsToMove.size()),
          originalModel.skeleton->getJoints(),
          false,
          0,
          verbose,
          false);
      originalModel.skeleton->setPositions(result.pose);
      originalModel.skeleton->setGroupScales(result.scale);
    }

    // 4.5. Now we can get the marker world locations on the original model, and
    // look for the nearest vertex on the target model (which hasn't changed
    // size, so we don't need to re-compute its vertices world locations), and
    // if any are within range we can assign them to the appropriate body.
    std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        failedMarkersMap;
    for (std::string failedMarker : failedMarkers)
    {
      failedMarkersMap[failedMarker] = originalModel.markersMap[failedMarker];
    }
    std::map<std::string, Eigen::Vector3s> failedWorldMarkers
        = originalModel.skeleton->getMarkerMapWorldPositions(failedMarkersMap);
    if (verbose)
    {
      std::cout << "Now that the skeletons are aligned, trying to find markers "
                   "to assign to the target model..."
                << std::endl;
    }

    targetVertices.clear();
    targetBodyName.clear();
    for (int j = 0; j < targetModel.skeleton->getNumBodyNodes(); j++)
    {
      dynamics::BodyNode* targetBody = targetModel.skeleton->getBodyNode(j);
      for (auto* targetShapeNode : targetBody->getShapeNodes())
      {
        if (targetShapeNode->getShape()->getType()
            == dynamics::MeshShape::getStaticType())
        {
          std::cout << "Found mesh on body \"" << targetBody->getName() << "\""
                    << std::endl;
          dynamics::MeshShape* targetMeshShape
              = static_cast<dynamics::MeshShape*>(
                  targetShapeNode->getShape().get());
          for (int m = 0; m < targetMeshShape->getMesh()->mNumMeshes; m++)
          {
            for (int v = 0;
                 v < targetMeshShape->getMesh()->mMeshes[m]->mNumVertices;
                 v++)
            {
              aiVector3D aiVertex
                  = targetMeshShape->getMesh()->mMeshes[m]->mVertices[v];
              Eigen::Vector3s rawVertex
                  = Eigen::Vector3s(aiVertex.x, aiVertex.y, aiVertex.z);
              Eigen::Vector3s vertex
                  = targetMeshShape->getScale().cwiseProduct(rawVertex);
              targetVertices.push_back(
                  targetShapeNode->getWorldTransform() * vertex);
              targetBodyName.push_back(targetBody->getName());
            }
          }
        }
      }
    }

    for (auto& pair : failedWorldMarkers)
    {
      // We only want to map markers that we couldn't map before (using precise
      // geometry matching)
      if (convertedMarkers.count(pair.first) == 0)
      {
        std::string markerName = pair.first;
        std::string originalBodyName
            = originalModel.markersMap[markerName].first->getName();

        dynamics::BodyNode* targetBodyGuess
            = targetModel.skeleton->getBodyNode(originalBodyName);
        if (targetBodyGuess != nullptr)
        {
          // If the target skeleton has a body with the same name (e.g.
          // "femur_r" and "femur_r"), then we want to short-circuit the vertex
          // matching, and just assign to that body
          Eigen::Vector3s targetMarker
              = targetBodyGuess->getWorldTransform().inverse() * pair.second;
          convertedMarkers[pair.first]
              = std::pair<std::string, Eigen::Vector3s>(
                  originalBodyName, targetMarker);
        }
        else
        {
          std::cout << "WARNING: Failed to find a body on the target model "
                       "with the same name as \""
                    << originalBodyName << "\"" << std::endl;
          s_t bestDistance = std::numeric_limits<double>::infinity();
          std::string bestBody = "";

          Eigen::Vector3s targetMarker
              = targetRootTransform.inverse() * pair.second;

          for (int v = 0; v < targetVertices.size(); v++)
          {
            Eigen::Vector3s point = targetVertices[v];
            s_t dist = (point - targetMarker).norm();

            if (dist < 0.15 && dist < bestDistance)
            {
              bestDistance = dist;
              bestBody = targetBodyName[v];
              break;
            }
          }

          if (std::isfinite(bestDistance))
          {
            if (verbose)
            {
              std::cout
                  << "Marker \"" << markerName
                  << "\", originally attached to a body named \""
                  << originalBodyName << "\" is closest to body \"" << bestBody
                  << "\" at distance " << bestDistance
                  << "m from nearest vertex on any mesh attached to that body"
                  << std::endl;
            }
            Eigen::Isometry3s bestBodyWorld
                = targetModel.skeleton->getBodyNode(bestBody)
                      ->getWorldTransform();
            convertedMarkers[pair.first]
                = std::pair<std::string, Eigen::Vector3s>(
                    bestBody, bestBodyWorld.inverse() * targetMarker);
          }
        }
      }
    }
  }

  std::map<std::string, bool> isAnatomical;
  for (auto& pair : convertedMarkers)
  {
    isAnatomical[pair.first] = false;
  }
  for (std::string marker : targetModel.anatomicalMarkers)
  {
    isAnatomical[marker] = true;
  }

  if (verbose)
  {
    std::cout << "Converted markers: " << std::endl;
    for (auto& pair : convertedMarkers)
    {
      std::cout << "  " << pair.first << " -> " << pair.second.first
                << std::endl;
    }
  }

  // 5. Finally, write the converted skeleton back to disk
  replaceOsimMarkers(
      targetModelPath, convertedMarkers, isAnatomical, outputPath);

  /*
  /// If we're saving to a GUI
  server::GUIRecording server;
  server.renderSkeleton(originalModel.skeleton, "original");
  server.renderSkeleton(
      targetModel.skeleton, "target", Eigen::Vector4s(1, 0, 0, 1));
  server.saveFrame();
  server.writeFramesJson("../../../javascript/src/data/movement2.bin");
  */

  // 6. We also find the markers that we couldn't deterministically convert with
  // equal meshes, and return them

  std::vector<std::string> guessedMarkers;
  std::vector<std::string> missingMarkers;
  for (std::string marker : failedMarkers)
  {
    if (convertedMarkers.count(marker) > 0)
    {
      guessedMarkers.push_back(marker);
    }
    else
    {
      missingMarkers.push_back(marker);
    }
  }

  return std::make_pair(guessedMarkers, missingMarkers);
}

//==============================================================================
/// This method will use several heuristics, including the names of joints,
/// meshes, and bones to determine if this body on this skeleton should be
/// considered an "arm," which means basically anything after an articulated
/// shoulder joint. Importantly, a torso with fixed meshes normally associated
/// with the arms attached to the torso is NOT considered an arm.
bool OpenSimParser::isArmBodyHeuristic(
    std::shared_ptr<dynamics::Skeleton> skel, const std::string& bodyName)
{
  dynamics::BodyNode* body = skel->getBodyNode(bodyName);
  // Default to "no, it's not an arm" if it doesn't exist
  if (body == nullptr)
  {
    return false;
  }

  // Trace out the path to the root of the tree
  std::vector<dynamics::Joint*> jointsToRoot;
  std::vector<dynamics::BodyNode*> bodiesToRoot;
  dynamics::Joint* cursor = body->getParentJoint();
  while (cursor != nullptr)
  {
    jointsToRoot.push_back(cursor);
    if (cursor->getParentBodyNode() != nullptr)
    {
      bodiesToRoot.push_back(cursor->getParentBodyNode());
      cursor = cursor->getParentBodyNode()->getParentJoint();
    }
    else
    {
      break;
    }
  }

  // Check all the names of the joints on the way up the chain, looking for dead
  // giveaways
  std::vector<std::string> jointNameHeuristics;
  jointNameHeuristics.push_back("shoulder");
  jointNameHeuristics.push_back("elbow");
  jointNameHeuristics.push_back("wrist");
  jointNameHeuristics.push_back("hand");
  jointNameHeuristics.push_back("thumb");
  jointNameHeuristics.push_back("shld");
  jointNameHeuristics.push_back("elb");
  for (dynamics::Joint* joint : jointsToRoot)
  {
    std::string name = joint->getName();
    for (std::string heuristic : jointNameHeuristics)
    {
      if (name.find(heuristic) != std::string::npos)
      {
        return true;
      }
    }
  }

  // Check all the names of the bodies on the way up the chain, looking for dead
  // giveaways
  std::vector<std::string> bodyNameHeuristics;
  bodyNameHeuristics.push_back("radius");
  bodyNameHeuristics.push_back("ulna");
  bodyNameHeuristics.push_back("humerus");
  bodyNameHeuristics.push_back("scapula");
  bodyNameHeuristics.push_back("clavicle");
  bodyNameHeuristics.push_back("hand");
  bodyNameHeuristics.push_back("forearm");
  bodyNameHeuristics.push_back("arm");
  bodyNameHeuristics.push_back("shoulder");
  bodyNameHeuristics.push_back("elbow");
  bodyNameHeuristics.push_back("wrist");
  for (dynamics::BodyNode* body : bodiesToRoot)
  {
    std::string name = body->getName();
    for (std::string heuristic : bodyNameHeuristics)
    {
      if (name.find(heuristic) != std::string::npos)
      {
        return true;
      }
    }
  }

  // Lastly, we want to check if we're not a torso, then if we or any parent
  // body has any meshes on it we typically associate with arms,
  if (!isTorsoBodyHeuristic(skel, bodyName))
  {
    std::vector<std::string> meshNameHeuristics;
    meshNameHeuristics.push_back("radius");
    meshNameHeuristics.push_back("ulna");
    meshNameHeuristics.push_back("forearm");
    meshNameHeuristics.push_back("index_distal");
    meshNameHeuristics.push_back("index_medial");
    meshNameHeuristics.push_back("index_proximal");
    meshNameHeuristics.push_back("metacarpal");
    meshNameHeuristics.push_back("humerus");
    meshNameHeuristics.push_back("ring");
    meshNameHeuristics.push_back("thumb");
    meshNameHeuristics.push_back("pinky");
    meshNameHeuristics.push_back("scapula");
    meshNameHeuristics.push_back("scaphoid");
    meshNameHeuristics.push_back("trapezium");
    meshNameHeuristics.push_back("metacarpal");
    bodiesToRoot.push_back(body);
    for (dynamics::BodyNode* body : bodiesToRoot)
    {
      for (auto* shape : body->getShapeNodes())
      {
        if (shape->getShape()->getType()
            == dynamics::MeshShape::getStaticType())
        {
          dynamics::MeshShape* meshShape
              = static_cast<dynamics::MeshShape*>(shape->getShape().get());
          std::string meshPath = meshShape->getMeshPath();
          for (std::string heuristic : meshNameHeuristics)
          {
            if (meshPath.find(heuristic) != std::string::npos)
            {
              return true;
            }
          }
        }
      }
    }
  }

  return false;
}

//==============================================================================
/// This method will use several heuristics, including the names of joints,
/// meshes, and bones to determine if this body on this skeleton should be
/// considered a "torso"
bool OpenSimParser::isTorsoBodyHeuristic(
    std::shared_ptr<dynamics::Skeleton> skel, const std::string& bodyName)
{
  dynamics::BodyNode* body = skel->getBodyNode(bodyName);
  // Default to "no, it's not a torso" if it doesn't exist
  if (body == nullptr)
  {
    return false;
  }

  // Check all the names of the joints on the way up the chain, looking for dead
  // giveaways
  std::vector<std::string> jointNameHeuristics;
  jointNameHeuristics.push_back("spine");
  jointNameHeuristics.push_back("lumbar");
  jointNameHeuristics.push_back("thoracic");
  jointNameHeuristics.push_back("cervical");
  jointNameHeuristics.push_back("torso");
  std::string jointName = body->getParentJoint()->getName();
  for (std::string heuristic : jointNameHeuristics)
  {
    if (jointName.find(heuristic) != std::string::npos)
    {
#ifndef NDEBUG
      std::cout << "Joint " << jointName
                << " is a torso joint based on name heuristics" << std::endl;
#endif
      return true;
    }
  }

  // Check all the names of the bodies on the way up the chain, looking for dead
  // giveaways
  std::vector<std::string> bodyNameHeuristics;
  // bodyNameHeuristics.push_back("torso"); <- turns out the gait2392 model has
  // a "torso" body, which is really just the pelvis, so this isn't indicative
  bodyNameHeuristics.push_back("thorax");
  bodyNameHeuristics.push_back("head");
  bodyNameHeuristics.push_back("ribs");
  bodyNameHeuristics.push_back("lumbar");
  bodyNameHeuristics.push_back("neck");
  bodyNameHeuristics.push_back("skull");
  for (std::string heuristic : bodyNameHeuristics)
  {
    if (bodyName.find(heuristic) != std::string::npos)
    {
#ifndef NDEBUG
      std::cout << "Body " << bodyName
                << " is a torso body based on name heuristics" << std::endl;
#endif
      return true;
    }
  }

  std::vector<std::string> meshNameHeuristics;
  meshNameHeuristics.push_back("ribs");
  meshNameHeuristics.push_back("thorax");
  meshNameHeuristics.push_back("spine");
  meshNameHeuristics.push_back("skull");
  meshNameHeuristics.push_back("hat");
  meshNameHeuristics.push_back("jaw");
  meshNameHeuristics.push_back("neck");
  meshNameHeuristics.push_back("vertibrae");
  for (auto* shape : body->getShapeNodes())
  {
    if (shape->getShape()->getType() == dynamics::MeshShape::getStaticType())
    {
      dynamics::MeshShape* meshShape
          = static_cast<dynamics::MeshShape*>(shape->getShape().get());
      std::string meshPath = meshShape->getMeshPath();
      for (std::string heuristic : meshNameHeuristics)
      {
        if (meshPath.find(heuristic) != std::string::npos)
        {
#ifndef NDEBUG
          std::cout << "Mesh " << meshPath
                    << " is a torso mesh based on name heuristics" << std::endl;
#endif

          return true;
        }
      }
    }
  }

  return false;
}

//==============================================================================
/// This method will return true if this skeleton has any bodies that return
/// true for isArmBodyHeuristic().
bool OpenSimParser::hasArms(std::shared_ptr<dynamics::Skeleton> skel)
{
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    if (isArmBodyHeuristic(skel, skel->getBodyNode(i)->getName()))
    {
#ifndef NDEBUG
      std::cout << "Skeleton " << skel->getName() << " has arm body "
                << skel->getBodyNode(i)->getName() << std::endl;
#endif
      return true;
    }
  }
  return false;
}

//==============================================================================
/// This method will return true if this skeleton has any bodies that return
/// true for isTorsoBodyHeuristic().
bool OpenSimParser::hasTorso(std::shared_ptr<dynamics::Skeleton> skel)
{
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    if (isTorsoBodyHeuristic(skel, skel->getBodyNode(i)->getName()))
    {
#ifndef NDEBUG
      std::cout << "Skeleton " << skel->getName() << " has torso body "
                << skel->getBodyNode(i)->getName() << std::endl;
#endif
      return true;
    }
  }
  return false;
}

//==============================================================================
/// Read an *.osim file, change the mass/COM/MOI for everything, and write it
/// out to a new *.osim file
void OpenSimParser::replaceOsimInertia(
    const common::Uri& uri,
    const std::shared_ptr<dynamics::Skeleton> skel,
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
  // Go through and adjust the custom joints

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    const char* name_c = bodyCursor->Attribute("name");
    if (name_c != nullptr)
    {
      std::string name(name_c);

      dynamics::BodyNode* correspondingBody = skel->getBodyNode(name);
      if (correspondingBody != nullptr)
      {
        tinyxml2::XMLElement* mass = bodyCursor->FirstChildElement("mass");
        if (mass != nullptr)
        {
          mass->SetText(to_string(correspondingBody->getMass()).c_str());
        }

        tinyxml2::XMLElement* massCenter
            = bodyCursor->FirstChildElement("mass_center");
        if (massCenter != nullptr)
        {
          massCenter->SetText(
              writeVec3(correspondingBody->getLocalCOM()).c_str());
        }

        tinyxml2::XMLElement* inertia
            = bodyCursor->FirstChildElement("inertia");
        if (inertia != nullptr)
        {
          inertia->SetText(
              writeVec6(correspondingBody->getInertia().getMomentVector())
                  .c_str());
        }

        tinyxml2::XMLElement* inertiaXX
            = bodyCursor->FirstChildElement("inertia_xx");
        if (inertiaXX != nullptr)
        {
          inertiaXX->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(0))
                  .c_str());
        }
        tinyxml2::XMLElement* inertiaYY
            = bodyCursor->FirstChildElement("inertia_yy");
        if (inertiaYY != nullptr)
        {
          inertiaYY->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(1))
                  .c_str());
        }
        tinyxml2::XMLElement* inertiaZZ
            = bodyCursor->FirstChildElement("inertia_zz");
        if (inertiaZZ != nullptr)
        {
          inertiaZZ->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(2))
                  .c_str());
        }
        tinyxml2::XMLElement* inertiaXY
            = bodyCursor->FirstChildElement("inertia_xy");
        if (inertiaXY != nullptr)
        {
          inertiaXY->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(3))
                  .c_str());
        }
        tinyxml2::XMLElement* inertiaXZ
            = bodyCursor->FirstChildElement("inertia_xz");
        if (inertiaXZ != nullptr)
        {
          inertiaXZ->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(4))
                  .c_str());
        }
        tinyxml2::XMLElement* inertiaYZ
            = bodyCursor->FirstChildElement("inertia_yz");
        if (inertiaYZ != nullptr)
        {
          inertiaYZ->SetText(
              to_string(correspondingBody->getInertia().getMomentVector()(5))
                  .c_str());
        }
      }
    }

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Save out the result
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
            if (!markerSwapSpace.hasNaN()
                && (markerSwapSpace != Eigen::Vector3s::Zero()))
            {
              if (markerNames.empty())
              {
                NIMBLE_THROW(
                    "No marker names found in TRC file. Please check "
                    "that the file is formatted correctly.");
              }
              if (markerNumber >= (int)markerNames.size())
              {
                NIMBLE_THROW(
                    "Marker number exceeds number of marker names in "
                    "TRC file. Please check that the file is formatted "
                    "correctly.");
              }
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
    result.framesPerSecond
        = roundToNearestMultiple((int)(frames / elapsed), 10);
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
    markerNames.push_back(trim(pair.first));
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
    motFile << "\t" << trim(skel->getDof(i)->getName());
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
/// This saves the *.mot file from the inverse dynamics solved for the
/// skeleton
void OpenSimParser::saveIDMot(
    std::shared_ptr<dynamics::Skeleton> skel,
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const Eigen::MatrixXs& controlForces)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "Coordinates\n";
  motFile << "version=1\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "nColumns=" << controlForces.rows() + 1 << "\n";
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
    motFile << "\t" << trim(skel->getDof(i)->getName());
    if (3 <= i && i < 6)
    {
      motFile << "_force";
    }
    else
    {
      motFile << "_moment";
    }
  }
  motFile << "\n";

  for (int t = 0; t < timestamps.size(); t++)
  {
    motFile << timestamps[t];
    for (int i = 0; i < skel->getNumDofs(); i++)
    {
      motFile << "\t" << controlForces(i, t);
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
void OpenSimParser::saveRawGRFMot(
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
      motFile << "\t" << zeroIfNan((double)forcePlates[i].forces[t](0));
      motFile << "\t" << zeroIfNan((double)forcePlates[i].forces[t](1));
      motFile << "\t" << zeroIfNan((double)forcePlates[i].forces[t](2));
      motFile << "\t"
              << zeroIfNan((double)forcePlates[i].centersOfPressure[t](0));
      motFile << "\t"
              << zeroIfNan((double)forcePlates[i].centersOfPressure[t](1));
      motFile << "\t"
              << zeroIfNan((double)forcePlates[i].centersOfPressure[t](2));
      motFile << "\t" << zeroIfNan((double)forcePlates[i].moments[t](0));
      motFile << "\t" << zeroIfNan((double)forcePlates[i].moments[t](1));
      motFile << "\t" << zeroIfNan((double)forcePlates[i].moments[t](2));
    }
    motFile << "\n";
  }

  motFile.close();
}

//==============================================================================
/// This saves the *.mot file for the ground reaction forces we've processed
/// through our dynamics fitter.
void OpenSimParser::saveProcessedGRFMot(
    const std::string& outputPath,
    const std::vector<double>& timestamps,
    const std::vector<dynamics::BodyNode*> contactBodies,
    std::shared_ptr<dynamics::Skeleton> skel,
    const Eigen::MatrixXs& poses,
    const std::vector<biomechanics::ForcePlate>& forcePlates,
    const Eigen::MatrixXs wrenches)
{
  std::ofstream motFile;
  motFile.open(outputPath);
  motFile << "nColumns=" << (9 * contactBodies.size()) + 1 << "\n";
  motFile << "nRows=" << timestamps.size() << "\n";
  motFile << "DataType=double\n";
  motFile << "version=3\n";
  motFile << "OpenSimVersion=4.1\n";
  motFile << "endheader\n";

  motFile << "time";
  for (int i = 0; i < contactBodies.size(); i++)
  {
    std::string name = trim(contactBodies[i]->getName());
    motFile << "\t"
            << "ground_force_" + name + "_vx";
    motFile << "\t"
            << "ground_force_" + name + "_vy";
    motFile << "\t"
            << "ground_force_" + name + "_vz";
    motFile << "\t"
            << "ground_force_" + name + "_px";
    motFile << "\t"
            << "ground_force_" + name + "_py";
    motFile << "\t"
            << "ground_force_" + name + "_pz";
    motFile << "\t"
            << "ground_force_" + name + "_mx";
    motFile << "\t"
            << "ground_force_" + name + "_my";
    motFile << "\t"
            << "ground_force_" + name + "_mz";
  }
  motFile << "\n";

  Eigen::VectorXs originalPose = skel->getPositions();

  for (int t = 0; t < timestamps.size(); t++)
  {
    skel->setPositions(poses.col(t));

    motFile << timestamps[t];
    for (int i = 0; i < contactBodies.size(); i++)
    {
      Eigen::Vector3s bodyPosition
          = contactBodies[i]->getWorldTransform().translation();
      s_t groundHeight = bodyPosition(1);

      s_t closestForcePlateDistance = std::numeric_limits<double>::infinity();
      for (auto& forcePlate : forcePlates)
      {
        if (forcePlate.forces[t].norm() > 0)
        {
          s_t distance
              = (forcePlate.centersOfPressure[t] - bodyPosition).norm();
          if (distance < closestForcePlateDistance)
          {
            closestForcePlateDistance = distance;
            groundHeight = forcePlate.centersOfPressure[t](1);
          }
        }
      }

      Eigen::Vector6s worldWrench = wrenches.block<6, 1>(i * 6, t);
      Eigen::Vector3s worldTau = worldWrench.head<3>();
      Eigen::Vector3s worldF = worldWrench.tail<3>();
      Eigen::Matrix3s crossF = math::makeSkewSymmetric(worldF);
      Eigen::Vector3s rightSide = worldTau - crossF.col(1) * groundHeight;
      Eigen::Matrix3s leftSide = -crossF;
      leftSide.col(1) = worldF;
      Eigen::Vector3s p
          = leftSide.completeOrthogonalDecomposition().solve(rightSide);
      s_t k = p(1);
      p(1) = 0;
      Eigen::Vector3s expectedTau = worldF * k;
      Eigen::Vector3s cop = p;
      cop(1) = groundHeight;

      motFile << "\t" << zeroIfNan((double)worldF(0));
      motFile << "\t" << zeroIfNan((double)worldF(1));
      motFile << "\t" << zeroIfNan((double)worldF(2));
      motFile << "\t" << zeroIfNan((double)cop(0));
      motFile << "\t" << zeroIfNan((double)cop(1));
      motFile << "\t" << zeroIfNan((double)cop(2));
      motFile << "\t" << zeroIfNan((double)expectedTau(0));
      motFile << "\t" << zeroIfNan((double)expectedTau(1));
      motFile << "\t" << zeroIfNan((double)expectedTau(2));
    }
    motFile << "\n";
  }

  skel->setPositions(originalPose);

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
    std::string bodyName = trim(skel->getBodyNode(i)->getName());

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
    std::string markerName = trim(pair.first);
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
    const std::vector<double>& targetTimestamps,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  OpenSimGRF result;
  const std::string content = retriever->readAll(uri);

  bool inHeader = true;

  std::vector<std::string> colNames = std::vector<std::string>();
  std::vector<int> colToPlate = std::vector<int>();
  std::vector<int> colToCOP = std::vector<int>();
  std::vector<int> colToWrench = std::vector<int>();
  int numPlates = 0;

  std::vector<s_t> timestamps;
  std::vector<std::vector<Eigen::Vector3s>> copRows;
  std::vector<std::vector<Eigen::Vector6s>> wrenchRows;

  int lineNumber = 0;
  auto start = 0U;
  auto end = content.find("\n");
  while (true)
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
    // If we're past the header and encounter an empty line, don't parse it
    // and skip to the next line. This will skip extra lines at the end of the
    // file.
    else if (!line.empty())
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
          std::string underscore = "_";
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
          if (token.find("moment_x") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("moment_x"), std::string("moment_x").size(), empty);
            wrench = 0;
          }
          if (token.find("moment_y") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("moment_y"), std::string("moment_y").size(), empty);
            wrench = 1;
          }
          if (token.find("moment_z") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("moment_z"), std::string("moment_z").size(), empty);
            wrench = 2;
          }
          if (token.find("torque_r_x") != std::string::npos
              || token.find("torque_l_x") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("_x"), std::string("_x").size(), underscore);
            wrench = 0;
          }
          if (token.find("torque_r_y") != std::string::npos
              || token.find("torque_l_y") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("_y"), std::string("_y").size(), underscore);
            wrench = 1;
          }
          if (token.find("torque_r_z") != std::string::npos
              || token.find("torque_l_z") != std::string::npos)
          {
            prefixSuffix.replace(
                token.find("_z"), std::string("_z").size(), underscore);
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

          if (token == "time" || token == "Time")
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
      // Ignore whitespace in a .MOT file between "endheader" and the names of
      // the columns
      if (lineNumber > 0 || colNames.size() > 0)
      {
        lineNumber++;
      }
    }

    if (end == std::string::npos)
    {
      break;
    }
    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }

  NIMBLE_THROW_IF(inHeader, 
    "Parsed the entire file '" + uri.toString() + "' and never found a line "
    "with string 'endheader'. Please check the file to ensure it's formatted "
    "correctly.");

  assert(timestamps.size() == copRows.size());
  assert(timestamps.size() == wrenchRows.size());

  if (!targetTimestamps.empty())
  {
    // If we've got a list of targetTimestamps, we want to just go ahead and
    // find the values for the GRF at exactly those timestamps, interpolating if
    // necessary. For the temporal out-of-bounds case, which does sometimes
    // happen (e.g. in the Tiziana 2019 Nature dataset), we just provide zero
    // GRFs on the OOB timesteps.
    std::vector<ForcePlate> forcePlates;
    for (int i = 0; i < numPlates; i++)
    {
      forcePlates.emplace_back();
    }

    for (int t = 0; t < (int)targetTimestamps.size(); t++)
    {
      s_t targetTimestamp = targetTimestamps[t];
      if (targetTimestamp < timestamps[0] - 1e-3
          || targetTimestamp > timestamps[timestamps.size() - 1] + 1e-3)
      {
        // If we're requesting a timestamp that's out of bounds, then just pad
        // the GRFs with 0s
        for (int i = 0; i < numPlates; i++)
        {
          forcePlates[i].timestamps.push_back(targetTimestamp);
          forcePlates[i].centersOfPressure.push_back(Eigen::Vector3s::Zero());
          forcePlates[i].moments.push_back(Eigen::Vector3s::Zero());
          forcePlates[i].forces.push_back(Eigen::Vector3s::Zero());
        }
      }
      else
      {
        int closestIndex = 0;
        s_t closestDistance = std::numeric_limits<s_t>::infinity();
        for (int j = 0; j < (int)timestamps.size(); j++)
        {
          s_t distance = std::abs(timestamps[j] - targetTimestamp);
          if (distance < closestDistance)
          {
            closestDistance = distance;
            closestIndex = j;
          }
        }
        if (std::abs(timestamps[closestIndex] - targetTimestamp) <= 1e-3)
        {
          // This means we just requested a timestamp pretty much directly, so
          // no blending is required
          for (int i = 0; i < numPlates; i++)
          {
            forcePlates[i].timestamps.push_back(targetTimestamp);
            forcePlates[i].centersOfPressure.push_back(
                copRows[closestIndex][i]);
            forcePlates[i].moments.push_back(
                wrenchRows[closestIndex][i].segment<3>(0));
            forcePlates[i].forces.push_back(
                wrenchRows[closestIndex][i].segment<3>(3));
          }
        }
        else
        {
          // We're going to have to interpolate (linearly) between two
          // timestamps, because we didn't find an exact match for the requested
          // timestamp in our loaded data.
          int highIndex = closestIndex;
          int lowIndex = closestIndex - 1;
          if (timestamps[closestIndex] < targetTimestamp)
          {
            highIndex = closestIndex + 1;
            lowIndex = closestIndex;
          }
          // Avoid divide by zero exceptions by clipping timestep size to be at
          // least 1e-4
          s_t timestep
              = std::max(timestamps[highIndex] - timestamps[lowIndex], 1e-4);
          s_t alpha = (targetTimestamp - timestamps[lowIndex]) / timestep;
          // Push an interpolated value between the two timestamps onto the
          // force plates
          for (int i = 0; i < numPlates; i++)
          {
            forcePlates[i].timestamps.push_back(targetTimestamp);
            forcePlates[i].centersOfPressure.push_back(
                copRows[lowIndex][i] * (1 - alpha)
                + copRows[highIndex][i] * alpha);
            forcePlates[i].moments.push_back(
                wrenchRows[lowIndex][i].segment<3>(0) * (1 - alpha)
                + wrenchRows[highIndex][i].segment<3>(0) * alpha);
            forcePlates[i].forces.push_back(
                wrenchRows[lowIndex][i].segment<3>(3) * (1 - alpha)
                + wrenchRows[highIndex][i].segment<3>(3) * alpha);
          }
        }
      }
    }

    return forcePlates;
  }
  else
  {
    // If we didn't get any target timestamps, then just put the data directly
    // into an array and return it.
    std::vector<ForcePlate> forcePlates;
    for (int i = 0; i < numPlates; i++)
    {
      forcePlates.emplace_back();
      ForcePlate& forcePlate = forcePlates[forcePlates.size() - 1];
      for (int t = 0; t < timestamps.size(); t++)
      {
        forcePlate.timestamps.push_back(timestamps[t]);
        forcePlate.centersOfPressure.push_back(copRows[t][i]);
        forcePlate.moments.push_back(wrenchRows[t][i].segment<3>(0));
        forcePlate.forces.push_back(wrenchRows[t][i].segment<3>(3));
      }
    }

    return forcePlates;
  }
}

//==============================================================================
/// This loads IMU data from a CSV file, where the headers of columns are
/// names of IMUs with axis suffixes (e.g. "IMU1_Accel_X", "IMU1_Accel_Y",
/// "IMU1_Accel_Z", "IMU1_Gyro_X", "IMU1_Gyro_Y", "IMU1_Gyro_Z")
OpenSimIMUData OpenSimParser::loadIMUFromCSV(
    const common::Uri& uri,
    bool isAccelInG,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);
  const std::string content = retriever->readAll(uri);

  OpenSimIMUData imus;

  const s_t g = 9.80665;
  const s_t accelScale = isAccelInG ? g : 1.0;

  // Split content into an array of lines
  std::vector<std::string> lines;
  std::istringstream f(content);
  std::string line;
  bool firstLine = true;
  std::vector<std::string> colNames;
  std::vector<std::string> colToIMU;
  std::vector<int> colToAxis;
  std::vector<bool> colToIsAcc;
  while (std::getline(f, line))
  {
    std::istringstream lineF(line);
    std::string token;

    if (firstLine)
    {
      while (std::getline(lineF, token, ','))
      {
        colNames.push_back(token);
        std::string imuName;
        int imuAxis = -1;
        bool isAcc = false;
        if (endsWith(token, "_Accel_X"))
        {
          imuName = token.substr(0, token.size() - strlen("_Accel_X"));
          imuAxis = 0;
          isAcc = true;
        }
        else if (endsWith(token, "_Accel_Y"))
        {
          imuName = token.substr(0, token.size() - strlen("_Accel_Y"));
          imuAxis = 1;
          isAcc = true;
        }
        else if (endsWith(token, "_Accel_Z"))
        {
          imuName = token.substr(0, token.size() - strlen("_Accel_Z"));
          imuAxis = 2;
          isAcc = true;
        }
        if (endsWith(token, "_Gyro_X"))
        {
          imuName = token.substr(0, token.size() - strlen("_Gyro_X"));
          imuAxis = 0;
          isAcc = false;
        }
        else if (endsWith(token, "_Gyro_Y"))
        {
          imuName = token.substr(0, token.size() - strlen("_Gyro_Y"));
          imuAxis = 1;
          isAcc = false;
        }
        else if (endsWith(token, "_Gyro_Z"))
        {
          imuName = token.substr(0, token.size() - strlen("_Gyro_Z"));
          imuAxis = 2;
          isAcc = false;
        }
        colToIMU.push_back(imuName);
        colToAxis.push_back(imuAxis);
        colToIsAcc.push_back(isAcc);
      }
    }
    else
    {
      int tokenIndex = 0;
      std::map<std::string, Eigen::Vector3s> accelerometerData;
      std::map<std::string, Eigen::Vector3s> gyroData;
      while (std::getline(lineF, token, ','))
      {
        // First column is always time
        if (tokenIndex == 0)
        {
          imus.timestamps.push_back(std::stod(token));
          tokenIndex++;
          continue;
        }

        std::string imuName = colToIMU[tokenIndex];
        bool isAcc = colToIsAcc[tokenIndex];
        int axis = colToAxis[tokenIndex];

        if (isAcc)
        {
          if (accelerometerData.count(imuName) == 0)
          {
            accelerometerData[imuName] = Eigen::Vector3s::Zero();
          }
          accelerometerData[imuName](axis) = accelScale * std::stod(token);
        }
        else
        {
          if (gyroData.count(imuName) == 0)
          {
            gyroData[imuName] = Eigen::Vector3s::Zero();
          }
          gyroData[imuName](axis) = std::stod(token);
        }

        tokenIndex++;
      }
      imus.accReadings.push_back(accelerometerData);
      imus.gyroReadings.push_back(gyroData);
    }

    firstLine = false;
  }

  return imus;
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

  auto axisOrderAndErrorFlag = getAxisOrder(eulerAxisOrder);
  dynamics::EulerJoint::AxisOrder axisOrder = axisOrderAndErrorFlag.first;
  bool axisOrderError = axisOrderAndErrorFlag.second;
  if (axisOrderError)
  {
    NIMBLE_THROW("Invalid axis order for Euler joint: " + jointName);
  }
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
/// Load excitations and activations from a MocoTrajectory *.sto file.
OpenSimMocoTrajectory OpenSimParser::loadMocoTrajectory(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);

  const std::string content = retriever->readAll(uri);
  std::vector<double> timestamps;
  std::vector<Eigen::VectorXs> excitations;
  std::vector<Eigen::VectorXs> activations;
  std::vector<std::string> excitationNames;
  std::vector<std::string> activationNames;
  std::vector<int> columnToExcitation;
  std::vector<int> columnToActivation;

  bool inHeader = true;

  int activationCount = 0;
  int excitationCount = 0;
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
      }
    }
    else
    {
      int tokenNumber = 0;
      std::string whitespace = " \t";
      auto tokenStart = line.find_first_not_of(whitespace);
      Eigen::VectorXs activationVec;
      Eigen::VectorXs excitationVec;
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
            // token = column label
            if (beginsWith(token, "/forceset/"))
            {
              int n = strlen("/forceset/");
              std::string actuName = token.substr(n, token.length() - n);
              if (endsWith(token, "/activation"))
              {
                columnToActivation.push_back(tokenNumber);
                activationCount++;
                activationNames.push_back(actuName);
              }
              else
              {
                columnToExcitation.push_back(tokenNumber);
                excitationCount++;
                excitationNames.push_back(actuName + "/excitation");
              }
            }
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
            int excIndex = findIndex(columnToExcitation, tokenNumber);
            int actIndex = findIndex(columnToActivation, tokenNumber);
            if (excIndex != -1)
            {
              if (excitationVec.size() == 0)
              {
                excitationVec = Eigen::VectorXs::Zero(excitationCount);
              }
              excitationVec(excIndex) = value;
            }
            else if (actIndex != -1)
            {
              if (activationVec.size() == 0)
              {
                activationVec = Eigen::VectorXs::Zero(activationCount);
              }
              activationVec(actIndex) = value;
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
        excitations.push_back(excitationVec);
        activations.push_back(activationVec);
        timestamps.push_back(timestamp);
      }
      lineNumber++;
    }

    start = end + 1; // "\n".length()
    end = content.find("\n", start);
  }

  Eigen::MatrixXs excitationsMatrix
      = Eigen::MatrixXs::Zero(excitationCount, excitations.size());
  for (int i = 0; i < (int)excitations.size(); i++)
  {
    excitationsMatrix.col(i) = excitations[i];
  }
  Eigen::MatrixXs activationsMatrix
      = Eigen::MatrixXs::Zero(activationCount, activations.size());
  for (int i = 0; i < (int)activations.size(); i++)
  {
    activationsMatrix.col(i) = activations[i];
  }
  OpenSimMocoTrajectory mocoTraj;
  mocoTraj.timestamps = timestamps;
  mocoTraj.excitations = excitationsMatrix;
  mocoTraj.activations = activationsMatrix;
  mocoTraj.excitationNames = excitationNames;
  mocoTraj.activationNames = activationNames;

  return mocoTraj;
}

//==============================================================================
/// Append excitations and activations from a MocoTrajectory to a CSV file and
/// save it.
void OpenSimParser::appendMocoTrajectoryAndSaveCSV(
    const common::Uri& uri,
    const OpenSimMocoTrajectory& mocoTraj,
    std::string path,
    const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever
      = ensureRetriever(nullOrRetriever);
  const std::string content = retriever->readAll(uri);

  std::ofstream csvFile;
  csvFile.open(path);

  // Split content into an array of lines
  std::vector<std::string> lines;
  std::istringstream f(content);
  std::string line;
  int lineNumber = 0;
  while (std::getline(f, line))
  {
    std::istringstream lineF(line);
    std::string token;

    if (lineNumber == 0)
    {
      int tokenIndex = 0;
      while (std::getline(lineF, token, ','))
      {
        if (tokenIndex == 0)
        {
          csvFile << token;
        }
        else
        {
          csvFile << "," << token;
        }
        tokenIndex++;
      }
      for (int i = 0; i < (int)mocoTraj.activationNames.size(); i++)
      {
        csvFile << "," << mocoTraj.activationNames[i];
      }
      for (int i = 0; i < (int)mocoTraj.excitationNames.size(); i++)
      {
        csvFile << "," << mocoTraj.excitationNames[i];
      }
    }
    else
    {
      int tokenIndex = 0;
      while (std::getline(lineF, token, ','))
      {
        // First column is always time
        if (tokenIndex == 0)
        {
          csvFile << std::endl << std::stod(token);
        }
        else
        {
          csvFile << "," << std::stod(token);
        }
        tokenIndex++;
      }
      for (int i = 0; i < (int)mocoTraj.activationNames.size(); i++)
      {
        csvFile << "," << mocoTraj.activations(i, lineNumber - 1);
      }
      for (int i = 0; i < (int)mocoTraj.excitationNames.size(); i++)
      {
        csvFile << "," << mocoTraj.excitations(i, lineNumber - 1);
      }
    }

    lineNumber++;
  }
  csvFile.close();
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
/// This does its best to convert a *.osim file to a URDF file.
bool OpenSimParser::convertOsimToSDF(
    const common::Uri& uri,
    const std::string& outputPath,
    std::map<std::string, std::string> mergeBodiesInto)
{
  OpenSimFile file = parseOsim(uri);

  std::shared_ptr<dynamics::Skeleton> simplified
      = file.skeleton->simplifySkeleton(
          file.skeleton->getName(), mergeBodiesInto);
  if (simplified)
  {
    SdfParser::writeSkeleton(outputPath, simplified);
    return true;
  }
  else
  {
    // Unable to simplify skeleton, so unable to write
    return false;
  }
}

/// This does its best to convert a *.osim file to an MJCF file. It will
/// simplify the skeleton by merging any bodies that are requested, and
/// deleting any joints linking those bodies.
bool OpenSimParser::convertOsimToMJCF(
    const common::Uri& uri,
    const std::string& outputPath,
    std::map<std::string, std::string> mergeBodiesInto)
{
  OpenSimFile file = parseOsim(uri);

  std::shared_ptr<dynamics::Skeleton> simplified
      = file.skeleton->simplifySkeleton(
          file.skeleton->getName(), mergeBodiesInto);
  if (simplified)
  {
    MJCFExporter::writeSkeleton(outputPath, simplified);
    return true;
  }
  else
  {
    // Unable to simplify skeleton, so unable to write
    return false;
  }
}

Eigen::Vector3s readAttachedGeometry(
    tinyxml2::XMLElement* attachedGeometry,
    dynamics::BodyNode* childBody,
    Eigen::Isometry3s relativeT,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    const common::ResourceRetrieverPtr& geometryRetriever,
    bool ignoreGeometry)
{
  (void)fileNameForErrorDisplay;

  Eigen::Vector3s avgScale = Eigen::Vector3s::Zero();
  int numScales = 0;

  tinyxml2::XMLElement* meshCursor
      = attachedGeometry->FirstChildElement("Mesh");
  while (meshCursor)
  {
    if (meshCursor->FirstChildElement("mesh_file")->GetText() == nullptr)
    {
      std::cout << "Body Node " << childBody->getName()
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

    if (!scale.hasNaN())
    {
      avgScale += scale;
      numScales++;
    }

    if (!ignoreGeometry)
    {
      common::Uri meshUri = common::Uri::createFromRelativeUri(
          geometryFolder, "./" + mesh_file + ".ply");
      std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr = nullptr;
      try
      {
        meshPtr = dynamics::MeshShape::loadMesh(meshUri, geometryRetriever);
        if (meshPtr)
        {
          std::shared_ptr<dynamics::MeshShape> meshShape
              = std::make_shared<dynamics::MeshShape>(
                  scale, meshPtr, meshUri, geometryRetriever);

          dynamics::ShapeNode* meshShapeNode
              = childBody->createShapeNodeWith<dynamics::VisualAspect>(
                  meshShape);

          Eigen::Isometry3s localT = Eigen::Isometry3s::Identity();
          tinyxml2::XMLElement* transformElem
              = meshCursor->FirstChildElement("transform");
          if (transformElem != nullptr)
          {
            Eigen::Vector6s transformVec = readVec6(transformElem);
            localT.linear() = math::eulerXYZToMatrix(transformVec.head<3>());
            localT.translation() = transformVec.tail<3>();
          }

          meshShapeNode->setRelativeTransform(relativeT * localT);

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
      }
      catch (std::exception& e)
      {
        std::cout << "Error loading mesh " << meshUri.toString() << ": "
                  << e.what() << std::endl;
      }
    }

    meshCursor = meshCursor->NextSiblingElement("Mesh");
  }

  avgScale /= numScales;
  return avgScale;
}

//==============================================================================
std::tuple<dynamics::Joint*, dynamics::BodyNode*, Eigen::Vector3s> createJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    tinyxml2::XMLElement* bodyCursor,
    tinyxml2::XMLElement* jointDetail,
    Eigen::Isometry3s transformFromParent,
    Eigen::Isometry3s transformFromChild,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    const common::ResourceRetrieverPtr& geometryRetriever,
    bool ignoreGeometry)
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
    int firstLinearIndex = -1;
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
      tinyxml2::XMLElement* piecewiseLinear
          = function->FirstChildElement("PiecewiseLinearFunction");
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
          piecewiseLinear
              = childFunction->FirstChildElement("PiecewiseLinearFunction");
          linearFunction = childFunction->FirstChildElement("LinearFunction");
          polynomialFunction
              = childFunction->FirstChildElement("PolynomialFunction");
          assert(
              constant || simmSpline || piecewiseLinear || linearFunction
              || polynomialFunction);
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

        s_t value
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
        if (firstLinearIndex == -1)
        {
          firstLinearIndex = dofIndex;
        }
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
        std::reverse(coeffs.begin(), coeffs.end());
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
      else if (piecewiseLinear != nullptr)
      {
        anySpline = true;
        allLocked = false;
        if (dofIndex < 3)
        {
          first3Linear = false;
        }

        std::vector<s_t> x = readVecX(piecewiseLinear->FirstChildElement("x"));
        std::vector<s_t> y = readVecX(piecewiseLinear->FirstChildElement("y"));
        for (int i = 0; i < y.size(); i++)
        {
          y[i] *= scale;
        }
        customFunctions.push_back(
            std::make_shared<math::PiecewiseLinearFunction>(x, y));
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
    else if (allLinear && !anySpline && dofNames.size() == 6)
    {
      auto axisOrderAndErrorFlag = getAxisOrder(eulerAxisOrder);
      dynamics::EulerJoint::AxisOrder axisOrder = axisOrderAndErrorFlag.first;
      bool axisOrderError = axisOrderAndErrorFlag.second;
      if (axisOrderError)
      {
        NIMBLE_THROW("Invalid axis order for Euler joint: " + jointName);
      }
      dynamics::EulerJoint::AxisOrder transOrder
          = getAxisOrder(transformAxisOrder).first;
      if (transOrder != dynamics::EulerJoint::AxisOrder::XYZ)
      {
        NIMBLE_THROW("Invalid transform order for Euler joint: " + jointName);
      }

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
    else if (
        first3Linear && !anySpline
        && (dofNames.size() == 3 || dofNames.size() == 6))
    {
      bool anyNonUnit = false;
      for (int i = 0; i < 3; i++)
      {
        bool isUnit = false;
        for (int j = 0; j < 3; j++)
        {
          if (eulerAxisOrder[i] == Eigen::Vector3s::Unit(j)
              || eulerAxisOrder[i] == -Eigen::Vector3s::Unit(j))
          {
            isUnit = true;
            break;
          }
        }
        if (!isUnit)
        {
          anyNonUnit = true;
          break;
        }
      }
      if (anyNonUnit)
      {
        s_t xDotY = abs(eulerAxisOrder[0].dot(eulerAxisOrder[1]));
        s_t yDotZ = abs(eulerAxisOrder[1].dot(eulerAxisOrder[2]));
        s_t zDotX = abs(eulerAxisOrder[2].dot(eulerAxisOrder[0]));
        if (xDotY < 1e-4 && yDotZ < 1e-4 && zDotX < 1e-4)
        {
          // Construct a rotation matrix to get these rotations back to normal
          // euler angles
          Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
          R.col(0) = eulerAxisOrder[0].normalized();
          R.col(1) = eulerAxisOrder[1].normalized();
          R.col(2) = eulerAxisOrder[2].normalized();
          for (int i = 0; i < 10; i++)
          {
            R.col(0) = R.col(0) - (R.col(0).dot(R.col(1)) * R.col(1))
                       - (R.col(0).dot(R.col(2)) * R.col(2));
            R.col(1) = R.col(1) - (R.col(1).dot(R.col(0)) * R.col(0))
                       - (R.col(1).dot(R.col(2)) * R.col(2));
            R.col(2) = R.col(2) - (R.col(2).dot(R.col(0)) * R.col(0))
                       - (R.col(2).dot(R.col(1)) * R.col(1));
          }
          // Now R contains an approximately orthonormal basis to the euler axis
          // we were originally passed.
          s_t errorOfCross = (R.col(0).cross(R.col(1)) - R.col(2)).norm();

          // We need to check the "handedness" of R
          dynamics::EulerJoint::AxisOrder axisOrder;
          if (errorOfCross < 1e-4)
          {
            axisOrder = dynamics::EulerJoint::AxisOrder::XYZ;
          }
          else
          {
            axisOrder = dynamics::EulerJoint::AxisOrder::XZY;
            // Flip the order of the columns to make the handed-ness work
            Eigen::Vector3s tmp = R.col(2);
            R.col(2) = R.col(1);
            R.col(1) = tmp;
          }

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
                = parentBody
                      ->createChildJointAndBodyNodePair<dynamics::EulerJoint>(
                          props, bodyProps);
            eulerJoint = pair.first;
            childBody = pair.second;
          }
          eulerJoint->setFlipAxisMap(Eigen::Vector3s::Ones());
          eulerJoint->setAxisOrder(axisOrder);
          Eigen::Isometry3s inFrame = Eigen::Isometry3s::Identity();
          inFrame.linear() = R;
          eulerJoint->setTransformFromChildBodyNode(inFrame);
          eulerJoint->setTransformFromParentBodyNode(
              inFrame); // this is inFrame.inverse().inverse()

          joint = eulerJoint;
        }
        else
        {
          assert(false && "3 rotation axis are not mutually orthogonal");
        }
      }
      else
      {
        auto axisOrderAndErrorFlag = getAxisOrder(eulerAxisOrder);
        dynamics::EulerJoint::AxisOrder axisOrder = axisOrderAndErrorFlag.first;
        bool axisOrderError = axisOrderAndErrorFlag.second;
        if (axisOrderError)
        {
          NIMBLE_THROW("Invalid axis order for Euler joint: " + jointName);
        }
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
              = parentBody
                    ->createChildJointAndBodyNodePair<dynamics::EulerJoint>(
                        props, bodyProps);
          eulerJoint = pair.first;
          childBody = pair.second;
        }
        eulerJoint->setFlipAxisMap(flips);
        eulerJoint->setAxisOrder(axisOrder);
        joint = eulerJoint;
      }
    }
    else if (numLinear == 1 && numConstant == 5 && !anySpline)
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
    else if (
        numLinear == 2 && numConstant == 4 && lastLinearIndex < 3 && !anySpline)
    {
      // Create a RevoluteJoint
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
      universalJoint->setAxis1(eulerAxisOrder[firstLinearIndex]);
      universalJoint->setAxis2(eulerAxisOrder[lastLinearIndex]);

      joint = universalJoint;
    }
    else
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
      else if (dofNames.size() == 4)
      {
        auto pair = createCustomJoint<4>(
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
      else if (dofNames.size() == 5)
      {
        auto pair = createCustomJoint<5>(
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
      else if (dofNames.size() == 6)
      {
        auto pair = createCustomJoint<6>(
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
  }
  else if (jointType == "WeldJoint")
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
  else if (jointType == "PinJoint")
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
  else if (jointType == "UniversalJoint")
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
  else if (jointType == "EllipsoidJoint")
  {
    // Create a EllipsoidJoint
    dynamics::EllipsoidJoint* ellipsoidJoint = nullptr;
    dynamics::EllipsoidJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<dynamics::EllipsoidJoint>(
          nullptr, props, bodyProps);
      ellipsoidJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair
          = parentBody
                ->createChildJointAndBodyNodePair<dynamics::EllipsoidJoint>(
                    props, bodyProps);
      ellipsoidJoint = pair.first;
      childBody = pair.second;
    }
    ellipsoidJoint->setAxisOrder(dynamics::EulerJoint::AxisOrder::XYZ);
    auto* radiiElem = jointDetail->FirstChildElement("radii_x_y_z");
    if (radiiElem != nullptr)
    {
      ellipsoidJoint->setEllipsoidRadii(readVec3(radiiElem));
    }

    joint = ellipsoidJoint;
  }
  else if (jointType == "ScapulothoracicJoint")
  {
    // Create a ScapulathorasicJoint
    dynamics::ScapulathoracicJoint* scapulothoracicJoint = nullptr;
    dynamics::ScapulathoracicJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair
          = skel->createJointAndBodyNodePair<dynamics::ScapulathoracicJoint>(
              nullptr, props, bodyProps);
      scapulothoracicJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair = parentBody->createChildJointAndBodyNodePair<
          dynamics::ScapulathoracicJoint>(props, bodyProps);
      scapulothoracicJoint = pair.first;
      childBody = pair.second;
    }
    auto* radiiElem
        = jointDetail->FirstChildElement("thoracic_ellipsoid_radii_x_y_z");
    if (radiiElem != nullptr)
    {
      scapulothoracicJoint->setEllipsoidRadii(readVec3(radiiElem));
    }
    auto* wingingOffsetElem
        = jointDetail->FirstChildElement("scapula_winging_axis_origin");
    if (wingingOffsetElem != nullptr)
    {
      scapulothoracicJoint->setWingingAxisOffset(readVec2(wingingOffsetElem));
    }
    auto* wingingNeutralDirectionElem
        = jointDetail->FirstChildElement("scapula_winging_axis_direction");
    if (wingingNeutralDirectionElem != nullptr)
    {
      scapulothoracicJoint->setWingingAxisDirection(
          atof(wingingNeutralDirectionElem->GetText()));
    }

    joint = scapulothoracicJoint;
  }
  else if (jointType == "ConstantCurvatureJoint")
  {
    // Create a ConstantCurvatureJoint
    dynamics::ConstantCurveIncompressibleJoint* curveJoint = nullptr;
    dynamics::ConstantCurveIncompressibleJoint::Properties props;
    props.mName = jointName;
    if (parentBody == nullptr)
    {
      auto pair = skel->createJointAndBodyNodePair<
          dynamics::ConstantCurveIncompressibleJoint>(
          nullptr, props, bodyProps);
      curveJoint = pair.first;
      childBody = pair.second;
    }
    else
    {
      auto pair = parentBody->createChildJointAndBodyNodePair<
          dynamics::ConstantCurveIncompressibleJoint>(props, bodyProps);
      curveJoint = pair.first;
      childBody = pair.second;
    }
    auto* lengthElem = jointDetail->FirstChildElement("length");
    if (lengthElem != nullptr)
    {
      s_t len = atof(lengthElem->GetText());
      std::cout << "Setting len to " << len << std::endl;
      curveJoint->setLength(len);
    }
    auto* neutralPos = jointDetail->FirstChildElement("neutral_angle_x_z_y");
    if (neutralPos != nullptr)
    {
      Eigen::Vector3s neutralVec = readVec3(neutralPos);
      std::cout << "Setting neutral pos to " << neutralVec << std::endl;
      curveJoint->setNeutralPos(neutralVec);
      curveJoint->setPositions(neutralVec);
    }

    joint = curveJoint;
  }
  else
  {
    NIMBLE_THROW("Nimble OpenSimParser doesn't yet support joint "
        "type \"" + jointType + "\". Nimble's support for OpenSim features is "
        "still under construction, so support may be added in the future. For "
        "now, though, we're exiting with failure.");
  }

  assert(childBody != nullptr);
  joint->setName(jointName);
  // std::cout << jointName << std::endl;

  // If there's a non-zero tranformation from the parent or child, it's almost
  // certainly because we're compensating for some squirrely specification of a
  // EulerJoint in a different frame, so we want to preserve that offset.
  joint->setTransformFromChildBodyNode(
      transformFromChild * joint->getTransformFromChildBodyNode());
  joint->setTransformFromParentBodyNode(
      transformFromParent * joint->getTransformFromParentBodyNode());

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
    bool locked
        = coordinateCursor->FirstChildElement("locked") != nullptr
          && trim(std::string(
                 coordinateCursor->FirstChildElement("locked")->GetText()))
                 == "true";
    bool clamped
        = coordinateCursor->FirstChildElement("clamped") != nullptr
          && trim(std::string(
                 coordinateCursor->FirstChildElement("clamped")->GetText()))
                 == "true";

    if (joint->getNumDofs() <= i)
    {
      break;
    }

    dynamics::DegreeOfFreedom* dof = joint->getDof(i);
    dof->setName(dofName);

    if (coordinateCursor->FirstChildElement("default_value"))
    {
      double defaultValue = atof(
          coordinateCursor->FirstChildElement("default_value")->GetText());
      dof->setPosition(defaultValue);
      dof->setInitialPosition(defaultValue);
    }

    if (coordinateCursor->FirstChildElement("default_speed_value"))
    {
      double defaultSpeedValue
          = atof(coordinateCursor->FirstChildElement("default_speed_value")
                     ->GetText());
      dof->setVelocity(defaultSpeedValue);
    }

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
      if (coordinateCursor->FirstChildElement("range"))
      {
        Eigen::Vector2s range
            = readVec2(coordinateCursor->FirstChildElement("range"));
        dof->setPositionLowerLimit(range(0));
        dof->setPositionUpperLimit(range(1));

        // This prevents warnings when copying the skeleton about out-of-bounds
        // rest positions
        dof->setRestPosition((range(0) + range(1)) / 2.0);
      }
    }
    if (locked)
    {
      // TODO: Just replace with a Weld joint
      /*
      dof->setVelocityUpperLimit(0);
      dof->setVelocityLowerLimit(0);
      */
      // std::cout << "Locking dof " << dof->getName() << std::endl;
      if (coordinateCursor->FirstChildElement("default_value"))
      {
        double defaultValue = atof(
            coordinateCursor->FirstChildElement("default_value")->GetText());
        dof->setPositionLowerLimit(defaultValue);
        dof->setPositionUpperLimit(defaultValue);
        dof->setVelocityLowerLimit(0);
        dof->setVelocityUpperLimit(0);
        dof->setAccelerationLowerLimit(0);
        dof->setAccelerationUpperLimit(0);
        // This prevents warnings when copying the skeleton about out-of-bounds
        // rest positions
        dof->setRestPosition(defaultValue);
      }
    }

    i++;
    coordinateCursor = coordinateCursor->NextSiblingElement("Coordinate");
  }

  Eigen::Vector3s avgScale = Eigen::Vector3s::Zero();

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
        int numScalesCounted = 0;
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

          if (!scale.hasNaN())
          {
            avgScale += scale;
            numScalesCounted++;
          }

          if (!ignoreGeometry)
          {
            common::Uri meshUri = common::Uri::createFromRelativeUri(
                geometryFolder, "./" + mesh_file + ".ply");
            std::shared_ptr<dynamics::SharedMeshWrapper> meshPtr
                = dynamics::MeshShape::loadMesh(meshUri, geometryRetriever);

            if (meshPtr)
            {
              std::shared_ptr<dynamics::MeshShape> meshShape
                  = std::make_shared<dynamics::MeshShape>(
                      scale, meshPtr, meshUri, geometryRetriever);

              dynamics::ShapeNode* meshShapeNode
                  = childBody->createShapeNodeWith<dynamics::VisualAspect>(
                      meshShape);
              meshShapeNode->setRelativeTransform(transform);

              dynamics::VisualAspect* meshVisualAspect
                  = meshShapeNode->getVisualAspect();
              meshVisualAspect->setColor(colors);
              meshVisualAspect->setAlpha(opacity);
            }
          }

          displayGeometryCursor
              = displayGeometryCursor->NextSiblingElement("DisplayGeometry");
        }

        avgScale /= numScalesCounted;
      }
    }
  }
  // OpenSim v4 files can also specify visible geometry this way
  tinyxml2::XMLElement* componentsElem
      = bodyCursor->FirstChildElement("components");
  if (componentsElem != nullptr)
  {
    tinyxml2::XMLElement* frameCursor
        = componentsElem->FirstChildElement("PhysicalOffsetFrame");
    while (frameCursor != nullptr)
    {
      Eigen::Isometry3s relativeT = Eigen::Isometry3s::Identity();

      tinyxml2::XMLElement* frameTranslationElem
          = frameCursor->FirstChildElement("translation");
      if (frameTranslationElem != nullptr)
      {
        relativeT.translation() = readVec3(frameTranslationElem);
      }
      tinyxml2::XMLElement* frameOrientationElem
          = frameCursor->FirstChildElement("orientation");
      if (frameOrientationElem != nullptr)
      {
        relativeT.linear()
            = math::eulerXYZToMatrix(readVec3(frameOrientationElem));
      }
      // OpenSim v3 files specify visible geometry this way
      tinyxml2::XMLElement* frameAttachedGeometry
          = frameCursor->FirstChildElement("attached_geometry");
      if (frameAttachedGeometry && childBody != nullptr)
      {
        avgScale = readAttachedGeometry(
            frameAttachedGeometry,
            childBody,
            relativeT,
            fileNameForErrorDisplay,
            geometryFolder,
            geometryRetriever,
            ignoreGeometry);
      }

      frameCursor = frameCursor->NextSiblingElement("PhysicalOffsetFrame");
    }
  }
  // OpenSim v3 files specify visible geometry this way
  tinyxml2::XMLElement* attachedGeometry
      = bodyCursor->FirstChildElement("attached_geometry");
  if (attachedGeometry && childBody != nullptr)
  {
    avgScale = readAttachedGeometry(
        attachedGeometry,
        childBody,
        Eigen::Isometry3s::Identity(),
        fileNameForErrorDisplay,
        geometryFolder,
        geometryRetriever,
        ignoreGeometry);
  }

  if (childBody == nullptr)
  {
    NIMBLE_THROW("Nimble OpenSimParser caught an error reading Joint \"" + jointName + "\". It has no child body. Please check that your OpenSim file is valid.");
  }

  return std::tuple<dynamics::Joint*, dynamics::BodyNode*, Eigen::Vector3s>(
      joint, childBody, avgScale);
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim30(
    tinyxml2::XMLElement* docElement,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    const common::ResourceRetrieverPtr& geometryRetriever,
    bool ignoreGeometry)
{
  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");

  if (bodySet == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] missing <BodySet> group.\n";
    OpenSimFile file;
    file.skeleton = nullptr;
    return file;
  }

  OpenSimFile file;

  //--------------------------------------------------------------------------
  // Build out the physical structure
  map<string, dynamics::BodyNode*> bodyLookupMap;
  map<string, pair<string, Eigen::Isometry3s>> frameTransforms;
  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor)
  {
    std::string name = trim(std::string(bodyCursor->Attribute("name")));
    // std::cout << name << std::endl;

    if (name == "ground")
    {
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }
    // Skip the kneecaps
    if (name == "patella_r" || name == "patella_l")
    {
      file.ignoredBodies.push_back(name);
      file.warnings.push_back(
          "Ignoring body \"" + name
          + "\" because it is driven by a constrained joint.");
      bodyCursor = bodyCursor->NextSiblingElement();
      continue;
    }

    // Collect any frames that are attached to this body, which may be used for
    // IMUs
    tinyxml2::XMLElement* components
        = bodyCursor->FirstChildElement("components");
    if (components != nullptr)
    {
      tinyxml2::XMLElement* frameCursor
          = components->FirstChildElement("PhysicalOffsetFrame");
      while (frameCursor != nullptr)
      {
        std::string frameName
            = trim(std::string(frameCursor->Attribute("name")));
        std::string frameUri = name + "/" + frameName;
        tinyxml2::XMLElement* frameTranslationElem
            = frameCursor->FirstChildElement("translation");
        Eigen::Isometry3s relativeT = Eigen::Isometry3s::Identity();
        if (frameTranslationElem != nullptr)
        {
          Eigen::Vector3s translation = readVec3(frameTranslationElem);
          relativeT.translation() = translation;
        }
        tinyxml2::XMLElement* frameOrientationElem
            = frameCursor->FirstChildElement("orientation");
        if (frameOrientationElem != nullptr)
        {
          Eigen::Vector3s orientation = readVec3(frameOrientationElem);
          relativeT.linear() = math::eulerXYZToMatrix(orientation);
        }
        frameTransforms[frameUri]
            = std::pair<std::string, Eigen::Isometry3s>(name, relativeT);
        frameCursor = frameCursor->NextSiblingElement("PhysicalOffsetFrame");
      }
    }

    tinyxml2::XMLElement* joint = bodyCursor->FirstChildElement("Joint");

    dynamics::BodyNode* childBody = nullptr;

    if (joint)
    {
      // Implicit WeldJoint
      if (joint->FirstChild() == nullptr)
      {
      }
      tinyxml2::XMLElement* jointDetail = nullptr;
      // Explicit WeldJoint
      tinyxml2::XMLElement* weldJoint = joint->FirstChildElement("WeldJoint");
      if (weldJoint)
      {
        jointDetail = weldJoint;
      }
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
      // ScapulathoracicJoint
      tinyxml2::XMLElement* scapulothoracicJoint
          = joint->FirstChildElement("ScapulothoracicJoint");
      if (scapulothoracicJoint)
      {
        jointDetail = scapulothoracicJoint;
      }

      if (jointDetail != nullptr)
      {
        std::string parentName = trim(std::string(
            jointDetail->FirstChildElement("parent_body")->GetText()));
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

        auto tuple = createJoint(
            skel,
            parentBody,
            bodyCursor,
            jointDetail,
            transformFromParent,
            transformFromChild,
            fileNameForErrorDisplay,
            geometryFolder,
            geometryRetriever,
            ignoreGeometry);
        childBody = std::get<1>(tuple);
        Eigen::Vector3s avgScale = std::get<2>(tuple);
        file.bodyScales[name] = avgScale;
      }
    }
    assert(childBody != nullptr);

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
            = markerCursor->FirstChildElement("fixed") == nullptr
              || markerCursor->FirstChildElement("fixed")->GetText() == nullptr
              || trim(std::string(
                     markerCursor->FirstChildElement("fixed")->GetText()))
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

  tinyxml2::XMLElement* componentSet
      = modelElement->FirstChildElement("ComponentSet");
  if (componentSet != nullptr)
  {
    tinyxml2::XMLElement* objects = componentSet->FirstChildElement("objects");
    if (objects != nullptr)
    {
      tinyxml2::XMLElement* imuCursor = objects->FirstChildElement("IMU");
      while (imuCursor != nullptr)
      {
        std::string name = trim(std::string(imuCursor->Attribute("name")));
        tinyxml2::XMLElement* socketFrame
            = imuCursor->FirstChildElement("socket_frame");
        if (socketFrame != nullptr)
        {
          std::string uri = std::string(socketFrame->GetText());
          std::string socketFrameName = uri.substr(strlen("/bodyset/"));
          if (frameTransforms.count(socketFrameName) > 0)
          {
            file.imuMap[name] = frameTransforms[socketFrameName];
          }
          frameTransforms[socketFrameName];
        }
        imuCursor = imuCursor->NextSiblingElement("IMU");
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
int recursiveCountChildren(OpenSimJointXML* joint)
{
  int numChildren = 1; // every joint has one body as a child
  if (joint->child != nullptr)
  {
    for (auto* childJoint : joint->child->children)
    {
      numChildren += recursiveCountChildren(childJoint);
    }
  }
  return numChildren;
}

//==============================================================================
void recursiveCreateJoint(
    dynamics::SkeletonPtr skel,
    dynamics::BodyNode* parentBody,
    OpenSimJointXML* joint,
    OpenSimFile& file,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    const common::ResourceRetrieverPtr& geometryRetriever,
    bool ignoreGeometry)
{
  (void)skel;
  (void)parentBody;
  (void)joint;

  // std::cout << "Building joint: " << joint->name << std::endl;

  tinyxml2::XMLElement* jointNode = joint->xml;
  tinyxml2::XMLElement* bodyNode = joint->child->xml;

  auto tuple = createJoint(
      skel,
      parentBody,
      bodyNode,
      jointNode,
      joint->fromParent,
      joint->fromChild,
      fileNameForErrorDisplay,
      geometryFolder,
      geometryRetriever,
      ignoreGeometry);

  dynamics::BodyNode* childBody = std::get<1>(tuple);
  file.bodyScales[childBody->getName()] = std::get<2>(tuple);

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
    recursiveCreateJoint(
        skel,
        childBody,
        grandChildJoint,
        file,
        fileNameForErrorDisplay,
        geometryFolder,
        geometryRetriever,
        ignoreGeometry);
  }
}

//==============================================================================
OpenSimFile OpenSimParser::readOsim40(
    tinyxml2::XMLElement* docElement,
    const std::string fileNameForErrorDisplay,
    const std::string geometryFolder,
    const common::ResourceRetrieverPtr& geometryRetriever,
    bool ignoreGeometry)
{
  OpenSimFile null_file;
  null_file.skeleton = nullptr;

  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] does not contain <Model> as the child of the root "
             "<OpenSimDocument> element.\n";
    return null_file;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (bodySet == nullptr || jointSet == nullptr)
  {
    dterr << "OpenSim file[" << fileNameForErrorDisplay
          << "] missing <BodySet> or <JointSet> groups.\n";
    return null_file;
  }

  //--------------------------------------------------------------------------
  // Read BodySet
  map<string, OpenSimBodyXML> bodyLookupMap;
  map<string, pair<string, Eigen::Isometry3s>> frameTransforms;

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

    // Collect any frames that are attached to this body, which may be used for
    // IMUs
    tinyxml2::XMLElement* components
        = bodyCursor->FirstChildElement("components");
    if (components != nullptr)
    {
      tinyxml2::XMLElement* frameCursor
          = components->FirstChildElement("PhysicalOffsetFrame");
      while (frameCursor != nullptr)
      {
        std::string frameName
            = trim(std::string(frameCursor->Attribute("name")));
        std::string frameUri = name + "/" + frameName;
        tinyxml2::XMLElement* frameTranslationElem
            = frameCursor->FirstChildElement("translation");
        Eigen::Isometry3s relativeT = Eigen::Isometry3s::Identity();
        if (frameTranslationElem != nullptr)
        {
          Eigen::Vector3s translation = readVec3(frameTranslationElem);
          relativeT.translation() = translation;
        }
        tinyxml2::XMLElement* frameOrientationElem
            = frameCursor->FirstChildElement("orientation");
        if (frameOrientationElem != nullptr)
        {
          Eigen::Vector3s orientation = readVec3(frameOrientationElem);
          relativeT.linear() = math::eulerXYZToMatrix(orientation);
        }
        frameTransforms[frameUri]
            = std::pair<std::string, Eigen::Isometry3s>(name, relativeT);
        frameCursor = frameCursor->NextSiblingElement("PhysicalOffsetFrame");
      }
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

    string parent_offset_frame;
    if (jointCursor->FirstChildElement("socket_parent_frame"))
    {
      parent_offset_frame = string(
          jointCursor->FirstChildElement("socket_parent_frame")->GetText());
    }
    else if (jointCursor->FirstChildElement(
                 "socket_parent_frame_connectee_name"))
    {
      parent_offset_frame = string(
          jointCursor->FirstChildElement("socket_parent_frame_connectee_name")
              ->GetText());
    }
    else
    {
      std::cout << "OpenSimParser encountered an error! Joint \"" << name
                << "\" does not specify either a <socket_parent_frame> or a "
                   "<socket_parent_frame_connectee_name>! This may be because "
                   "the OpenSim file is in an older and unsupported version. "
                   "Try a newer format."
                << std::endl;
      return null_file;
    }
    string child_offset_frame;
    if (jointCursor->FirstChildElement("socket_child_frame"))
    {
      child_offset_frame = string(
          jointCursor->FirstChildElement("socket_child_frame")->GetText());
    }
    else if (jointCursor->FirstChildElement(
                 "socket_child_frame_connectee_name"))
    {
      child_offset_frame = string(
          jointCursor->FirstChildElement("socket_child_frame_connectee_name")
              ->GetText());
    }
    else
    {
      std::cout << "OpenSimParser encountered an error! Joint \"" << name
                << "\" does not specify either a <socket_child_frame> or a "
                   "<socket_child_frame_connectee_name>! This may be because "
                   "the OpenSim file is particularly old. Try a newer format."
                << std::endl;
      return null_file;
    }

    string parentName = "";
    Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
    string childName = "";
    Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();

    /// Check if the parent and child linkages are defined in frames
    tinyxml2::XMLElement* frames = jointCursor->FirstChildElement("frames");
    if (frames)
    {
      tinyxml2::XMLElement* framesCursor = frames->FirstChildElement();

      while (framesCursor)
      {
        string parent_body;
        if (framesCursor->FirstChildElement("socket_parent"))
        {
          parent_body = string(
              framesCursor->FirstChildElement("socket_parent")->GetText());
        }
        else if (framesCursor->FirstChildElement(
                     "socket_parent_connectee_name"))
        {
          parent_body = string(
              framesCursor->FirstChildElement("socket_parent_connectee_name")
                  ->GetText());
        }
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
    }
    // If the linkages to children and parents weren't defined in the Frames
    // list, then the linkages are just the raw names
    if (childName == "")
    {
      childName = child_offset_frame;
    }
    if (parentName == "")
    {
      parentName = parent_offset_frame;
    }

    // Trim leading "../" because we don't care about relative paths
    while (childName.rfind("../", 0) == 0)
    {
      childName = childName.substr(3);
    }
    while (parentName.rfind("../", 0) == 0)
    {
      parentName = parentName.substr(3);
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
    else if (
        bodyLookupMap.find("/bodyset/" + parentName) != bodyLookupMap.end())
    {
      bodyLookupMap["/bodyset/" + parentName].children.push_back(&joint);
      joint.parent = &bodyLookupMap["/bodyset/" + parentName];
    }
    else
    {
      joint.parent = nullptr;
    }

    if (bodyLookupMap.find(childName) != bodyLookupMap.end())
    {
      joint.child = &bodyLookupMap[childName];
      bodyLookupMap[childName].parent = &joint;
    }
    else if (bodyLookupMap.find("/bodyset/" + childName) != bodyLookupMap.end())
    {
      joint.child = &bodyLookupMap["/bodyset/" + childName];
      bodyLookupMap["/bodyset/" + childName].parent = &joint;
    }
    else
    {
      NIMBLE_THROW("Loading *.osim file: Joint " + name + " has child "
                   "body \"" + childName + "\", yet the child was not "
                   "found in the <BodySet>");
    }

    jointCursor = jointCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Check tree invarients

  std::vector<OpenSimJointXML*> roots;
  for (auto& pair : jointLookupMap)
  {
    if (pair.second.parent == nullptr)
    {
      roots.push_back(&pair.second);
    }
  }

  OpenSimJointXML* root = nullptr;
  if (roots.size() == 0)
  {
    std::cout
        << "Error reading OpenSim file, looks like the joints form a loop "
           "rather than a tree. This is unsupported. Returning a null skeleton."
        << std::endl;
    return null_file;
  }
  else if (roots.size() == 1)
  {
    root = roots[0];
  }
  else if (roots.size() > 1)
  {
    std::cout
        << "WARNING: There is more than one kinematic tree in the OpenSim file:"
        << std::endl;
    root = roots[0];
    int mostChildren = 0;
    for (int i = 0; i < roots.size(); i++)
    {
      int numChildren = recursiveCountChildren(roots[i]);
      std::cout << " - " << roots[i]->name << ": " << numChildren << " children"
                << std::endl;
      if (numChildren > mostChildren)
      {
        root = roots[i];
        mostChildren = numChildren;
      }
    }
    std::cout << "Choosing " << root->name
              << " as root kinematic tree because it has the most children, "
                 "ignoring the others."
              << std::endl;
  }

  assert(root != nullptr);

  //--------------------------------------------------------------------------
  // Build out the physical structure

  OpenSimFile file;

  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();
  root->parentBody = nullptr;
  recursiveCreateJoint(
      skel,
      nullptr,
      root,
      file,
      fileNameForErrorDisplay,
      geometryFolder,
      geometryRetriever,
      ignoreGeometry);

  /*
  std::cout << "Num dofs: " << skel->getNumDofs() << std::endl;
  std::cout << "Num bodies: " << skel->getNumBodyNodes() << std::endl;
  */

  file.skeleton = skel;

  //--------------------------------------------------------------------------
  // Read Marker set

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
        if (offset.hasNaN())
        {
          throw std::runtime_error("Marker " + name + " has a NaN offset in the OpenSim file we are attempting to load!");
        }
        assert(!offset.hasNaN());
        std::string socketName;

        if (markerCursor->FirstChildElement("socket_parent_frame"))
        {
          socketName = markerCursor->FirstChildElement("socket_parent_frame")
                           ->GetText();
        }
        else if (markerCursor->FirstChildElement(
                     "socket_parent_frame_connectee_name"))
        {
          socketName
              = markerCursor
                    ->FirstChildElement("socket_parent_frame_connectee_name")
                    ->GetText();
        }
        else
        {
          std::cout
              << "OpenSimParser encountered an error! Marker \"" << name
              << "\" does not specify either a <socket_parent_frame> or a "
                 "<socket_parent_frame_connectee_name>! This may be because "
                 "the OpenSim file is in an older and unsupported version. "
                 "Try a newer format."
              << std::endl;
          return null_file;
        }

        while (socketName.rfind("../", 0) == 0)
        {
          socketName = socketName.substr(strlen("../"));
        }

        std::string bodyName;
        if (bodyLookupMap.find(socketName) != bodyLookupMap.end())
        {
          bodyName = bodyLookupMap[socketName].name;
        }
        else if (
            bodyLookupMap.find("/bodyset/" + socketName) != bodyLookupMap.end())
        {
          bodyName = bodyLookupMap["/bodyset/" + socketName].name;
        }
        else
        {
          std::cout << "Warning: OpenSimParser attempting to read marker \""
                    << name << "\" attached to socket \"" << socketName
                    << "\" which does not exist! As a last ditch effort to "
                       "recover, we'll "
                       "assume the socket name is the body name."
                    << std::endl;
          bodyName = socketName;
        }

        tinyxml2::XMLElement* fixedElem
            = markerCursor->FirstChildElement("fixed");
        bool fixed = fixedElem == nullptr || fixedElem->GetText() == nullptr
                     || trim(std::string(fixedElem->GetText())) == "true";
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

  //--------------------------------------------------------------------------
  // Read IMUs

  tinyxml2::XMLElement* componentSet
      = modelElement->FirstChildElement("ComponentSet");
  if (componentSet != nullptr)
  {
    tinyxml2::XMLElement* objects = componentSet->FirstChildElement("objects");
    if (objects != nullptr)
    {
      tinyxml2::XMLElement* imuCursor = objects->FirstChildElement("IMU");
      while (imuCursor != nullptr)
      {
        std::string name = trim(std::string(imuCursor->Attribute("name")));
        tinyxml2::XMLElement* socketFrame
            = imuCursor->FirstChildElement("socket_frame");
        if (socketFrame != nullptr)
        {
          std::string uri = std::string(socketFrame->GetText());
          std::string socketFrameName = uri.substr(strlen("/bodyset/"));
          if (frameTransforms.count(socketFrameName) > 0)
          {
            file.imuMap[name] = frameTransforms[socketFrameName];
          }
          frameTransforms[socketFrameName];
        }
        imuCursor = imuCursor->NextSiblingElement("IMU");
      }
    }
  }

  return file;
}

}; // namespace biomechanics
}; // namespace dart
