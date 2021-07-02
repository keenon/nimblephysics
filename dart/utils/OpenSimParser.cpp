#include "dart/utils/OpenSimParser.hpp"

#include "dart/utils/XmlHelpers.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/dynamics/FreeJoint.hpp"

#include <unordered_map>

using namespace std;

namespace dart {
namespace utils {

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
Eigen::Vector2s readVec2(tinyxml2::XMLElement* elem) {
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector2s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  return vec;
}

Eigen::Vector3s readVec3(tinyxml2::XMLElement* elem) {
  const char* c = elem->GetText();
  char* q;
  Eigen::Vector3s vec;
  vec(0) = strtod(c, &q);
  vec(1) = strtod(q, &q);
  vec(2) = strtod(q, &q);
  return vec;
}

Eigen::Vector6s readVec6(tinyxml2::XMLElement* elem) {
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

struct OpenSimBodyXML;

struct OpenSimJointXML {
  string name;
  OpenSimBodyXML* parent;
  OpenSimBodyXML* child;
  Eigen::Isometry3s fromParent;
  Eigen::Isometry3s fromChild;
  tinyxml2::XMLElement* xml;
  dynamics::BodyNode* parentBody;
};

struct OpenSimBodyXML {
  string name;
  OpenSimJointXML* parent;
  std::vector<OpenSimJointXML*> children;
  tinyxml2::XMLElement* xml;
};

//==============================================================================
void buildJoint(dynamics::SkeletonPtr /*skel*/, OpenSimJointXML* joint) {
  // TODO: actually construct the joint
  std::cout << "Building joint: " << joint->name << std::endl;
}

//==============================================================================
dynamics::SkeletonPtr OpenSimParser::readSkeleton(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const common::ResourceRetrieverPtr retriever = ensureRetriever(nullOrRetriever);

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
    return nullptr;
  }

  //--------------------------------------------------------------------------
  tinyxml2::XMLElement* docElement = osimFile.FirstChildElement("OpenSimDocument");
  if (docElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <OpenSimDocument> as the root element.\n";
    return nullptr;
  }

  tinyxml2::XMLElement* modelElement = docElement->FirstChildElement("Model");
  if (modelElement == nullptr)
  {
    dterr << "OpenSim file[" << uri.toString()
          << "] does not contain <Model> as the child of the root <OpenSimDocument> element.\n";
    return nullptr;
  }

  tinyxml2::XMLElement* bodySet = modelElement->FirstChildElement("BodySet");
  tinyxml2::XMLElement* jointSet = modelElement->FirstChildElement("JointSet");

  if (bodySet == nullptr || jointSet == nullptr) {
    dterr << "OpenSim file[" << uri.toString()
          << "] missing <BodySet> or <JointSet> groups.\n";
    return nullptr;
  }

  //--------------------------------------------------------------------------
  // Read BodySet
  unordered_map<string, OpenSimBodyXML> bodyLookupMap;

  tinyxml2::XMLElement* bodySetList = bodySet->FirstChildElement("objects");
  tinyxml2::XMLElement* bodyCursor = bodySetList->FirstChildElement("Body");
  while (bodyCursor) {
    std::string name(bodyCursor->Attribute("name"));
    std::cout << name << std::endl;

    bodyLookupMap["/bodyset/"+name].xml = bodyCursor;
    bodyLookupMap["/bodyset/"+name].parent = nullptr;
    bodyLookupMap["/bodyset/"+name].children.clear();
    bodyLookupMap["/bodyset/"+name].name = "/bodyset/"+name;

    double mass = atof(bodyCursor->FirstChildElement("mass")->GetText());
    std::cout << "Mass: " << mass << std::endl;
    Eigen::Vector3s massCenter = readVec3(bodyCursor->FirstChildElement("mass_center"));
    std::cout << "Mass Center: " << massCenter << std::endl;
    Eigen::Vector6s inertia = readVec6(bodyCursor->FirstChildElement("inertia"));
    std::cout << "Inertia: " << inertia << std::endl;

    tinyxml2::XMLElement* attachedGeometry = bodyCursor->FirstChildElement("attached_geometry");
    if (attachedGeometry) {
      tinyxml2::XMLElement* meshCursor = attachedGeometry->FirstChildElement("Mesh");
      while (meshCursor) {
        std::string mesh_file(meshCursor->FirstChildElement("mesh_file")->GetText());
        std::cout << "\t" << mesh_file << std::endl;

        meshCursor = meshCursor->NextSiblingElement("Mesh");
      }
    }

    bodyCursor = bodyCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Read JointSet
  unordered_map<string, OpenSimJointXML> jointLookupMap;

  tinyxml2::XMLElement* jointSetList = jointSet->FirstChildElement("objects");
  tinyxml2::XMLElement* jointCursor = jointSetList->FirstChildElement();
  while (jointCursor) {
    std::string type(jointCursor->Name());
    std::string name(jointCursor->Attribute("name"));

    string parent_offset_frame = string(jointCursor->FirstChildElement("socket_parent_frame")->GetText());
    string child_offset_frame = string(jointCursor->FirstChildElement("socket_child_frame")->GetText());

    tinyxml2::XMLElement* frames = jointCursor->FirstChildElement("frames");
    tinyxml2::XMLElement* framesCursor = frames->FirstChildElement();

    string parentName = "";
    Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
    string childName = "";
    Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();

    while (framesCursor) {
      string parent_body = string(framesCursor->FirstChildElement("socket_parent")->GetText());
      Eigen::Vector3s translation = readVec3(framesCursor->FirstChildElement("translation"));
      Eigen::Vector3s rotationXYZ = readVec3(framesCursor->FirstChildElement("orientation"));
      Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
      T.linear() = math::eulerXYZToMatrix(rotationXYZ);
      T.translation() = translation;

      string name(framesCursor->Attribute("name"));
      if (name == parent_offset_frame) {
        parentName = parent_body;
        fromParent = T;
      }
      else if (name == child_offset_frame) {
        childName = parent_body;
        // TODO: do we want T.inverse() here?
        fromChild = T;
      }

      framesCursor = framesCursor->NextSiblingElement();
    }

    OpenSimJointXML& joint = jointLookupMap[name];
    joint.name = name;
    joint.fromParent = fromParent;
    joint.fromChild = fromChild;
    joint.xml = jointCursor;

    // A body can have only one parent joint
    if (bodyLookupMap.find(parentName) != bodyLookupMap.end()) {
      bodyLookupMap[parentName].children.push_back(&joint);
      joint.parent = &bodyLookupMap[parentName];
    }
    else {
      joint.parent = nullptr;
    }

    assert(bodyLookupMap.find(childName) != bodyLookupMap.end());
    assert(bodyLookupMap[childName].parent == nullptr);
    joint.child = &bodyLookupMap[childName];
    bodyLookupMap[childName].parent = &joint;

    if (type == "CustomJoint") {
      tinyxml2::XMLElement* coordinates = jointCursor->FirstChildElement("coordinates");
      tinyxml2::XMLElement* coordinatesCursor = coordinates->FirstChildElement("Coordinate");
      int numDofs = 0;
      while (coordinatesCursor) {
        numDofs++;
        coordinatesCursor = coordinatesCursor->NextSiblingElement();
      }

      coordinatesCursor = coordinates->FirstChildElement("Coordinate");
      while (coordinatesCursor) {
        /*
        std::string name(coordinatesCursor->Attribute("name"));
        s_t default_value = static_cast<s_t>(atof(coordinatesCursor->FirstChildElement("default_value")->GetText()));
        bool clamped = std::string(coordinatesCursor->FirstChildElement("clamped")->GetText()) == "true";
        bool locked = std::string(coordinatesCursor->FirstChildElement("locked")->GetText()) == "true";
        Eigen::Vector2s bounds = readVec2(coordinatesCursor->FirstChildElement("range"));
        std::cout << "\t" << name << " - " << default_value << " - " << clamped << " - " << locked << "-" << bounds << std::endl;
        */
        coordinatesCursor = coordinatesCursor->NextSiblingElement();
      }
    }
    else if (type == "PinJoint") {

    }
    else if (type == "UniversalJoint") {

    }
    jointCursor = jointCursor->NextSiblingElement();
  }

  //--------------------------------------------------------------------------
  // Check tree invarients

  OpenSimJointXML* root = nullptr;
  for (auto& pair: jointLookupMap) {
    if (pair.second.parent == nullptr) {
      assert(root == nullptr);
      root = &pair.second;
    }
  }

  assert(root != nullptr);

  //--------------------------------------------------------------------------
  // Build out the physical structure

  dynamics::SkeletonPtr skel = dynamics::Skeleton::create();
  root->parentBody = nullptr;
  buildJoint(skel, root);

  return skel;
}

};
};