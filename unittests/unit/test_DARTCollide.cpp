#define _USE_MATH_DEFINES
#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/collision/dart/DARTCollide.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace realtime;

// #define ALL_TESTS

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_SUPPORT_X)
{
  Eigen::Vector3d boxSize = Eigen::Vector3d::Ones();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();

  ccdBox box;
  box.size = &boxSize;
  box.transform = &boxTransform;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.0;
  dir.v[2] = 0.0;

  ccd_vec3_t out;

  ccdSupportBox(&box, &dir, &out);

  EXPECT_EQ(out.v[0], 0.5);
  EXPECT_EQ(out.v[1], 0.0);
  EXPECT_EQ(out.v[2], 0.0);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_SUPPORT_ROTATED_X_Y)
{
  Eigen::Vector3d boxSize = Eigen::Vector3d::Ones();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();
  // Swap X and Y axis in rotation
  boxTransform.matrix()(0, 0) = 0;
  boxTransform.matrix()(1, 1) = 0;
  boxTransform.matrix()(1, 0) = 1;
  boxTransform.matrix()(0, 1) = 1;

  ccdBox box;
  box.size = &boxSize;
  box.transform = &boxTransform;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.0;
  dir.v[2] = 0.0;

  ccd_vec3_t out;

  ccdSupportBox(&box, &dir, &out);

  EXPECT_EQ(out.v[0], 0.5);
  EXPECT_EQ(out.v[1], 0.0);
  EXPECT_EQ(out.v[2], 0.0);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_SUPPORT_ROTATED_45)
{
  Eigen::Vector3d boxSize = Eigen::Vector3d::Ones();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();
  boxTransform = boxTransform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));

  ccdBox box;
  box.size = &boxSize;
  box.transform = &boxTransform;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.0;
  dir.v[2] = 0.0;

  ccd_vec3_t out;

  ccdSupportBox(&box, &dir, &out);

  EXPECT_NEAR(out.v[0], sqrt(0.5), 1e-8);
  EXPECT_NEAR(out.v[1], 0.0, 1e-8);
  EXPECT_NEAR(out.v[2], 0.0, 1e-8);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_SUPPORT_ROTATED_45_TRANSLATED)
{
  Eigen::Vector3d boxSize = Eigen::Vector3d::Ones();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();
  boxTransform = boxTransform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));
  boxTransform.translation() = Eigen::Vector3d::Random() * 100;

  ccdBox box;
  box.size = &boxSize;
  box.transform = &boxTransform;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.0;
  dir.v[2] = 0.0;

  ccd_vec3_t out;

  ccdSupportBox(&box, &dir, &out);

  EXPECT_NEAR(out.v[0], boxTransform.translation()(0) + sqrt(0.5), 1e-8);
  EXPECT_NEAR(out.v[1], boxTransform.translation()(1), 1e-8);
  EXPECT_NEAR(out.v[2], boxTransform.translation()(2), 1e-8);
}
#endif

aiScene* createBoxMeshUnsafe()
{
  aiScene* scene = (aiScene*)malloc(sizeof(aiScene));
  scene->mNumMeshes = 1;
  aiMesh* mesh = (aiMesh*)malloc(sizeof(aiMesh));
  aiMesh** arr = (aiMesh**)malloc(sizeof(void*) * scene->mNumMeshes);
  scene->mMeshes = arr;
  scene->mMeshes[0] = mesh;
  scene->mMaterials = nullptr;

  mesh->mNormals = nullptr;
  mesh->mNumUVComponents[0] = 0;

  // Create vertices

  mesh->mNumVertices = 8;
  mesh->mVertices = (aiVector3D*)malloc(sizeof(aiVector3D) * 10);

  // Bottom face

  mesh->mVertices[0].x = -.5;
  mesh->mVertices[0].y = -.5;
  mesh->mVertices[0].z = -.5;

  mesh->mVertices[1].x = .5;
  mesh->mVertices[1].y = -.5;
  mesh->mVertices[1].z = -.5;

  mesh->mVertices[2].x = .5;
  mesh->mVertices[2].y = .5;
  mesh->mVertices[2].z = -.5;

  mesh->mVertices[3].x = -.5;
  mesh->mVertices[3].y = .5;
  mesh->mVertices[3].z = -.5;

  // Top face

  mesh->mVertices[4].x = -.5;
  mesh->mVertices[4].y = -.5;
  mesh->mVertices[4].z = .5;

  mesh->mVertices[5].x = .5;
  mesh->mVertices[5].y = -.5;
  mesh->mVertices[5].z = .5;

  mesh->mVertices[6].x = .5;
  mesh->mVertices[6].y = .5;
  mesh->mVertices[6].z = .5;

  mesh->mVertices[7].x = -.5;
  mesh->mVertices[7].y = .5;
  mesh->mVertices[7].z = .5;

  // Create faces

  mesh->mNumFaces = 12;
  mesh->mFaces = (aiFace*)malloc(sizeof(aiFace) * mesh->mNumFaces);
  for (int i = 0; i < mesh->mNumFaces; i++)
  {
    mesh->mFaces[i].mIndices = (unsigned int*)malloc(sizeof(unsigned int) * 3);
    mesh->mFaces[i].mNumIndices = 3;
  }

  // Bottom face

  mesh->mFaces[0].mIndices[0] = 0;
  mesh->mFaces[0].mIndices[1] = 1;
  mesh->mFaces[0].mIndices[2] = 2;

  mesh->mFaces[1].mIndices[0] = 0;
  mesh->mFaces[1].mIndices[1] = 2;
  mesh->mFaces[1].mIndices[2] = 3;

  // Top face

  mesh->mFaces[2].mIndices[0] = 4;
  mesh->mFaces[2].mIndices[1] = 5;
  mesh->mFaces[2].mIndices[2] = 6;

  mesh->mFaces[3].mIndices[0] = 4;
  mesh->mFaces[3].mIndices[1] = 6;
  mesh->mFaces[3].mIndices[2] = 7;

  // Left face

  mesh->mFaces[4].mIndices[0] = 0;
  mesh->mFaces[4].mIndices[1] = 1;
  mesh->mFaces[4].mIndices[2] = 5;

  mesh->mFaces[5].mIndices[0] = 0;
  mesh->mFaces[5].mIndices[1] = 4;
  mesh->mFaces[5].mIndices[2] = 5;

  // Right face

  mesh->mFaces[6].mIndices[0] = 2;
  mesh->mFaces[6].mIndices[1] = 3;
  mesh->mFaces[6].mIndices[2] = 6;

  mesh->mFaces[7].mIndices[0] = 3;
  mesh->mFaces[7].mIndices[1] = 6;
  mesh->mFaces[7].mIndices[2] = 7;

  // Back face

  mesh->mFaces[8].mIndices[0] = 1;
  mesh->mFaces[8].mIndices[1] = 2;
  mesh->mFaces[8].mIndices[2] = 5;

  mesh->mFaces[9].mIndices[0] = 2;
  mesh->mFaces[9].mIndices[1] = 5;
  mesh->mFaces[9].mIndices[2] = 6;

  // Front face

  mesh->mFaces[10].mIndices[0] = 0;
  mesh->mFaces[10].mIndices[1] = 3;
  mesh->mFaces[10].mIndices[2] = 4;

  mesh->mFaces[11].mIndices[0] = 3;
  mesh->mFaces[11].mIndices[1] = 4;
  mesh->mFaces[11].mIndices[2] = 7;

  mesh->mNumFaces = 12;

  return scene;
}

void verifyBoxMeshResultsIdenticalToAnalytical(ccdBox* box1, ccdBox* box2)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(box1, box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(box2, &dir, true);

  CollisionResult meshResult;
  createMeshMeshContacts(nullptr, nullptr, meshResult, &dir, pointsA, pointsB);

  CollisionResult analyticalResult;
  collideBoxBox(
      nullptr,
      nullptr,
      *box1->size,
      *box1->transform,
      *box2->size,
      *box2->transform,
      analyticalResult);

  EXPECT_EQ(meshResult.getNumContacts(), analyticalResult.getNumContacts());

  Eigen::Vector3d randDirection = Eigen::Vector3d::Random();
  meshResult.sortContacts(randDirection);
  analyticalResult.sortContacts(randDirection);

  for (int i = 0; i < meshResult.getNumContacts(); i++)
  {
    Contact& meshContact = meshResult.getContact(i);
    Contact& analyticalContact = analyticalResult.getContact(i);

    // Don't actually enforce this, cause there are places where the meshes are
    // actually more specific and correct.

    // EXPECT_EQ(meshContact.type, analyticalContact.type);

    // The tolerances on the point equality is pretty huge, because again the
    // meshes are actually more specific, because they pick each contact point
    // at specific vertices, which can be offset by the intersection distance.

    if (!equals(meshContact.point, analyticalContact.point, 2e-2))
    {
      std::cout << "Analytical got point: " << std::endl
                << analyticalContact.point << std::endl
                << "Mesh got point: " << std::endl
                << meshContact.point << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.point, analyticalContact.point, 2e-2));
    if (!equals(meshContact.normal, analyticalContact.normal))
    {
      std::cout << "Analytical got normal: " << std::endl
                << analyticalContact.normal << std::endl
                << "Mesh got normal: " << std::endl
                << meshContact.normal << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.normal, analyticalContact.normal));
    EXPECT_NEAR(
        meshContact.penetrationDepth, analyticalContact.penetrationDepth, 1e-8);
  }
}

void verifyMeshAndBoxResultsIdentical(
    Eigen::Vector3d size1,
    Eigen::Isometry3d T1,
    Eigen::Vector3d size2,
    Eigen::Isometry3d T2)
{
  aiScene* boxMesh = createBoxMeshUnsafe();

  CollisionResult meshResult;
  collideMeshMesh(
      nullptr, nullptr, boxMesh, size1, T1, boxMesh, size2, T2, meshResult);

  CollisionResult analyticalResult;
  collideBoxBox(nullptr, nullptr, size1, T1, size2, T2, analyticalResult);

  EXPECT_EQ(meshResult.getNumContacts(), analyticalResult.getNumContacts());

  CollisionResult meshResultBackwards;
  collideMeshMesh(
      nullptr,
      nullptr,
      boxMesh,
      size2,
      T2,
      boxMesh,
      size1,
      T1,
      meshResultBackwards);

  CollisionResult analyticalResultBackwards;
  collideBoxBox(
      nullptr, nullptr, size2, T2, size1, T1, analyticalResultBackwards);

  Eigen::Vector3d randDirection = Eigen::Vector3d::Random();
  meshResult.sortContacts(randDirection);
  analyticalResult.sortContacts(randDirection);
  meshResultBackwards.sortContacts(randDirection);
  analyticalResultBackwards.sortContacts(randDirection);

  for (int i = 0; i < meshResult.getNumContacts(); i++)
  {
    Contact& meshContact = meshResult.getContact(i);
    Contact& analyticalContact = analyticalResult.getContact(i);
    Contact& meshContactBackwards = meshResultBackwards.getContact(i);
    Contact& analyticalContactBackwards
        = analyticalResultBackwards.getContact(i);

    // Don't actually enforce this, cause there are places where the meshes are
    // actually more specific and correct.

    // EXPECT_EQ(meshContact.type, analyticalContact.type);

    // The tolerances on the point equality is pretty huge, because again the
    // meshes are actually more specific, because they pick each contact point
    // at specific vertices, which can be offset by the intersection distance.

    if (!equals(meshContact.point, analyticalContact.point, 2e-2))
    {
      std::cout << "Analytical got point: " << std::endl
                << analyticalContact.point << std::endl
                << "Mesh got point: " << std::endl
                << meshContact.point << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.point, analyticalContact.point, 2e-2));
    EXPECT_TRUE(
        equals(meshContactBackwards.point, analyticalContact.point, 2e-2));
    EXPECT_TRUE(equals(
        analyticalContactBackwards.point, analyticalContact.point, 2e-2));
    if (!equals(meshContact.normal, analyticalContact.normal))
    {
      std::cout << "Analytical got normal: " << std::endl
                << analyticalContact.normal << std::endl
                << "Mesh got normal: " << std::endl
                << meshContact.normal << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.normal, analyticalContact.normal));

    Eigen::Vector3d reverseMeshBackwardsNormal
        = meshContactBackwards.normal * -1;
    if (!equals(reverseMeshBackwardsNormal, analyticalContact.normal))
    {
      std::cout << "Analytical got normal: " << std::endl
                << analyticalContact.normal << std::endl
                << "(Backwards Mesh * -1) got normal: " << std::endl
                << reverseMeshBackwardsNormal << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.normal, reverseMeshBackwardsNormal));

    Eigen::Vector3d reverseAnalyticalBackwardsNormal
        = analyticalContactBackwards.normal * -1;
    if (!equals(reverseAnalyticalBackwardsNormal, analyticalContact.normal))
    {
      std::cout << "Analytical got normal: " << std::endl
                << analyticalContact.normal << std::endl
                << "(Backwards Analytical * -1) got normal: " << std::endl
                << reverseAnalyticalBackwardsNormal << std::endl;
    }
    EXPECT_TRUE(equals(meshContact.normal, reverseAnalyticalBackwardsNormal));

    EXPECT_NEAR(
        meshContact.penetrationDepth, analyticalContact.penetrationDepth, 1e-8);
  }
}

#ifdef ALL_TESTS
TEST(DARTCollide, EDGE_EDGE_2D_COLLISION_TEST)
{
  Eigen::Vector2d a1 = Eigen::Vector2d(-1, 0);
  Eigen::Vector2d a2 = Eigen::Vector2d(1, 0);
  Eigen::Vector2d b1 = Eigen::Vector2d(0, -1);
  Eigen::Vector2d b2 = Eigen::Vector2d(0, 1);

  Eigen::Vector2d out = Eigen::Vector2d::Random();

  bool hit = get2DLineIntersection(a1, a2, b1, b2, out);
  // they do cross
  EXPECT_TRUE(hit);
  // at the origin
  EXPECT_NEAR(0, out(0), 1e-8);
  EXPECT_NEAR(0, out(1), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, EDGE_EDGE_2D_COLLISION_TEST_PARALLEL_NON_INTERSECT)
{
  Eigen::Vector2d a1 = Eigen::Vector2d(-1, 0);
  Eigen::Vector2d a2 = Eigen::Vector2d(1, 0);
  Eigen::Vector2d b1 = Eigen::Vector2d(-1, 1);
  Eigen::Vector2d b2 = Eigen::Vector2d(1, 1);

  Eigen::Vector2d out = Eigen::Vector2d::Random();

  bool hit = get2DLineIntersection(a1, a2, b1, b2, out);
  // they don't cross
  EXPECT_FALSE(hit);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, EDGE_EDGE_2D_COLLISION_TEST_COLINEAR_OVERLAP)
{
  Eigen::Vector2d a1 = Eigen::Vector2d(-1, 0);
  Eigen::Vector2d a2 = Eigen::Vector2d(0.1, 0);
  Eigen::Vector2d b1 = Eigen::Vector2d(-0.1, 0);
  Eigen::Vector2d b2 = Eigen::Vector2d(1, 0);

  Eigen::Vector2d out = Eigen::Vector2d::Random();

  bool hit = get2DLineIntersection(a1, a2, b1, b2, out);
  // they do cross
  EXPECT_TRUE(hit);
  // at the beginning of the overlap
  EXPECT_NEAR(-0.1, out(0), 1e-8);
  EXPECT_NEAR(0, out(1), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, EDGE_EDGE_2D_NON_PARALLEL_NON_OVERLAP)
{
  Eigen::Vector2d a1 = Eigen::Vector2d(-1, 0);
  Eigen::Vector2d a2 = Eigen::Vector2d(1, 0);
  Eigen::Vector2d b1 = Eigen::Vector2d(-1, 2);
  Eigen::Vector2d b2 = Eigen::Vector2d(1, 1);

  Eigen::Vector2d out = Eigen::Vector2d::Random();

  bool hit = get2DLineIntersection(a1, a2, b1, b2, out);
  // they don't cross
  EXPECT_FALSE(hit);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_SUPPORT_TINY_OFF_AXIS)
{
  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  // flip the x and y axis
  box1Transform.linear()(0, 0) = 0;
  box1Transform.linear()(1, 1) = 0;
  box1Transform.linear()(1, 0) = 1;
  box1Transform.linear()(0, 1) = 1;
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  ccd_vec3_t dir;
  dir.v[0] = 0;
  dir.v[1] = 3.79054e-16;
  dir.v[2] = -1;

  ccd_vec3_t out;

  ccdSupportBox(&box1, &dir, &out);

  EXPECT_EQ(out.v[0], 0);
  EXPECT_EQ(out.v[1], 0.5);
  EXPECT_EQ(out.v[2], -0.5);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_VERTEX_FACE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  // flip the x and y axis
  box1Transform.linear()(0, 0) = 0;
  box1Transform.linear()(1, 1) = 0;
  box1Transform.linear()(1, 0) = 1;
  box1Transform.linear()(0, 1) = 1;
  double radOff = 0.001;
  box1Transform = box1Transform.rotate(
      Eigen::AngleAxis<double>(radOff, Eigen::Vector3d::UnitZ()));
  box1Transform = box1Transform.rotate(
      Eigen::AngleAxis<double>(radOff, Eigen::Vector3d::UnitY()));
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitY()));
  box2Transform.translation()(0) = (0.5 + sqrt(3 * 0.25)) - 0.02;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // Randomly translate both boxes in the scene
  Eigen::Vector3d translation = Eigen::Vector3d::Random();
  box1Transform.translation() += translation;
  box2Transform.translation() += translation;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  /*
  Eigen::Vector3d outPos = Eigen::Vector3d(pos.v[0], pos.v[1], pos.v[2]);
  Eigen::Vector3d outDir = Eigen::Vector3d(dir.v[0], dir.v[1], dir.v[2]);
  std::vector<Eigen::Vector3d> outLine;
  outLine.push_back(outPos);
  outLine.push_back(outPos + outDir);

  server::GUIWebsocketServer server;
  server.createBox(
      "box1",
      box1Size,
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()));
  server.createBox(
      "box2",
      box2Size,
      box2Transform.translation(),
      math::matrixToEulerXYZ(box2Transform.linear()));
  server.createLine(
      "contact",
      outLine);
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  EXPECT_EQ(intersect, 0);
  Eigen::Vector3d expected = box1Transform.linear() * Eigen::Vector3d::UnitY();
  EXPECT_NEAR(dir.v[0], expected(0), 1e-8);
  EXPECT_NEAR(dir.v[1], expected(1), 1e-8);
  EXPECT_NEAR(dir.v[2], expected(2), 1e-8);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 1);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_EDGE_EDGE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  // flip the x and z axis
  box1Transform.linear()(0, 0) = 0;
  box1Transform.linear()(2, 2) = 0;
  box1Transform.linear()(2, 0) = 1;
  box1Transform.linear()(0, 2) = 1;
  box1Transform = box1Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitY()));
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));
  box2Transform.translation()(0)
      = (sqrt(0.5 * 0.5 * 2) + sqrt(0.25 * 0.25 * 2)) - 0.01;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // Randomly translate both boxes in the scene
  Eigen::Vector3d translation = Eigen::Vector3d::Random();
  box1Transform.translation() += translation;
  box2Transform.translation() += translation;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  /*
  Eigen::Vector3d outPos = Eigen::Vector3d(pos.v[0], pos.v[1], pos.v[2]);
  Eigen::Vector3d outDir = Eigen::Vector3d(dir.v[0], dir.v[1], dir.v[2]);
  std::vector<Eigen::Vector3d> outLine;
  outLine.push_back(outPos);
  outLine.push_back(outPos + outDir);

  server::GUIWebsocketServer server;
  server.createBox(
      "box1",
      box1Size,
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()));
  server.createBox(
      "box2",
      box2Size,
      box2Transform.translation(),
      math::matrixToEulerXYZ(box2Transform.linear()));
  server.createLine("contact", outLine);
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 2);
  EXPECT_EQ(pointsB.size(), 2);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
#endif

/*
// While it's theoretically possible to get edge-vertex collisions, these seem
// to not get returned libccd.

TEST(DARTCollide, BOX_BOX_MESH_EDGE_VERTEX_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  box1Transform = box1Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitY()));
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitY()));
  box2Transform.translation()(0)
      = (sqrt(0.5 * 0.5 * 2) + sqrt(0.25 * 0.25 * 3)) - 0.01;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 2);
  EXPECT_EQ(pointsB.size(), 1);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
*/

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_EDGE_FACE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform = box2Transform.rotate(Eigen::AngleAxis<double>(
      45.0 * ((double)M_PI / 180.0), Eigen::Vector3d::UnitZ()));
  box2Transform.translation()(0) = (0.5 + sqrt(0.25 * 0.25 * 2)) - 0.01;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 2);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_FACE_SMALL_FACE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform.translation()(0) = (0.5 + 0.25) - 0.01;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 4);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_SMALL_FACE_FACE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform.translation()(0) = (0.5 + 0.25) - 0.01;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 4);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);

  verifyMeshAndBoxResultsIdentical(
      box1Size, box1Transform, box2Size, box2Transform);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, BOX_BOX_MESH_FACE_FACE_OFFSET_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdBox box1;
  box1.size = &box1Size;
  box1.transform = &box1Transform;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform.translation()(0) = (0.5 + 0.25) - 0.01;
  // Shift laterally so we get two edge-edge collisions, and two vertex-face
  // collisions
  box2Transform.translation()(1) = 0.5;
  box2Transform.translation()(2) = 0.5;

  ccdBox box2;
  box2.size = &box2Size;
  box2.transform = &box2Transform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  ccd.mpr_tolerance = 0.0001;   // maximal tolerance

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessBox(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessBox(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 4);

  verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, PREPARE_2D_CVX_SHAPE)
{
  std::vector<Eigen::Vector2d> shape;
  shape.emplace_back(0.0, 0.0);
  shape.emplace_back(0.0, 1.0);
  shape.emplace_back(1.0, 0.0);
  shape.emplace_back(1.0, 1.0);
  // In the inside of the convex shape
  shape.emplace_back(0.6, 0.5);

  prepareConvex2DShape(shape);
  EXPECT_EQ(4, shape.size());

  std::vector<double> angles;
  double lastAngle = -450;
  for (Eigen::Vector2d pt : shape)
  {
    double angle = angle2D(Eigen::Vector2d(0.5, 0.5), pt) * 180 / 3.14159;
    EXPECT_TRUE(angle > lastAngle);
    lastAngle = angle;
    angles.push_back(angle);
  }
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CVX_2D_SHAPE_CONTAINS)
{
  std::vector<Eigen::Vector2d> shape;
  shape.emplace_back(0.0, 0.0);
  shape.emplace_back(0.0, 1.0);
  shape.emplace_back(1.0, 1.0);
  shape.emplace_back(1.0, 0.0);
  // Add a point in the inside of the shape
  shape.emplace_back(0.6, 0.7);

  prepareConvex2DShape(shape);
  EXPECT_EQ(4, shape.size());

  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(0.5, 0.5), shape));
  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(0.8, 0.2), shape));
  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(0.2, 0.8), shape));
  EXPECT_FALSE(convex2DShapeContains(Eigen::Vector2d(1.2, 0.8), shape));
  EXPECT_FALSE(convex2DShapeContains(Eigen::Vector2d(0.2, -0.8), shape));
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CVX_2D_SHAPE_CONTAINS_OFFSET_BOX_EXAMPLE)
{
  std::vector<Eigen::Vector2d> shape;
  shape.emplace_back(-0.5, -0.5);
  shape.emplace_back(0.5, -0.5);
  shape.emplace_back(0.5, 0.5);
  shape.emplace_back(-0.5, 0.5);

  EXPECT_FALSE(convex2DShapeContains(Eigen::Vector2d(-0.75, -0.25), shape));
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CVX_2D_SHAPE_CONTAINS_OFFSET)
{
  std::vector<Eigen::Vector2d> shape;
  shape.emplace_back(2.0, 2.0);
  shape.emplace_back(2.0, 3.0);
  shape.emplace_back(3.0, 3.0);
  shape.emplace_back(3.0, 2.0);
  prepareConvex2DShape(shape);

  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(2.5, 2.5), shape));
  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(2.8, 2.2), shape));
  EXPECT_TRUE(convex2DShapeContains(Eigen::Vector2d(2.2, 2.8), shape));
  EXPECT_FALSE(convex2DShapeContains(Eigen::Vector2d(3.2, 2.8), shape));
  EXPECT_FALSE(convex2DShapeContains(Eigen::Vector2d(2.2, 1.2), shape));
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, MESH_SUPPORT_PLANE)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d boxSize = Eigen::Vector3d(2.0, 4.0, 1.0);
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();

  ccdMesh mesh;
  mesh.mesh = boxMesh;
  mesh.transform = &boxTransform;
  mesh.scale = &boxSize;

  ccdBox box;
  box.transform = &boxTransform;
  box.size = &boxSize;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.1;
  dir.v[2] = 0.1;

  ccd_vec3_t outBox;
  ccd_vec3_t outMesh;

  ccdSupportBox(&box, &dir, &outBox);
  ccdSupportMesh(&mesh, &dir, &outMesh);

  EXPECT_EQ(outBox.v[0], outMesh.v[0]);
  EXPECT_EQ(outBox.v[1], outMesh.v[1]);
  EXPECT_EQ(outBox.v[2], outMesh.v[2]);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, MESH_RANDOM_SUPPORT_PLANES)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d boxSize = Eigen::Vector3d(2.0, 4.0, 1.0);
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();

  ccdMesh mesh;
  mesh.mesh = boxMesh;
  mesh.transform = &boxTransform;
  mesh.scale = &boxSize;

  ccdBox box;
  box.transform = &boxTransform;
  box.size = &boxSize;

  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector3d dirVec = Eigen::Vector3d::Random();

    ccd_vec3_t dir;
    dir.v[0] = dirVec(0);
    dir.v[1] = dirVec(1);
    dir.v[2] = dirVec(2);

    ccd_vec3_t outBox;
    ccd_vec3_t outMesh;

    ccdSupportBox(&box, &dir, &outBox);
    ccdSupportMesh(&mesh, &dir, &outMesh);

    EXPECT_EQ(outBox.v[0], outMesh.v[0]);
    EXPECT_EQ(outBox.v[1], outMesh.v[1]);
    EXPECT_EQ(outBox.v[2], outMesh.v[2]);
  }
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, MESH_WITNESS_POINTS)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d boxSize = Eigen::Vector3d(2.0, 4.0, 1.0);
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();

  ccdMesh mesh;
  mesh.mesh = boxMesh;
  mesh.transform = &boxTransform;
  mesh.scale = &boxSize;

  ccdBox box;
  box.transform = &boxTransform;
  box.size = &boxSize;

  ccd_vec3_t dir;
  dir.v[0] = 1.0;
  dir.v[1] = 0.0;
  dir.v[2] = 0.0;

  ccd_vec3_t outBox;
  ccd_vec3_t outMesh;

  // Start with the positive direction

  std::vector<Eigen::Vector3d> boxWitness
      = ccdPointsAtWitnessBox(&box, &dir, false);
  std::vector<Eigen::Vector3d> meshWitness
      = ccdPointsAtWitnessMesh(&mesh, &dir, false);

  Eigen::Vector3d randomDir = Eigen::Vector3d::Random();
  std::sort(
      boxWitness.begin(),
      boxWitness.end(),
      [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        return a.dot(randomDir) < b.dot(randomDir);
      });
  std::sort(
      meshWitness.begin(),
      meshWitness.end(),
      [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        return a.dot(randomDir) < b.dot(randomDir);
      });

  EXPECT_EQ(boxWitness.size(), meshWitness.size());
  for (int i = 0; i < boxWitness.size(); i++)
  {
    if (!equals(boxWitness[i], meshWitness[i]))
    {
      std::cout << "neg=false Mismatch at [" << i << "]" << std::endl
                << "Expected:" << std::endl
                << boxWitness[i] << std::endl
                << "Got:" << std::endl
                << meshWitness[i] << std::endl;
    }
    EXPECT_TRUE(equals(boxWitness[i], meshWitness[i]));
  }

  // Rerun for the negative direction

  boxWitness = ccdPointsAtWitnessBox(&box, &dir, true);
  meshWitness = ccdPointsAtWitnessMesh(&mesh, &dir, true);

  std::sort(
      boxWitness.begin(),
      boxWitness.end(),
      [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        return a.dot(randomDir) < b.dot(randomDir);
      });
  std::sort(
      meshWitness.begin(),
      meshWitness.end(),
      [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        return a.dot(randomDir) < b.dot(randomDir);
      });

  EXPECT_EQ(boxWitness.size(), meshWitness.size());
  for (int i = 0; i < boxWitness.size(); i++)
  {
    if (!equals(boxWitness[i], meshWitness[i]))
    {
      std::cout << "neg=true Mismatch at [" << i << "]" << std::endl
                << "Expected:" << std::endl
                << boxWitness[i] << std::endl
                << "Got:" << std::endl
                << meshWitness[i] << std::endl;
    }
    EXPECT_TRUE(equals(boxWitness[i], meshWitness[i]));
  }
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, MESH_WITNESS_POINTS_RANDOM)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d boxSize = Eigen::Vector3d(2.0, 4.0, 1.0);
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Isometry3d boxTransform = Eigen::Isometry3d::Identity();

  ccdMesh mesh;
  mesh.mesh = boxMesh;
  mesh.transform = &boxTransform;
  mesh.scale = &boxSize;

  ccdBox box;
  box.transform = &boxTransform;
  box.size = &boxSize;

  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector3d dirVec = Eigen::Vector3d::Random();

    ccd_vec3_t dir;
    dir.v[0] = dirVec(0);
    dir.v[1] = dirVec(1);
    dir.v[2] = dirVec(2);

    ccd_vec3_t outBox;
    ccd_vec3_t outMesh;

    // Start with the positive direction

    std::vector<Eigen::Vector3d> boxWitness
        = ccdPointsAtWitnessBox(&box, &dir, false);
    std::vector<Eigen::Vector3d> meshWitness
        = ccdPointsAtWitnessMesh(&mesh, &dir, false);

    Eigen::Vector3d randomDir = Eigen::Vector3d::Random();
    std::sort(
        boxWitness.begin(),
        boxWitness.end(),
        [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          return a.dot(randomDir) < b.dot(randomDir);
        });
    std::sort(
        meshWitness.begin(),
        meshWitness.end(),
        [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          return a.dot(randomDir) < b.dot(randomDir);
        });

    EXPECT_EQ(boxWitness.size(), meshWitness.size());
    for (int i = 0; i < boxWitness.size(); i++)
    {
      if (!equals(boxWitness[i], meshWitness[i]))
      {
        std::cout << "neg=false Mismatch at [" << i << "]" << std::endl
                  << "Expected:" << std::endl
                  << boxWitness[i] << std::endl
                  << "Got:" << std::endl
                  << meshWitness[i] << std::endl;
      }
      EXPECT_TRUE(equals(boxWitness[i], meshWitness[i]));
    }

    // Rerun for the negative direction

    boxWitness = ccdPointsAtWitnessBox(&box, &dir, true);
    meshWitness = ccdPointsAtWitnessMesh(&mesh, &dir, true);

    std::sort(
        boxWitness.begin(),
        boxWitness.end(),
        [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          return a.dot(randomDir) < b.dot(randomDir);
        });
    std::sort(
        meshWitness.begin(),
        meshWitness.end(),
        [&randomDir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          return a.dot(randomDir) < b.dot(randomDir);
        });

    EXPECT_EQ(boxWitness.size(), meshWitness.size());
    for (int i = 0; i < boxWitness.size(); i++)
    {
      if (!equals(boxWitness[i], meshWitness[i]))
      {
        std::cout << "neg=true Mismatch at [" << i << "]" << std::endl
                  << "Expected:" << std::endl
                  << boxWitness[i] << std::endl
                  << "Got:" << std::endl
                  << meshWitness[i] << std::endl;
      }
      EXPECT_TRUE(equals(boxWitness[i], meshWitness[i]));
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, MESH_MESH_FACE_FACE_OFFSET_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box1;
  box1.mesh = box1Mesh;
  box1.transform = &box1Transform;
  box1.scale = &box1Size;

  Eigen::Vector3d box2Size = Eigen::Vector3d::Ones() * 0.5;
  aiScene* box2Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box2Transform = Eigen::Isometry3d::Identity();
  box2Transform.translation()(0) = (0.5 + 0.25) - 0.01;
  // Shift laterally so we get two edge-edge collisions, and two vertex-face
  // collisions
  box2Transform.translation()(1) = 0.5;
  box2Transform.translation()(2) = 0.5;

  ccdMesh box2;
  box2.mesh = box2Mesh;
  box2.transform = &box2Transform;
  box2.scale = &box2Size;

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh; // support function for first object
  ccd.support2 = ccdSupportMesh; // support function for second object
  ccd.center1 = ccdCenterMesh;   // center function for first object
  ccd.center2 = ccdCenterMesh;   // center function for second object
  ccd.mpr_tolerance = 0.0001;    // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box1",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createMeshASSIMP(
      "box2",
      box2Mesh,
      "",
      box2Transform.translation(),
      math::matrixToEulerXYZ(box2Transform.linear()),
      box2Size);
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> pointsA
      = ccdPointsAtWitnessMesh(&box1, &dir, false);
  std::vector<Eigen::Vector3d> pointsB
      = ccdPointsAtWitnessMesh(&box2, &dir, true);

  EXPECT_EQ(pointsA.size(), 4);
  EXPECT_EQ(pointsB.size(), 4);

  // verifyBoxMeshResultsIdenticalToAnalytical(&box1, &box2);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, WEIRD_GRAVITY)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // world->setPenetrationCorrectionEnabled(true);
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.81, 0);
  world->setGravity(gravity);

  std::shared_ptr<dynamics::Skeleton> box = dynamics::Skeleton::create("box");
  auto pair = box->createJointAndBodyNodePair<dynamics::FreeJoint>();
  world->addSkeleton(box);

  box->setPosition(0, 3.141 * 0.4);

  world->step();

  Eigen::Vector3d vel = box->getVelocities().tail<3>();
  Eigen::Vector3d transformedGravity
      = pair.second->getWorldTransform() * gravity;
  std::cout << "Vel: " << vel << std::endl;
  std::cout << "Gravity: " << transformedGravity << std::endl;
}
#endif

// #ifdef ALL_TESTS
TEST(DARTCollide, ATLAS_5)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  world->setConstraintForceMixingEnabled(true);
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0));

  // Load ground and Atlas robot and add them to the world
  dart::utils::DartLoader urdfLoader;
  std::shared_ptr<dynamics::Skeleton> ground
      = urdfLoader.parseSkeleton("dart://sample/sdf/atlas/ground.urdf");

  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  // std::shared_ptr<SphereShape> boxShape(new SphereShape(1.0));

  /*
  auto retriever = std::make_shared<utils::CompositeResourceRetriever>();
  retriever->addSchemaRetriever(
      "file", std::make_shared<common::LocalResourceRetriever>());
  retriever->addSchemaRetriever("dart", utils::DartResourceRetriever::create());
  std::string meshURI = "dart://sample/sdf/atlas/l_foot.dae";
  const aiScene* model = dynamics::MeshShape::loadMesh(meshURI, retriever);
  dynamics::ShapePtr meshShape = std::make_shared<dynamics::MeshShape>(
      Eigen::Vector3d::Ones(), model, meshURI, retriever);
      */

  std::shared_ptr<dynamics::Skeleton> box = dynamics::Skeleton::create("box");
  auto pair = box->createJointAndBodyNodePair<dynamics::FreeJoint>();
  pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  pair.second->setFrictionCoeff(0.0);

  std::shared_ptr<dynamics::Skeleton> groundBox
      = dynamics::Skeleton::create("groundBox");
  auto groundPair
      = groundBox->createJointAndBodyNodePair<dynamics::WeldJoint>();
  std::shared_ptr<BoxShape> groundShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  groundPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      groundShape);
  groundPair.second->setFrictionCoeff(1.0);
  Eigen::Isometry3d groundTransform = Eigen::Isometry3d::Identity();
  groundTransform.translation()(1) = -0.999;
  groundPair.first->setTransformFromParentBodyNode(groundTransform);

  world->addSkeleton(ground);
  world->addSkeleton(atlas);
  // world->setPenetrationCorrectionEnabled(true);
  // world->addSkeleton(box);
  // world->addSkeleton(groundBox);

  box->setPosition(0, 3.141 * 0.4);
  box->setPosition(4, 1.0);

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  // atlas->setPosition(3, 0.75);

  // Disable the ground from casting its own shadows
  ground->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  world->step();

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);

  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());
  server.serve(8070);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    double diff = sin(((double)time / 2000));
    // atlas->setPosition(0, diff * dart::math::constantsd::pi());
    // double diff2 = sin(((double)time / 4000));
    // atlas->setPosition(4, diff2 * 1);
    world->step();
    server.renderWorld(world);
  });
  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
  }
}
// #endif