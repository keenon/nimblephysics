#define _USE_MATH_DEFINES
#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/collision/CollisionResult.hpp"
#include "dart/collision/dart/DARTCollide.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace realtime;
using namespace collision;

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

  for (int i = 0;
       i < std::min(
           analyticalResult.getNumContacts(), meshResult.getNumContacts());
       i++)
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

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_CATAPULT_EDGE_EDGE_PARALLEL_CASE)
{
  // This bug appeared in the Catapult test case, in the wild. This test is here
  // to catch regressions at the source.

  Eigen::Vector3d size0 = Eigen::Vector3d(0.1, 0.1, 0.1);
  Eigen::Matrix4d M_T0_T;
  // clang-format off
  M_T0_T << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            -2.7807346909040666e-29, -0.11941045999999991, 0, 1;
  // clang-format on
  Eigen::Isometry3d T0(M_T0_T.transpose());

  Eigen::Vector3d size1 = Eigen::Vector3d(0.05, 0.25, 0.05);
  Eigen::Matrix4d M_T1_T;
  // clang-format off
  M_T1_T << -0.016937176236502371, 0.99985655574243948, 0, 0,
            -0.99985655574243948, -0.016937176236502371, 0, 0,
            0, 0, 1, 0,
            0.009234186353806767, -0.11678049047941066, 0, 1;
  // clang-format on
  Eigen::Isometry3d T1(M_T1_T.transpose());

  verifyMeshAndBoxResultsIdentical(size0, T0, size1, T1);

  server::GUIWebsocketServer server;
  server.renderBasis();
  server.createBox(
      "box1", size0, T0.translation(), math::matrixToEulerXYZ(T0.linear()));
  server.createBox(
      "box2", size1, T1.translation(), math::matrixToEulerXYZ(T1.linear()));
  server.serve(8070);
  while (server.isServing())
  {
    // spin
  }
  /*
  CollisionResult result;
  collideBoxBoxAsMesh(nullptr, nullptr, size0, T0, size1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  */
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_CATAPULT_INTER_PENETRATE_CASE)
{
  // This bug appeared in the Catapult test case, in the wild. This test is here
  // to catch regressions at the source.

  Eigen::Vector3d size0 = Eigen::Vector3d(0.1, 0.1, 0.1);
  Eigen::Matrix4d M_T0_T;
  // clang-format off
  M_T0_T << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, -0.051074000000000175, 0, 1;
  // clang-format on
  Eigen::Isometry3d T0(M_T0_T.transpose());

  Eigen::Vector3d size1 = Eigen::Vector3d(0.05, 0.25, 0.05);
  Eigen::Matrix4d M_T1_T;
  // clang-format off
  M_T1_T << -0.016196940812207683, 0.99986882095019136, 0, 0,
            -0.99986882095019136, -0.016196940812207683, 0, 0,
            0, 0, 1, 0,
            0.0061889905849920879, -0.11447217037238604, 0, 1;
  // clang-format on
  Eigen::Isometry3d T1(M_T1_T.transpose());

  verifyMeshAndBoxResultsIdentical(size0, T0, size1, T1);

  server::GUIWebsocketServer server;
  server.renderBasis();
  server.createBox(
      "box1", size0, T0.translation(), math::matrixToEulerXYZ(T0.linear()));
  server.createBox(
      "box2", size1, T1.translation(), math::matrixToEulerXYZ(T1.linear()));
  server.serve(8070);
  while (server.isServing())
  {
    // spin
  }
  /*
  CollisionResult result;
  collideBoxBoxAsMesh(nullptr, nullptr, size0, T0, size1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  */
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(DARTCollide, BOX_CATAPULT_BROKEN_CASE)
{
  // This bug appeared in the Catapult test case, in the wild. This test is here
  // to catch regressions at the source.

  Eigen::Vector3d size0 = Eigen::Vector3d(0.1, 0.1, 0.1);
  Eigen::Matrix4d M_T0;
  // clang-format off
  M_T0 << 1,0,0,0.27082,
          0,1,0,0.189111,
          0,0,1,0,
          0,0,0,1;
  // clang-format on
  Eigen::Isometry3d T0(M_T0);
  Eigen::Vector3d size1 = Eigen::Vector3d(0.05, 0.25, 0.05);
  Eigen::Matrix4d M_T1;
  // clang-format off
  M_T1 << 0.862653,-0.505797,0,0.331802,
          0.505797,0.862653,0,0.143934,
          0,0,1,0,
          0,0,0,1;
  // clang-format on
  Eigen::Isometry3d T1(M_T1);

  verifyMeshAndBoxResultsIdentical(size0, T0, size1, T1);

  server::GUIWebsocketServer server;

  aiScene* boxMesh = createBoxMeshUnsafe();

  CollisionResult meshResult;
  collideMeshMesh(
      nullptr, nullptr, boxMesh, size0, T0, boxMesh, size1, T1, meshResult);
  CollisionResult analyticalResult;
  collideBoxBox(nullptr, nullptr, size0, T0, size1, T1, analyticalResult);

  for (int i = 0; i < analyticalResult.getNumContacts(); i++)
  {
    std::vector<Eigen::Vector3d> points;
    points.push_back(analyticalResult.getContact(i).point);
    points.push_back(
        analyticalResult.getContact(i).point
        + analyticalResult.getContact(i).normal);
    server.createLine("analytical_" + i, points, Eigen::Vector3d::UnitX());
  }
  for (int i = 0; i < meshResult.getNumContacts(); i++)
  {
    std::vector<Eigen::Vector3d> points;
    points.push_back(meshResult.getContact(i).point);
    points.push_back(
        meshResult.getContact(i).point + meshResult.getContact(i).normal);
    server.createLine("mesh_" + i, points, Eigen::Vector3d::UnitZ());
  }

  meshResult.clear();
  collideMeshMesh(
      nullptr, nullptr, boxMesh, size1, T1, boxMesh, size0, T0, meshResult);
  analyticalResult.clear();
  collideBoxBox(nullptr, nullptr, size1, T1, size0, T0, analyticalResult);

  for (int i = 0; i < analyticalResult.getNumContacts(); i++)
  {
    std::vector<Eigen::Vector3d> points;
    points.push_back(analyticalResult.getContact(i).point);
    points.push_back(
        analyticalResult.getContact(i).point
        + analyticalResult.getContact(i).normal);
    server.createLine("back_analytical_" + i, points, Eigen::Vector3d::UnitY());
  }
  for (int i = 0; i < meshResult.getNumContacts(); i++)
  {
    std::vector<Eigen::Vector3d> points;
    points.push_back(meshResult.getContact(i).point);
    points.push_back(
        meshResult.getContact(i).point + meshResult.getContact(i).normal);
    server.createLine("back_mesh_" + i, points, Eigen::Vector3d(1.0, 0.5, 0));
  }

  server.renderBasis();
  server.createBox(
      "box1", size0, T0.translation(), math::matrixToEulerXYZ(T0.linear()));
  server.createBox(
      "box2", size1, T1.translation(), math::matrixToEulerXYZ(T1.linear()));
  server.serve(8070);
  while (server.isServing())
  {
    // spin
  }
  /*
  CollisionResult result;
  collideBoxBoxAsMesh(nullptr, nullptr, size0, T0, size1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  */
}
#endif

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
  box2Transform.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 0.0001, 0));

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
  std::vector<Eigen::Vector3d> shape;
  shape.emplace_back(0.0, 0.0, 0.0);
  shape.emplace_back(0.0, 1.0, 0.0);
  shape.emplace_back(1.0, 0.0, 0.0);
  shape.emplace_back(1.0, 1.0, 0.0);
  // In the inside of the convex shape
  shape.emplace_back(0.6, 0.5, 0.0);

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d basisX = Eigen::Vector3d::UnitX();
  Eigen::Vector3d basisY = Eigen::Vector3d::UnitY();

  keepOnlyConvex2DHull(shape, origin, basisX, basisY);
  prepareConvex2DShape(shape, origin, basisX, basisY);
  EXPECT_EQ(4, shape.size());

  std::vector<double> angles;
  double lastAngle = -450;
  for (Eigen::Vector3d pt3d : shape)
  {
    Eigen::Vector2d pt = pointInPlane(pt3d, origin, basisX, basisY);
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
  std::vector<Eigen::Vector3d> shape;
  shape.emplace_back(0.0, 0.0, 0.0);
  shape.emplace_back(0.0, 1.0, 0.0);
  shape.emplace_back(1.0, 1.0, 0.0);
  shape.emplace_back(1.0, 0.0, 0.0);
  // Add a point in the inside of the shape
  shape.emplace_back(0.6, 0.7, 0.0);

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d basisX = Eigen::Vector3d::UnitX();
  Eigen::Vector3d basisY = Eigen::Vector3d::UnitY();

  keepOnlyConvex2DHull(shape, origin, basisX, basisY);
  prepareConvex2DShape(shape, origin, basisX, basisY);
  EXPECT_EQ(4, shape.size());

  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(0.5, 0.5, 0.0), shape, origin, basisX, basisY));
  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(0.8, 0.2, 0.0), shape, origin, basisX, basisY));
  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(0.2, 0.8, 0.0), shape, origin, basisX, basisY));
  EXPECT_FALSE(convex2DShapeContains(
      Eigen::Vector3d(1.2, 0.8, 0.0), shape, origin, basisX, basisY));
  EXPECT_FALSE(convex2DShapeContains(
      Eigen::Vector3d(0.2, -0.8, 0.0), shape, origin, basisX, basisY));
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, PREPARE_2D_CVX_SHAPE_COLINEAR)
{
  std::vector<Eigen::Vector3d> shape;
  // Create a box with vertices at the midpoint of every edge
  shape.emplace_back(0.0, 0.0, 0.0);
  shape.emplace_back(0.0, 0.5, 0.0);
  shape.emplace_back(0.0, 1.0, 0.0);
  shape.emplace_back(0.5, 1.0, 0.0);
  shape.emplace_back(1.0, 1.0, 0.0);
  shape.emplace_back(1.0, 0.5, 0.0);
  shape.emplace_back(1.0, 0.0, 0.0);
  shape.emplace_back(0.5, 0.0, 0.0);
  // In the inside of the convex shape
  shape.emplace_back(0.6, 0.5, 0.0);
  shape.emplace_back(0.3, 0.7, 0.0);
  shape.emplace_back(0.6, 0.2, 0.0);

  // std::random_device rd;
  std::mt19937 g(42L);
  std::shuffle(shape.begin(), shape.end(), g);

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d basisX = Eigen::Vector3d::UnitX();
  Eigen::Vector3d basisY = Eigen::Vector3d::UnitY();

  keepOnlyConvex2DHull(shape, origin, basisX, basisY);
  prepareConvex2DShape(shape, origin, basisX, basisY);
  EXPECT_EQ(8, shape.size());

  std::vector<double> angles;
  double lastAngle = -450;
  for (Eigen::Vector3d pt3d : shape)
  {
    Eigen::Vector2d pt = pointInPlane(pt3d, origin, basisX, basisY);
    double angle = angle2D(Eigen::Vector2d(0.5, 0.5), pt) * 180 / 3.14159;
    EXPECT_TRUE(angle > lastAngle);
    lastAngle = angle;
    angles.push_back(angle);
  }
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CVX_2D_SHAPE_CONTAINS_OFFSET_BOX_EXAMPLE)
{
  std::vector<Eigen::Vector3d> shape;
  shape.emplace_back(-0.5, -0.5, 0.0);
  shape.emplace_back(0.5, -0.5, 0.0);
  shape.emplace_back(0.5, 0.5, 0.0);
  shape.emplace_back(-0.5, 0.5, 0.0);

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d basisX = Eigen::Vector3d::UnitX();
  Eigen::Vector3d basisY = Eigen::Vector3d::UnitY();

  EXPECT_FALSE(convex2DShapeContains(
      Eigen::Vector3d(-0.75, -0.25, 0.0), shape, origin, basisX, basisY));
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CVX_2D_SHAPE_CONTAINS_OFFSET)
{
  std::vector<Eigen::Vector3d> shape;
  shape.emplace_back(2.0, 2.0, 0.0);
  shape.emplace_back(2.0, 3.0, 0.0);
  shape.emplace_back(3.0, 3.0, 0.0);
  shape.emplace_back(3.0, 2.0, 0.0);

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d basisX = Eigen::Vector3d::UnitX();
  Eigen::Vector3d basisY = Eigen::Vector3d::UnitY();

  keepOnlyConvex2DHull(shape, origin, basisX, basisY);
  prepareConvex2DShape(shape, origin, basisX, basisY);

  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(2.5, 2.5, 0.0), shape, origin, basisX, basisY));
  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(2.8, 2.2, 0.0), shape, origin, basisX, basisY));
  EXPECT_TRUE(convex2DShapeContains(
      Eigen::Vector3d(2.2, 2.8, 0.0), shape, origin, basisX, basisY));
  EXPECT_FALSE(convex2DShapeContains(
      Eigen::Vector3d(3.2, 2.8, 0.0), shape, origin, basisX, basisY));
  EXPECT_FALSE(convex2DShapeContains(
      Eigen::Vector3d(2.2, 1.2, 0.0), shape, origin, basisX, basisY));
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

  for (int test = 0; test < 20; test++)
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
TEST(DARTCollide, VERTEX_SPHERE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 0.5 + sqrt(0.25 / 3) - 0.01;
  sphereTransform.translation()(1) = 0.5 + sqrt(0.25 / 3) - 0.01;
  sphereTransform.translation()(2) = 0.5 + sqrt(0.25 / 3) - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh;   // support function for first object
  ccd.support2 = ccdSupportSphere; // support function for second object
  ccd.center1 = ccdCenterMesh;     // center function for first object
  ccd.center2 = ccdCenterSphere;   // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box, &sphere, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, false);

  EXPECT_EQ(meshPoints.size(), 1);

  CollisionResult collisionResult;
  createMeshSphereContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      meshPoints,
      sphereTransform.translation(),
      sphere.radius);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::VERTEX_SPHERE);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(-1, -1, -1).normalized();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0.5);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, sqrt(3 * 0.01 * 0.01), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, EDGE_SPHERE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 0.5 + sqrt(0.125) - 0.01;
  sphereTransform.translation()(1) = 0.5 + sqrt(0.125) - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh;   // support function for first object
  ccd.support2 = ccdSupportSphere; // support function for second object
  ccd.center1 = ccdCenterMesh;     // center function for first object
  ccd.center2 = ccdCenterSphere;   // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box, &sphere, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, false);

  EXPECT_EQ(meshPoints.size(), 2);

  CollisionResult collisionResult;
  createMeshSphereContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      meshPoints,
      sphereTransform.translation(),
      sphere.radius);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::EDGE_SPHERE);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(-1, -1, 0).normalized();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, sqrt(2 * 0.01 * 0.01), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, FACE_SPHERE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 1.0 - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh;   // support function for first object
  ccd.support2 = ccdSupportSphere; // support function for second object
  ccd.center1 = ccdCenterMesh;     // center function for first object
  ccd.center2 = ccdCenterSphere;   // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&box, &sphere, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, false);

  EXPECT_EQ(meshPoints.size(), 4);

  CollisionResult collisionResult;
  createMeshSphereContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      meshPoints,
      sphereTransform.translation(),
      sphere.radius);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::FACE_SPHERE);
  Eigen::Vector3d expectedNormal = -1 * Eigen::Vector3d::UnitX();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5 - 0.01, 0, 0);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, SPHERE_VERTEX_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 0.5 + sqrt(0.25 / 3) - 0.01;
  sphereTransform.translation()(1) = 0.5 + sqrt(0.25 / 3) - 0.01;
  sphereTransform.translation()(2) = 0.5 + sqrt(0.25 / 3) - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportSphere; // support function for first object
  ccd.support2 = ccdSupportMesh;   // support function for second object
  ccd.center1 = ccdCenterSphere;   // center function for first object
  ccd.center2 = ccdCenterMesh;     // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&sphere, &box, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, true);

  EXPECT_EQ(meshPoints.size(), 1);

  CollisionResult collisionResult;
  createSphereMeshContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      sphereTransform.translation(),
      sphere.radius,
      meshPoints);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::SPHERE_VERTEX);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 1).normalized();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0.5);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, sqrt(3 * 0.01 * 0.01), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, SPHERE_EDGE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 0.5 + sqrt(0.125) - 0.01;
  sphereTransform.translation()(1) = 0.5 + sqrt(0.125) - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportSphere; // support function for first object
  ccd.support2 = ccdSupportMesh;   // support function for second object
  ccd.center1 = ccdCenterSphere;   // center function for first object
  ccd.center2 = ccdCenterMesh;     // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&sphere, &box, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, true);

  EXPECT_EQ(meshPoints.size(), 2);

  CollisionResult collisionResult;
  createSphereMeshContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      sphereTransform.translation(),
      sphere.radius,
      meshPoints);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::SPHERE_EDGE);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 0).normalized();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, sqrt(2 * 0.01 * 0.01), 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, SPHERE_FACE_COLLISION)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Vector3d box1Size = Eigen::Vector3d::Ones();
  aiScene* box1Mesh = createBoxMeshUnsafe();
  Eigen::Isometry3d box1Transform = Eigen::Isometry3d::Identity();
  ccdMesh box;
  box.mesh = box1Mesh;
  box.transform = &box1Transform;
  box.scale = &box1Size;

  Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
  sphereTransform.translation()(0) = 1.0 - 0.01;

  ccdSphere sphere;
  sphere.radius = 0.5;
  sphere.transform = &sphereTransform;

  // set up ccd_t struct
  ccd.support1 = ccdSupportSphere; // support function for first object
  ccd.support2 = ccdSupportMesh;   // support function for second object
  ccd.center1 = ccdCenterSphere;   // center function for first object
  ccd.center2 = ccdCenterMesh;     // center function for second object
  ccd.mpr_tolerance = 0.0001;      // maximal tolerance

  /*
  server::GUIWebsocketServer server;
  server.createMeshASSIMP(
      "box",
      box1Mesh,
      "",
      box1Transform.translation(),
      math::matrixToEulerXYZ(box1Transform.linear()),
      box1Size);
  server.createSphere("sphere", sphere.radius, sphereTransform.translation());
  server.serve(8070);
  while (server.isServing())
  {
  }
  */

  ccd_real_t depth;
  ccd_vec3_t dir, pos;
  int intersect = ccdMPRPenetration(&sphere, &box, &ccd, &depth, &dir, &pos);

  EXPECT_EQ(intersect, 0);
  /*
  std::cout << "Dir: " << dir.v[0] << "," << dir.v[1] << "," << dir.v[2]
            << std::endl;
  std::cout << "Pos: " << pos.v[0] << "," << pos.v[1] << "," << pos.v[2]
            << std::endl;
  */

  std::vector<Eigen::Vector3d> meshPoints
      = ccdPointsAtWitnessMesh(&box, &dir, true);

  EXPECT_EQ(meshPoints.size(), 4);

  CollisionResult collisionResult;
  createSphereMeshContact(
      nullptr,
      nullptr,
      collisionResult,
      &dir,
      sphereTransform.translation(),
      sphere.radius,
      meshPoints);

  EXPECT_EQ(collisionResult.getNumContacts(), 1);
  if (collisionResult.getNumContacts() == 0)
    return;

  Contact& contact = collisionResult.getContact(0);
  EXPECT_EQ(contact.type, ContactType::SPHERE_FACE);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitX();
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-9));
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5 - 0.01, 0, 0);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-9));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-8);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_CAPSULE_T_SHAPED_COLLISION)
{
  double height = 1.0;
  double radius1 = 0.4;
  double radius2 = 0.3;

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = radius1 + radius2 + (height / 2) - 0.01;
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(0, M_PI_2, 0));

  CollisionResult result;
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius1, T1, height, radius2, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint
      = Eigen::Vector3d::UnitX()
        * (radius1 - (0.01 * radius1 / (radius1 + radius2)));
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitX() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, PIPE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius2, T2, height, radius1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitX();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_PIPE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_CAPSULE_X_SHAPED_COLLISION)
{
  double height = 1.0;
  double radius1 = 0.4;
  double radius2 = 0.3;

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = radius1 + radius2 - 0.01;
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(0, M_PI_2, 0));

  CollisionResult result;
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius1, T1, height, radius2, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint
      = Eigen::Vector3d::UnitY()
        * (radius1 - (0.01 * radius1 / (radius1 + radius2)));
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitY() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, PIPE_PIPE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius2, T2, height, radius1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitY();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, PIPE_PIPE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_CAPSULE_L_SHAPED_COLLISION)
{
  double height = 1.0;
  double radius1 = 0.4;
  double radius2 = 0.3;

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = sqrt(2) * height / 4;
  T2.translation()(2)
      = height / 2 + (sqrt(2) * height / 4) + radius1 + radius2 - 0.01;
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(0, M_PI_4, 0));

  CollisionResult result;
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius1, T1, height, radius2, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint
      = Eigen::Vector3d::UnitZ()
        * (height / 2 + radius1 - (0.01 * radius1 / (radius1 + radius2)));
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitZ() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideCapsuleCapsule(
      nullptr, nullptr, height, radius2, T2, height, radius1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitZ();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_SPHERE_END_COLLISION)
{
  double height = 1.0;
  double radius1 = 0.4;
  double radius2 = 0.3;

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(2) = height / 2 + radius1 + radius2 - 0.01;

  CollisionResult result;
  collideCapsuleSphere(
      nullptr, nullptr, height, radius1, T1, radius2, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint
      = Eigen::Vector3d::UnitZ()
        * (height / 2 + radius1 - (0.01 * radius1 / (radius1 + radius2)));
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitZ() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideSphereCapsule(
      nullptr, nullptr, radius2, T2, height, radius1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitZ();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_SPHERE_SIDE_COLLISION)
{
  double height = 1.0;
  double radius1 = 0.4;
  double radius2 = 0.3;

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = radius1 + radius2 - 0.01;

  CollisionResult result;
  collideCapsuleSphere(
      nullptr, nullptr, height, radius1, T1, radius2, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint
      = Eigen::Vector3d::UnitX()
        * (radius1 - (0.01 * radius1 / (radius1 + radius2)));
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitX() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, PIPE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideSphereCapsule(
      nullptr, nullptr, radius2, T2, height, radius1, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitX();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_PIPE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_AS_SPHERE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d::Ones();
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.4;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(2) = size(0) / 2 + height / 2 + radius - 0.01;

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint = Eigen::Vector3d::UnitZ() * 0.5;
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitZ();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_BOX);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitZ() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, BOX_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  /*
  server::GUIWebsocketServer server;
  server.createCapsule(
      "capsule",
      radius,
      height,
      T2.translation(),
      math::matrixToEulerXYZ(T2.linear()));
  server.createBox(
      "box", size, T1.translation(), math::matrixToEulerXYZ(T1.linear()));

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());

  server.serve(8070);

  while (server.isServing())
  {
  }
  */
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_AS_SPHERE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d::Ones();
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.4;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(2) = size(0) / 2 + height / 2 + radius - 0.01;

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  auto contact = result.getContact(0);

  // Normals always go from B -> A
  Eigen::Vector3d expectedPoint = Eigen::Vector3d::UnitZ() * (0.5 - 0.01);
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitZ();
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, SPHERE_FACE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  // Check the results in the backwards direction

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() == 0)
    return;
  contact = result.getContact(0);

  // Normals always go from B -> A
  expectedNormal = Eigen::Vector3d::UnitZ() * -1;
  if (!equals(expectedPoint, contact.point, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Real point: " << std::endl << contact.point << std::endl;
  }
  EXPECT_EQ(contact.type, FACE_SPHERE);
  EXPECT_TRUE(equals(expectedPoint, contact.point, 1e-10));
  if (!equals(expectedNormal, contact.normal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Real normal: " << std::endl << contact.normal << std::endl;
  }
  EXPECT_TRUE(equals(expectedNormal, contact.normal, 1e-10));
  EXPECT_NEAR(contact.penetrationDepth, 0.01, 1e-10);

  /*
  server::GUIWebsocketServer server;
  server.createCapsule(
      "capsule",
      radius,
      height,
      T2.translation(),
      math::matrixToEulerXYZ(T2.linear()));
  server.createBox(
      "box", size, T1.translation(), math::matrixToEulerXYZ(T1.linear()));

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());

  server.serve(8070);

  while (server.isServing())
  {
  }
  */
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_PIPE_FACE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(10, 1, 10);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 1.0 - 0.01;

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitY();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0, 0.5 - 0.01, -height / 2);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0, 0.5 - 0.01, height / 2);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = -1 * Eigen::Vector3d::UnitY();

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_PIPE_FACE_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(10, 1, 10);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 1.0 - 0.01;

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitY();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0, 0.5 - 0.01, -height / 2);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0, 0.5 - 0.01, height / 2);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = -1 * Eigen::Vector3d::UnitY();

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_SPHERE_AND_PIPE_EDGE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(2, 1, 2);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 1.0 - 0.01;
  T2.translation()(2) = 1.0;

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitY();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0, 0.5 - 0.01, radius);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0, 0.5 - (0.01 / 2), 1.0);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, PIPE_EDGE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d::UnitY() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, EDGE_PIPE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_SPHERE_AND_PIPE_EDGE_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(2, 1, 2);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 1.0 - 0.01;
  T2.translation()(2) = 1.0;

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d::UnitY();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0, 0.5 - 0.01, radius);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0, 0.5 - (0.01 / 2), 1.0);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_FACE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, PIPE_EDGE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() < 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d::UnitY() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, FACE_SPHERE);
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, EDGE_PIPE);
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_PIPE_EDGE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(2) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(M_PI_4, 0, 0));

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(0, 1, 1).normalized();
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_EDGE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  expectedNormal = Eigen::Vector3d(0, 1, 1).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, EDGE_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_PIPE_EDGE_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(2) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(M_PI_4, 0, 0));

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(0, 1, 1).normalized();
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_EDGE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  expectedNormal = Eigen::Vector3d(0, 1, 1).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, EDGE_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_PIPE_VERTEX_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.translation()(2) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(M_PI_4, 0, 0));

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 1).normalized();
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_VERTEX);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 1).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, VERTEX_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_PIPE_VERTEX_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.translation()(2) = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
  T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3d(M_PI_4, 0, 0));

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 1).normalized();
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0.5, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_VERTEX);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 1);
  if (result.getNumContacts() != 1)
    return;

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 1).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, VERTEX_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPoint, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPoint << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPoint, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_PIPE_EDGE_PARALLEL_VERTEX_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 2.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 0).normalized();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0.5, 0.5, -0.5);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0.5, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_VERTEX);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, PIPE_VERTEX);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 0).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, VERTEX_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, VERTEX_PIPE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_PIPE_EDGE_PARALLEL_VERTEX_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 1);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 2.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 0).normalized();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(0.5, 0.5, -0.5);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(0.5, 0.5, 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, PIPE_VERTEX);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, PIPE_VERTEX);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 0).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, VERTEX_PIPE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, VERTEX_PIPE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_BOX_PIPE_EDGE_PARALLEL_SPHERE_COLLISION)
{
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 2.0);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);

  CollisionResult result;
  collideCapsuleBox(nullptr, nullptr, height, radius, T2, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 0).normalized();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(
      0.5 - sqrt(0.01 * 0.01 / 2), 0.5 - sqrt(0.01 * 0.01 / 2), -0.5);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(
      0.5 - sqrt(0.01 * 0.01 / 2), 0.5 - sqrt(0.01 * 0.01 / 2), 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_EDGE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, SPHERE_EDGE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideBoxCapsule(nullptr, nullptr, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 0).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, EDGE_SPHERE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, EDGE_SPHERE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /*
  server::GUIWebsocketServer server;
  server.createCapsule(
      "capsule",
      radius,
      height,
      T2.translation(),
      math::matrixToEulerXYZ(T2.linear()));
  server.createBox(
      "box", size, T1.translation(), math::matrixToEulerXYZ(T1.linear()));

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());

  server.serve(8070);

  while (server.isServing())
  {
  }
  */
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_MESH_PIPE_EDGE_PARALLEL_SPHERE_COLLISION)
{
  aiScene* boxMesh = createBoxMeshUnsafe();
  Eigen::Vector3d size = Eigen::Vector3d(1, 1, 2.0);
  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();

  double height = 1.0;
  double radius = 0.5;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
  T2.translation()(0) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);
  T2.translation()(1) = 0.5 + sqrt(radius * radius / 2) - sqrt(0.01 * 0.01 / 2);

  CollisionResult result;
  collideCapsuleMesh(
      nullptr, nullptr, height, radius, T2, boxMesh, size, T1, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  Eigen::Vector3d sortDir = Eigen::Vector3d::UnitZ();
  result.sortContacts(sortDir);

  // Points from B to A
  Eigen::Vector3d expectedNormal = Eigen::Vector3d(1, 1, 0).normalized();
  Eigen::Vector3d expectedPointA = Eigen::Vector3d(
      0.5 - sqrt(0.01 * 0.01 / 2), 0.5 - sqrt(0.01 * 0.01 / 2), -0.5);
  Eigen::Vector3d expectedPointB = Eigen::Vector3d(
      0.5 - sqrt(0.01 * 0.01 / 2), 0.5 - sqrt(0.01 * 0.01 / 2), 0.5);

  Contact contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, SPHERE_EDGE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  Contact contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, SPHERE_EDGE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);

  /////////////////////////////////////////////////////////
  // Try the reverse direction
  /////////////////////////////////////////////////////////

  result.clear();
  collideMeshCapsule(
      nullptr, nullptr, boxMesh, size, T1, height, radius, T2, result);

  EXPECT_EQ(result.getNumContacts(), 2);
  if (result.getNumContacts() != 2)
    return;

  result.sortContacts(sortDir);

  // Points from B to A
  expectedNormal = Eigen::Vector3d(1, 1, 0).normalized() * -1;

  contact1 = result.getContact(0);
  EXPECT_EQ(contact1.type, EDGE_SPHERE);
  if (!equals(contact1.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact1.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact1.normal, expectedNormal, 1e-10));
  if (!equals(contact1.point, expectedPointA, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointA << std::endl;
    std::cout << "Actual point: " << std::endl << contact1.point << std::endl;
  }
  EXPECT_TRUE(equals(contact1.point, expectedPointA, 1e-10));
  EXPECT_NEAR(contact1.penetrationDepth, 0.01, 1e-10);

  contact2 = result.getContact(1);
  EXPECT_EQ(contact2.type, EDGE_SPHERE);
  if (!equals(contact2.normal, expectedNormal, 1e-10))
  {
    std::cout << "Expected normal: " << std::endl
              << expectedNormal << std::endl;
    std::cout << "Actual normal: " << std::endl << contact2.normal << std::endl;
  }
  EXPECT_TRUE(equals(contact2.normal, expectedNormal, 1e-10));
  if (!equals(contact2.point, expectedPointB, 1e-10))
  {
    std::cout << "Expected point: " << std::endl << expectedPointB << std::endl;
    std::cout << "Actual point: " << std::endl << contact2.point << std::endl;
  }
  EXPECT_TRUE(equals(contact2.point, expectedPointB, 1e-10));
  EXPECT_NEAR(contact2.penetrationDepth, 0.01, 1e-10);
}
#endif

#ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_CCD_SUPPORT_FN)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  Eigen::Isometry3d capsuleTransform = Eigen::Isometry3d::Identity();
  capsuleTransform.translation()(0) = 0.5 + sqrt(0.25 / 3) - 0.01;
  capsuleTransform.translation()(1) = 0.5 + sqrt(0.25 / 3) - 0.01;
  capsuleTransform.translation()(2) = 0.5 + sqrt(0.25 / 3) - 0.01;

  ccdCapsule capsule;
  capsule.radius = 0.5;
  capsule.height = 2.0;
  capsule.transform = &capsuleTransform;

  ccd_vec3_t dirRaw;
  ccd_vec3_t outRaw;

  Eigen::Map<Eigen::Vector3d> dir(dirRaw.v);
  Eigen::Map<Eigen::Vector3d> out(outRaw.v);

  // Straight up

  dir = Eigen::Vector3d::UnitZ();

  ccdSupportCapsule(&capsule, &dirRaw, &outRaw);

  Eigen::Vector3d expectedOut
      = capsuleTransform
        * (Eigen::Vector3d::UnitZ() * (capsule.height / 2 + capsule.radius));
  if (!out.isApprox(expectedOut, 1e-7))
  {
    std::cout << "Expected out: " << std::endl << expectedOut << std::endl;
    std::cout << "Actual out: " << std::endl << out << std::endl;
  }
  EXPECT_TRUE(out.isApprox(expectedOut, 1e-7));

  // Straight out

  dir = Eigen::Vector3d::UnitX();

  ccdSupportCapsule(&capsule, &dirRaw, &outRaw);

  expectedOut = capsuleTransform * (Eigen::Vector3d::UnitX() * capsule.radius);
  if (!out.isApprox(expectedOut, 1e-7))
  {
    std::cout << "Expected out: " << std::endl << expectedOut << std::endl;
    std::cout << "Actual out: " << std::endl << out << std::endl;
  }
  EXPECT_TRUE(out.isApprox(expectedOut, 1e-7));

  /*
  server::GUIWebsocketServer server;
  server.createCapsule(
      "capsule",
      capsule.radius,
      capsule.height,
      capsuleTransform.translation(),
      math::matrixToEulerXYZ(capsuleTransform.linear()));
  server.createSphere(
      "marker", 0.1, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());

  server.serve(8070);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    double diff = 10 * sin(((double)time / 2000));
    dir = math::eulerXYZToMatrix(Eigen::Vector3d(diff, 0, 0))
          * Eigen::Vector3d::UnitY();
    ccdSupportCapsule(&capsule, &dirRaw, &outRaw);
    server.setObjectPosition("marker", out);
  });
  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
  }
  */
}
#endif

// The number of contacts shouldn't change under tiny perturbations to position,
// and the contacts should move in predictable ways.

#ifdef ALL_TESTS
TEST(DARTCollide, ATLAS_5_STABILITY)
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

  world->addSkeleton(ground);
  world->addSkeleton(atlas);

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  neural::RestorableSnapshot snapshot(world);

  // Get the first snapshot
  world->step();
  std::vector<collision::Contact> originalContacts
      = world->getLastCollisionResult().getContacts();

  snapshot.restore();

  // Perturb by a tiny bit
  atlas->setPosition(0, atlas->getPosition(0) + 1e-8);

  // Get the second snapshot
  world->step();
  std::vector<collision::Contact> perturbedContacts
      = world->getLastCollisionResult().getContacts();

  snapshot.restore();

  EXPECT_EQ(originalContacts.size(), perturbedContacts.size());

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  int counter = 0;
  for (collision::Contact col : originalContacts)
  {
    bool foundDuplicate = false;
    for (collision::Contact col2 : perturbedContacts)
    {
      if ((col.point - col2.point).norm() < 1e-8)
      {
        foundDuplicate = true;
      }
    }
    if (foundDuplicate)
      continue;

    std::vector<Eigen::Vector3d> points;
    points.push_back(col.point);
    points.push_back(col.point - col.normal);
    server.createLine("contact_" + counter, points, Eigen::Vector3d::UnitY());
    counter++;
  }
  for (collision::Contact col : perturbedContacts)
  {
    std::vector<Eigen::Vector3d> points;
    points.push_back(col.point);
    points.push_back(col.point - col.normal);
    server.createLine("contact_" + counter, points, Eigen::Vector3d::UnitZ());
    counter++;
  }

  server.serve(8070);

  while (server.isServing())
  {
  }
  */
}
#endif

/*
// #ifdef ALL_TESTS
TEST(DARTCollide, CAPSULE_REALTIME)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  // world->setConstraintForceMixingEnabled(true);
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0));

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  std::shared_ptr<CapsuleShape> capsuleShape(new CapsuleShape(0.5, 1.0));

  std::shared_ptr<dynamics::Skeleton> capsule
      = dynamics::Skeleton::create("capsule");
  auto pair = capsule->createJointAndBodyNodePair<dynamics::FreeJoint>();
  pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(capsuleShape);
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

  world->addSkeleton(capsule);
  world->addSkeleton(groundBox);
  // world->setPenetrationCorrectionEnabled(true);
  // world->addSkeleton(box);
  // world->addSkeleton(groundBox);

  capsule->setPosition(0, 3.141 * 0.4);
  capsule->setPosition(4, 1.0);

  // Disable the ground from casting its own shadows
  groundBox->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
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
  bool anyContact = false;
  ticker.registerTickListener([&](long time) {
    double diff = sin(((double)time / 2000));
    // atlas->setPosition(0, diff * dart::math::constantsd::pi());
    // double diff2 = sin(((double)time / 4000));
    // atlas->setPosition(4, diff2 * 1);
    world->step();
    auto result = world->getLastCollisionResult();
    if (result.getNumContacts() > 0)
    {
      anyContact = true;
    }
    else
    {
      if (anyContact)
      {
        std::cout << "Contact has disappeared!" << std::endl;
      }
    }
    server.renderWorld(world);
  });
  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
  }
}
// #endif
*/

/*
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

  // auto retriever = std::make_shared<utils::CompositeResourceRetriever>();
  // retriever->addSchemaRetriever(
  //    "file", std::make_shared<common::LocalResourceRetriever>());
  // retriever->addSchemaRetriever("dart",
  //    utils::DartResourceRetriever::create());
  // std::string meshURI = "dart://sample/sdf/atlas/l_foot.dae";
  // const aiScene* model = dynamics::MeshShape::loadMesh(meshURI, retriever);
  // dynamics::ShapePtr meshShape = std::make_shared<dynamics::MeshShape>(
  //    Eigen::Vector3d::Ones(), model, meshURI, retriever);

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
*/