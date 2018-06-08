#include "dart/gui/glfw/Scene.hpp"

#include "dart/common/StlHelpers.hpp"
#include "dart/gui/glfw/Entity.hpp"
#include "dart/math/Constants.hpp"

namespace dart {
namespace gui {
namespace glfw {

// Constants
const int SHADOWMAP_WIDTH = 2048;
const int SHADOWMAP_HEIGHT = 2048;
// TODO(JS): Better name

//==============================================================================
Scene::Scene(const std::string& name, GLFWwindow* window)
  : mName{name}, mGlfwWindow{window}
{
  notifyMainWindowChanged(window);
}

//==============================================================================
Scene::~Scene()
{
  removeAllEntities();
}

//==============================================================================
void Scene::addEntity(const std::shared_ptr<Entity>& entity)
{
  if (nullptr == entity)
    return;

  mEntities.insert(entity);

  if (nullptr == mGlfwWindow)
    return;

  glfwMakeContextCurrent(mGlfwWindow);

  entity->createGlObjectsFor(mGlfwWindow);

  for (auto* glfwSubWindow : mGlfwSubWindows)
    entity->createGlObjectsFor(glfwSubWindow, mGlfwWindow);
}

//==============================================================================
void Scene::removeEntity(const std::shared_ptr<Entity>& entity)
{
  if (nullptr == entity)
    return;

  if (nullptr == mGlfwWindow)
    return;

  glfwMakeContextCurrent(mGlfwWindow);

  entity->destroyGlObjectsFor(mGlfwWindow);

  for (auto* glfwSubWindow : mGlfwSubWindows)
    entity->destroyGlObjectsFor(glfwSubWindow, mGlfwWindow);

  mEntities.erase(entity);
}

//==============================================================================
void Scene::removeAllEntities()
{
  for (const auto& entity : mEntities)
    removeEntity(entity);
}

//==============================================================================
// const std::vector<std::shared_ptr<Entity>>& Scene::getEntities() const
//{
//  return mEntities;
//}

//==============================================================================
void Scene::notifyMainWindowChanged(GLFWwindow* window)
{
  if (mGlfwWindow == window)
    return;

  if (nullptr != mGlfwWindow)
  {
    glfwMakeContextCurrent(mGlfwWindow);

    for (const auto& entity : mEntities)
      entity->destroyGlObjectsFor(mGlfwWindow);
  }

  mGlfwWindow = window;

  if (nullptr == mGlfwWindow)
    return;

  glfwMakeContextCurrent(mGlfwWindow);

  for (const auto& entity : mEntities)
    entity->createGlObjectsFor(mGlfwWindow);
}

//==============================================================================
void Scene::notifySubWindowAdded(GLFWwindow* /*window*/)
{
  // TODO(JS): Not implemented
}

//==============================================================================
void Scene::notifySubWindowRemoved(GLFWwindow* /*window*/)
{
  // TODO(JS): Not implemented
}

//==============================================================================
void Scene::renderSinglePass(
    Program& shader, const Eigen::Isometry3f& worldToCameraMatrix)
{
  shader.bind();

  for (auto& entity : mEntities)
    entity->render(shader, worldToCameraMatrix);

  shader.unbind();
}

} // namespace glfw
} // namespace gui
} // namespace dart