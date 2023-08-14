#ifndef DART_GUI_STATE_MACHINE
#define DART_GUI_STATE_MACHINE

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <asio/io_service.hpp>
#include <assimp/Importer.hpp>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/proto/GUI.pb.h"
#include "dart/server/WebsocketServer.hpp"

namespace dart {

namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
}

namespace trajectory {
class TrajectoryRollout;
}

namespace server {

class GUIStateMachine
{
public:
  GUIStateMachine();

  virtual ~GUIStateMachine();

  /// This returns the current state of the GUI as a JSON blob, for example if
  /// we have a websocket reconnecting
  std::string getCurrentStateAsJson();

  /// This formats the latest set of commands as JSON, and clears the buffer
  std::string flushJson();

  /// This is a high-level command that creates/updates all the shapes in a
  /// world by calling the lower-level commands
  void renderWorld(
      const std::shared_ptr<simulation::World>& world,
      const std::string& prefix = "world",
      bool renderForces = true,
      bool renderForceMagnitudes = true,
      const std::string& layer = "");

  /// This is a high-level command that creates a basis
  void renderBasis(
      s_t scale = 10.0,
      const std::string& prefix = "basis",
      const Eigen::Vector3s pos = Eigen::Vector3s::Zero(),
      const Eigen::Vector3s euler = Eigen::Vector3s::Zero(),
      const std::string& layer = "");

  /// This is a high-level command that creates/updates all the shapes in a
  /// world by calling the lower-level commands
  void renderSkeleton(
      const std::shared_ptr<dynamics::Skeleton>& skel,
      const std::string& prefix = "skel",
      Eigen::Vector4s overrideColor = -1 * Eigen::Vector4s::Ones(),
      const std::string& layer = "");

  /// This is a high-level command that creates/updates all the shapes in a
  /// world by calling the lower-level commands
  void renderSkeletonInertiaCubes(
      const std::shared_ptr<dynamics::Skeleton>& skel,
      const std::string& prefix = "skel_inertia",
      Eigen::Vector4s overrideColor = Eigen::Vector4s(0, 0, 1, 0.3),
      const std::string& layer = "");

  /// This is a high-level command that renders a given trajectory as a
  /// bunch of lines in the world, one per body
  void renderTrajectoryLines(
      std::shared_ptr<simulation::World> world,
      Eigen::MatrixXs positions,
      std::string prefix = "trajectory",
      const std::string& layer = "");

  /// This either creates or moves an arrow to have the new start and end points
  void renderArrow(
      Eigen::Vector3s start,
      Eigen::Vector3s end,
      s_t bodyRadius,
      s_t tipRadius,
      Eigen::Vector4s color = Eigen::Vector4s(1, 0, 0, 0.5),
      const std::string& prefix = "arrow",
      const std::string& layer = "");

  /// This is a high-level command that renders a wrench on a body node
  void renderBodyWrench(
      const dynamics::BodyNode* body,
      Eigen::Vector6s wrench,
      s_t scaleFactor = 0.1,
      std::string prefix = "wrench",
      const std::string& layer = "");

  /// This renders little velocity lines starting at every vertex in the passed
  /// in body
  void renderMovingBodyNodeVertices(
      const dynamics::BodyNode* body,
      s_t scaleFactor = 0.1,
      std::string prefix = "vert-vel",
      const std::string& layer = "");

  /// This is a high-level command that removes the lines rendering a wrench on
  /// a body node
  void clearBodyWrench(
      const dynamics::BodyNode* body, std::string prefix = "wrench");

  /// This completely resets the web GUI, deleting all objects, UI elements,
  /// and listeners
  virtual void clear();

  /// Set frames per second
  void setFramesPerSecond(int framesPerSecond);

  /// This creates a layer in the web GUI
  void createLayer(
      std::string key,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      bool defaultShow = true);

  /// This creates a box in the web GUI under a specified key
  void createBox(
      std::string key,
      const Eigen::Vector3s& size,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a sphere in the web GUI under a specified key
  void createSphere(
      std::string key,
      s_t radius,
      const Eigen::Vector3s& pos,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a sphere in the web GUI under a specified key
  void createSphere(
      std::string key,
      Eigen::Vector3s radii,
      const Eigen::Vector3s& pos,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a cone in the web GUI under a specified key
  void createCone(
      std::string key,
      s_t radius,
      s_t height,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a cylinder in the web GUI under a specified key
  void createCylinder(
      std::string key,
      s_t radius,
      s_t height,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a capsule in the web GUI under a specified key
  void createCapsule(
      std::string key,
      s_t radius,
      s_t height,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a line in the web GUI under a specified key
  void createLine(
      std::string key,
      const std::vector<Eigen::Vector3s>& points,
      const Eigen::Vector4s& color = Eigen::Vector4s(1.0, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      const std::vector<s_t>& width = std::vector<s_t>());

  /// This creates a mesh in the web GUI under a specified key, using raw shape
  /// data
  void createMesh(
      std::string key,
      const std::vector<Eigen::Vector3s>& vertices,
      const std::vector<Eigen::Vector3s>& vertexNormals,
      const std::vector<Eigen::Vector3i>& faces,
      const std::vector<Eigen::Vector2s>& uv,
      const std::vector<std::string>& textures,
      const std::vector<int>& textureStartIndices,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector3s& scale = Eigen::Vector3s::Ones(),
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a mesh in the web GUI under a specified key, from the ASSIMP
  /// mesh
  void createMeshASSIMP(
      const std::string& key,
      const aiScene* mesh,
      const std::string& meshPath,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector3s& scale = Eigen::Vector3s::Ones(),
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a mesh in the web GUI under a specified key, from the ASSIMP
  /// mesh
  void createMeshFromShape(
      const std::string& key,
      const std::shared_ptr<dynamics::MeshShape> mesh,
      const Eigen::Vector3s& pos,
      const Eigen::Vector3s& euler,
      const Eigen::Vector3s& scale = Eigen::Vector3s::Ones(),
      const Eigen::Vector4s& color = Eigen::Vector4s(0.5, 0.5, 0.5, 1.0),
      const std::string& layer = "",
      bool castShadows = false,
      bool receiveShadows = false);

  /// This creates a texture object, to be sent to the web frontend
  void createTexture(const std::string& key, const std::string& base64);

  /// This creates a texture object by loading it from a file
  void createTextureFromFile(const std::string& key, const std::string& path);

  /// This returns true if we've already got an object with the key "key"
  bool hasObject(const std::string& key);

  /// This returns the position of an object, if we've got it (and it's not
  /// a line). Otherwise it returns Vector3s::Zero().
  Eigen::Vector3s getObjectPosition(const std::string& key);

  /// This returns the rotation of an object, if we've got it (and it's not
  /// a line or a sphere). Otherwise it returns Vector3s::Zero().
  Eigen::Vector3s getObjectRotation(const std::string& key);

  /// This returns the color of an object, if we've got it. Otherwise it
  /// returns Vector4s::Zero().
  Eigen::Vector4s getObjectColor(const std::string& key);

  /// This returns the size of a box, scale of a mesh, 3vec of [radius, radius,
  /// radius] for a sphere, and [radius, radius, height] for a capsule. Returns
  /// 0 for lines.
  Eigen::Vector3s getObjectScale(const std::string& key);

  /// This moves an object (e.g. box, sphere, line) to a specified position
  void setObjectPosition(const std::string& key, const Eigen::Vector3s& pos);

  /// This moves an object (e.g. box, sphere, line) to a specified
  /// orientation
  void setObjectRotation(const std::string& key, const Eigen::Vector3s& euler);

  /// This changes an object (e.g. box, sphere, line) color
  void setObjectColor(const std::string& key, const Eigen::Vector4s& color);

  /// This changes an object (e.g. box, sphere, mesh) size. Has no effect on
  /// lines.
  void setObjectScale(const std::string& key, const Eigen::Vector3s& scale);

  /// This sets a tooltip for the object at key.
  void setObjectTooltip(const std::string& key, const std::string& tooltip);

  /// This removes a tooltip for the object at key.
  void deleteObjectTooltip(const std::string& key);

  /// This sets an object to allow dragging around by the mouse on the GUI
  void setObjectDragEnabled(const std::string& key);

  /// This sets an object to allow editing the tooltip by double-clicking on the
  /// object
  void setObjectTooltipEditable(const std::string& key);

  /// This deletes an object by key
  void deleteObject(const std::string& key);

  /// This deletes all the objects that match a given prefix
  void deleteObjectsByPrefix(const std::string& prefix);

  /// This gets the current screen size
  Eigen::Vector2i getScreenSize();

  /// This registers a callback to get called whenever the screen size changes.
  void registerScreenResizeListener(
      std::function<void(Eigen::Vector2i)> listener);

  /// This places some text on the screen at the specified coordinates
  void createText(
      const std::string& key,
      const std::string& contents,
      const Eigen::Vector2i& fromTopLeft,
      const Eigen::Vector2i& size,
      const std::string& layer = "");

  /// This changes the contents of text on the screen
  void setTextContents(const std::string& key, const std::string& newContents);

  /// This places a clickable button on the screen at the specified
  /// coordinates
  void createButton(
      const std::string& key,
      const std::string& label,
      const Eigen::Vector2i& fromTopLeft,
      const Eigen::Vector2i& size,
      std::function<void()> onClick,
      const std::string& layer = "");

  /// This changes the contents of text on the screen
  void setButtonLabel(const std::string& key, const std::string& newLabel);

  /// This creates a slider
  void createSlider(
      const std::string& key,
      const Eigen::Vector2i& fromTopLeft,
      const Eigen::Vector2i& size,
      s_t min,
      s_t max,
      s_t value,
      bool onlyInts,
      bool horizontal,
      std::function<void(s_t)> onChange,
      const std::string& layer = "");

  /// This changes the contents of text on the screen
  void setSliderValue(const std::string& key, s_t value);

  /// This changes the contents of text on the screen
  void setSliderMin(const std::string& key, s_t min);

  /// This changes the contents of text on the screen
  void setSliderMax(const std::string& key, s_t max);

  /// This creates a plot to display data on the GUI
  void createPlot(
      const std::string& key,
      const Eigen::Vector2i& fromTopLeft,
      const Eigen::Vector2i& size,
      const std::vector<s_t>& xs,
      s_t minX,
      s_t maxX,
      const std::vector<s_t>& ys,
      s_t minY,
      s_t maxY,
      const std::string& type,
      const std::string& layer = "");

  /// This changes the contents of a plot, along with its display limits
  void setPlotData(
      const std::string& key,
      const std::vector<s_t>& xs,
      s_t minX,
      s_t maxX,
      const std::vector<s_t>& ys,
      s_t minY,
      s_t maxY);

  /// This creates a rich plot with axis labels, a title, tickmarks, and
  /// multiple simultaneous lines
  void createRichPlot(
      const std::string& key,
      const Eigen::Vector2i& fromTopLeft,
      const Eigen::Vector2i& size,
      s_t minX,
      s_t maxX,
      s_t minY,
      s_t maxY,
      const std::string& title,
      const std::string& xAxisLabel,
      const std::string& yAxisLabel,
      const std::string& layer = "");

  /// This sets a single data stream for a rich plot. `name` should be human
  /// readable and unique. You can overwrite data by using the same `name` with
  /// multiple calls to `setRichPlotData`.
  void setRichPlotData(
      const std::string& key,
      const std::string& name,
      const std::string& color,
      const std::string& type,
      const std::vector<s_t>& xs,
      const std::vector<s_t>& ys);

  /// This sets a single data stream for a rich plot
  void setRichPlotBounds(
      const std::string& key, s_t minX, s_t maxX, s_t minY, s_t maxY);

  /// This moves a UI element on the screen
  void setUIElementPosition(
      const std::string& key, const Eigen::Vector2i& fromTopLeft);

  /// This changes the size of a UI element
  void setUIElementSize(const std::string& key, const Eigen::Vector2i& size);

  /// This deletes a UI element by key
  void deleteUIElement(const std::string& key);

  /// This gets an integer code for a string
  int getStringCode(const std::string& key);

  /// This gets a string code for an integer
  std::string getCodeString(int code);

protected:
  // protects the buffered JSON message (mJson) from getting
  // corrupted if we queue messages while trying to flush()
  std::recursive_mutex globalMutex;
  std::recursive_mutex mProtoMutex;
  int mMessagesQueued;
  proto::CommandList mCommandList;
  std::string mCommandListOutputBuffer;
  // This is a list of all the objects with mouse interaction enabled
  std::unordered_set<std::string> mDragEnabled;
  std::unordered_set<std::string> mTooltipEditable;

  std::unordered_map<std::string, int> mStringCodes;
  std::unordered_map<int, std::string> mCodeStrings;

  struct Layer
  {
    std::string key;
    Eigen::Vector4s color;
    bool defaultShow;
  };
  std::unordered_map<std::string, Layer> mLayers;

  struct Box
  {
    std::string key;
    std::string layer;
    Eigen::Vector3s size;
    Eigen::Vector3s pos;
    Eigen::Vector3s euler;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Box> mBoxes;
  struct Sphere
  {
    std::string key;
    std::string layer;
    Eigen::Vector3s radii;
    Eigen::Vector3s pos;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Sphere> mSpheres;

  struct Cone
  {
    std::string key;
    std::string layer;
    s_t radius;
    s_t height;
    Eigen::Vector3s pos;
    Eigen::Vector3s euler;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Cone> mCones;

  struct Cylinder
  {
    std::string key;
    std::string layer;
    s_t radius;
    s_t height;
    Eigen::Vector3s pos;
    Eigen::Vector3s euler;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Cylinder> mCylinders;

  struct Capsule
  {
    std::string key;
    std::string layer;
    s_t radius;
    s_t height;
    Eigen::Vector3s pos;
    Eigen::Vector3s euler;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Capsule> mCapsules;
  struct Line
  {
    std::string key;
    std::string layer;
    std::vector<Eigen::Vector3s> points;
    Eigen::Vector4s color;
    std::vector<s_t> width;
  };
  std::unordered_map<std::string, Line> mLines;

  struct Tooltip
  {
    std::string key;
    std::string tooltip;
  };
  std::unordered_map<std::string, Tooltip> mTooltips;

  struct Mesh
  {
    std::string key;
    std::string layer;
    std::vector<Eigen::Vector3s> vertices;
    std::vector<Eigen::Vector3s> vertexNormals;
    std::vector<Eigen::Vector3i> faces;
    std::vector<Eigen::Vector2s> uv;
    std::vector<std::string> textures;
    std::vector<int> textureStartIndices;
    Eigen::Vector3s pos;
    Eigen::Vector3s euler;
    Eigen::Vector3s scale;
    Eigen::Vector4s color;
    bool castShadows;
    bool receiveShadows;
  };
  std::unordered_map<std::string, Mesh> mMeshes;

  struct Texture
  {
    std::string key;
    std::string base64;
  };
  std::unordered_map<std::string, Texture> mTextures;

  struct Text
  {
    std::string key;
    std::string layer;
    std::string contents;
    Eigen::Vector2i fromTopLeft;
    Eigen::Vector2i size;
  };
  std::unordered_map<std::string, Text> mText;

  struct Button
  {
    std::string key;
    std::string layer;
    std::string label;
    Eigen::Vector2i fromTopLeft;
    Eigen::Vector2i size;
    std::function<void()> onClick;
  };
  std::unordered_map<std::string, Button> mButtons;

  struct Slider
  {
    std::string key;
    std::string layer;
    Eigen::Vector2i fromTopLeft;
    Eigen::Vector2i size;
    s_t min;
    s_t max;
    s_t value;
    bool onlyInts;
    bool horizontal;
    std::function<void(s_t)> onChange;
  };
  std::unordered_map<std::string, Slider> mSliders;

  struct Plot
  {
    std::string key;
    std::string layer;
    Eigen::Vector2i fromTopLeft;
    Eigen::Vector2i size;
    std::vector<s_t> xs;
    s_t minX;
    s_t maxX;
    std::vector<s_t> ys;
    s_t minY;
    s_t maxY;
    std::string type;
  };
  std::unordered_map<std::string, Plot> mPlots;

  struct RichPlotData
  {
    std::string name;
    std::string color;
    std::vector<s_t> ys;
    std::vector<s_t> xs;
    std::string type;
  };
  struct RichPlot
  {
    std::string key;
    std::string layer;
    Eigen::Vector2i fromTopLeft;
    Eigen::Vector2i size;
    s_t minX;
    s_t maxX;
    s_t minY;
    s_t maxY;
    std::string title;
    std::string xAxisLabel;
    std::string yAxisLabel;
    std::unordered_map<std::string, RichPlotData> data;
  };
  std::unordered_map<std::string, RichPlot> mRichPlots;

  void queueCommand(std::function<void(proto::CommandList&)> writeCommand);

  void encodeSetFramesPerSecond(proto::CommandList& list, int framesPerSecond);
  void encodeCreateLayer(proto::CommandList& list, Layer& layer);
  void encodeCreateBox(proto::CommandList& list, Box& box);
  void encodeCreateSphere(proto::CommandList& list, Sphere& sphere);
  void encodeCreateCone(proto::CommandList& list, Cone& cone);
  void encodeCreateCylinder(proto::CommandList& list, Cylinder& cylinder);
  void encodeCreateCapsule(proto::CommandList& list, Capsule& capsule);
  void encodeCreateLine(proto::CommandList& list, Line& line);
  void encodeCreateMesh(proto::CommandList& list, Mesh& mesh);
  void encodeSetTooltip(proto::CommandList& list, Tooltip& tooltip);
  void encodeCreateTexture(proto::CommandList& list, Texture& texture);
  void encodeEnableDrag(proto::CommandList& list, const std::string& key);
  void encodeEnableEditTooltip(
      proto::CommandList& list, const std::string& key);
  void encodeCreateText(proto::CommandList& list, Text& text);
  void encodeCreateButton(proto::CommandList& list, Button& button);
  void encodeCreateSlider(proto::CommandList& list, Slider& slider);
  void encodeCreatePlot(proto::CommandList& list, Plot& plot);
  void encodeCreateRichPlot(proto::CommandList& list, RichPlot& plot);
  void encodeSetRichPlotData(
      proto::CommandList& list,
      const std::string& plotKey,
      const RichPlotData& data);
};

} // namespace server
} // namespace dart

#endif