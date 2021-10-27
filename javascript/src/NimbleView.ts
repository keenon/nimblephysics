import * as THREE from "three";
import { CapsuleBufferGeometry } from "./threejs_lib/three-js-capsule-geometry/CapsuleBufferGeometry";
import "./style.scss";

import View from "./components/View";
import Slider from "./components/Slider";
import Plot from "./components/Plot";
import VERSION_NUM from "../../VERSION.txt";

const SCALE_FACTOR = 100;

type Text = {
  type: "text";
  container: HTMLElement;
  key: string;
  from_top_left: number[];
  size: number[];
  contents: string;
};

type Button = {
  type: "button";
  container: HTMLElement;
  buttonElem: HTMLButtonElement;
  key: string;
  from_top_left: number[];
  size: number[];
  label: string;
};

class DARTView {
  scene: THREE.Scene;
  container: HTMLElement;
  glContainer: HTMLElement;
  uiContainer: HTMLElement;
  view: View;
  running: boolean;

  objects: Map<string, THREE.Group | THREE.Mesh | THREE.Line>;
  keys: Map<THREE.Object3D, string>;
  textures: Map<string, THREE.Texture>;
  disposeHandlers: Map<string, () => void>;

  uiElements: Map<string, Text | Button | Slider | Plot>;

  dragListeners: ((key: string, pos: number[]) => void)[];

  connected: boolean;
  notConnectedWarning: HTMLElement;

  constructor(container: HTMLElement, startConnected: boolean = false) {
    container.className += " DARTWindow";
    this.container = container;
    this.glContainer = document.createElement("div");
    this.glContainer.className = "DARTWindow-gl";
    this.uiContainer = document.createElement("div");
    this.uiContainer.className = "DARTWindow-ui";
    this.container.appendChild(this.glContainer);
    this.container.appendChild(this.uiContainer);

    this.objects = new Map();
    this.keys = new Map();
    this.disposeHandlers = new Map();
    this.textures = new Map();
    this.uiElements = new Map();
    this.dragListeners = [];

    this.scene = new THREE.Scene();
    // this.scene.background = new THREE.Color(0xf8f8f8);
    this.scene.background = new THREE.Color(0xffffff);

    const light = new THREE.DirectionalLight();
    light.castShadow = true;
    light.position.set(0, 1000, 0);
    light.target.position.set(0, 0, 0);
    light.color = new THREE.Color(1.0, 1.0, 1.0);
    light.intensity = 0.2;

    this.scene.add(light);
    this.scene.add(new THREE.HemisphereLight());
    const shadowDim = 500;
    light.shadow.camera.left = -(shadowDim / 2);
    light.shadow.camera.right = shadowDim / 2;
    light.shadow.camera.top = -(shadowDim / 2);
    light.shadow.camera.bottom = shadowDim / 2;
    light.shadow.camera.far = 3000;
    light.shadow.mapSize.width = 2048;
    light.shadow.mapSize.height = 2048;

    /*
    // Just to debug shadow attributes

    const cameraHelper = new THREE.CameraHelper(light.shadow.camera);
    scene.add(cameraHelper);
    const helper = new THREE.DirectionalLightHelper(light);
    scene.add(helper);
    function updateCamera() {
      // update the light target's matrixWorld because it's needed by the helper
      light.target.updateMatrixWorld();
      helper.update();
      // update the light's shadow camera's projection matrix
      light.shadow.camera.updateProjectionMatrix();
      // and now update the camera helper we're using to show the light's shadow camera
      cameraHelper.update();
    }
    updateCamera();
    */

    this.view = new View(this.scene, this.glContainer);
    this.running = false;

    this.glContainer.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
      }
    });

    /// Get ready to deal with object dragging

    this.view.setDragHandler((obj: THREE.Object3D, posVec: THREE.Vector3) => {
      let key = this.keys.get(obj);
      let pos = [
        posVec.x / SCALE_FACTOR,
        posVec.y / SCALE_FACTOR,
        posVec.z / SCALE_FACTOR,
      ];
      if (key != null) {
        this.dragListeners.forEach((listener) => listener(key, pos));
      }
    });

    /// Random GUI stuff

    const title = document.createElement("div");
    title.innerHTML = "Nimble Visualizer - v" + VERSION_NUM;
    title.className = "GUI_title";
    this.uiContainer.appendChild(title);
    this.setConnected(startConnected);
  }

  /**
   * This marks the GUI as connected or not, which allows us to clearly display connection status on the GUI.
   */
  setConnected = (connected: boolean) => {
    this.connected = connected;

    if (!this.connected) {
      if (this.notConnectedWarning == null) {
        this.notConnectedWarning = document.createElement("div");
        this.notConnectedWarning.innerHTML = "Connecting to GUI server...";
        this.notConnectedWarning.className = "GUI_not_connected";
        this.uiContainer.appendChild(this.notConnectedWarning);
      }
    } else {
      if (this.notConnectedWarning != null) {
        this.notConnectedWarning.remove();
        this.notConnectedWarning = null;
      }
    }
  };

  /**
   * This adds a listener for dragging events
   */
  addDragListener = (dragListener: (key: string, pos: number[]) => void) => {
    this.dragListeners.push(dragListener);
  };

  /**
   * This enables mouse interaction on a specific object by key
   */
  enableMouseInteraction = (key: string) => {
    const obj = this.objects.get(key);
    if (obj != null) {
      this.view.enableMouseInteraction(obj);
    }
  };

  /**
   * This enables mouse interaction on a specific object by key
   */
  disableMouseInteraction = (key: string) => {
    const obj = this.objects.get(key);
    if (obj != null) {
      this.view.disableMouseInteraction(obj);
    }
  };

  /**
   * This adds a cube to the scene
   *
   * Must call render() to see results!
   */
  createBox = (
    key: string,
    size: number[],
    pos: number[],
    euler: number[],
    color: number[],
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    const geometry = new THREE.BoxBufferGeometry(
      SCALE_FACTOR,
      SCALE_FACTOR,
      SCALE_FACTOR
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.rotation.x = euler[0];
    mesh.rotation.y = euler[1];
    mesh.rotation.z = euler[2];
    mesh.castShadow = castShadows;
    mesh.receiveShadow = receiveShadows;
    mesh.scale.set(size[0], size[1], size[2]);

    this.objects.set(key, mesh);
    this.disposeHandlers.set(key, () => {
      material.dispose();
      geometry.dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(mesh);
  };

  /**
   * This adds a sphere to the scene
   *
   * Must call render() to see results!
   */
  createSphere = (
    key: string,
    radius: number,
    pos: number[],
    color: number[],
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    const NUM_SPHERE_SEGMENTS = 18;
    const geometry = new THREE.SphereBufferGeometry(
      SCALE_FACTOR,
      NUM_SPHERE_SEGMENTS,
      NUM_SPHERE_SEGMENTS
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.castShadow = castShadows;
    mesh.receiveShadow = receiveShadows;
    mesh.scale.set(radius, radius, radius);

    this.objects.set(key, mesh);
    this.disposeHandlers.set(key, () => {
      material.dispose();
      geometry.dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(mesh);
  };

  /**
   * This adds a capsule to the scene
   *
   * Must call render() to see results!
   */
  createCapsule = (
    key: string,
    radius: number,
    height: number,
    pos: number[],
    euler: number[],
    color: number[],
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    const NUM_SPHERE_SEGMENTS = 18;
    const geometry = new CapsuleBufferGeometry(
      radius * SCALE_FACTOR,
      radius * SCALE_FACTOR,
      height * SCALE_FACTOR,
      NUM_SPHERE_SEGMENTS,
      1,
      NUM_SPHERE_SEGMENTS,
      NUM_SPHERE_SEGMENTS
    );

    // By default, this extends the capsule along the Y axis, we want the Z axis instead
    const vertexArray: Float32Array = geometry.getAttribute("position")
      .array as Float32Array;
    for (var i = 0; i < vertexArray.length / 3; i++) {
      const index = i * 3;
      const swapY = vertexArray[index + 1];
      vertexArray[index + 1] = -vertexArray[index + 2];
      vertexArray[index + 2] = swapY;
    }
    geometry.computeVertexNormals();

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.rotation.x = euler[0];
    mesh.rotation.y = euler[1];
    mesh.rotation.z = euler[2];
    mesh.castShadow = castShadows;
    mesh.receiveShadow = receiveShadows;

    this.objects.set(key, mesh);
    this.disposeHandlers.set(key, () => {
      material.dispose();
      geometry.dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(mesh);
  };

  /**
   * This adds a line to the scene
   *
   * Must call render() to see results!
   */
  createLine = (key: string, points: number[][], color: number[]) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    const pathMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      linewidth: 2,
    });
    const pathPoints = [];
    for (let i = 0; i < points.length; i++) {
      pathPoints.push(
        new THREE.Vector3(
          points[i][0] * SCALE_FACTOR,
          points[i][1] * SCALE_FACTOR,
          points[i][2] * SCALE_FACTOR
        )
      );
    }
    const pathGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    const path = new THREE.Line(pathGeometry, pathMaterial);

    this.objects.set(key, path);
    this.disposeHandlers.set(key, () => {
      pathMaterial.dispose();
      pathGeometry.dispose();
    });
    this.keys.set(path, key);

    this.view.add(path);
  };

  /**
   * This loads a texture from a Base64 string encoding of it
   */
  createTexture = (key: string, base64: string) => {
    this.textures.set(key, new THREE.TextureLoader().load(base64));
  };

  /**
   * This adds a line to the scene
   *
   * Must call render() to see results!
   */
  createMesh = (
    key: string,
    vertices: number[][],
    vertexNormals: number[][],
    faces: number[][],
    uv: number[][],
    texture_starts: { key: string; start: number }[],
    pos: number[],
    euler: number[],
    scale: number[],
    color: number[],
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    let meshMaterial;
    if (texture_starts.length > 0 && uv.length > 0) {
      // TODO: respect multiple texture_starts per object
      meshMaterial = new THREE.MeshLambertMaterial({
        color: new THREE.Color(1, 1, 1),
        map: this.textures.get(texture_starts[0].key),
      });
    } else {
      meshMaterial = new THREE.MeshLambertMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
      });
    }

    const meshPoints = [];
    const rawUVs = [];
    const rawNormals = [];

    for (let i = 0; i < faces.length; i++) {
      for (let j = 0; j < 3; j++) {
        let vertexIndex = faces[i][j];
        meshPoints.push(
          new THREE.Vector3(
            vertices[vertexIndex][0] * SCALE_FACTOR,
            vertices[vertexIndex][1] * SCALE_FACTOR,
            vertices[vertexIndex][2] * SCALE_FACTOR
          )
        );
        if (uv != null && uv.length > vertexIndex) {
          rawUVs.push(uv[vertexIndex][0]);
          rawUVs.push(uv[vertexIndex][1]);
        }
        if (vertexNormals != null && vertexNormals.length > vertexIndex) {
          rawNormals.push(vertexNormals[vertexIndex][0]);
          rawNormals.push(vertexNormals[vertexIndex][1]);
          rawNormals.push(vertexNormals[vertexIndex][2]);
        }
      }
    }

    const meshGeometry = new THREE.BufferGeometry().setFromPoints(meshPoints);
    if (rawUVs.length > 0) {
      meshGeometry.setAttribute(
        "uv",
        new THREE.BufferAttribute(new Float32Array(rawUVs), 2)
      );
    }
    if (rawNormals.length > 0) {
      meshGeometry.setAttribute(
        "normal",
        new THREE.BufferAttribute(new Float32Array(rawNormals), 3)
      );
    } else {
      meshGeometry.computeVertexNormals();
    }
    meshGeometry.computeBoundingBox();

    const mesh = new THREE.Mesh(meshGeometry, meshMaterial);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.rotation.x = euler[0];
    mesh.rotation.y = euler[1];
    mesh.rotation.z = euler[2];
    mesh.castShadow = castShadows;
    mesh.receiveShadow = receiveShadows;
    mesh.scale.set(scale[0], scale[1], scale[2]);

    this.objects.set(key, mesh);
    this.disposeHandlers.set(key, () => {
      meshMaterial.dispose();
      meshGeometry.dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(mesh);
  };

  /**
   * Moves an object.
   *
   * Must call render() to see results!
   */
  setObjectPos = (key: string, pos: number[]) => {
    const obj = this.objects.get(key);
    if (obj) {
      obj.position.x = pos[0] * SCALE_FACTOR;
      obj.position.y = pos[1] * SCALE_FACTOR;
      obj.position.z = pos[2] * SCALE_FACTOR;
    }
  };

  /**
   * Rotates an object.
   *
   * Must call render() to see results!
   */
  setObjectRotation = (key: string, euler: number[]) => {
    const obj = this.objects.get(key);
    if (obj) {
      obj.rotation.x = euler[0];
      obj.rotation.y = euler[1];
      obj.rotation.z = euler[2];
    }
  };

  /**
   * Changes the scale of an object
   *
   * Must call render() to see results!
   */
  setObjectScale = (key: string, scale: number[]) => {
    const obj = this.objects.get(key);
    if (obj) {
      (obj as any).scale.set(scale[0], scale[1], scale[2]);
    }
  };

  /**
   * Removes an object from the scene, if it exists.
   *
   * @param key The key of the object (box, sphere, line, mesh) to be removed
   */
  deleteObject = (key: string) => {
    const obj = this.objects.get(key);
    if (obj) {
      this.disposeHandlers.get(key)();
      this.view.remove(obj);
      this.keys.delete(obj);
      this.scene.remove(obj);
      this.objects.delete(key);
    }
  };

  _createUIElementContainer = (
    key: string,
    from_top_left: number[],
    size: number[]
  ) => {
    const div: HTMLDivElement = document.createElement("div");
    div.style.position = "absolute";
    div.style.left = from_top_left[0] + "px";
    div.style.top = from_top_left[1] + "px";
    div.style.width = size[0] + "px";
    div.style.height = size[1] + "px";
    div.className = "DARTWindow-ui-elem";
    this.uiContainer.appendChild(div);
    return div;
  };

  /**
   * This adds a text box to the GUI. This is visible immediately even if you don't call render()
   */
  createText = (
    key: string,
    from_top_left: number[],
    size: number[],
    contents: string
  ) => {
    this.deleteUIElement(key);
    let text: Text = {
      type: "text",
      container: this._createUIElementContainer(key, from_top_left, size),
      key,
      from_top_left,
      size,
      contents,
    };
    text.container.innerHTML = contents;
    this.uiElements.set(key, text);
  };

  /**
   * This adds a button to the GUI. This is visible immediately even if you don't call render()
   */
  createButton = (
    key: string,
    from_top_left: number[],
    size: number[],
    label: string,
    onClick: () => void
  ) => {
    this.deleteUIElement(key);
    let container: HTMLDivElement = this._createUIElementContainer(
      key,
      from_top_left,
      size
    );
    let buttonElem: HTMLButtonElement = document.createElement("button");
    buttonElem.innerHTML = label;
    buttonElem.onclick = onClick;
    buttonElem.className = "DARTWindow-button";
    container.appendChild(buttonElem);
    let button: Button = {
      type: "button",
      container,
      buttonElem,
      key,
      from_top_left,
      size,
      label,
    };
    this.uiElements.set(key, button);
  };

  /**
   * This adds a slider to the GUI. This is visible immediately even if you don't call render()
   */
  createSlider = (
    key: string,
    from_top_left: number[],
    size: number[],
    min: number,
    max: number,
    value: number,
    onlyInts: boolean,
    horizontal: boolean,
    onChange: (value) => void
  ) => {
    this.deleteUIElement(key);
    let container: HTMLDivElement = this._createUIElementContainer(
      key,
      from_top_left,
      size
    );
    let slider = new Slider(
      container,
      key,
      from_top_left,
      size,
      min,
      max,
      value,
      onlyInts,
      horizontal,
      onChange
    );
    this.uiElements.set(key, slider);
  };

  /**
   * This adds a plot to the GUI. This is visible immediately even if you don't call render()
   */
  createPlot = (
    key: string,
    from_top_left: number[],
    size: number[],
    minX: number,
    maxX: number,
    xs: number[],
    minY: number,
    maxY: number,
    ys: number[],
    plotType: "line" | "scatter"
  ) => {
    this.deleteUIElement(key);
    let container: HTMLDivElement = this._createUIElementContainer(
      key,
      from_top_left,
      size
    );
    let plot = new Plot(
      container,
      key,
      from_top_left,
      size,
      minX,
      maxX,
      xs,
      minY,
      maxY,
      ys,
      plotType
    );
    this.uiElements.set(key, plot);
  };

  /**
   * This deletes a UI element (e.g. text, button, slider, plot) by key
   */
  deleteUIElement = (key: string) => {
    if (this.uiElements.has(key)) {
      this.uiElements.get(key).container.remove();
      this.uiElements.delete(key);
    }
  };

  /**
   * This moves a UI element (e.g. text, button, slider, plot) by key
   */
  setUIElementPosition = (key: string, fromTopLeft: number[]) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      elem.from_top_left = fromTopLeft;
      elem.container.style.left = fromTopLeft[0] + "px";
      elem.container.style.top = fromTopLeft[1] + "px";
      if (elem.type === "plot") elem.redraw();
    }
  };

  /**
   * This resizes a UI element (e.g. text, button, slider, plot) by key
   */
  setUIElementSize = (key: string, size: number[]) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      elem.size = size;
      elem.container.style.width = size[0] + "px";
      elem.container.style.height = size[1] + "px";
      if (elem.type === "plot") elem.redraw();
    }
  };

  setTextContents = (key: string, contents: string) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "text") elem.container.innerHTML = contents;
    }
  };

  setButtonLabel = (key: string, label: string) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "button") elem.buttonElem.innerHTML = label;
    }
  };

  setSliderValue = (key: string, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setValue(value);
    }
  };

  setSliderMin = (key: string, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setMin(value);
    }
  };

  setSliderMax = (key: string, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setMax(value);
    }
  };

  setPlotData = (
    key: string,
    minX: number,
    maxX: number,
    xs: number[],
    minY: number,
    maxY: number,
    ys: number[]
  ) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "plot") elem.setData(minX, maxX, xs, minY, maxY, ys);
    }
  };

  render() {
    this.view.render();
  }

  /**
   * This runs the main loop. If you don't call this, the view won't update!
   */
  clear() {
    this.objects.forEach((v, k) => {
      this.view.remove(v);
    });
    this.objects.clear();
    this.keys.clear();

    this.view.render();
  }

  stop() {
    this.running = false;
  }
}

export default DARTView;
