import * as THREE from "three";
import "./style.scss";

import View from "./components/View";

const SCALE_FACTOR = 100;

class DARTView {
  scene: THREE.Scene;
  container: HTMLElement;
  view: View;
  running: boolean;
  objects: Map<string, THREE.Group | THREE.Mesh | THREE.Line>;
  names: Map<THREE.Object3D, string>;

  dragListeners: ((name: string, pos: number[]) => void)[];

  constructor(container: HTMLElement) {
    container.className += " DARTWindow";
    this.container = container;
    this.objects = new Map();
    this.names = new Map();
    this.dragListeners = [];

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf8f8f8);

    const light = new THREE.DirectionalLight();
    light.castShadow = true;
    light.position.set(0, 1000, 0);
    light.target.position.set(0, 0, 0);
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

    this.view = new View(this.scene, container);
    this.running = false;

    container.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
      }
    });

    /// Get ready to deal with object dragging

    this.view.setDragHandler((obj: THREE.Object3D, posVec: THREE.Vector3) => {
      let name = this.names.get(obj);
      let pos = [
        posVec.x / SCALE_FACTOR,
        posVec.y / SCALE_FACTOR,
        posVec.z / SCALE_FACTOR,
      ];
      if (name != null) {
        this.dragListeners.forEach((listener) => listener(name, pos));
      }
    });

    /// Random GUI stuff

    const title = document.createElement("div");
    title.innerHTML = "DiffDART Visualizer - v0.1.0";
    title.className = "GUI_title";
    container.appendChild(title);
  }

  /**
   * This adds a listener for dragging events
   */
  addDragListener = (dragListener: (name: string, pos: number[]) => void) => {
    this.dragListeners.push(dragListener);
  };

  /**
   * This enables mouse interaction on a specific object by name
   */
  enableMouseInteraction = (name: string) => {
    const obj = this.objects.get(name);
    if (obj != null) {
      this.view.enableMouseInteraction(obj);
    }
  };

  /**
   * This enables mouse interaction on a specific object by name
   */
  disableMouseInteraction = (name: string) => {
    const obj = this.objects.get(name);
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
    name: string,
    size: number[],
    pos: number[],
    euler: number[],
    color: number[]
  ) => {
    if (this.objects.has(name)) {
      this.view.remove(this.objects.get(name));
    }
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    const geometry = new THREE.BoxBufferGeometry(
      size[0] * SCALE_FACTOR,
      size[1] * SCALE_FACTOR,
      size[2] * SCALE_FACTOR
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.rotation.x = euler[0];
    mesh.rotation.y = euler[1];
    mesh.rotation.z = euler[2];
    this.objects.set(name, mesh);
    this.names.set(mesh, name);

    this.view.add(mesh);
  };

  /**
   * This adds a sphere to the scene
   *
   * Must call render() to see results!
   */
  createSphere = (
    name: string,
    radius: number,
    pos: number[],
    color: number[]
  ) => {
    if (this.objects.has(name)) {
      this.view.remove(this.objects.get(name));
    }
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    const NUM_SPHERE_SEGMENTS = 18;
    const geometry = new THREE.SphereBufferGeometry(
      radius * SCALE_FACTOR,
      NUM_SPHERE_SEGMENTS,
      NUM_SPHERE_SEGMENTS
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    this.objects.set(name, mesh);
    this.names.set(mesh, name);

    this.view.add(mesh);
  };

  /**
   * This adds a line to the scene
   *
   * Must call render() to see results!
   */
  createLine = (name: string, points: number[][], color: number[]) => {
    if (this.objects.has(name)) {
      this.view.remove(this.objects.get(name));
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
    this.objects.set(name, path);
    this.names.set(path, name);

    this.view.add(path);
  };

  /**
   * Moves an object.
   *
   * Must call render() to see results!
   */
  setObjectPos = (name: string, pos: number[]) => {
    const obj = this.objects.get(name);
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
  setObjectRotation = (name: string, euler: number[]) => {
    const obj = this.objects.get(name);
    if (obj) {
      obj.rotation.x = euler[0];
      obj.rotation.y = euler[1];
      obj.rotation.z = euler[2];
    }
  };

  /**
   * Changes the color of an object
   *
   * Must call render() to see results!
   */
  setObjectColor = (name: string, color: number[]) => {
    const obj = this.objects.get(name);
    if (obj) {
      (obj as any).material.color.x = color[0];
      (obj as any).material.color.y = color[1];
      (obj as any).material.color.z = color[2];
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
    this.names.clear();

    this.view.render();
  }

  stop() {
    this.running = false;
  }
}

export default DARTView;
