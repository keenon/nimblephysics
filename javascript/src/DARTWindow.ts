import * as THREE from "three";
import "./style.scss";

import View from "./View";
import WorldDisplay from "./WorldDisplay";
import Timeline from "./Timeline";
import DataSelector from "./DataSelector";

class DARTWindow {
  scene: THREE.Scene;
  container: HTMLElement;
  view: View;
  timeline: Timeline | null;
  world: WorldDisplay | null;
  dataSelector: DataSelector | null;

  constructor(container: HTMLElement) {
    container.className += " DARTWindow";
    this.container = container;

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

    container.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
      }
    });

    const mainLoop = () => {
      this.view.render();
      if (this.timeline != null) {
        this.timeline.update();
      }

      requestAnimationFrame(mainLoop);
    };
    mainLoop();

    /// Random GUI stuff

    const title = document.createElement("div");
    title.innerHTML = "DiffDART Visualizer - v0.0.1";
    title.className = "GUI_title";
    container.appendChild(title);

    const showPathsContainer = document.createElement("div");
    showPathsContainer.className = "GUI_show-paths";
    showPathsContainer.innerHTML = "Show paths: ";
    container.appendChild(showPathsContainer);
    const showPathsButton = document.createElement("input");
    showPathsButton.type = "checkbox";
    showPathsContainer.appendChild(showPathsButton);
    showPathsButton.checked = true;
    showPathsButton.onclick = () => {
      if (this.world != null) {
        this.world.setShowPaths(showPathsButton.checked);
        showPathsButton.blur();
      }
    };
  }

  /**
   * This loads some data remotely and registers it
   *
   * @param name
   * @param url
   */
  registerRemoteData = (name: string, url: string) => {
    return fetch(url)
      .then((result) => result.json())
      .then((json) => {
        this.registerData(name, json);
      });
  };

  /**
   * This adds a FullReport as an option with a button for users to be able to select.
   *
   * @param name The human readable name for this data, to add to our chooser menu
   * @param data The raw data
   */
  registerData = (name: string, data: FullReport) => {
    if (this.world == null) {
      this.world = new WorldDisplay(this.scene, data);
      this.timeline = new Timeline(this.world, this.container);
    }
    if (this.dataSelector == null) {
      this.dataSelector = new DataSelector(
        this.world,
        this.timeline,
        this.container
      );
    }
    this.dataSelector.registerData(name, data);
  };

  /**
   * This sets the data this Window is displaying.
   *
   * @param data The raw data
   */
  setData = (data: FullReport) => {
    if (this.world == null) {
      this.world = new WorldDisplay(this.scene, data);
      this.timeline = new Timeline(this.world, this.container);
    } else {
      this.world.setData(data);
    }
  };
}

export default DARTWindow;
