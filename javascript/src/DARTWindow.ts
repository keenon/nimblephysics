import * as THREE from "three";
import "./style.scss";

import View from "./components/View";
import WorldDisplay from "./components/WorldDisplay";
import Timeline from "./components/Timeline";
import DataSelector from "./components/DataSelector";
import RealtimeWorldDisplay from "./components/RealtimeWorldDisplay";
import TimingScreen from "./components/TimingScreen";

class DARTWindow {
  scene: THREE.Scene;
  container: HTMLElement;
  view: View;
  timeline: Timeline | null;
  world: WorldDisplay | null;
  dataSelector: DataSelector | null;
  realtimeWorld: RealtimeWorldDisplay | null;
  timingScreen: TimingScreen | null;

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
   * This connects a websocket to the provided remote, so that we can get a live view of what is going on.
   *
   * @param url The WebsocketURL to connect to, for example "ws://localhost:8080"
   */
  connectLiveRemote = (url: string) => {
    this.timingScreen = new TimingScreen(this.container);

    const socket = new WebSocket(url);

    // Connection opened
    socket.addEventListener("open", (event) => {
      this.realtimeWorld = new RealtimeWorldDisplay(this.scene);
    });

    // Listen for messages
    socket.addEventListener("message", (event) => {
      const data: RealtimeUpdate = JSON.parse(event.data);
      if (data.type == "init") {
        this.realtimeWorld.initWorld(data.world);
      } else if (data.type == "update") {
        if (data.positions != null) {
          this.realtimeWorld.setPositions(data.timestep, data.positions);
        }
        if (data.colors != null) {
          this.realtimeWorld.setColors(data.colors);
        }
        if (data.timings != null) {
          this.timingScreen.registerTimings(data.timings);
        }
      } else if (data.type == "new_plan") {
        this.realtimeWorld.displayMPCPlan(data.plan);
      } else if (data.type == "timings") {
        this.timingScreen.registerTimings(data.timings);
      }
    });

    socket.addEventListener("close", () => {
      this.timingScreen.stop();
    });

    window.addEventListener("keydown", (e: KeyboardEvent) => {
      const message = JSON.stringify({
        type: "keydown",
        key: e.key.toString(),
      });
      if (socket.readyState == WebSocket.OPEN) {
        socket.send(message);
      }
    });

    window.addEventListener("keyup", (e: KeyboardEvent) => {
      const message = JSON.stringify({
        type: "keyup",
        key: e.key.toString(),
      });
      if (socket.readyState == WebSocket.OPEN) {
        socket.send(message);
      }
    });
  };

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
