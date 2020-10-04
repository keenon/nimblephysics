import * as THREE from "three";
import "./style.scss";

import View from "./View";
import WorldDisplay from "./WorldDisplay";
import Timeline from "./Timeline";
import DataSelector from "./DataSelector";

import worm_data from "./data/worm.txt";
import cart_data from "./data/cartpole.txt";

const worm = JSON.parse(worm_data);
const cart = JSON.parse(cart_data);

var scene = new THREE.Scene();
scene.background = new THREE.Color(0xaaaaaa);

const light = new THREE.DirectionalLight();
light.castShadow = true;
light.position.set(0, 1000, 0);
light.target.position.set(0, 0, 0);
light.intensity = 0.2;

scene.add(light);
scene.add(new THREE.HemisphereLight());
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

const world = new WorldDisplay(scene, cart);

const view = new View(scene);

const timeline = new Timeline(world);

const dataSelector = new DataSelector(world, timeline);
dataSelector.registerData("Cartpole", cart);
dataSelector.registerData("Jumpworm", worm);

addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "r") {
  }
});

const mainLoop = () => {
  view.render();
  timeline.update();

  requestAnimationFrame(mainLoop);
};
mainLoop();

/// Random GUI stuff

const title = document.createElement("div");
title.innerHTML = "DiffDART Visualizer - v0.0.1";
title.className = "GUI_title";
document.body.appendChild(title);

const showPathsContainer = document.createElement("div");
showPathsContainer.className = "GUI_show-paths";
showPathsContainer.innerHTML = "Show paths: ";
document.body.appendChild(showPathsContainer);
const showPathsButton = document.createElement("input");
showPathsButton.type = "checkbox";
showPathsContainer.appendChild(showPathsButton);
showPathsButton.checked = true;
showPathsButton.onclick = function () {
  world.setShowPaths(showPathsButton.checked);
  showPathsButton.blur();
};

/*
fetch("http://localhost:9080")
  .then((res) => res.json())
  .then((res) => {
    console.log(res);
  });
  */
