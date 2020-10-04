import * as THREE from "three";

import { EffectComposer } from "./threejs_lib/postprocessing/EffectComposer.js";
import { SSAOPass } from "./threejs_lib/postprocessing/SSAOPass.js";
import { OrbitControls } from "./threejs_lib/controls/OrbitControls.js";

class View {
  container: any;
  scene: THREE.Scene;
  renderer: THREE.Renderer;
  camera: THREE.Camera;
  composer: EffectComposer;
  width: number;
  height: number;
  controls: OrbitControls;

  constructor(scene: THREE.Scene) {
    this.refreshSize();

    const container: any = document.createElement("div");
    container.className = "View__container";
    document.body.appendChild(container);

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setSize(this.width, this.height);
    (this.renderer as any).shadowMap.enabled = true;
    (this.renderer as any).shadowMapType = THREE.PCFSoftShadowMap;

    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(
      65,
      this.width / this.height,
      10,
      100000
    );
    this.camera.position.z = 500;

    this.composer = new EffectComposer(this.renderer);

    var ssaoPass = new SSAOPass(scene, this.camera, this.width, this.height);
    ssaoPass.kernelRadius = 5;
    ssaoPass.minDistance = 0.00001;
    ssaoPass.maxDistance = 0.00006;
    this.composer.addPass(ssaoPass);
    ssaoPass.output = SSAOPass.OUTPUT.Default;

    window.addEventListener("resize", this.onWindowResize, false);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
  }

  refreshSize = () => {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
  };

  onWindowResize = () => {
    this.refreshSize();

    (this.camera as any).aspect = this.width / this.height;
    (this.camera as any).updateProjectionMatrix();

    this.renderer.setSize(this.width, this.height);
    this.composer.setSize(this.width, this.height);
  };

  render = () => {
    this.controls.update();
    this.composer.render();
  };
}

export default View;
