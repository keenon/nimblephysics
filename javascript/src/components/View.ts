import * as THREE from "three";

import { EffectComposer } from "../threejs_lib/postprocessing/EffectComposer.js";
import { SSAOPass } from "../threejs_lib/postprocessing/SSAOPass.js";
import { OrbitControls } from "../threejs_lib/controls/OrbitControls.js";
import { DragControls } from "../threejs_lib/controls/DragControls.js";

class View {
  container: HTMLDivElement;
  scene: THREE.Scene;
  renderer: THREE.Renderer;
  camera: THREE.Camera;
  composer: EffectComposer;
  width: number;
  height: number;
  orbitControls: OrbitControls;
  dragControls: DragControls;
  parent: HTMLElement;

  constructor(scene: THREE.Scene, parent: HTMLElement) {
    this.scene = scene;
    this.container = document.createElement("div");
    this.container.className = "View__container";
    parent.appendChild(this.container);
    this.parent = parent;

    this.refreshSize();

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setSize(this.width, this.height);
    (this.renderer as any).shadowMap.enabled = true;
    (this.renderer as any).shadowMapType = THREE.PCFSoftShadowMap;
    (this.renderer as any).outputEncoding = THREE.sRGBEncoding;

    this.container.appendChild(this.renderer.domElement);

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

    this.orbitControls = new OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    this.dragControls = new DragControls(
      [],
      this.camera,
      this.renderer.domElement
    );

    this.dragControls.addEventListener("dragstart", () => {
      this.orbitControls.enabled = false;
    });
    this.dragControls.addEventListener("dragend", () => {
      this.orbitControls.enabled = true;
    });

    this.orbitControls.addEventListener("change", () => {
      this.composer.render();
    });
    this.dragControls.addEventListener("drag", () => {
      this.composer.render();
    });
  }

  setDragHandler = (
    handler: (obj: THREE.Object3D, pos: THREE.Vector3) => void
  ) => {
    this.dragControls.setDragHandler(handler);
  };

  add = (obj: THREE.Object3D) => {
    this.scene.add(obj);
  };

  remove = (obj: THREE.Object3D) => {
    this.scene.remove(obj);
    this.dragControls.remove(obj);
  };

  enableMouseInteraction = (obj: THREE.Object3D) => {
    this.dragControls.add(obj);
  };

  disableMouseInteraction = (obj: THREE.Object3D) => {
    this.dragControls.remove(obj);
  };

  refreshSize = () => {
    this.width = this.parent.getBoundingClientRect().width;
    this.height = this.parent.getBoundingClientRect().height;
  };

  onWindowResize = () => {
    this.refreshSize();

    (this.camera as any).aspect = this.width / this.height;
    (this.camera as any).updateProjectionMatrix();

    this.renderer.setSize(this.width, this.height);
    this.composer.setSize(this.width, this.height);
    this.composer.render();
  };

  render = () => {
    this.orbitControls.update();
    this.composer.render();
  };
}

export default View;
