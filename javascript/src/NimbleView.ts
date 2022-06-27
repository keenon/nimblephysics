import * as THREE from "three";
import { CapsuleBufferGeometry } from "./threejs_lib/three-js-capsule-geometry/CapsuleBufferGeometry";
import "./style.scss";

import View from "./components/View";
import Slider from "./components/Slider";
import SimplePlot from "./components/SimplePlot";
import RichPlot from "./components/RichPlot";
import { dart } from './proto/GUI';
import VERSION_NUM from "../../VERSION.txt";
import logoSvg from "!!raw-loader!./nimblelogo.svg";
import leftMouseSvg from "!!raw-loader!./leftMouse.svg";
import rightMouseSvg from "!!raw-loader!./rightMouse.svg";
import scrollMouseSvg from "!!raw-loader!./scrollMouse.svg";

const SCALE_FACTOR = 100;

type Text = {
  type: "text";
  container: HTMLElement;
  key: number;
  from_top_left: number[];
  size: number[];
  contents: string;
};

type Button = {
  type: "button";
  container: HTMLElement;
  buttonElem: HTMLButtonElement;
  key: number;
  from_top_left: number[];
  size: number[];
  label: string;
};

class Layer {
  view: DARTView;
  shown: boolean;
  key: number;
  name: string;
  color: number[];
  objects: Set<number>;
  uiElements: Set<number>;

  constructor(key: number, name: string, color: number[], shown: boolean, view: DARTView) {
    this.key = key;
    this.name = name;
    this.color = color;
    this.view = view;
    this.objects = new Set();
    this.uiElements = new Set();
    this.shown = shown;

    const row = document.createElement("tr");

    const checkCell = document.createElement("td");
    row.appendChild(checkCell);
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = shown;
    checkCell.appendChild(checkbox);

    const nameCell = document.createElement("td");
    row.appendChild(nameCell);
    const nameColor = document.createElement('div');
    nameColor.style.width = '20px';
    nameColor.style.height = '20px';
    nameColor.style.marginRight = '7px';
    nameColor.style.display = 'inline-block';
    nameColor.style.verticalAlign = 'middle';
    nameColor.style.backgroundColor = 'rgba('+(color[0]*255)+','+(color[1]*255)+','+(color[2]*255)+','+color[3]+')';
    nameCell.appendChild(nameColor);
    const nameSpan = document.createElement("span");
    nameSpan.innerHTML = this.name === '' ? 'Default' : this.name;
    nameCell.appendChild(nameSpan);

    this.view.layersTable.appendChild(row);

    checkbox.onclick = () => {
      if (!checkbox.checked) {
        this.hide();
      }
      else {
        this.show();
      }
    };
  }

  addObject = (key: number) => {
    this.objects.add(key);
    if (this.shown) {
      this.view.showObject(key);
    }
    else {
      this.view.hideObject(key);
    }
  };

  addUIElement = (key: number) => {
    this.uiElements.add(key);
  };

  show = () => {
    console.log("Show: "+this.name);
    this.shown = true;
    this.objects.forEach((key) => {
      this.view.showObject(key);
    });
    this.view.render();
  };

  hide = () => {
    this.shown = false;
    this.objects.forEach((key) => {
      this.view.hideObject(key);
    });
    this.view.render();
  };
}

class DARTView {
  scene: THREE.Scene;
  container: HTMLElement;
  glContainer: HTMLElement;
  uiContainer: HTMLElement;
  view: View;
  running: boolean;

  objects: Map<number, THREE.Group | THREE.Mesh | THREE.Line>;
  objectColors: Map<number, number[]>;
  keys: Map<THREE.Object3D, number>;
  textures: Map<number, THREE.Texture>;
  disposeHandlers: Map<number, () => void>;
  objectType: Map<number, string>;

  uiElements: Map<number, Text | Button | Slider | SimplePlot | RichPlot>;

  dragListeners: ((key: number, pos: number[]) => void)[];

  sphereGeometry: THREE.SphereBufferGeometry;

  layers: Map<number, Layer>;

  connected: boolean;
  notConnectedWarning: HTMLElement;
  layersTable: HTMLElement;

  tooltip: HTMLElement;
  hovering: number[];

  constructor(container: HTMLElement, startConnected: boolean = false) {
    container.className += " DARTWindow";
    this.container = container;
    this.glContainer = document.createElement("div");
    this.glContainer.className = "DARTWindow-gl";
    this.uiContainer = document.createElement("div");
    this.uiContainer.className = "DARTWindow-ui";
    this.container.appendChild(this.glContainer);
    this.container.appendChild(this.uiContainer);

    this.tooltip = document.createElement("div");
    this.tooltip.className = 'Tooltip';
    this.uiContainer.appendChild(this.tooltip);
    this.hovering = []

    this.objects = new Map();
    this.objectColors = new Map();
    this.keys = new Map();
    this.disposeHandlers = new Map();
    this.textures = new Map();
    this.uiElements = new Map();
    this.objectType = new Map();
    this.dragListeners = [];
    this.layers = new Map();

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

    this.view = new View(this.scene, this.glContainer, this.onTooltipHoveron, this.onTooltipHoveroff);
    this.running = false;

    this.glContainer.addEventListener("keydown", this.glContainerKeyboardEventListener);

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

    const title = document.createElement("a");
    title.href = "https://www.nimblephysics.org";
    title.target = "#";
    title.className = "GUI_title";

    const logo = document.createElement("svg");
    logo.innerHTML = logoSvg;
    title.appendChild(logo);

    const titleText = document.createElement("span");
    titleText.innerHTML = "nimble<b>visualizer</b> v" + VERSION_NUM;
    title.appendChild(titleText);

    this.uiContainer.appendChild(title);
    this.setConnected(startConnected);

    const instructions = document.createElement("table");
    instructions.className = "GUI_instruction";

    const instructionsRow1 = document.createElement("tr");
    instructions.appendChild(instructionsRow1);
    const leftMouseCell = document.createElement("td");
    instructionsRow1.appendChild(leftMouseCell);
    const leftMouseImg = document.createElement("svg");
    leftMouseImg.innerHTML = leftMouseSvg;
    leftMouseCell.appendChild(leftMouseImg);
    const leftMouseInstructionsCell = document.createElement("td");
    instructionsRow1.appendChild(leftMouseInstructionsCell);
    leftMouseInstructionsCell.innerHTML = "Rotate view";

    const instructionsRow2 = document.createElement("tr");
    instructions.appendChild(instructionsRow2);
    const rightMouseCell = document.createElement("td");
    instructionsRow2.appendChild(rightMouseCell);
    const rightMouseImg = document.createElement("svg");
    rightMouseImg.innerHTML = rightMouseSvg;
    rightMouseCell.appendChild(rightMouseImg);
    const rightMouseInstructionsCell = document.createElement("td");
    instructionsRow2.appendChild(rightMouseInstructionsCell);
    rightMouseInstructionsCell.innerHTML = "Translate view";

    const instructionsRow3 = document.createElement("tr");
    instructions.appendChild(instructionsRow3);
    const scrollMouseCell = document.createElement("td");
    instructionsRow3.appendChild(scrollMouseCell);
    const scrollMouseImg = document.createElement("svg");
    scrollMouseImg.innerHTML = scrollMouseSvg;
    scrollMouseCell.appendChild(scrollMouseImg);
    const scrollMouseInstructionsCell = document.createElement("td");
    instructionsRow3.appendChild(scrollMouseInstructionsCell);
    scrollMouseInstructionsCell.innerHTML = "Zoom view";

    this.uiContainer.appendChild(instructions);

    this.layersTable = document.createElement("table");
    this.layersTable.className = "GUI_layers";

    this.uiContainer.appendChild(this.layersTable);

    // Set up the reusable sphere geometry

    const NUM_SPHERE_SEGMENTS = 18;
    this.sphereGeometry = new THREE.SphereBufferGeometry(
      SCALE_FACTOR,
      NUM_SPHERE_SEGMENTS,
      NUM_SPHERE_SEGMENTS
    );
  }

  glContainerKeyboardEventListener = (e: KeyboardEvent) => {
    if (e.key === " ") {
      e.preventDefault();
    }
  };

  onTooltipHoveron = (keys: number[], tooltip: string, top_x: number, top_y: number) => {
    this.tooltip.innerHTML = tooltip;
    this.tooltip.style.top = top_y+'px';
    this.tooltip.style.left = top_x+'px';
    this.tooltip.style.opacity = '1.0';
    if (JSON.stringify(keys) !== JSON.stringify(this.hovering)) {
      if (this.hovering.length > 0) {
        this.hovering.forEach(k => {
          this.resetObjectColor(k);
        });
      }

      keys.forEach((key) => {
        const currentColor = this.objectColors.get(key);
        let hoverColor = [currentColor[0]*0.7, currentColor[1]*0.7, currentColor[2]*0.7, 1];
        this.setObjectColor(key, hoverColor, false);
      })

      this.hovering = keys;
      this.render();
    }
    window.addEventListener('mousemove', this.tooltipMousemoveListener);
  };

  tooltipMousemoveListener = (e: MouseEvent) => {
    const rect = this.container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    this.tooltip.style.top = mouseY+'px';
    this.tooltip.style.left = mouseX+'px';
  }

  onTooltipHoveroff = () => {
    if (this.hovering.length > 0) {
      this.hovering.forEach(k => {
        this.resetObjectColor(k);
      });
      this.hovering = [];
      this.render();
    }
    this.tooltip.style.opacity = '0.0';
    window.removeEventListener('mousemove', this.tooltipMousemoveListener);
  };

  /**
   * This cleans up any resources that the view is using
   */
  dispose = () => {
    this.clear();

    // Clean up leftover callbacks that could cause a leak
    this.disposeHandlers.clear();
    this.view.setDragHandler(null);
    this.glContainer.removeEventListener("keydown", this.glContainerKeyboardEventListener);

    this.scene = null;
    this.view.dispose();
    this.view = null;
    this.glContainer.remove();
    this.uiContainer.remove();
  };

  /**
   * This reads and handles a command sent from the backend
   */
  handleCommand = (command: dart.proto.Command) => {
    if (command.layer != null) {
      const key = command.layer.key;
      const name = command.layer.name;
      const color = command.layer.color;
      const show = command.layer.default_show === true;
      this.createLayer(key, name, color, show);
    }
    else if (command.box != null) {
      const data = command.box.data;
      const size: number[] = [data[0], data[1], data[2]];
      const pos: number[] = [data[3], data[4], data[5]];
      const euler: number[] = [data[6], data[7], data[8]];
      const color: number[] = [data[9], data[10], data[11], data[12]];
      this.createBox(
        command.box.key,
        size,
        pos,
        euler,
        color,
        command.box.layer,
        command.box.cast_shadows === true,
        command.box.receive_shadows === true
      );
    }
    else if (command.sphere != null) {
      const data = command.sphere.data;
      const radius: number = data[0];
      const pos: number[] = [data[1], data[2], data[3]];
      const color: number[] = [data[4], data[5], data[6], data[7]];
      this.createSphere(
        command.sphere.key,
        radius,
        pos,
        color,
        command.sphere.layer,
        command.sphere.cast_shadows === true,
        command.sphere.receive_shadows === true
      );
    }
    else if (command.capsule != null) {
      const data = command.capsule.data;
      const radius: number = data[0];
      const height: number = data[1];
      const pos: number[] = [data[2], data[3], data[4]];
      const euler: number[] = [data[5], data[6], data[7]];
      const color: number[] = [data[8], data[9], data[10], data[11]];
      this.createCapsule(
        command.capsule.key,
        radius,
        height,
        pos,
        euler,
        color,
        command.capsule.layer,
        command.capsule.cast_shadows === true,
        command.capsule.receive_shadows === true
      );
    }
    else if (command.set_object_tooltip != null) {
      this.setTooltip(command.set_object_tooltip.key, command.set_object_tooltip.tooltip);
    }
    else if (command.delete_object_tooltip != null) {
      this.deleteTooltip(command.delete_object_tooltip.key);
    }
    else if (command.line != null) {
      const color: number[] = [command.line.color[0], command.line.color[1], command.line.color[2], command.line.color[3]];
      const vertices: number[][] = [];
      for (let i = 0; i < command.line.points.length; i++) {
        if (i % 3 == 0) {
          vertices.push([]);
        }
        vertices[vertices.length-1].push(command.line.points[i]);
      }

      this.createLine(
        command.line.key,
        vertices,
        color,
        command.line.layer
      );
    }
    else if (command.mesh != null) {
      const vertices: number[][] = [];
      const vertexNormals: number[][] = [];
      for (let i = 0; i < command.mesh.vertex.length; i++) {
        if (i % 3 == 0) {
          vertices.push([]);
          vertexNormals.push([]);
        }
        vertices[vertices.length-1].push(command.mesh.vertex[i]);
        vertexNormals[vertexNormals.length-1].push(command.mesh.vertex_normal[i]);
      }
      const faces: number[][] = [];
      for (let i = 0; i < command.mesh.face.length; i++) {
        if (i % 3 == 0) {
          faces.push([]);
        }
        faces[faces.length-1].push(command.mesh.face[i]);
      }
      const uvs: number[][] = [];
      for (let i = 0; i < command.mesh.uv.length; i++) {
        if (i % 2 == 0) {
          uvs.push([]);
        }
        uvs[uvs.length-1].push(command.mesh.uv[i]);
      }
      const texture_starts: {
        key: number,
        start: number
      }[] = [];
      for (let i = 0; i < command.mesh.texture.length; i++) {
        texture_starts.push({
          key: command.mesh.texture[i],
          start: command.mesh.texture_start[i]
        });
      }
      const data = command.mesh.data;
      const scale: number[] = [data[0], data[1], data[2]];
      const pos: number[] = [data[3], data[4], data[5]];
      const euler: number[] = [data[6], data[7], data[8]];
      const color: number[] = [data[9], data[10], data[11], data[12]];
      this.createMesh(
        command.mesh.key,
        vertices,
        vertexNormals,
        faces,
        uvs,
        texture_starts,
        pos,
        euler,
        scale,
        color,
        command.mesh.layer,
        command.mesh.cast_shadows === true,
        command.mesh.receive_shadows === true
      );
    }
    else if (command.texture != null) {
      this.createTexture(command.texture.key, command.texture.base64);
    }
    else if (command.set_object_position != null) {
      const data = command.set_object_position.data;
      const pos: number[] = [data[0], data[1], data[2]];
      this.setObjectPos(command.set_object_position.key, pos);
    }
    else if (command.set_object_rotation != null) {
      const data = command.set_object_rotation.data;
      const euler: number[] = [data[0], data[1], data[2]];
      this.setObjectRotation(command.set_object_rotation.key, euler);
    }
    else if (command.set_object_scale != null) {
      const data = command.set_object_scale.data;
      const scale: number[] = [data[0], data[1], data[2]];
      this.setObjectScale(command.set_object_scale.key, scale);
    }
    else if (command.set_object_color != null) {
      const data = command.set_object_color.data;
      const color: number[] = [data[0], data[1], data[2], data[3]];
      this.setObjectColor(command.set_object_color.key, color, true);
    }
    else if (command.enable_mouse_interaction != null) {
      this.enableMouseInteraction(command.enable_mouse_interaction.key);
    }
    else if (command.text != null) {
      const from_top_left: number[] = [command.text.pos[0], command.text.pos[1]];
      const size: number[] = [command.text.pos[2], command.text.pos[3]];
      this.createText(
        command.text.key,
        from_top_left,
        size,
        command.text.contents,
        command.text.layer
      );
    }
    else if (command.plot != null) {
      const from_top_left: number[] = [command.plot.pos[0], command.plot.pos[1]];
      const size: number[] = [command.plot.pos[2], command.plot.pos[3]];
      const minX = command.plot.bounds[0];
      const maxX = command.plot.bounds[1];
      const minY = command.plot.bounds[2];
      const maxY = command.plot.bounds[3];
      this.createSimplePlot(
        command.plot.key,
        from_top_left,
        size,
        minX,
        maxX,
        command.plot.xs,
        minY,
        maxY,
        command.plot.ys,
        command.plot.plot_type as any,
        command.plot.layer
      );
    }
    else if (command.rich_plot != null) {
      const from_top_left: number[] = [command.rich_plot.pos[0], command.rich_plot.pos[1]];
      const size: number[] = [command.rich_plot.pos[2], command.rich_plot.pos[3]];
      const minX = command.rich_plot.bounds[0];
      const maxX = command.rich_plot.bounds[1];
      const minY = command.rich_plot.bounds[2];
      const maxY = command.rich_plot.bounds[3];
      this.createRichPlot(
        command.rich_plot.key,
        from_top_left,
        size,
        minX,
        maxX,
        minY,
        maxY,
        command.rich_plot.title,
        command.rich_plot.x_axis_label,
        command.rich_plot.y_axis_label,
        command.rich_plot.layer
      );
    }
    else if (command.set_rich_plot_data != null) {
      this.setRichPlotData(
        command.set_rich_plot_data.key,
        command.set_rich_plot_data.name,
        command.set_rich_plot_data.xs,
        command.set_rich_plot_data.ys,
        command.set_rich_plot_data.color,
        command.set_rich_plot_data.plot_type as any
      );
    }
    else if (command.set_rich_plot_bounds != null) {
      const minX = command.set_rich_plot_bounds.bounds[0];
      const maxX = command.set_rich_plot_bounds.bounds[1];
      const minY = command.set_rich_plot_bounds.bounds[2];
      const maxY = command.set_rich_plot_bounds.bounds[3];
      this.setRichPlotBounds(
        command.set_rich_plot_bounds.key,
        minX,
        maxX,
        minY,
        maxY,
      );
    } else if (command.set_ui_elem_pos != null) {
      this.setUIElementPosition(command.set_ui_elem_pos.key, command.set_ui_elem_pos.fromTopLeft);
    } else if (command.set_ui_elem_size != null) {
      this.setUIElementSize(command.set_ui_elem_size.key, command.set_ui_elem_size.size);
    } else if (command.delete_ui_elem != null) {
      this.deleteUIElement(command.delete_ui_elem.key);
    } else if (command.delete_object != null) {
      this.deleteObject(command.delete_object.key);
    } else if (command.set_text_contents != null) {
      this.setTextContents(command.set_text_contents.key, command.set_text_contents.contents);
    } else if (command.set_button_label != null) {
      this.setButtonLabel(command.set_button_label.key, command.set_button_label.label);
    } else if (command.set_slider_value != null) {
      this.setSliderValue(command.set_slider_value.key, command.set_slider_value.value);
    } else if (command.set_slider_min != null) {
      this.setSliderMin(command.set_slider_min.key, command.set_slider_min.value);
    } else if (command.set_slider_max != null) {
      this.setSliderMax(command.set_slider_max.key, command.set_slider_max.value);
    } else if (command.set_plot_data != null) {
      const minX = command.set_plot_data.bounds[0];
      const maxX = command.set_plot_data.bounds[1];
      const minY = command.set_plot_data.bounds[2];
      const maxY = command.set_plot_data.bounds[3];
      this.setPlotData(
        command.set_plot_data.key,
        minX,
        maxX,
        command.set_plot_data.xs,
        minY,
        maxY,
        command.set_plot_data.ys
      );
    }
  };

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
  addDragListener = (dragListener: (key: number, pos: number[]) => void) => {
    this.dragListeners.push(dragListener);
  };

  /**
   * This enables mouse interaction on a specific object by key
   */
  enableMouseInteraction = (key: number) => {
    const obj = this.objects.get(key);
    if (obj != null) {
      this.view.enableMouseInteraction(key);
    }
  };

  /**
   * This enables mouse interaction on a specific object by key
   */
  disableMouseInteraction = (key: number) => {
    const obj = this.objects.get(key);
    if (obj != null) {
      this.view.disableMouseInteraction(key);
    }
  };

  /**
   * This creates a layer
   */
  createLayer = (
    key: number,
    name: string,
    color: number[],
    shown: boolean
  ) => {
    let layer = this.layers.get(key);
    if (layer == null) {
      layer = new Layer(key, name, color, shown, this);
      this.layers.set(key, layer);
    }
  };

  /**
   * This adds a cube to the scene
   *
   * Must call render() to see results!
   */
  createBox = (
    key: number,
    size: number[],
    pos: number[],
    euler: number[],
    color: number[],
    layer: number | undefined,
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    this.objectColors.set(key, color);
    if (this.objects.has(key)) {
      const mesh = this.objects.get(key) as THREE.Mesh;
      mesh.position.x = pos[0] * SCALE_FACTOR;
      mesh.position.y = pos[1] * SCALE_FACTOR;
      mesh.position.z = pos[2] * SCALE_FACTOR;
      mesh.rotation.x = euler[0];
      mesh.rotation.y = euler[1];
      mesh.rotation.z = euler[2];
      mesh.castShadow = castShadows;
      mesh.receiveShadow = receiveShadows;
      mesh.scale.set(size[0], size[1], size[2]);

      const material = mesh.material as THREE.MeshLambertMaterial;
      material.color.r = color[0];
      material.color.g = color[1];
      material.color.b = color[2];
      if (color.length > 3 && color[3] < 1.0) {
        material.transparent = true;
        material.opacity = color[3];
      }
    }
    else {
      const material = new THREE.MeshLambertMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
      });
      if (color.length > 3 && color[3] < 1.0) {
        material.transparent = true;
        material.opacity = color[3];
      }
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

      this.view.add(key, mesh);

      if (layer != null && this.layers.get(layer) != null) {
        this.layers.get(layer).addObject(key);
      }
    }
  };

  /**
   * This adds a sphere to the scene
   *
   * Must call render() to see results!
   */
  createSphere = (
    key: number,
    radius: number,
    pos: number[],
    color: number[],
    layer: number | undefined,
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    this.objectColors.set(key, color);
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    if (color.length > 3 && color[3] < 1.0) {
      material.transparent = true;
      material.opacity = color[3];
    }
    const mesh = new THREE.Mesh(this.sphereGeometry, material);
    mesh.position.x = pos[0] * SCALE_FACTOR;
    mesh.position.y = pos[1] * SCALE_FACTOR;
    mesh.position.z = pos[2] * SCALE_FACTOR;
    mesh.castShadow = castShadows;
    mesh.receiveShadow = receiveShadows;
    mesh.scale.set(radius, radius, radius);

    this.objects.set(key, mesh);
    this.disposeHandlers.set(key, () => {
      material.dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(key, mesh);

    if (layer != null && this.layers.has(layer)) {
      this.layers.get(layer).addObject(key);
    }
  };

  /**
   * This adds a capsule to the scene
   *
   * Must call render() to see results!
   */
  createCapsule = (
    key: number,
    radius: number,
    height: number,
    pos: number[],
    euler: number[],
    color: number[],
    layer: number | undefined,
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    if (this.objects.has(key)) {
      this.deleteObject(key);
    }
    this.objectType.set(key, "capsule");
    this.objectColors.set(key, color);
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
    });
    if (color.length > 3 && color[3] < 1.0) {
      material.transparent = true;
      material.opacity = color[3];
    }
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
      (geometry as any).dispose();
    });
    this.keys.set(mesh, key);

    this.view.add(key, mesh);

    if (layer != null && this.layers.has(layer)) {
      this.layers.get(layer).addObject(key);
    }
  };

  /**
   * This adds a line to the scene
   *
   * Must call render() to see results!
   */
  createLine = (key: number, points: number[][], color: number[], layer: number | undefined) => {
    this.objectType.set(key, "line");
    this.objectColors.set(key, color);
    // Try not to recreate geometry. If we already created a line in the past,
    // let's just update its buffers instead of creating fresh ones.
    if (this.objects.has(key)) {
      // console.log("Not creating line " + key);
      const line: THREE.Line = this.objects.get(key) as any;
      const positions = (line as any).geometry.attributes.position.array;
      (line as any).material.color = new THREE.Color(
        color[0],
        color[1],
        color[2]
      );

      let cursor = 0;
      for (let i = 0; i < points.length; i++) {
        positions[cursor++] = points[i][0] * SCALE_FACTOR;
        positions[cursor++] = points[i][1] * SCALE_FACTOR;
        positions[cursor++] = points[i][2] * SCALE_FACTOR;
      }

      (line as any).geometry.attributes.position.needsUpdate = true; // required after the first render
      // line.geometry.computeBoundingBox();
      // line.geometry.computeBoundingSphere();

      this.view.add(key, line);
    } else {
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

      this.view.add(key, path);
    }

    if (layer != null && this.layers.has(layer)) {
      // console.log(this.layers.get(layer).name + ": " + this.layers.get(layer).shown);
      this.layers.get(layer).addObject(key);
    }
  };

  /**
   * This registers a tooltip for a specific object in the view
   */
  setTooltip = (key: number, tooltip: string) => {
    this.view.setTooltip(key, tooltip);
  }

  /**
   * This removes the tooltip for a specific object in the view
   */
  deleteTooltip = (key: number) => {
    this.view.removeTooltip(key);
  }

  /**
   * This loads a texture from a Base64 string encoding of it
   */
  createTexture = (key: number, base64: string) => {
    if (!this.textures.has(key)) {
      this.textures.set(key, new THREE.TextureLoader().load(base64));
    }
  };

  /**
   * This adds a line to the scene
   *
   * Must call render() to see results!
   */
  createMesh = (
    key: number,
    vertices: number[][],
    vertexNormals: number[][],
    faces: number[][],
    uv: number[][],
    texture_starts: { key: number; start: number }[],
    pos: number[],
    euler: number[],
    scale: number[],
    color: number[],
    layer: number | undefined,
    castShadows: boolean,
    receiveShadows: boolean
  ) => {
    // Try not to recreate geometry. If we already created a mesh in the past,
    // let's just update its state instead of creating a new one.
    if (this.objects.has(key)) {
      const mesh: THREE.Mesh = this.objects.get(key) as THREE.Mesh;
      mesh.position.x = pos[0] * SCALE_FACTOR;
      mesh.position.y = pos[1] * SCALE_FACTOR;
      mesh.position.z = pos[2] * SCALE_FACTOR;
      mesh.rotation.x = euler[0];
      mesh.rotation.y = euler[1];
      mesh.rotation.z = euler[2];
      mesh.castShadow = castShadows;
      mesh.receiveShadow = receiveShadows;
      mesh.scale.set(scale[0], scale[1], scale[2]);
      this.view.add(key, mesh);
    } else {
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
      this.objectColors.set(key, color);
      if (color.length > 3 && color[3] < 1.0) {
        meshMaterial.transparent = true;
        meshMaterial.opacity = color[3];
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

      this.view.add(key, mesh);
    }

    if (layer != null && this.layers.has(layer)) {
      this.layers.get(layer).addObject(key);
    }
  };

  /**
   * Moves an object.
   *
   * Must call render() to see results!
   */
  setObjectPos = (key: number, pos: number[]) => {
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
  setObjectRotation = (key: number, euler: number[]) => {
    const obj = this.objects.get(key);
    if (obj) {
      obj.rotation.x = euler[0];
      obj.rotation.y = euler[1];
      obj.rotation.z = euler[2];
    }
  };

  /**
   * Sets the color of an object
   *
   * Must call render() to see results!
   */
  setObjectColor = (key: number, color: number[], save: boolean = true) => {
    const obj = this.objects.get(key);
    if (save) {
      this.objectColors.set(key, color);
    }
    (obj as any).material.color = new THREE.Color(color[0], color[1], color[2]);
    if (color.length > 3) {
      if (color[3] == 1.0) {
        (obj as any).material.transparent = false;
      }
      else {
        (obj as any).material.transparent = true;
        (obj as any).material.opacity = color[3];
      }
    }
  };

  /**
   * This resets an object to the saved color
   */
  resetObjectColor = (key: number) => {
    this.setObjectColor(key, this.objectColors.get(key));
  };

  /**
   * Changes the scale of an object
   *
   * Must call render() to see results!
   */
  setObjectScale = (key: number, scale: number[]) => {
    const obj = this.objects.get(key);
    // Don't dynamically set scales on capsules
    if (obj && this.objectType.get(key) != "capsule") {
      (obj as any).scale.set(scale[0], scale[1], scale[2]);
    }
  };

  /**
   * This removes an objects from the view, without deleting the reference to the object.
   * 
   * @param key 
   */
  hideObject = (key: number) => {
    const obj = this.objects.get(key);
    if (obj) {
      this.view.remove(key);
    }
  };

  /**
   * This shows an object in the view, if there's a reference to the object in maps.
   * 
   * @param key 
   */
  showObject = (key: number) => {
    const obj = this.objects.get(key);
    if (obj) {
      this.view.add(key, obj);
    }
  };

  /**
   * Removes an object from the scene, if it exists.
   *
   * @param key The key of the object (box, sphere, line, mesh) to be removed
   */
  deleteObject = (key: number) => {
    const obj = this.objects.get(key);
    if (obj) {
      this.view.remove(key);
      this.keys.delete(obj);
      this.scene.remove(obj);
      // Keep lines around, and just update the buffers if they ever get recreated
      if (this.objectType.get(key) != "line") {
        // console.log("Deleting object " + key);
        this.objects.delete(key);
        this.disposeHandlers.get(key)();
      } else {
        // console.log("Not deleting line " + key);
      }
    }
  };

  _createUIElementContainer = (
    key: number,
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
    key: number,
    from_top_left: number[],
    size: number[],
    contents: string,
    layer: number
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
    key: number,
    from_top_left: number[],
    size: number[],
    label: string,
    onClick: () => void,
    layer: number
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
    key: number,
    from_top_left: number[],
    size: number[],
    min: number,
    max: number,
    value: number,
    onlyInts: boolean,
    horizontal: boolean,
    onChange: (value) => void,
    layer: number
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
  createSimplePlot = (
    key: number,
    from_top_left: number[],
    size: number[],
    minX: number,
    maxX: number,
    xs: number[],
    minY: number,
    maxY: number,
    ys: number[],
    plotType: "line" | "scatter",
    layer: number
  ) => {
    this.deleteUIElement(key);
    let container: HTMLDivElement = this._createUIElementContainer(
      key,
      from_top_left,
      size
    );
    let plot = new SimplePlot(
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
   * This adds a plot to the GUI. This is visible immediately even if you don't call render()
   */
  createRichPlot = (
    key: number,
    from_top_left: number[],
    size: number[],
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
    title: string,
    xAxisLabel: string,
    yAxisLabel: string,
    layer: number
  ) => {
    this.deleteUIElement(key);
    let container: HTMLDivElement = this._createUIElementContainer(
      key,
      from_top_left,
      size
    );
    let plot = new RichPlot(
      container,
      key,
      from_top_left,
      size,
      minX,
      maxX,
      minY,
      maxY,
      title,
      xAxisLabel,
      yAxisLabel
    );
    this.uiElements.set(key, plot);
  };

  /**
   * This sets one data seriese on a rich plot. If there is no rich plot at "key", then this is a no-op.
   * 
   * @param key the key for the rich plot
   * @param dataName the name for the data series (must be unique)
   * @param xs the array of x points to plot
   * @param ys the array of y points to plot
   * @param color the color of the line to plot
   * @param plotType the type of plot (line or scatter)
   */
  setRichPlotData = (
    key: number,
    dataName: string,
    xs: number[],
    ys: number[],
    color: string,
    plotType: "line" | "scatter"
  ) => {
    const element = this.uiElements.get(key);
    if (element != null && element.type === 'rich_plot') {
      const richPlot: RichPlot = element as RichPlot;
      richPlot.setLineData(dataName, xs, ys, color, plotType);
    }
  };

  /**
   * This sets the bounds for a rich plot. If there is no rich plot at "key", then this is a no-op.
   * 
   * @param key 
   * @param minX 
   * @param maxX 
   * @param minY 
   * @param maxY 
   */
  setRichPlotBounds = (
    key: number,
    minX: number,
    maxX: number,
    minY: number,
    maxY: number
  ) => {
    const element = this.uiElements.get(key);
    if (element != null && element.type === 'rich_plot') {
      const richPlot: RichPlot = element as RichPlot;
      richPlot.setBounds(minX, maxX, minY, maxY);
    }
  };

  /**
   * This deletes a UI element (e.g. text, button, slider, plot) by key
   */
  deleteUIElement = (key: number) => {
    if (this.uiElements.has(key)) {
      this.uiElements.get(key).container.remove();
      this.uiElements.delete(key);
    }
  };

  /**
   * This moves a UI element (e.g. text, button, slider, plot) by key
   */
  setUIElementPosition = (key: number, fromTopLeft: number[]) => {
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
  setUIElementSize = (key: number, size: number[]) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      elem.size = size;
      elem.container.style.width = size[0] + "px";
      elem.container.style.height = size[1] + "px";
      if (elem.type === "plot") elem.redraw();
    }
  };

  setTextContents = (key: number, contents: string) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "text") elem.container.innerHTML = contents;
    }
  };

  setButtonLabel = (key: number, label: string) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "button") elem.buttonElem.innerHTML = label;
    }
  };

  setSliderValue = (key: number, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setValue(value);
    }
  };

  setSliderMin = (key: number, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setMin(value);
    }
  };

  setSliderMax = (key: number, value: number) => {
    if (this.uiElements.has(key)) {
      const elem = this.uiElements.get(key);
      if (elem.type === "slider") elem.setMax(value);
    }
  };

  setPlotData = (
    key: number,
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
      this.view.remove(k);
    });
    this.disposeHandlers.forEach((v, k) => v());
    this.objects.clear();
    this.keys.clear();

    this.view.render();
  }

  stop() {
    this.running = false;
  }
}

export default DARTView;
