import * as THREE from "three";
import { Mesh } from "three";
import { bodyToGroup } from "./ThreeJSUtils";

const SCALE_FACTOR = 100;

class RealtimeWorldDisplay {
  scene: THREE.Scene;
  world: BodyNode[];
  objects: Map<string, THREE.Group>;
  colors: Map<string, THREE.Color>;
  lines: Map<string, THREE.Line>;
  timestep: number;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.objects = new Map();
    this.colors = new Map();
    this.lines = new Map();
    this.timestep = -1;
  }

  initWorld = (world: BodyNode[]) => {
    this.world = world;
    this.objects.forEach((v) => {
      this.scene.remove(v);
    });
    this.objects.clear();

    for (let i = 0; i < world.length; i++) {
      const body: BodyNode = world[i];
      const bodyGroupAndColor = bodyToGroup(body, SCALE_FACTOR);
      this.objects.set(body.name, bodyGroupAndColor.group);
      this.scene.add(bodyGroupAndColor.group);
      this.colors.set(body.name, bodyGroupAndColor.color);
    }

    console.log("Initialized!");
  };

  setPositions = (timestep: number, positons: SetPositions) => {
    if (timestep < this.timestep) {
      // Reject out of order messages
      return;
    }
    this.timestep = timestep;
    for (let key in positons) {
      const posData = positons[key];
      const group = this.objects.get(key);
      group.position.x = posData.pos[0] * SCALE_FACTOR;
      group.position.y = posData.pos[1] * SCALE_FACTOR;
      group.position.z = posData.pos[2] * SCALE_FACTOR;
      group.rotation.x = posData.angle[0];
      group.rotation.y = posData.angle[1];
      group.rotation.z = posData.angle[2];
    }
  };

  setColors = (colors: SetColors) => {
    for (let key in colors) {
      const colorData = colors[key];
      const group = this.objects.get(key);
      for (let i = 0; i < group.children.length; i++) {
        const color: number[] = colorData[i];
        (group.children[i] as any).material.color.r = color[0];
        (group.children[i] as any).material.color.g = color[1];
        (group.children[i] as any).material.color.b = color[2];
      }
    }
  };

  displayMPCPlan = (trajectory: WorldTrajectory) => {
    for (let key in trajectory) {
      if (this.lines.has(key)) {
        this.scene.remove(this.lines.get(key));
      }

      const path: Trajectory = trajectory[key];

      const pathMaterial = new THREE.LineBasicMaterial({
        color: this.colors.get(key),
        linewidth: 2,
      });
      const pathPoints = [];
      for (let i = 0; i < path.pos_x.length; i++) {
        pathPoints.push(
          new THREE.Vector3(
            path.pos_x[i] * SCALE_FACTOR,
            path.pos_y[i] * SCALE_FACTOR,
            path.pos_z[i] * SCALE_FACTOR
          )
        );
      }
      const pathGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
      const pathLine = new THREE.Line(pathGeometry, pathMaterial);

      this.scene.add(pathLine);
      this.lines.set(key, pathLine);
    }
  };
}

export default RealtimeWorldDisplay;
