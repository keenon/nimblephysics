import * as THREE from "three";

const SCALE_FACTOR = 100;

class MeshTrajectory {
  mesh: THREE.Group;
  path: THREE.Line;
  pos_xs: number[];
  pos_ys: number[];
  pos_zs: number[];
  rot_xs: number[];
  rot_ys: number[];
  rot_zs: number[];

  constructor(
    mesh: THREE.Group,
    pos_xs: number[],
    pos_ys: number[],
    pos_zs: number[],
    rot_xs: number[],
    rot_ys: number[],
    rot_zs: number[],
    color: THREE.Color
  ) {
    this.mesh = mesh;
    this.pos_xs = pos_xs;
    this.pos_ys = pos_ys;
    this.pos_zs = pos_zs;
    this.rot_xs = rot_xs;
    this.rot_ys = rot_ys;
    this.rot_zs = rot_zs;

    const pathMaterial = new THREE.LineBasicMaterial({
      color: color,
      linewidth: 2,
    });
    const pathPoints = [];
    for (let i = 0; i < pos_xs.length; i++) {
      pathPoints.push(new THREE.Vector3(pos_xs[i], pos_ys[i], pos_zs[i]));
    }
    const pathGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    this.path = new THREE.Line(pathGeometry, pathMaterial);
  }

  addToScene(scene: THREE.Scene, showPaths: boolean) {
    scene.add(this.mesh);
    if (showPaths) scene.add(this.path);
  }

  removeFromScene(scene: THREE.Scene) {
    scene.remove(this.mesh);
    scene.remove(this.path);
  }

  getTimesteps() {
    return this.pos_xs.length;
  }

  setTimestep(t: number) {
    this.mesh.position.x = this.pos_xs[t];
    this.mesh.position.y = this.pos_ys[t];
    this.mesh.position.z = this.pos_zs[t];
    this.mesh.rotation.x = this.rot_xs[t];
    this.mesh.rotation.y = this.rot_ys[t];
    this.mesh.rotation.z = this.rot_zs[t];
  }
}

class WorldDisplay {
  scene: THREE.Scene;
  report: FullReport;
  objects: Map<string, MeshTrajectory>;
  // The iteration of gradient descent we're showing
  i: number;
  // The timestep we're showing
  t: number;

  timesteps: number;
  showPaths: boolean;

  constructor(scene: THREE.Scene, report: FullReport) {
    this.scene = scene;
    this.report = report;
    this.objects = new Map();
    this.t = 0;
    this.showPaths = true;

    this.timesteps = report.record[0].timesteps;
    this.setIteration(0);
  }

  setData = (report: FullReport) => {
    this.objects.forEach((v) => {
      v.removeFromScene(this.scene);
    });
    this.report = report;
    this.t = 0;
    this.timesteps = report.record[0].timesteps;

    // Force setIteration to run
    this.i = -1;
    this.setIteration(0);
  };

  setShowPaths = (showPaths: boolean) => {
    this.showPaths = showPaths;
    this.objects.forEach((v) => {
      v.removeFromScene(this.scene);
      v.addToScene(this.scene, this.showPaths);
    });
  };

  setMesh = (key: string, trajectory: MeshTrajectory) => {
    if (this.objects.has(key)) {
      this.objects.get(key).removeFromScene(this.scene);
    }
    this.objects.set(key, trajectory);
    trajectory.setTimestep(this.t);
    trajectory.addToScene(this.scene, this.showPaths);
  };

  getTimestep = () => {
    return this.t;
  };

  getTimesteps = () => {
    return this.timesteps;
  };

  setTimestep = (t: number) => {
    if (this.t === t) return;
    this.t = t;
    this.objects.forEach((v) => v.setTimestep(t));
  };

  getIteration = () => {
    return this.i;
  };

  getNumIterations = () => {
    return this.report.record.length;
  };

  getLoss = () => {
    return this.report.record[this.i].loss;
  };

  getConstraintViolation = () => {
    return this.report.record[this.i].constraintViolation;
  };

  setIteration = (i: number) => {
    if (i < 0 || i >= this.report.record.length) return;
    if (this.i === i) return;
    this.i = i;

    const record = this.report.record[i];

    for (const body of this.report.world) {
      const bodyGroup = new THREE.Group();
      // default to black
      let color = new THREE.Color(0, 0, 0);
      body.shapes.forEach((shape) => {
        let shapeColor = new THREE.Color(
          shape.color[0],
          shape.color[1],
          shape.color[2]
        );
        color = shapeColor;
        const material = new THREE.MeshLambertMaterial({
          color: shapeColor,
        });
        const geometry = new THREE.BoxBufferGeometry(
          shape.size[0] * SCALE_FACTOR,
          shape.size[1] * SCALE_FACTOR,
          shape.size[2] * SCALE_FACTOR
        );
        const mesh = new THREE.Mesh(geometry, material);
        if (body.name.toLowerCase().includes("floor")) {
          mesh.receiveShadow = true;
        } else {
          mesh.castShadow = true;
        }
        bodyGroup.add(mesh);
        mesh.position.x = shape.pos[0] * SCALE_FACTOR;
        mesh.position.y = shape.pos[1] * SCALE_FACTOR;
        mesh.position.z = shape.pos[2] * SCALE_FACTOR;
        mesh.rotation.x = shape.angle[0];
        mesh.rotation.y = shape.angle[1];
        mesh.rotation.z = shape.angle[2];
      });
      bodyGroup.position.x = body.pos[0] * SCALE_FACTOR;
      bodyGroup.position.y = body.pos[1] * SCALE_FACTOR;
      bodyGroup.position.z = body.pos[2] * SCALE_FACTOR;
      bodyGroup.rotation.x = body.angle[0];
      bodyGroup.rotation.y = body.angle[1];
      bodyGroup.rotation.z = body.angle[2];

      const trajectory = record.trajectory[body.name];

      const bodyMeshTrajectory = new MeshTrajectory(
        bodyGroup,
        trajectory.pos_x.map((x) => x * SCALE_FACTOR),
        trajectory.pos_y.map((y) => y * SCALE_FACTOR),
        trajectory.pos_z.map((z) => z * SCALE_FACTOR),
        trajectory.rot_x,
        trajectory.rot_y,
        trajectory.rot_z,
        color
      );
      this.setMesh(body.name, bodyMeshTrajectory);
    }
  };
}

export default WorldDisplay;
