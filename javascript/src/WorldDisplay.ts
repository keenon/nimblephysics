import * as THREE from "three";

class MeshTrajectory {
  mesh: THREE.Mesh;
  path: THREE.Line;
  pos_xs: number[];
  pos_ys: number[];
  pos_zs: number[];
  rot_xs: number[];
  rot_ys: number[];
  rot_zs: number[];

  constructor(
    mesh: THREE.Mesh,
    pos_xs: number[],
    pos_ys: number[],
    pos_zs: number[],
    rot_xs: number[],
    rot_ys: number[],
    rot_zs: number[]
  ) {
    this.mesh = mesh;
    this.pos_xs = pos_xs;
    this.pos_ys = pos_ys;
    this.pos_zs = pos_zs;
    this.rot_xs = rot_xs;
    this.rot_ys = rot_ys;
    this.rot_zs = rot_zs;

    const color: THREE.Color = (this.mesh.material as any).color;
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
  objects: Map<string, MeshTrajectory>;
  t: number;
  timesteps: number;
  showPaths: boolean;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.objects = new Map();
    this.t = 0;
    this.timesteps = 1;
    this.showPaths = true;
  }

  randomCubes = () => {
    var geometry = new THREE.BoxBufferGeometry(10, 10, 10);

    const interpolate = (start: number, end: number) => {
      if (this.timesteps == 1) return [(start + end) / 2];

      let arr: number[] = [];
      let diff: number = (end - start) / (this.timesteps - 1);
      let cursor: number = start;
      for (let i = 0; i < this.timesteps; i++) {
        arr.push(cursor);
        cursor += diff;
      }
      return arr;
    };

    for (let i = 0; i < 10; i++) {
      var material = new THREE.MeshLambertMaterial({
        color: Math.random() * 0xffffff,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const pos_xs = interpolate(
        Math.random() * 400 - 200,
        Math.random() * 400 - 200
      );
      const pos_ys = interpolate(
        Math.random() * 400 - 200,
        Math.random() * 400 - 200
      );
      const pos_zs = interpolate(
        Math.random() * 400 - 200,
        Math.random() * 400 - 200
      );
      const rot_xs = interpolate(Math.random(), Math.random());
      const rot_ys = interpolate(Math.random(), Math.random());
      const rot_zs = interpolate(Math.random(), Math.random());

      mesh.scale.setScalar(Math.random() * 10 + 2);

      const trajectory = new MeshTrajectory(
        mesh,
        pos_xs,
        pos_ys,
        pos_zs,
        rot_xs,
        rot_ys,
        rot_zs
      );

      this.setMesh(i.toString(), trajectory);
    }
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
    this.t = t;
    this.objects.forEach((v) => v.setTimestep(t));
  };

  setTimesteps = (timesteps: number) => {
    this.timesteps = timesteps;
  };
}

export default WorldDisplay;
