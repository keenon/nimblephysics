type Shape = {
  type: string;
  size: number[];
  color: number[];
  pos: number[];
  angle: number[];
};

type BodyNode = {
  name: string;
  shapes: Shape[];
  pos: number[];
  angle: number[];
};

type Trajectory = {
  pos_x: number[];
  pos_y: number[];
  pos_z: number[];
  rot_x: number[];
  rot_y: number[];
  rot_z: number[];
};

interface WorldTrajectory {
  [key: string]: Trajectory;
}

type OptRecord = {
  index: number;
  timesteps: number;
  loss: number;
  constraintViolation: number;
  trajectory: WorldTrajectory;
};

type FullReport = {
  world: BodyNode[];
  record: OptRecord[];
};

type ObjectPosition = {
  pos: number[];
  angle: number[];
};

interface SetPositions {
  [key: string]: ObjectPosition;
}

type RealtimeUpdate = {
  type: "init" | "update" | "new_plan";
  // for type: "init"
  world?: BodyNode[];
  // for type: "update"
  positions?: SetPositions;
  timestep?: number;
  // for type: "new_plan"
  plan?: WorldTrajectory;
};

type BodyGroupAndColor = {
  group: THREE.Group;
  color: THREE.Color;
};
