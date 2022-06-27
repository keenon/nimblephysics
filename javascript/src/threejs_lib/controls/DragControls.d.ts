import { Camera, EventDispatcher, Object3D, Vector3 } from "three";

export class DragControls extends EventDispatcher {
  constructor(objects: Object3D[], camera: Camera, domElement?: HTMLElement);

  object: Camera;

  // API

  enabled: boolean;
  transformGroup: boolean;

  activate(): void;
  deactivate(): void;
  dispose(): void;
  getObjects(): Object3D[];

  add(object: Object3D);
  remove(object: Object3D);
  setDragHandler(handler: (obj: Object3D, pos: Vector3) => void);

  // EventDispatcher mixins
  addEventListener(type: string, listener: (event: any) => void): void;

  hasEventListener(type: string, listener: (event: any) => void): boolean;

  removeEventListener(type: string, listener: (event: any) => void): void;

  dispatchEvent(event: { type: string; target: any }): void;
}
