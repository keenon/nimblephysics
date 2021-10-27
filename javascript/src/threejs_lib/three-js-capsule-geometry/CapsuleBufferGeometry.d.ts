import { BufferGeometry } from "three";

export class CapsuleBufferGeometry extends BufferGeometry {
  constructor(
    radiusTop?: number,
    radiusBottom?: number,
    height?: number,
    radialSegments?: number,
    heightSegments?: number,
    capsTopSegments?: number,
    capsBottomSegments?: number,
    thetaStart?: number,
    thetaLength?: number
  );
}
