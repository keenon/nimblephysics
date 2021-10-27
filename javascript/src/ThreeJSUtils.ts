import * as THREE from "three";

export function bodyToGroup(
  body: BodyNode,
  scaleFactor: number
): BodyGroupAndColor {
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
      shape.size[0] * scaleFactor,
      shape.size[1] * scaleFactor,
      shape.size[2] * scaleFactor
    );
    const mesh = new THREE.Mesh(geometry, material);
    if (body.name.toLowerCase().includes("floor")) {
      mesh.receiveShadow = true;
    } else {
      mesh.castShadow = true;
    }
    bodyGroup.add(mesh);
    mesh.position.x = shape.pos[0] * scaleFactor;
    mesh.position.y = shape.pos[1] * scaleFactor;
    mesh.position.z = shape.pos[2] * scaleFactor;
    mesh.rotation.x = shape.angle[0];
    mesh.rotation.y = shape.angle[1];
    mesh.rotation.z = shape.angle[2];
  });
  bodyGroup.position.x = body.pos[0] * scaleFactor;
  bodyGroup.position.y = body.pos[1] * scaleFactor;
  bodyGroup.position.z = body.pos[2] * scaleFactor;
  bodyGroup.rotation.x = body.angle[0];
  bodyGroup.rotation.y = body.angle[1];
  bodyGroup.rotation.z = body.angle[2];

  return {
    group: bodyGroup,
    color: color,
  };
}
