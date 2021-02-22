import { ContextReplacementPlugin } from "webpack";
import "./style.scss";

const height = 400;
const width = 600;

// Problem statement
let A11 = 0.348223;
let A12 = 0.223228;
let A21 = 0.223228;
let A22 = 0.348223;
let b1 = 0.035896;
let b2 = -0.035895;

// Solution values
let x1 = 0.0;
let x2 = 0.0;
let w1 = 0.0;
let w2 = 0.0;

// HTML structure
const A11_elem = document.getElementById("A11") as HTMLInputElement;
A11_elem.valueAsNumber = A11;
const A12_elem = document.getElementById("A12") as HTMLInputElement;
A12_elem.valueAsNumber = A12;
const A21_elem = document.getElementById("A21") as HTMLInputElement;
A21_elem.valueAsNumber = A21;
const A22_elem = document.getElementById("A22") as HTMLInputElement;
A22_elem.valueAsNumber = A22;
const x1_elem = document.getElementById("x1") as HTMLTableDataCellElement;
x1_elem.innerHTML = x1.toString();
const x2_elem = document.getElementById("x2") as HTMLTableDataCellElement;
x2_elem.innerHTML = x2.toString();
const b1_elem = document.getElementById("b1") as HTMLInputElement;
b1_elem.valueAsNumber = b1;
const b2_elem = document.getElementById("b2") as HTMLInputElement;
b2_elem.valueAsNumber = b2;
const w1_elem = document.getElementById("w1") as HTMLTableDataCellElement;
w1_elem.innerHTML = w1.toString();
const w2_elem = document.getElementById("w2") as HTMLTableDataCellElement;
w2_elem.innerHTML = w2.toString();

// Colors and legend
const xGeqZero = "green";
const wGeqZero = "blue";
const comp0 = "orange";
const comp1 = "purple";
document.getElementById("w-geq").style.backgroundColor = wGeqZero;
document.getElementById("x-geq").style.backgroundColor = xGeqZero;
document.getElementById("comp-0").style.backgroundColor = comp0;
document.getElementById("comp-1").style.backgroundColor = comp1;

A11_elem.style.border = "1px solid " + comp1;
A21_elem.style.border = "1px solid " + comp1;
x1_elem.style.border = "1px solid " + comp1;

A12_elem.style.border = "1px solid " + comp0;
A22_elem.style.border = "1px solid " + comp0;
x2_elem.style.border = "1px solid " + comp0;

// View parameters
let centerX = 0.0;
let centerY = 0.0;
let scaleX = 1.0;
let scaleY = 1.0;

type Coordinate = {
  x: number;
  y: number;
};

function canvasPxToNumbers(pt: Coordinate): Coordinate {
  return {
    x: (pt.x / width - 0.5) * scaleX + centerX,
    y: (pt.y / height - 0.5) * -1 * scaleY + centerY,
  };
}

function numbersToCanvasPx(pt: Coordinate): Coordinate {
  return {
    x: ((pt.x - centerX) / scaleX + 0.5) * width,
    y: (((pt.y - centerY) * -1) / scaleY + 0.5) * height,
  };
}

const canvas = document.createElement("canvas");
canvas.style.height = height + "px";
canvas.style.width = width + "px";
canvas.height = height;
canvas.width = width;
canvas.style.margin = "0px";
canvas.style.border = "1px solid black";

document.body.style.margin = "0px";
document.body.style.padding = "0px";
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

function isValidPoint(x: Coordinate): boolean {
  const _w1 = A11 * x.x + A12 * x.y + b1;
  const _w2 = A21 * x.x + A22 * x.y + b2;
  const EPS = 1e-7;
  const wGeq = _w1 >= -EPS && _w2 >= -EPS;
  const xGeq = x.x >= -EPS && x.y >= -EPS;
  const comp1 = Math.abs(_w1 * x.x) < EPS;
  const comp2 = Math.abs(_w2 * x.y) < EPS;
  console.log(
    JSON.stringify({
      wGeq,
      xGeq,
      comp1,
      comp2,
      x,
      w: {
        x: _w1,
        y: _w2,
      },
    })
  );
  const valid: boolean = wGeq && xGeq && comp1 && comp2;
  if (valid) {
    x1 = x.x;
    x1_elem.innerHTML = x1.toString();
    x2 = x.y;
    x2_elem.innerHTML = x2.toString();
    w1 = _w1;
    w1_elem.innerHTML = w1.toString();
    w2 = _w2;
    w2_elem.innerHTML = w2.toString();
  }
  return valid;
}

function debugPoint(x: Coordinate) {
  const w = {
    x: A11 * x.x + A12 * x.y + b1,
    y: A21 * x.x + A22 * x.y + b2,
  };
  ctx.beginPath();
  ctx.strokeStyle = isValidPoint(x) ? "green" : "red";
  const solutionRadius = 5;
  const wPx = numbersToCanvasPx(w);
  ctx.arc(wPx.x, wPx.y, solutionRadius, 0, 2 * Math.PI);
  ctx.stroke();
}

function updateLCP() {
  A11 = A11_elem.valueAsNumber;
  A12 = A12_elem.valueAsNumber;
  A21 = A21_elem.valueAsNumber;
  A22 = A22_elem.valueAsNumber;
  b1 = b1_elem.valueAsNumber;
  b2 = b2_elem.valueAsNumber;

  ctx.clearRect(0, 0, width, height);
  ctx.globalAlpha = 0.5;

  // Draw the "feasible region" for w >= 0
  ctx.beginPath();
  const originPx: Coordinate = numbersToCanvasPx({
    x: 0,
    y: 0,
  });
  ctx.moveTo(originPx.x, originPx.y);
  ctx.fillStyle = wGeqZero;
  ctx.fillRect(originPx.x, originPx.y - 1000, 1000, 1000);

  // Get the b point
  const bPt: Coordinate = {
    x: b1,
    y: b2,
  };
  const bPx: Coordinate = numbersToCanvasPx(bPt);

  // Draw the "feasible region" for x >= 0
  const colA1px: Coordinate = numbersToCanvasPx({
    x: b1 + A11 * 100,
    y: b2 + A21 * 100,
  });
  const colA2px: Coordinate = numbersToCanvasPx({
    x: b1 + A12 * 100,
    y: b2 + A22 * 100,
  });
  ctx.beginPath();
  ctx.moveTo(bPx.x, bPx.y);
  ctx.lineTo(colA1px.x, colA1px.y);
  ctx.lineTo(colA2px.x, colA2px.y);
  ctx.lineTo(bPx.x, bPx.y);
  ctx.fillStyle = xGeqZero;
  ctx.fill();

  const xLeft: Coordinate = numbersToCanvasPx({
    x: -scaleX / 2,
    y: 0,
  });
  const xRight: Coordinate = numbersToCanvasPx({
    x: scaleX / 2,
    y: 0,
  });
  const yTop: Coordinate = numbersToCanvasPx({
    x: 0,
    y: scaleY / 2,
  });
  const yBottom: Coordinate = numbersToCanvasPx({
    x: 0,
    y: -scaleY / 2,
  });

  // Draw the first complimentarity constraint
  // Any spots with X[0]==0
  ctx.beginPath();
  ctx.strokeStyle = comp0;
  ctx.moveTo(bPx.x, bPx.y);
  ctx.lineTo(colA2px.x, colA2px.y);
  ctx.lineWidth = 2;
  ctx.stroke();
  // Any spots with W[0]==0
  ctx.beginPath();
  ctx.strokeStyle = comp0;
  ctx.moveTo(yTop.x, yTop.y);
  ctx.lineTo(yBottom.x, yBottom.y);
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw the second complimentarity constraint
  // Any spots with X[1]==0
  ctx.beginPath();
  ctx.strokeStyle = comp1;
  ctx.moveTo(bPx.x, bPx.y);
  ctx.lineTo(colA1px.x, colA1px.y);
  ctx.lineWidth = 2;
  ctx.stroke();
  // Any spots with W[1]==0
  ctx.beginPath();
  ctx.strokeStyle = comp1;
  ctx.moveTo(xLeft.x, xLeft.y);
  ctx.lineTo(xRight.x, xRight.y);
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw the B point
  ctx.beginPath();
  const bRadius = 3;
  ctx.arc(bPx.x, bPx.y, bRadius, 0, 2 * Math.PI);
  ctx.fillStyle = "black";
  ctx.fill();

  const solutionRadius = 5;
  // Compute the intersection of colA1 with y=0
  // b1 + A11*p = 0
  // p = -b2 / A21
  // x = b1 + A11*p
  let p1 = -b1 / A11;
  if (p1 > 0) {
    debugPoint({ x: p1, y: 0 });
    /*
    const col1Intersect = {
      x: b1 - A11 * (b2 / A21),
      y: 0,
    };
    const col1IntersectPx = numbersToCanvasPx(col1Intersect);
    ctx.beginPath();
    ctx.strokeStyle = isValidPoint({
      x: p1,
      y: 0,
    })
      ? "green"
      : "red";
    ctx.arc(
      col1IntersectPx.x,
      col1IntersectPx.y,
      solutionRadius,
      0,
      2 * Math.PI
    );
    ctx.stroke();
    */
  }
  // Compute the intersection of colA2 with x=0
  // b1 + A12*p = 0
  // p = -b1 / A12
  // x = b2 + A22*p
  let p2 = -b2 / A22;
  if (p2 > 0) {
    debugPoint({ x: 0, y: p2 });
    /*
    const col2Intersect = {
      x: 0,
      y: b2 - A22 * (b1 / A12),
    };
    const col2IntersectPx = numbersToCanvasPx(col2Intersect);
    ctx.beginPath();
    ctx.strokeStyle = isValidPoint({
      x: 0,
      y: p2,
    })
      ? "green"
      : "red";
    ctx.arc(
      col2IntersectPx.x,
      col2IntersectPx.y,
      solutionRadius,
      0,
      2 * Math.PI
    );
    ctx.stroke();
    */
  }
  // Draw at b
  debugPoint({ x: 0, y: 0 });
  /*
  ctx.beginPath();
  ctx.strokeStyle = isValidPoint({ x: 0, y: 0 }) ? "green" : "red";
  ctx.arc(bPx.x, bPx.y, solutionRadius, 0, 2 * Math.PI);
  ctx.stroke();
  */
  // Draw at origin
  ctx.beginPath();
  ctx.strokeStyle = "black";
  ctx.arc(originPx.x, originPx.y, solutionRadius, 0, 2 * Math.PI);
  ctx.stroke();
}
updateLCP();

canvas.addEventListener("mousedown", (ev: MouseEvent) => {
  var rect = canvas.getBoundingClientRect();
  const canvasPx: Coordinate = {
    x: ev.clientX - rect.left,
    y: ev.clientY - rect.top,
  };
  const numbers = canvasPxToNumbers(canvasPx);
  b1_elem.valueAsNumber = numbers.x;
  b2_elem.valueAsNumber = numbers.y;
  updateLCP();
});
