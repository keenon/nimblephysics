import NimbleStandalone from "./NimbleStandalone";

// Publish this so that any downstream object can use it
(document as any).NimbleStandalone = NimbleStandalone;

const container = document.createElement("div");
container.style.width = "90vw";
container.style.height = "90vh";
container.style.marginTop = "5vh";
container.style.marginLeft = "5vw";
container.style.border = "1px solid grey";
document.body.appendChild(container);
const standalone: NimbleStandalone = new NimbleStandalone(container);
// Much larger file:
// https://mocap-processed.s3.us-west-2.amazonaws.com/michael.json
standalone.loadRecording(
  "https://mocap-processed.s3.us-west-2.amazonaws.com/laiArnold.json"
);
