import NimbleScreenshot from './NimbleScreenshot';
import previewJson from './data/preview.json';

const container = document.createElement("div");
container.style.width = "100vw";
container.style.height = "100vh";
document.body.appendChild(container);
document.body.style.margin = "0";

const screenshot: NimbleScreenshot = new NimbleScreenshot(container);
screenshot.setRecording(previewJson as any);