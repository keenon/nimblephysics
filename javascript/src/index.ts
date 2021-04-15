import NimbleView from "./NimbleView";
import NimbleRemote from "./NimbleRemote";

const container = document.createElement("div");
container.style.height = "100vh";
container.style.width = "100vw";
container.style.margin = "0px";
document.body.style.margin = "0px";
document.body.style.padding = "0px";
document.body.appendChild(container);
const view = new NimbleView(container);
const remote = new NimbleRemote("ws://localhost:8070", view);
