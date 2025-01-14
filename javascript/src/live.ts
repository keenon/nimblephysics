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
// Use the current host and protocol for the WebSocket connection
const wsProtocol = location.protocol === "https:" ? "wss://" : "ws://";
const wsHost = location.hostname; // Current hostname (e.g., domain or IP)
const wsPort = 8070; // Fixed WebSocket port
const wsUrl = `${wsProtocol}${wsHost}:${wsPort}`;
const remote = new NimbleRemote(wsUrl, view);
