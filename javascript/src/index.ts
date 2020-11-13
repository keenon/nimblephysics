import Window from "./DARTWindow";

// import worm_data from "./data/worm.txt";
// import cart_data from "./data/cartpole.txt";
// import realtime_data from "./data/realtime.txt";

const container = document.createElement("div");
container.style.height = "100vh";
container.style.width = "100vw";
container.style.margin = "0px";
document.body.style.margin = "0px";
document.body.style.padding = "0px";
document.body.appendChild(container);
const window = new Window(container);

window.connectLiveRemote("ws://localhost:8070");

// const worm = JSON.parse(worm_data);
// const cart = JSON.parse(cart_data);
// window.registerData("Jumpworm", worm);
// window.registerData("Cartpole", cart);
// const realtime = JSON.parse(realtime_data);
// window.registerData("Realtime", realtime);

// Publish this so that any downstream object can use it
(document as any).DARTWindow = Window;

/*
fetch("http://localhost:9080")
  .then((res) => res.json())
  .then((res) => {
    console.log(res);
  });
  */
