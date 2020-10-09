import Window from "./DARTWindow";

/*
import worm_data from "./data/worm.txt";
import cart_data from "./data/cartpole.txt";

const worm = JSON.parse(worm_data);
const cart = JSON.parse(cart_data);

const window = new Window(document.body);
window.registerData("Jumpworm", worm);
window.registerData("Cartpole", cart);
*/

// Publish this so that any downstream object can use it
(document as any).DARTWindow = Window;

/*
fetch("http://localhost:9080")
  .then((res) => res.json())
  .then((res) => {
    console.log(res);
  });
  */
