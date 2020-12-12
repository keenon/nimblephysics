import DARTView from "./DARTView";

type CreateBoxCommand = {
  type: "create_box";
  name: string;
  size: number[];
  pos: number[];
  euler: number[];
  color: number[];
};

type CreateSphereCommand = {
  type: "create_sphere";
  name: string;
  radius: number;
  pos: number[];
  color: number[];
};

type CreateLineCommand = {
  type: "create_line";
  name: string;
  points: number[][];
  color: number[];
};

type SetObjectPosCommand = {
  type: "set_object_pos";
  name: string;
  pos: number[];
};

type SetObjectRotationCommand = {
  type: "set_object_rotation";
  name: string;
  euler: number[];
};

type SetObjectColorCommand = {
  type: "set_object_color";
  name: string;
  color: number[];
};

type EnableMouseInteractionCommand = {
  type: "enable_mouse";
  name: string;
};

type DisableMouseInteractionCommand = {
  type: "disable_mouse";
  name: string;
};

type Command =
  | CreateBoxCommand
  | CreateSphereCommand
  | CreateLineCommand
  | SetObjectPosCommand
  | SetObjectRotationCommand
  | SetObjectColorCommand
  | EnableMouseInteractionCommand
  | DisableMouseInteractionCommand;

class DARTRemote {
  url: string;
  view: DARTView;
  socket: WebSocket | null;

  constructor(url: string, view: DARTView) {
    this.url = url;
    this.view = view;

    this.trySocket();

    window.addEventListener("keydown", (e: KeyboardEvent) => {
      const message = JSON.stringify({
        type: "keydown",
        key: e.key.toString(),
      });
      if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
        this.socket.send(message);
      }
    });

    window.addEventListener("keyup", (e: KeyboardEvent) => {
      const message = JSON.stringify({
        type: "keyup",
        key: e.key.toString(),
      });
      if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
        this.socket.send(message);
      }
    });

    this.view.addDragListener((name: string, pos: number[]) => {
      const message = JSON.stringify({
        type: "drag",
        name,
        pos,
      });
      if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
        this.socket.send(message);
      }
    });
  }

  /**
   * This reads and handles a command sent from the backend
   */
  handleCommand = (command: Command) => {
    console.log("Handling command: " + JSON.stringify(command, null, 2));
    if (command.type === "create_box") {
      this.view.createBox(
        command.name,
        command.size,
        command.pos,
        command.euler,
        command.color
      );
    } else if (command.type === "create_sphere") {
      this.view.createSphere(
        command.name,
        command.radius,
        command.pos,
        command.color
      );
    } else if (command.type === "create_line") {
      this.view.createLine(command.name, command.points, command.color);
    } else if (command.type === "set_object_pos") {
      this.view.setObjectPos(command.name, command.pos);
    } else if (command.type === "set_object_rotation") {
      this.view.setObjectRotation(command.name, command.euler);
    } else if (command.type === "set_object_color") {
      this.view.setObjectColor(command.name, command.color);
    } else if (command.type === "enable_mouse") {
      this.view.enableMouseInteraction(command.name);
    } else if (command.type === "disable_mouse") {
      this.view.disableMouseInteraction(command.name);
    }
  };

  /**
   * This attempts to connect a socket to the backend.
   */
  trySocket = () => {
    this.socket = new WebSocket(this.url);

    // Connection opened
    this.socket.addEventListener("open", (event) => {
      // Clear the view on a reconnect, the socket will broadcast us new data
      this.view.clear();
    });

    // Listen for messages
    this.socket.addEventListener("message", (event) => {
      const data: Command[] = JSON.parse(event.data);
      data.forEach(this.handleCommand);
      this.view.render();
    });

    this.socket.addEventListener("close", () => {
      // do nothing
      console.log("Socket closed. Retrying in 1s");
      this.view.stop();
      setTimeout(this.trySocket, 1000);
    });
  };
}

export default DARTRemote;
