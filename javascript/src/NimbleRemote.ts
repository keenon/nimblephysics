import NimbleView from "./NimbleView";
import { dart } from './proto/GUI';

class DARTRemote {
  url: string;
  view: NimbleView;
  socket: WebSocket | null;

  constructor(url: string, view: NimbleView) {
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

    this.view.addDragListener((key: number, pos: number[]) => {
      const message = JSON.stringify({
        type: "drag",
        key,
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
  handleCommand = (command: dart.proto.Command) => {
    // We manually handle any "interactive" commands here, since the NimbleView object by itself doesn't know what to do with those.
    if (command.button != null) {
      const from_top_left: number[] = [command.button.pos[0], command.button.pos[1]];
      const size: number[] = [command.button.pos[2], command.button.pos[3]];
      this.view.createButton(
        command.button.key,
        from_top_left,
        size,
        command.button.label,
        () => {
          const message = JSON.stringify({
            type: "button_click",
            key: command.button.key,
          });
          if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
            this.socket.send(message);
          }
        },
        command.button.layer
      );
    }
    else if (command.slider != null) {
      const from_top_left: number[] = [command.slider.pos[0], command.slider.pos[1]];
      const size: number[] = [command.slider.pos[2], command.slider.pos[3]];
      const min = command.slider.data[0];
      const max = command.slider.data[1];
      const value = command.slider.data[2];
      this.view.createSlider(
        command.slider.key,
        from_top_left,
        size,
        min,
        max,
        value,
        command.slider.only_ints,
        command.slider.horizontal,
        (new_value: number) => {
          const message = JSON.stringify({
            type: "slider_set_value",
            key: command.slider.key,
            value: new_value,
          });
          if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
            this.socket.send(message);
          }
        },
        command.slider.layer
      );
    } else {
      // Otherwise, the command doesn't require any interactive callbacks, so NimbleView can handle it directly.
      this.view.handleCommand(command);
    }
  };

  /**
   * This attempts to connect a socket to the backend.
   */
  trySocket = () => {
    this.socket = new WebSocket(this.url);

    // Connection opened
    this.socket.addEventListener("open", (event) => {
      console.log("Socket connected!");
      // Clear the view on a reconnect, the socket will broadcast us new data
      this.view.setConnected(true);
      this.view.clear();
    });

    // Listen for messages
    this.socket.addEventListener("message", (event) => {
      try {
        const list: dart.proto.CommandList = dart.proto.CommandList.deserialize(event.data);
        list.command.forEach(this.handleCommand);
        this.view.render();
      } catch (e) {
        console.error(
          "Something went wrong on command:\n\n" + event.data + "\n\n",
          e
        );
      }
    });

    this.socket.addEventListener("close", () => {
      // do nothing
      console.log("Socket closed. Retrying in 1s");
      this.view.stop();
      this.view.setConnected(false);
      setTimeout(this.trySocket, 1000);
    });
  };
}

export default DARTRemote;
