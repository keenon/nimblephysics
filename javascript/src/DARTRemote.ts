import DARTView from "./DARTView";

type CreateBoxCommand = {
  type: "create_box";
  key: string;
  size: number[];
  pos: number[];
  euler: number[];
  color: number[];
};

type CreateSphereCommand = {
  type: "create_sphere";
  key: string;
  radius: number;
  pos: number[];
  color: number[];
};

type CreateLineCommand = {
  type: "create_line";
  key: string;
  points: number[][];
  color: number[];
};

type SetObjectPosCommand = {
  type: "set_object_pos";
  key: string;
  pos: number[];
};

type SetObjectRotationCommand = {
  type: "set_object_rotation";
  key: string;
  euler: number[];
};

type SetObjectColorCommand = {
  type: "set_object_color";
  key: string;
  color: number[];
};

type EnableMouseInteractionCommand = {
  type: "enable_mouse";
  key: string;
};

type DisableMouseInteractionCommand = {
  type: "disable_mouse";
  key: string;
};

type CreateTextCommand = {
  type: "create_text";
  key: string;
  from_top_left: number[];
  size: number[];
  contents: string;
};

type CreateButtonCommand = {
  type: "create_button";
  key: string;
  from_top_left: number[];
  size: number[];
  label: string;
};

type CreateSliderCommand = {
  type: "create_slider";
  key: string;
  from_top_left: number[];
  size: number[];
  min: number;
  max: number;
  value: number;
  only_ints: boolean;
  horizontal: boolean;
};

type CreatePlotCommand = {
  type: "create_plot";
  key: string;
  from_top_left: number[];
  size: number[];
  min_x: number;
  max_x: number;
  xs: number[];
  min_y: number;
  max_y: number;
  ys: number[];
  plot_type: "line" | "scatter";
};

type SetUIElementPositionCommmand = {
  type: "set_ui_elem_pos";
  key: string;
  from_top_left: number[];
};

type SetUIElementSizeCommmand = {
  type: "set_ui_elem_size";
  key: string;
  size: number[];
};

type DeleteUIElementCommmand = {
  type: "delete_ui_elem";
  key: string;
};

type SetTextContents = {
  type: "set_text_contents";
  key: string;
  contents: string;
};

type SetButtonLabel = {
  type: "set_button_label";
  key: string;
  label: string;
};

type SetSliderValue = {
  type: "set_slider_value";
  key: string;
  value: number;
};

type SetSliderMin = {
  type: "set_slider_min";
  key: string;
  min: number;
};

type SetSliderMax = {
  type: "set_slider_max";
  key: string;
  max: number;
};

type SetPlotData = {
  type: "set_plot_data";
  key: string;
  min_x: number;
  max_x: number;
  xs: number[];
  min_y: number;
  max_y: number;
  ys: number[];
};

type Command =
  | CreateBoxCommand
  | CreateSphereCommand
  | CreateLineCommand
  | SetObjectPosCommand
  | SetObjectRotationCommand
  | SetObjectColorCommand
  | EnableMouseInteractionCommand
  | DisableMouseInteractionCommand
  | CreateTextCommand
  | CreateButtonCommand
  | CreateSliderCommand
  | CreatePlotCommand
  | SetUIElementPositionCommmand
  | SetUIElementSizeCommmand
  | DeleteUIElementCommmand
  | SetTextContents
  | SetButtonLabel
  | SetSliderValue
  | SetSliderMin
  | SetSliderMax
  | SetPlotData;

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

    this.view.addDragListener((key: string, pos: number[]) => {
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
  handleCommand = (command: Command) => {
    if (command.type === "create_box") {
      this.view.createBox(
        command.key,
        command.size,
        command.pos,
        command.euler,
        command.color
      );
    } else if (command.type === "create_sphere") {
      this.view.createSphere(
        command.key,
        command.radius,
        command.pos,
        command.color
      );
    } else if (command.type === "create_line") {
      this.view.createLine(command.key, command.points, command.color);
    } else if (command.type === "set_object_pos") {
      this.view.setObjectPos(command.key, command.pos);
    } else if (command.type === "set_object_rotation") {
      this.view.setObjectRotation(command.key, command.euler);
    } else if (command.type === "set_object_color") {
      this.view.setObjectColor(command.key, command.color);
    } else if (command.type === "enable_mouse") {
      this.view.enableMouseInteraction(command.key);
    } else if (command.type === "disable_mouse") {
      this.view.disableMouseInteraction(command.key);
    } else if (command.type === "create_text") {
      this.view.createText(
        command.key,
        command.from_top_left,
        command.size,
        command.contents
      );
    } else if (command.type === "create_button") {
      this.view.createButton(
        command.key,
        command.from_top_left,
        command.size,
        command.label,
        () => {
          const message = JSON.stringify({
            type: "button_click",
            key: command.key,
          });
          if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
            this.socket.send(message);
          }
        }
      );
    } else if (command.type === "create_slider") {
      this.view.createSlider(
        command.key,
        command.from_top_left,
        command.size,
        command.min,
        command.max,
        command.value,
        command.only_ints,
        command.horizontal,
        (new_value: number) => {
          const message = JSON.stringify({
            type: "slider_set_value",
            key: command.key,
            value: new_value,
          });
          if (this.socket != null && this.socket.readyState == WebSocket.OPEN) {
            this.socket.send(message);
          }
        }
      );
    } else if (command.type === "create_plot") {
      this.view.createPlot(
        command.key,
        command.from_top_left,
        command.size,
        command.min_x,
        command.max_x,
        command.xs,
        command.min_y,
        command.max_y,
        command.ys,
        command.plot_type
      );
    } else if (command.type === "set_ui_elem_pos") {
      this.view.setUIElementPosition(command.key, command.from_top_left);
    } else if (command.type === "set_ui_elem_size") {
      this.view.setUIElementSize(command.key, command.size);
    } else if (command.type === "delete_ui_elem") {
      this.view.deleteUIElement(command.key);
    } else if (command.type === "set_text_contents") {
      this.view.setTextContents(command.key, command.contents);
    } else if (command.type === "set_button_label") {
      this.view.setButtonLabel(command.key, command.label);
    } else if (command.type === "set_slider_value") {
      this.view.setSliderValue(command.key, command.value);
    } else if (command.type === "set_slider_min") {
      this.view.setSliderMin(command.key, command.min);
    } else if (command.type === "set_slider_max") {
      this.view.setSliderMax(command.key, command.max);
    } else if (command.type === "set_plot_data") {
      this.view.setPlotData(
        command.key,
        command.min_x,
        command.max_x,
        command.xs,
        command.min_y,
        command.max_y,
        command.ys
      );
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
      try {
        const data: Command[] = JSON.parse(event.data);
        data.forEach(this.handleCommand);
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
      setTimeout(this.trySocket, 1000);
    });
  };
}

export default DARTRemote;
