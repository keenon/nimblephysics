import NimbleView from "./NimbleView";
// import logoSvg from "!!raw-loader!./nimblelogo.svg";
import { CommandRecording, Command } from "./types";
import * as THREE from "three";

class NimbleStandalone {
  view: NimbleView | null;
  recording: CommandRecording;
  playing: boolean;
  startedPlaying: number;
  msPerFrame: number;
  startFrame: number;
  lastFrame: number;

  viewContainer: HTMLDivElement;

  constructor(container: HTMLElement) {
    this.viewContainer = document.createElement("div");
    this.viewContainer.className = "NimbleStandalone-container";
    container.appendChild(this.viewContainer);
    this.view = new NimbleView(this.viewContainer, true);
    this.view.addDragListener((key: string, pos: number[]) => {
      if (this.view != null) {
        this.view.setObjectPos(key, pos);
      }
    });
    this.view.scene.background = new THREE.Color(0xffffff);

    this.recording = [];
    this.playing = false;
    this.startedPlaying = new Date().getTime();
    this.lastFrame = -1;
    this.msPerFrame = 20;
    this.startFrame = 0;
  }

  /**
   * This cleans up and kills the standalone player.
   */
  dispose = () => {
    if (this.view != null) {
      this.view.dispose();
      this.viewContainer.remove();
    }
    this.playing = false;
    this.view = null;
  };

  /**
   * This replaces the set of recorded commands we're replaying
   *
   * @param recording The JSON object representing a recording of timestep(s) command(s)
   */
  setRecording = (recording: CommandRecording) => {
    if (recording != this.recording) {
      this.recording = recording;
      this.view.view.onWindowResize();

      // Collect the scene from command 0
      let createCommands: Map<string, Command> = new Map();
      for (let i = 0; i < this.recording[0].length; i++) {
        let command: Command = this.recording[0][i];
        if (command.type === 'create_mesh' || command.type === 'create_box' || command.type === 'create_capsule' || command.type === 'create_sphere') {
          // Create a deep copy of each command
          createCommands.set(command.key, command);
        }
      }

      let renderTimestep = (timestep: number, alpha: number) => {
        let thisTimestepCreateCommands: Map<string, Command> = new Map();
        this.recording[timestep].forEach((command: Command) => {
          // Ensure we have the create commands translated to this timestep
          if (command.type === 'set_object_pos' || command.type === 'set_object_rotation' || command.type === 'set_object_scale' || command.type === 'set_object_color') {
            if (!thisTimestepCreateCommands.has(command.key) && createCommands.has(command.key)) {
              thisTimestepCreateCommands.set(command.key, JSON.parse(JSON.stringify(createCommands.get(command.key))));
            }
          }

          let createCommand: Command = thisTimestepCreateCommands.get(command.key);
          if (command.type === 'set_object_pos') {
            (createCommand as any).pos = command.pos;
          }
          if (command.type === 'set_object_rotation') {
            (createCommand as any).euler = command.euler;
          }
          if (command.type === 'set_object_color') {
            (createCommand as any).color = command.color;
          }
        });

        // Actually run all the commands
        thisTimestepCreateCommands.forEach((command: Command) => {
          command.key = 'T' + timestep + '_' + command.key;
          if ((command as any).color) {
            if ((command as any).color.length == 4) {
              (command as any).color[3] *= alpha;
            }
            else if ((command as any).color.length == 3) {
              (command as any).color.push(alpha);
            }
          }
          this.view.handleCommand(command);
        });
      };

      let numSteps = 5;
      let stepSize = Math.floor((recording.length - 5) / numSteps);
      for (let i = 0; i < numSteps; i++) {
        let alpha = (i + 1) * (1.0 / numSteps);
        console.log(alpha);
        renderTimestep(i * stepSize + 1, alpha);
      }

      this.view.render();
    }
  };
}

export default NimbleStandalone;
