import NimbleView from "./NimbleView";
// import logoSvg from "!!raw-loader!./nimblelogo.svg";
import { CommandRecording } from "./types";
import protobuf from 'google-protobuf';
import {dart} from './proto/GUI';
import { createHash } from 'sha256-uint8array';

class NimbleStandalone {
  view: NimbleView | null;
  lastRecordingHash: string;
  recording: dart.proto.CommandList[];
  playing: boolean;
  startedPlaying: number;
  msPerFrame: number;
  startFrame: number;
  lastFrame: number;

  viewContainer: HTMLDivElement;
  progressBarContainer: HTMLDivElement;
  progressBarBackground: HTMLDivElement;
  progressBar: HTMLDivElement;
  progressScrub: HTMLDivElement;

  loadingContainerMounted: boolean;
  loadingContainer: HTMLDivElement;
  loadingProgressBarContainer: HTMLDivElement;
  loadingProgressBarBg: HTMLDivElement;

  constructor(container: HTMLElement) {
    this.viewContainer = document.createElement("div");
    this.viewContainer.className = "NimbleStandalone-container";
    container.appendChild(this.viewContainer);
    this.view = new NimbleView(this.viewContainer, true);

    const instructions = document.createElement("div");
    instructions.innerHTML = "Press [Space] to Play/Pause"
    instructions.className =
      "NimbleStandalone-progress-instructions";
    this.viewContainer.appendChild(instructions);

    this.progressBarContainer = document.createElement("div");
    this.progressBarContainer.className =
      "NimbleStandalone-progress-bar-container";
    this.viewContainer.appendChild(this.progressBarContainer);

    this.progressBarBackground = document.createElement("div");
    this.progressBarBackground.className = "NimbleStandalone-progress-bar-bg";
    this.progressBarContainer.appendChild(this.progressBarBackground);

    this.progressBar = document.createElement("div");
    this.progressBar.className = "NimbleStandalone-progress-bar";
    this.progressBarContainer.appendChild(this.progressBar);

    this.progressScrub = document.createElement("div");
    this.progressScrub.className = "NimbleStandalone-progress-bar-scrub";
    this.progressBarContainer.appendChild(this.progressScrub);

    const processMouseEvent = (e: MouseEvent) => {
      const rect = this.progressBarContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      let percentage = x / rect.width;
      if (percentage < 0) percentage = 0;
      if (percentage > 1) percentage = 1;
      if (this.playing) this.togglePlay();
      this.lastFrame = Math.round(this.recording.length * percentage);
      if (this.view != null) {
        this.recording[this.lastFrame].command.forEach(this.view.handleCommand);
        this.view.render();
      }
      this.setProgress(percentage);
    };

    this.progressBarContainer.addEventListener("mousedown", (e: MouseEvent) => {
      processMouseEvent(e);

      window.addEventListener("mousemove", processMouseEvent);
      const mouseUp = () => {
        window.removeEventListener("mousemove", processMouseEvent);
        window.removeEventListener("mousup", mouseUp);
      };
      window.addEventListener("mouseup", mouseUp);
    });

    window.addEventListener("keydown", this.keyboardListener);

    this.view.addDragListener((key: string, pos: number[]) => {
      if (this.view != null) {
        this.view.setObjectPos(key, pos);
      }
    });

    this.lastRecordingHash = "";
    this.recording = [];
    this.playing = false;
    this.startedPlaying = new Date().getTime();
    this.lastFrame = -1;
    this.msPerFrame = 20;
    this.startFrame = 0;

    this.loadingContainerMounted = false;
    this.loadingContainer = document.createElement("div");
    this.loadingContainer.className = "NimbleStandalone-loading-overlay";

    /*
    const loadingLogo = document.createElement("svg");
    loadingLogo.innerHTML = logoSvg;
    this.loadingContainer.appendChild(loadingLogo);
    */

    const loadingTitle = document.createElement("div");
    loadingTitle.className = "NimbleStandalone-loading-text";
    loadingTitle.innerHTML = "nimble<b>viewer</b> loading...";
    this.loadingContainer.appendChild(loadingTitle);

    const loadingProgressBarOuterContainer = document.createElement("div");
    loadingProgressBarOuterContainer.className =
      "NimbleStandalone-loading-bar-container";
    this.loadingContainer.appendChild(loadingProgressBarOuterContainer);

    const loadingContainerInnerBg = document.createElement("div");
    loadingContainerInnerBg.className =
      "NimbleStandalone-loading-bar-container-inner-bg";
    loadingProgressBarOuterContainer.appendChild(loadingContainerInnerBg);

    this.loadingProgressBarContainer = document.createElement("div");
    this.loadingProgressBarContainer.className =
      "NimbleStandalone-loading-bar-container-inner-bar-container";
    loadingContainerInnerBg.appendChild(this.loadingProgressBarContainer);
    this.loadingProgressBarBg = document.createElement("div");
    this.loadingProgressBarBg.className =
      "NimbleStandalone-loading-bar-container-inner-bar-container-bg";
    this.loadingProgressBarContainer.appendChild(this.loadingProgressBarBg);
  }

  /**
   * This is our keyboard listener, which we keep around until we clean up the player.
   */
  keyboardListener = (e: KeyboardEvent) => {
    if (e.key.toString() == " ") {
      this.togglePlay();
    }
  };

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
    window.removeEventListener("keydown", this.keyboardListener);
  };

  setProgress = (percentage: number) => {
    this.progressBar.style.width = (1.0 - percentage) * 100 + "%";
    this.progressScrub.style.left = percentage * 100 + "%";

    const zeroRGB = [255, 184, 0];
    const oneThirdRGB = [245, 71, 71];
    const twoThirdRGB = [207, 50, 158];
    const fullRGB = [141, 25, 233];

    function pickHex(color1: number[], color2: number[], weight: number) {
      var w2 = weight;
      var w1 = 1 - w2;
      return (
        "rgb(" +
        Math.round(color1[0] * w1 + color2[0] * w2) +
        "," +
        Math.round(color1[1] * w1 + color2[1] * w2) +
        "," +
        Math.round(color1[2] * w1 + color2[2] * w2) +
        ")"
      );
    }

    if (percentage < 0.33) {
      this.progressScrub.style.backgroundColor = pickHex(
        zeroRGB,
        oneThirdRGB,
        (percentage - 0.0) / 0.33
      );
    } else if (percentage < 0.66) {
      this.progressScrub.style.backgroundColor = pickHex(
        oneThirdRGB,
        twoThirdRGB,
        (percentage - 0.33) / 0.33
      );
    } else {
      this.progressScrub.style.backgroundColor = pickHex(
        twoThirdRGB,
        fullRGB,
        (percentage - 0.66) / 0.33
      );
    }
  };

  /**
   * The loading progress
   *
   * @param progress The progress from 0-1 in loading
   */
  setLoadingProgress = (progress: number) => {
    if (!this.loadingContainerMounted) {
      this.loadingContainer.remove();
      this.viewContainer.appendChild(this.loadingContainer);
      this.loadingContainerMounted = true;
    }
    this.loadingProgressBarContainer.style.width = "calc(" + progress * 100 + "% - 6px)";
    this.loadingProgressBarBg.style.width = (1.0 / progress) * 100 + "%";
  };

  /**
   * This hides the loading bar, which unmounts it from the DOM (if it was previously mounted).
   */
  hideLoadingBar = () => {
    if (this.loadingContainerMounted) {
      this.loadingContainer.remove();
      this.loadingContainerMounted = false;
    }
  }

  /**
   * This loads a recording to play back. It attempts to display a progress bar while loading the model.
   *
   * @param url The URL to load a recording from, in order to play back
   */
  loadRecording = (url: string) => {
    this.setLoadingProgress(0.0);

    let xhr = new XMLHttpRequest();
    xhr.open("GET", url);

    xhr.onprogress = (event) => {
      console.log(`Received ${event.loaded} of ${event.total}`);
      this.setLoadingProgress(event.loaded / event.total);
    };

    xhr.onload = () => {
      if (xhr.status != 200) {
        console.log(`Error ${xhr.status}: ${xhr.statusText}`);
      } else {
        let response = JSON.parse(xhr.response);
        this.setRecording(response);
        setTimeout(() => {
          this.hideLoadingBar();
        }, 100);
      }
    };

    xhr.send();
  };

  /**
   * This replaces the set of recorded commands we're replaying
   *
   * @param recording The JSON object representing a recording of timestep(s) command(s)
   */
  setRecording = (rawBytes: Uint8Array) => {
    const hash = createHash().update(rawBytes).digest("hex");

    if (hash !== this.lastRecordingHash) {
      this.lastRecordingHash = hash;

      this.recording = [];
      let cursor = 0;
      while (cursor < rawBytes.length) {
        // Read thet size byte
        const u32bytes = rawBytes.buffer.slice(cursor, cursor+4); // last four bytes as a new `ArrayBuffer`
        const size = new Uint32Array(u32bytes)[0];
        cursor += 4;
        try {
          let command: dart.proto.CommandList = dart.proto.CommandList.deserialize(rawBytes.buffer.slice(cursor, cursor + size));
          this.recording.push(command);
        }
        catch (e) {
          console.error(e);
        }
        cursor += size;
      }

      this.view.view.onWindowResize();
      if (!this.playing) {
        this.togglePlay();
      }
    }
  };

  /**
   * This turns playback on or off.
   */
  togglePlay = () => {
    this.playing = !this.playing;
    if (this.playing) {
      this.startFrame = this.lastFrame;
      this.startedPlaying = new Date().getTime();
      this.animationFrame();
    }
  };

  handleCommand = (command: dart.proto.Command) => {
    if (command.set_frames_per_second) {
      console.log("Frames per second: " + command.set_frames_per_second);
      this.msPerFrame = 1000 / command.set_frames_per_second.framesPerSecond;
    }
    else {
      this.view.handleCommand(command);
    }
  };

  /**
   * This gets called within requestAnimationFrame(...), and handles replaying any commands necessary to jump to the appropriate time in the animation.
   */
  animationFrame = () => {
    // Avoid race conditions if we were stopped, then this frame fired later
    if (!this.playing) return;

    const elapsed: number = new Date().getTime() - this.startedPlaying;
    let frameNumber =
      (this.startFrame + Math.round(elapsed / this.msPerFrame)) %
      this.recording.length;
    if (frameNumber != this.lastFrame) {
      if (frameNumber < this.lastFrame) {
        // Reset at the beginning
        this.lastFrame = -1;
      }
      this.setProgress(frameNumber / this.recording.length);
      if (this.view != null) {
        for (let i = this.lastFrame + 1; i <= frameNumber; i++) {
          this.recording[i].command.forEach(this.handleCommand);
        }
        this.view.render();
      }
      this.lastFrame = frameNumber;
    }

    if (this.playing) {
      // Don't use requestAnimationFrame(), because that causes contention with the mouse interaction, which makes the whole UI feel sluggish
      setTimeout(this.animationFrame, this.msPerFrame);
    }
  };
}

export default NimbleStandalone;
