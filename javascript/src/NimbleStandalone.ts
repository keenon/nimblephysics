import NimbleView from "./NimbleView";
// import logoSvg from "!!raw-loader!./nimblelogo.svg";
import { CommandRecording } from "./types";
import protobuf from 'google-protobuf';
import {dart} from './proto/GUI';
import { createHash } from 'sha256-uint8array';

class NimbleStandalone {
  view: NimbleView | null;
  lastRecordingHash: string;
  rawBytes: Uint8Array;
  rawFrameBytes: Uint8Array[];
  estimatedTotalFrames: number;

  playing: boolean;
  scrubbing: boolean;
  startedPlaying: number;
  originalMsPerFrame: number;
  playbackMultiple: number;
  msPerFrame: number;
  startFrame: number;
  renderedFirstFrame: boolean;
  lastFrame: number;
  scrubFrame: number;

  animationKey: number;

  viewContainer: HTMLDivElement;
  playPauseButton: HTMLButtonElement;
  progressBarContainer: HTMLDivElement;
  progressBarBackground: HTMLDivElement;
  progressBarLoaded: HTMLDivElement;
  progressBar: HTMLDivElement;
  progressScrub: HTMLDivElement;

  loadingContainerMounted: boolean;
  loadingTitle: HTMLDivElement;
  loadingContainer: HTMLDivElement;
  loadingProgressBarOuterContainer: HTMLDivElement;
  loadingProgressBarContainer: HTMLDivElement;
  loadingProgressBarBg: HTMLDivElement;

  playbackSpeed: HTMLInputElement;
  playbackSpeedDisplay: HTMLDivElement;

  frameChangedListener: ((frame: number) => void) | null;
  playPausedListener: ((playing: boolean) => void) | null;

  cancelDownload: (() => void) | null;

  constructor(container: HTMLElement) {
    this.frameChangedListener = null;
    this.playPausedListener = null;
    this.cancelDownload = null;

    this.viewContainer = document.createElement("div");
    this.viewContainer.className = "NimbleStandalone-container";
    container.appendChild(this.viewContainer);
    this.view = new NimbleView(this.viewContainer, true);

    const instructions = document.createElement("div");
    this.playPauseButton = document.createElement("button");
    this.playPauseButton.innerHTML = "Play";
    this.playPauseButton.addEventListener("click", () => {
      this.togglePlay();
    });
    this.playPauseButton.className =
      "NimbleStandalone-progress-play-pause";
    // instructions.innerHTML = "Press [Space] to Play/Pause"
    instructions.appendChild(this.playPauseButton);
    instructions.className =
      "NimbleStandalone-progress-instructions";
    this.viewContainer.appendChild(instructions);

    const playbackSpeedContainer = document.createElement("div");
    playbackSpeedContainer.className = "NimbleStandalone-playback-speed-container";
    this.viewContainer.appendChild(playbackSpeedContainer);
    this.playbackSpeed = document.createElement("input");
    this.playbackSpeed.type = 'range';
    this.playbackSpeed.min = '0.01';
    this.playbackSpeed.max = '1.5';
    this.playbackSpeed.value = '1.0';
    this.playbackSpeed.step = '0.01';
    this.playbackSpeedDisplay = document.createElement('div');
    this.playbackSpeedDisplay.innerHTML = '1.00x speed';
    playbackSpeedContainer.appendChild(this.playbackSpeedDisplay);
    playbackSpeedContainer.appendChild(this.playbackSpeed);
    this.playbackSpeed.oninput = (e: Event) => {
      const val: number = parseFloat(this.playbackSpeed.value);
      this.setPlaybackSpeed(val);
    }

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

    this.progressBarLoaded = document.createElement("div");
    this.progressBarLoaded.className = "NimbleStandalone-progress-bar-loaded";
    this.progressBarContainer.appendChild(this.progressBarLoaded);

    this.progressScrub = document.createElement("div");
    this.progressScrub.className = "NimbleStandalone-progress-bar-scrub";
    this.progressBarContainer.appendChild(this.progressScrub);

    const processMouseEvent = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = this.progressBarContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      let percentage = x / rect.width;
      if (percentage < 0) percentage = 0;
      if (percentage > 1) percentage = 1;
      if (this.playing) this.togglePlay();
      this.scrubFrame = Math.round(this.estimatedTotalFrames * percentage); 
      this.setProgress(percentage);
    };

    this.progressBarContainer.addEventListener("mousedown", (e: MouseEvent) => {
      processMouseEvent(e);

      this.scrubbing = true;
      this.startAnimation();

      window.addEventListener("mousemove", processMouseEvent);
      const mouseUp = () => {
        window.removeEventListener("mousemove", processMouseEvent);
        window.removeEventListener("mousup", mouseUp);
        this.scrubbing = false;
      };
      window.addEventListener("mouseup", mouseUp);
    });

    window.addEventListener("keydown", this.keyboardListener);

    this.view.addDragListener((key: number, pos: number[]) => {
      if (this.view != null) {
        this.view.setObjectPos(key, pos);
      }
    });

    this.lastRecordingHash = "";
    this.rawFrameBytes = [];
    this.estimatedTotalFrames = 100;
    this.animationKey = 0;
    this.playing = false;
    this.scrubbing = false;
    this.startedPlaying = new Date().getTime();
    this.renderedFirstFrame = false;
    this.lastFrame = -1;
    this.scrubFrame = 0;
    this.originalMsPerFrame = 20.0;
    this.playbackMultiple = 1.0;
    this.msPerFrame = this.originalMsPerFrame / this.playbackMultiple;
    this.startFrame = 0;

    this.loadingContainerMounted = false;
    this.loadingContainer = document.createElement("div");
    this.loadingContainer.className = "NimbleStandalone-loading-overlay";

    /*
    const loadingLogo = document.createElement("svg");
    loadingLogo.innerHTML = logoSvg;
    this.loadingContainer.appendChild(loadingLogo);
    */

    this.loadingTitle = document.createElement("div");
    this.loadingTitle.className = "NimbleStandalone-loading-text";
    this.loadingTitle.innerHTML = "nimble<b>viewer</b> loading...";
    this.loadingContainer.appendChild(this.loadingTitle);

    this.loadingProgressBarOuterContainer = document.createElement("div");
    this.loadingProgressBarOuterContainer.className =
      "NimbleStandalone-loading-bar-container";
    this.loadingContainer.appendChild(this.loadingProgressBarOuterContainer);

    const loadingContainerInnerBg = document.createElement("div");
    loadingContainerInnerBg.className =
      "NimbleStandalone-loading-bar-container-inner-bg";
    this.loadingProgressBarOuterContainer.appendChild(loadingContainerInnerBg);

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
      e.preventDefault();
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
    if (this.cancelDownload != null) {
      this.cancelDownload();
    }
    this.playing = false;
    this.view = null;
    window.removeEventListener("keydown", this.keyboardListener);
  };

  setLoadedProgress = (percentage: number) => {
    this.progressBarLoaded.style.width = (1.0 - percentage) * 100 + "%";
    this.progressBarLoaded.style.left = percentage * 100 + "%";
  }

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
   * Sets the loading text we display to users
   */
  setLoadingType = (type: string) => {
    this.loadingTitle.innerHTML = "nimble<b>viewer</b> "+type+"...";
  }

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

  setLoadingProgressError = () => {
    this.loadingContainer.style.backgroundColor = "#eb7575";
    this.loadingProgressBarOuterContainer.remove();
    this.loadingTitle.innerHTML = "nimble<b>viewer</b> error loading file!";
  }

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
    if (this.cancelDownload != null) {
      // Ignore this call, user must cancel the existing download before starting a new one
      return;
    }

    const abortController = new AbortController();
    const abortSignal = abortController.signal;
    this.cancelDownload = () => {
      abortController.abort();
      this.cancelDownload = null;
    }

    this.rawFrameBytes = [];

    this.setLoadedProgress(0.0);

    // Until we get the first frame loaded, we're "loading"
    this.setLoadingType('loading');
    this.setLoadingProgress(0.0);

    fetch(url, {
      method: 'get',
      signal: abortSignal
    }).then((response) => {
      console.log(response);
      if (response != null && response.body != null && response.ok) {
        let body = response.body;
        if (url.endsWith('gz')) {
          console.log("Nimble Visualizer is unzipping the target recording, because it was compressed with Gzip.");
          body = body.pipeThrough(
            new DecompressionStream('gzip')
          );
        }
        const reader = body.getReader();

        const contentLength = parseInt(response.headers.get('Content-Length') ?? '0');

        let bytesReceived = 0;

        let currentFrameCursor: number[] = [0];
        let currentFrameBytes: Uint8Array[] = [new Uint8Array(4)];
        let isSizeFrame: boolean[] = [true];

        const processBytes = ({ done, value }) => {
            // Result objects contain two properties:
            // done  - true if the stream has already given all its data.
            // value - some data. 'undefined' if the reader is canceled.
            if (value == null) {
              this.estimatedTotalFrames = this.rawFrameBytes.length;
              this.setLoadedProgress(1.0);
              if (!this.playing) {
                this.togglePlay();
              }
              return;
            }

            let valueCursor = 0;

            while (valueCursor < value.length) {
              const valueRemainingBytes = value.length - valueCursor;
              const currentFrameRemainingBytes = currentFrameBytes[0].length - currentFrameCursor[0];
              // If we can finish this frame with this read:
              if (currentFrameRemainingBytes <= valueRemainingBytes) {
                for (let i = 0; i < currentFrameRemainingBytes; i++) {
                  currentFrameBytes[0][currentFrameCursor[0]] = value[valueCursor];
                  currentFrameCursor[0] += 1;
                  valueCursor += 1;
                }

                if (isSizeFrame[0]) {
                  const u32bytes = currentFrameBytes[0].buffer.slice(0, 4);
                  const size = new Uint32Array(u32bytes)[0];
                  if (size == 0) {
                    break;
                  }
                  currentFrameBytes[0] = new Uint8Array(size);
                  currentFrameCursor[0] = 0;
                  isSizeFrame[0] = false;
                }
                else {
                  this.rawFrameBytes.push(currentFrameBytes[0]);

                  if (this.rawFrameBytes.length > 1) {
                    // Average all except the first frame
                    let totalFrameBytes = 0;
                    for (let i = 1; i < this.rawFrameBytes.length; i++) {
                      totalFrameBytes += this.rawFrameBytes[i].length;
                    }
                    const avgFrameBytes = totalFrameBytes / (this.rawFrameBytes.length - 1);
                    // Include the first frame as a special case
                    this.estimatedTotalFrames = 1 + Math.round((contentLength - this.rawFrameBytes[0].length) / avgFrameBytes);

                    this.setLoadedProgress(this.rawFrameBytes.length / this.estimatedTotalFrames);
                  }

                  // Immediately show the first frame
                  if (this.rawFrameBytes.length == 1) {
                    // Clear the loading bar
                    this.setLoadingProgress(1.0);
                    this.hideLoadingBar();

                    this.lastFrame = -1;
                    this.setFrame(0);
                    this.view.view.onWindowResize();
                  }

                  currentFrameBytes[0] = new Uint8Array(4);
                  currentFrameCursor[0] = 0;
                  isSizeFrame[0] = true;
                }
              }
              // If we can't finish this frame with this read, just do as well as we can
              else {
                for (let i = 0; i < valueRemainingBytes; i++) {
                  currentFrameBytes[0][currentFrameCursor[0]] = value[valueCursor];
                  currentFrameCursor[0] += 1;
                  valueCursor += 1;
                }
                // If we're still loading the first frame (which can sometimes take a while, because it generally describes all the meshes and textures used)
                if (this.rawFrameBytes.length == 0) {
                  let progress = currentFrameCursor[0] / currentFrameBytes[0].length;
                  this.setLoadingProgress(progress);
                }
              }
            }

            bytesReceived += value.byteLength;

            if (done) {
              this.estimatedTotalFrames = this.rawFrameBytes.length;
              this.setLoadedProgress(1.0);
              if (!this.playing) {
                this.togglePlay();
              }
              return;
            }

            // Read some more, and call this function again
            // Note that here we create a new view over the original buffer.
            return reader
              .read()
              .then(processBytes);
        };

        // read() returns a promise that fulfills when a value has been received
        reader
          .read()
          .then(processBytes);
      }
      else {
        this.setLoadingProgressError();
      }
    }).catch((reason) => {
      console.error(reason);
      this.setLoadingProgressError();
    });
  };

  getRecordingFrame: (number) => dart.proto.CommandList = (index: number) => {
    let command: dart.proto.CommandList = dart.proto.CommandList.deserialize(this.rawFrameBytes[index]);
    return command;
  };

  registerPlayPauseListener = (playPausedListener: ((playing: boolean) => void) | null) => {
    this.playPausedListener = playPausedListener;
  };

  /**
   * This turns playback on or off.
   */
  togglePlay = () => {
    this.playing = !this.playing;
    if (this.playing) {
      this.playPauseButton.innerHTML = "Pause";
    }
    else {
      this.playPauseButton.innerHTML = "Play";
    }
    if (this.playPausedListener != null) {
      this.playPausedListener(this.playing);
    }
    if (this.playing) {
      this.startFrame = this.lastFrame;
      this.startedPlaying = new Date().getTime();
      this.startAnimation();
    }
  };

  /**
   * Sets whether or not we're currently playing.
   */
  setPlaying = (playing: boolean) => {
    if (this.rawBytes != null && playing != this.playing) {
      this.playing = playing;
      if (this.playing) {
        this.playPauseButton.innerHTML = "Pause";
      }
      else {
        this.playPauseButton.innerHTML = "Play";
      }
      if (this.playing) {
        this.startFrame = this.lastFrame;
        this.startedPlaying = new Date().getTime();
        this.startAnimation();
      }
    }
  }

  getPlaying = () => {
    return this.playing;
  }

  /**
   * This sets our playback speed to a multiple of the fundamental number for this data.
   */
  setPlaybackSpeed = (multiple: number) => {
    if (parseFloat(this.playbackSpeed.value) != multiple) {
      this.playbackSpeed.value = multiple.toString();
    }
    this.playbackSpeedDisplay.innerHTML = multiple+'x speed';
    this.startFrame = this.getCurrentFrame();
    this.startedPlaying = new Date().getTime();

    this.playbackMultiple = multiple;
    this.msPerFrame = this.originalMsPerFrame / this.playbackMultiple;
  };

  getCurrentFrame = () => {
    const elapsed: number = new Date().getTime() - this.startedPlaying;
    const currentFrame = (this.startFrame + Math.round(elapsed / this.msPerFrame)) %
      this.estimatedTotalFrames;
    return currentFrame;
  };

  registerFrameChangeListener = (frameChange: ((frame: number) => void) | null) => {
    this.frameChangedListener = frameChange;
  };

  setFrame = (frameNumber: number) => {
    if (frameNumber != this.lastFrame) {
      if (frameNumber < this.lastFrame) {
        // Reset at the beginning
        this.lastFrame = -1;
        // Deliberately skip the first frame when looping back, for efficiency. The first frame usually has a bunch of creation of meshes and stuff, which is expensive to decode and hangs the browser.
        if (this.renderedFirstFrame && frameNumber == 0) {
          frameNumber = 1;
        }
      }
      
      this.setProgress(frameNumber / this.estimatedTotalFrames);
      if (this.view != null) {
        if (frameNumber < this.rawFrameBytes.length) {
          this.getRecordingFrame(frameNumber).command.forEach(this.handleCommand);
        }

        // This is the slower but more correct method.
        /*
        for (let i = this.lastFrame + 1; i <= frameNumber; i++) {
          this.recording[i].command.forEach(this.handleCommand);
        }
        */
        this.view.render();
        if (frameNumber == 0) {
          this.renderedFirstFrame = true;
        }
      }
      this.lastFrame = frameNumber;
    }
  };

  getFrame = () => {
    return this.lastFrame;
  };

  handleCommand = (command: dart.proto.Command) => {
    if (command.set_frames_per_second) {
      console.log("Frames per second: " + command.set_frames_per_second);
      this.originalMsPerFrame = 1000.0 / command.set_frames_per_second.framesPerSecond;
      this.msPerFrame = this.originalMsPerFrame / this.playbackMultiple;
    }
    else {
      this.view.handleCommand(command);
    }
  };

  startAnimation = () => {
    const key = Math.floor(Math.random() * 10000000);
    this.animationKey = key;
    this.animationFrame(key);
  }

  /**
   * This gets called within requestAnimationFrame(...), and handles replaying any commands necessary to jump to the appropriate time in the animation.
   */
  animationFrame = (deduplicationKey: number) => {
    // Only allow a single thread of animationFrame() calls to run at a time
    if (deduplicationKey !== this.animationKey) {
      return;
    }
    // Avoid race conditions if we were stopped, then this frame fired later
    if (!this.playing && !this.scrubbing) return;

    if (this.playing) {
      let frameNumber = this.getCurrentFrame();
      if (frameNumber != this.lastFrame) {
        this.setFrame(frameNumber);
        // Always call this _after_ updating this.lastFrame, to avoid loops
        if (this.frameChangedListener != null) {
          this.frameChangedListener(frameNumber);
        }
      }
    }
    if (this.scrubbing) {
      if (this.scrubFrame !== this.lastFrame) {
        this.setFrame(this.scrubFrame);
        // Always call this _after_ updating this.lastFrame, to avoid loops
        if (this.frameChangedListener != null) {
          this.frameChangedListener(this.scrubFrame);
        }
      }
    }

    // Don't use requestAnimationFrame(), because that causes contention with the mouse interaction, which makes the whole UI feel sluggish
    setTimeout(() => this.animationFrame(deduplicationKey), this.msPerFrame);
  };
}

export default NimbleStandalone;
