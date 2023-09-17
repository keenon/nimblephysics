import NimbleView from "./NimbleView";
// import logoSvg from "!!raw-loader!./nimblelogo.svg";
import { CommandRecording } from "./types";
import protobuf from 'google-protobuf';
import {dart} from './proto/GUI';
import { createHash } from 'sha256-uint8array';
import playSvg from "!!raw-loader!./play.svg";
import pauseSvg from "!!raw-loader!./pause.svg";
import warningSvg from "!!raw-loader!./warning.svg";

class WarningSpan {
  standalone: NimbleStandalone;
  warningKey: number;
  start: number;
  end: number;
  isSpan: boolean;
  warning: string;
  div: HTMLDivElement;
  description: HTMLDivElement;
  dismissButton: HTMLButtonElement;

  constructor(standalone: NimbleStandalone, warningKey: number, start: number, end: number, warning: string) {
    this.standalone = standalone;
    this.warningKey = warningKey;
    this.start = start;
    this.end = end;
    this.isSpan = end - start > 5;
    this.warning = warning;
    this.div = document.createElement('div');
    this.div.className = this.isSpan ? 'NimbleStandalone-warning-span' : 'NimbleStandalone-warning-bubble';
    this.description = document.createElement('div');
    this.description.className = 'NimbleStandalone-warning-description';
    this.description.innerHTML = warning;
    this.dismissButton = document.createElement('button');
    this.dismissButton.innerHTML = 'Dismiss';
    this.dismissButton.className = 'NimbleStandalone-warning-dismiss';
    this.description.appendChild(this.dismissButton);

    this.div.appendChild(this.description);
    this.div.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Bubble clicked");
      if (this.standalone.playing) {
        this.standalone.togglePlay();
      }
      this.standalone.setFrame(this.start);
      this.standalone.view.focusOnWarning(this.warningKey);
    });
    this.update();
    this.standalone.progressBarContainer.appendChild(this.div);
  }

  updateTimestep = (currentFrame: number) => {
    if (this.isSpan) {
      if (currentFrame >= this.start && currentFrame <= this.end) {
        this.div.className = 'NimbleStandalone-warning-span NimbleStandalone-warning-span-active';
      }
      else {
        this.div.className = 'NimbleStandalone-warning-span';
      }
    }
    else {
      if (currentFrame >= this.start && currentFrame <= this.end) {
        this.div.className = 'NimbleStandalone-warning-bubble NimbleStandalone-warning-bubble-active';
      }
      else {
        this.div.className = 'NimbleStandalone-warning-bubble';
      }
    }
  }

  update = () => {
    const startPercentage = this.start / this.standalone.estimatedTotalFrames;
    const endPercentage = this.end / this.standalone.estimatedTotalFrames;
    const width = (endPercentage - startPercentage) * 100;
    const left = startPercentage * 100;
    this.isSpan = width > 1;
    if (this.isSpan) {
      this.div.style.width = width+'%';
      this.div.style.left = left+'%';
    }
    else {
      this.div.style.width = null;
      this.div.style.left = 'calc('+left+'% - 15px)';
    }
  }
}

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
  warningSpans: WarningSpan[];

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
    this.playPauseButton.innerHTML = playSvg;
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

    this.warningSpans = [];

    const processEvent = (e: MouseEvent | Touch) => {
      const rect = this.progressBarContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      let percentage = x / rect.width;
      if (percentage < 0) percentage = 0;
      if (percentage > 1) percentage = 1;
      if (this.playing) this.togglePlay();
      this.scrubFrame = Math.round(this.estimatedTotalFrames * percentage); 
      this.setProgress(percentage);
    };

    const startEvent = (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.touches) {
        processEvent(event.touches[0]); // Get the first touch
      } else {
        processEvent(event);
      }

      this.scrubbing = true;
      this.startAnimation();

      const moveEvent = (event) => {
        // try {
        //   event.preventDefault();
        // }
        // catch (e) {
        //   // Ignore
        // }

        if (event.touches) {
          processEvent(event.touches[0]); // Get the first touch
        } else {
          processEvent(event);
        }
      };

      const endEvent = () => {
        if (this.scrubbing) {
          window.removeEventListener("mousemove", moveEvent);
          window.removeEventListener("mouseup", endEvent);
          window.removeEventListener("touchmove", moveEvent);
          window.removeEventListener("touchend", endEvent);
          this.scrubbing = false;
        }
      };

      if (event.touches) {
        window.addEventListener("touchmove", moveEvent);
        window.addEventListener("touchend", endEvent);
      } else {
        window.addEventListener("mousemove", moveEvent);
        window.addEventListener("mouseup", endEvent);
      }
    };

    this.progressBarContainer.addEventListener("mousedown", startEvent);
    this.progressBarContainer.addEventListener("touchstart", startEvent);


    // this.progressBarContainer.addEventListener("mousedown", (e: MouseEvent) => {
    //   processMouseEvent(e);

    //   this.scrubbing = true;
    //   this.startAnimation();

    //   window.addEventListener("mousemove", processMouseEvent);
    //   const mouseUp = () => {
    //     window.removeEventListener("mousemove", processMouseEvent);
    //     window.removeEventListener("mousup", mouseUp);
    //     this.scrubbing = false;
    //   };
    //   window.addEventListener("mouseup", mouseUp);
    // });

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

  getRemainingLoadedMillis = () => {
    const currentFrame = this.lastFrame;
    const bufferFrames = this.rawFrameBytes.length;
    const availableFrames = bufferFrames - currentFrame;
    const availableTime = availableFrames * this.msPerFrame;
    return availableTime;
  };

  setLoadedProgress = (percentage: number) => {
    this.progressBarLoaded.style.width = (1.0 - percentage) * 100 + "%";
    this.progressBarLoaded.style.left = percentage * 100 + "%";

    // As we're buffering, if we buffer up enough for 1s of playback, start playing
    if (!this.playing) {
      if (this.getRemainingLoadedMillis() > 1000) {
        this.togglePlay();
      }
    }
  }

  setProgress = (percentage: number) => {
    this.progressBar.style.width = percentage * 100 + "%";
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
              // Update the spans now that we know exactly how many frames we have
              for (const span of this.warningSpans) {
                span.update();
              }
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
    this.setPlaying(!this.playing);
  };

  /**
   * Sets whether or not we're currently playing.
   */
  setPlaying = (playing: boolean) => {
    if (playing != this.playing) {
      this.playing = playing;
      if (this.playing) {
        if (this.getRemainingLoadedMillis() < 500) {
          // Start back at the beginning if we reached the end and we want to play anyways
          this.lastFrame = -1;
        }
        this.playPauseButton.innerHTML = pauseSvg;
      }
      else {
        this.playPauseButton.innerHTML = playSvg;
      }
      if (this.playPausedListener != null) {
        this.playPausedListener(this.playing);
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
      this.warningSpans.forEach((span) => {
        span.updateTimestep(frameNumber);
      });

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
        else {
          // Stop playing, if we've reached the end of our loaded content
          if (this.playing) {
            this.togglePlay();
          }
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
    else if (command.set_span_warning) {
      this.addSpanWarning(command.set_span_warning.warning_key, command.set_span_warning.start_timestep, command.set_span_warning.end_timestep, command.set_span_warning.warning);
    }
    else {
      this.view.handleCommand(command);
    }
  };

  addSpanWarning = (warningKey: number, start: number, end: number, warning: string) => {
    const warningSpan = new WarningSpan(this, warningKey, start, end, warning);
    this.warningSpans.push(warningSpan);
  }

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
