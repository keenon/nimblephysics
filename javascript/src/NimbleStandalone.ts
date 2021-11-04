import NimbleView from "./NimbleView";

class NimbleStandalone {
  view: NimbleView;
  recording: CommandRecording;
  playing: boolean;

  constructor(view: NimbleView) {
    this.view = view;

    window.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key.toString() == " ") {
        this.togglePlay();
      }
    });

    this.view.addDragListener((key: string, pos: number[]) => {
      this.view.setObjectPos(key, pos);
    });

    this.recording = [];
    this.playing = false;
  }

  /**
   * This loads a model, and a recording. The model is assumed to be a bunch of "createX" commands, which only need to be run once.
   * Then the recording is just commands like "setObjectPosition", etc.
   *
   * @param modelUrl The URL to load the model from
   * @param recordingUrl The URL to load the recording from
   */
  loadModelAndRecording = (modelUrl: string, recordingUrl: string) => {};

  /**
   * This loads a standalone recording to play back. That means it doesn't require a separate "model" file in order to create objects.
   *
   * @param url The URL to load a recording from, in order to play back
   */
  loadStandaloneRecording = (url: string) => {};

  /**
   * This replaces the set of recorded commands we're replaying
   *
   * @param recording The JSON object representing a recording of timestep(s) command(s)
   */
  setRecording = (recording: CommandRecording) => {
    this.setRecording(recording);
    if (!this.playing) {
      this.togglePlay();
    }
  };

  /**
   * This turns playback on or off.
   */
  togglePlay = () => {
    this.playing = true;
  };
}

export default NimbleStandalone;
