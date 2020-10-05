import WorldDisplay from "./WorldDisplay";

class Timeline {
  world: WorldDisplay;
  playing: boolean;
  playStartTime: number;
  playStartTimestep: number;

  playPauseButton: HTMLButtonElement;

  timeline: HTMLDivElement;
  timelineTick: HTMLDivElement;
  timelineText: HTMLDivElement;

  logline: HTMLDivElement;
  loglineTick: HTMLDivElement;
  loglineText: HTMLDivElement;

  constructor(world: WorldDisplay) {
    this.world = world;
    this.playing = false;

    const playPauseButtonHolder = document.createElement("div");
    playPauseButtonHolder.className = "Timeline__play-pause-holder";
    document.body.appendChild(playPauseButtonHolder);
    this.playPauseButton = document.createElement("button");
    this.playPauseButton.className = "Timeline__play-pause-button";
    this.playPauseButton.innerHTML = "Play";
    playPauseButtonHolder.appendChild(this.playPauseButton);
    this.playPauseButton.onclick = this.playPauseToggle;

    this.timeline = document.createElement("div");
    this.timeline.className = "Timeline__timeline";
    document.body.appendChild(this.timeline);

    this.timelineTick = document.createElement("div");
    this.timelineTick.className = "Timeline__timeline-tick";
    this.timeline.appendChild(this.timelineTick);

    this.timelineText = document.createElement("div");
    this.timelineText.className = "Timeline__timeline-text";
    document.body.appendChild(this.timelineText);

    this.logline = document.createElement("div");
    this.logline.className = "Timeline__logline";
    document.body.appendChild(this.logline);

    this.loglineTick = document.createElement("div");
    this.loglineTick.className = "Timeline__logline-tick";
    this.logline.appendChild(this.loglineTick);

    this.loglineText = document.createElement("div");
    this.loglineText.className = "Timeline__logline-text";
    document.body.appendChild(this.loglineText);
    this.loglineText.innerHTML = "iter: 0/100<br>loss: 1.23e-7<br>infeas: 1e-3";

    this.updateTimelineTick();
    this.updateLoglineTick();

    addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        this.playPauseToggle();
      }
    });

    window.addEventListener(
      "resize",
      () => {
        this.updateTimelineTick();
        this.updateLoglineTick();
      },
      false
    );

    // Make the timeline draggable
    this.timeline.addEventListener("mousedown", (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.playing = false;

      const setTickByMouse = (mouseX: number) => {
        const timelineRect = this.timeline.getBoundingClientRect();
        let percentage = (mouseX - timelineRect.left) / timelineRect.width;
        if (percentage < 0) percentage = 0;
        if (percentage > 1) percentage = 1;

        const timesteps = this.world.getTimesteps();
        let t = Math.round(timesteps * percentage);
        if (t >= timesteps) {
          t = timesteps - 1;
        }
        this.world.setTimestep(t);
        this.updateTimelineTick();
      };

      setTickByMouse(e.clientX);

      const moveListener = (e: MouseEvent) => {
        setTickByMouse(e.clientX);
      };
      const upListener = (e: MouseEvent) => {
        window.removeEventListener("mousemove", moveListener);
        window.removeEventListener("mouseup", upListener);
      };

      window.addEventListener("mousemove", moveListener);
      window.addEventListener("mouseup", upListener);
    });

    // Make the logline draggable
    this.logline.addEventListener("mousedown", (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.playing = false;

      const setLoglineTickByMouse = (mouseY: number) => {
        const loglineRect = this.logline.getBoundingClientRect();
        let percentage = 1.0 - (mouseY - loglineRect.top) / loglineRect.height;
        if (percentage < 0) percentage = 0;
        if (percentage > 1) percentage = 1;

        const iteration = this.world.getNumIterations();
        let i = Math.round(iteration * percentage);
        if (i >= this.world.getNumIterations()) {
          i = this.world.getNumIterations() - 1;
        }
        this.world.setIteration(i);
        this.updateLoglineTick();
      };

      setLoglineTickByMouse(e.clientY);

      const moveListener = (e: MouseEvent) => {
        setLoglineTickByMouse(e.clientY);
      };
      const upListener = (e: MouseEvent) => {
        window.removeEventListener("mousemove", moveListener);
        window.removeEventListener("mouseup", upListener);
      };

      window.addEventListener("mousemove", moveListener);
      window.addEventListener("mouseup", upListener);
    });
  }

  playPauseToggle = () => {
    this.playing = !this.playing;
    if (this.playing) {
      this.playPauseButton.innerHTML = "Pause";
      this.playStartTime = new Date().getTime();
      this.playStartTimestep = this.world.getTimestep();
    } else {
      this.playPauseButton.innerHTML = "Play";
    }
  };

  updateTimelineTick = () => {
    let percentage = this.world.getTimestep() / (this.world.getTimesteps() - 1);
    if (this.world.getTimesteps() === 1) percentage = 0.5;
    const timelineWidth =
      this.timeline.getBoundingClientRect().width -
      this.timelineTick.getBoundingClientRect().width -
      2;
    this.timelineTick.style.left = percentage * timelineWidth + "px";
    this.timelineText.innerHTML =
      "step " +
      (this.world.getTimestep() + 1) +
      "/" +
      this.world.getTimesteps();
  };

  updateLoglineTick = () => {
    let percentage =
      this.world.getIteration() / (this.world.getNumIterations() - 1);
    if (this.world.getNumIterations() === 1) percentage = 0.5;
    const loglineHeight =
      this.logline.getBoundingClientRect().height -
      this.loglineTick.getBoundingClientRect().height -
      2;
    this.loglineTick.style.bottom = percentage * loglineHeight + "px";
    const text = `<table>
  <tbody>
    <tr>
      <td>iter:</td>
      <td>${this.world.getIteration() + 1}/${this.world.getNumIterations()}</td>
    </tr>
    <tr>
      <td>loss:</td>
      <td>${this.world.getLoss().toExponential(5)}</td>
    </tr>
    <tr>
      <td>constraints:</td>
      <td class="${
        this.world.getConstraintViolation() > 1.0e-1
          ? "Timeline__big-constraint-violation"
          : this.world.getConstraintViolation() > 1.0e-2
          ? "Timeline__constraint-violation"
          : ""
      }">${this.world.getConstraintViolation().toExponential(5)}</td>
    </tr>
    <tr>
      <td colspan="2" id="next-step-holder">
      </td>
    </tr>
    <tr>
      <td colspan="2" id="prev-step-holder">
      </td>
    </tr>
  </tbody>
</table>`;
    this.loglineText.innerHTML = text;
    const nextStepHolder = document.getElementById("next-step-holder");
    const prevStepHolder = document.getElementById("prev-step-holder");

    const nextStepButton = document.createElement("button");
    nextStepButton.innerHTML = "Show next learning step";
    nextStepHolder.appendChild(nextStepButton);
    if (this.world.getIteration() < this.world.getNumIterations() - 1) {
      nextStepButton.onclick = () => {
        this.world.setIteration(this.world.getIteration() + 1);
        this.updateLoglineTick();
      };
    } else {
      nextStepButton.disabled = true;
    }

    const prevStepButton = document.createElement("button");
    prevStepButton.innerHTML = "Show prev learning step";
    prevStepHolder.appendChild(prevStepButton);
    if (this.world.getIteration() > 0) {
      prevStepButton.onclick = () => {
        this.world.setIteration(this.world.getIteration() - 1);
        this.updateLoglineTick();
      };
    } else {
      prevStepButton.disabled = true;
    }
  };

  update = () => {
    if (this.playing) {
      const msElapsed = new Date().getTime() - this.playStartTime;
      const FPS = 100;
      const advanceTimesteps = Math.round(msElapsed * (FPS / 1000));
      const timestep =
        (this.playStartTimestep + advanceTimesteps) % this.world.getTimesteps();
      this.world.setTimestep(timestep);
      this.updateTimelineTick();
    }
  };
}

export default Timeline;
