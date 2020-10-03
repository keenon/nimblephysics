import WorldDisplay from "./WorldDisplay";

class Timeline {
  world: WorldDisplay;
  playing: boolean;
  playStartTime: number;
  playStartTimestep: number;
  timeline: HTMLDivElement;
  timelineTick: HTMLDivElement;
  timelineText: HTMLDivElement;

  constructor(world: WorldDisplay) {
    this.world = world;
    this.playing = false;

    this.timeline = document.createElement("div");
    this.timeline.className = "Timeline__timeline";
    document.body.appendChild(this.timeline);

    this.timelineTick = document.createElement("div");
    this.timelineTick.className = "Timeline__timeline-tick";
    this.timeline.appendChild(this.timelineTick);

    this.timelineText = document.createElement("div");
    this.timelineText.className = "Timeline__timeline-text";
    document.body.appendChild(this.timelineText);

    this.updateTick();

    addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        this.playing = !this.playing;
        if (this.playing) {
          this.playStartTime = new Date().getTime();
          this.playStartTimestep = this.world.getTimestep();
        }
      }
    });

    window.addEventListener("resize", this.updateTick, false);

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
        this.updateTick();
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
  }

  updateTick = () => {
    const percentage = this.world.getTimestep() / this.world.getTimesteps();
    const timelineWidth = this.timeline.getBoundingClientRect().width;
    this.timelineTick.style.left = percentage * timelineWidth + "px";
    this.timelineText.innerHTML =
      "step " + this.world.getTimestep() + "/" + this.world.getTimesteps();
  };

  update = () => {
    if (this.playing) {
      const msElapsed = new Date().getTime() - this.playStartTime;
      const FPS = 100;
      const advanceTimesteps = Math.round(msElapsed * (FPS / 1000));
      const timestep =
        (this.playStartTimestep + advanceTimesteps) % this.world.getTimesteps();
      this.world.setTimestep(timestep);
      this.updateTick();
    }
  };
}

export default Timeline;
