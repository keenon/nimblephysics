import Timeline from "./Timeline";
import WorldDisplay from "./WorldDisplay";

class DataSelector {
  world: WorldDisplay;
  timeline: Timeline;
  data: Map<string, FullReport>;
  holder: HTMLDivElement;

  constructor(world: WorldDisplay, timeline: Timeline) {
    this.world = world;
    this.timeline = timeline;
    this.data = new Map();
    this.holder = document.createElement("div");
    this.holder.className = "DataSelector__holder";
    document.body.appendChild(this.holder);
  }

  registerData = (name: string, data: FullReport) => {
    this.data.set(name, data);
    const buttonHolder = document.createElement("div");
    const button = document.createElement("button");
    button.innerText = name;
    button.onclick = () => {
      this.world.setData(data);
      this.timeline.updateLoglineTick();
      this.timeline.updateTimelineTick();
    };
    buttonHolder.appendChild(button);
    this.holder.appendChild(buttonHolder);
  };
}

export default DataSelector;
