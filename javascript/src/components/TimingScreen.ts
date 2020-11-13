import LiveGraph from "./LiveGraph";

class TimingScreen {
  container: HTMLDivElement;
  plots: Map<string, LiveGraph>;

  constructor(parent: HTMLElement) {
    this.container = document.createElement("div");
    parent.appendChild(this.container);
    this.container.className = "TimingScreen";
    this.container.innerHTML = "";
    this.plots = new Map();
  }

  registerTimings = (timings: Timings) => {
    for (let key in timings) {
      const timing = timings[key];
      let plot: LiveGraph | null = this.plots.get(key);
      if (plot == null) {
        plot = new LiveGraph(this.container, key, 200, 60, 1000, timing.units);
        this.plots.set(key, plot);
      }
      plot.recordValue(timing.value);
    }
  };

  stop = () => {
    this.plots.forEach((plot) => {
      plot.stop();
    });
  };
}

export default TimingScreen;
