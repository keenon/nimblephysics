import { DirectGeometry } from "three";

type GraphEntry = {
  time: number;
  value: number;
};

class LiveGraph {
  container: HTMLDivElement;
  title: string;
  units: string;
  titleElem: HTMLDivElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  durationMs: number;

  value: number;
  history: GraphEntry[];

  max: number;
  min: number;

  running: boolean;

  constructor(
    parent: HTMLElement,
    title: string,
    width: number,
    height: number,
    durationMs: number = 10 * 1000,
    units: string = ""
  ) {
    this.durationMs = durationMs;
    this.history = [];

    this.container = document.createElement("div");
    parent.appendChild(this.container);
    this.title = title;
    this.titleElem = document.createElement("div");
    this.titleElem.className = "LiveGraph__title";
    this.container.appendChild(this.titleElem);
    this.titleElem.innerHTML = this.title;
    this.units = units;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "LiveGraph__canvas";
    this.container.appendChild(this.canvas);
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d");
    this.width = width;
    this.height = height;

    this.max = 1;
    this.min = 0;

    this.running = true;
    this.redrawCanvas();
  }

  recordValue = (value: number) => {
    const time: number = new Date().getTime();
    this.history.push({
      time,
      value,
    });
    this.value = value;
  };

  trimHistory = () => {
    if (this.history.length == 0) return;

    const now: number = new Date().getTime();
    const cutoff: number = now - this.durationMs;
    let cursor: number = -1;
    while (
      cursor + 1 < this.history.length &&
      this.history[cursor + 1].time < cutoff
    ) {
      cursor++;
    }
    if (cursor > -1) {
      this.history.splice(0, cursor);
    }
  };

  redrawCanvas = () => {
    this.trimHistory();
    if (this.history.length > 0) {
      this.titleElem.innerHTML =
        this.title + ": " + this.value.toFixed(5) + this.units;

      for (let i = 0; i < this.history.length; i++) {
        if (this.history[i].value < this.min) this.min = this.history[i].value;
        if (this.history[i].value > this.max) this.max = this.history[i].value;
      }

      const getY = (value: number) => {
        const fromBottom = value - this.min;
        const percentage = fromBottom / (this.max - this.min);
        const pixel = this.height * (1.0 - percentage);
        return pixel;
      };

      const now: number = new Date().getTime();
      const pixelPerMs = this.width / this.durationMs;

      const getX = (time: number) => {
        const timeSinceNow = now - time;
        const pixelsSinceNow = timeSinceNow * pixelPerMs;
        return this.width - pixelsSinceNow;
      };

      this.ctx.clearRect(0, 0, this.width, this.height);

      // Draw a zero-line
      let zeroHeight = getY(0);
      this.ctx.beginPath();
      this.ctx.moveTo(0, zeroHeight);
      this.ctx.lineTo(this.width, zeroHeight);
      this.ctx.strokeStyle = "#ddd";
      this.ctx.stroke();

      this.ctx.strokeStyle = "white";
      this.ctx.beginPath();
      this.ctx.moveTo(getX(this.history[0].time), getY(this.history[0].value));
      for (let i = 0; i < this.history.length; i++) {
        this.ctx.lineTo(
          getX(this.history[i].time),
          getY(this.history[i].value)
        );
      }
      this.ctx.stroke();
    }
    if (this.running) {
      requestAnimationFrame(this.redrawCanvas);
    }
  };

  stop = () => {
    this.running = false;
  };
}

export default LiveGraph;
