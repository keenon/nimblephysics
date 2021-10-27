class Plot {
  type: "plot";
  key: string;
  container: HTMLElement;
  from_top_left: number[];
  size: number[];
  maxX: number;
  minX: number;
  xs: number[];
  maxY: number;
  minY: number;
  ys: number[];
  plotType: "line" | "scatter";

  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;

  constructor(
    container: HTMLElement,
    key: string,
    from_top_left: number[],
    size: number[],
    minX: number,
    maxX: number,
    xs: number[],
    minY: number,
    maxY: number,
    ys: number[],
    plotType: "line" | "scatter"
  ) {
    this.key = key;
    this.type = "plot";
    this.container = container;
    this.from_top_left = from_top_left;
    this.size = size;
    this.maxX = maxX;
    this.minX = minX;
    this.xs = xs;
    this.maxY = maxY;
    this.minY = minY;
    this.ys = ys;
    this.plotType = plotType;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "DARTWindow-plot-canvas";
    this.container.appendChild(this.canvas);
    this.canvas.width = this.container.getBoundingClientRect().width;
    this.canvas.height = this.container.getBoundingClientRect().height;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");

    this.redraw();
  }

  setData = (
    minX: number,
    maxX: number,
    xs: number[],
    minY: number,
    maxY: number,
    ys: number[]
  ) => {
    this.minX = minX;
    this.maxX = maxX;
    this.xs = xs;
    this.minY = minY;
    this.maxY = maxY;
    this.ys = ys;
    this.redraw();
  };

  redraw = () => {
    const width = this.container.getBoundingClientRect().width;
    const height = this.container.getBoundingClientRect().height;

    this.canvas.width = width;
    this.canvas.height = height;

    this.ctx.clearRect(0, 0, width, height);

    const getY = (value: number) => {
      const percentage = (value - this.minY) / (this.maxY - this.minY);
      const pixel = height * (1.0 - percentage);
      return pixel;
    };

    const getX = (value: number) => {
      const percentage = (value - this.minX) / (this.maxX - this.minX);
      return percentage * width;
    };

    // Draw a zero-line
    let zeroHeight = getY(0);
    this.ctx.beginPath();
    this.ctx.moveTo(0, zeroHeight);
    this.ctx.lineTo(width, zeroHeight);
    this.ctx.strokeStyle = "#ddd";
    this.ctx.stroke();

    // Draw the data
    this.ctx.strokeStyle = "white";
    this.ctx.beginPath();
    this.ctx.moveTo(getX(this.xs[0]), getY(this.ys[0]));
    for (let i = 0; i < this.xs.length; i++) {
      this.ctx.lineTo(getX(this.xs[i]), getY(this.ys[i]));
    }
    this.ctx.stroke();
  };
}

export default Plot;
