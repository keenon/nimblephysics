class Line {
  name: string;
  color: string;
  xs: number[];
  ys: number[];
  plotType: "line" | "scatter";

  constructor(
    name: string,
    color: string,
    xs: number[],
    ys: number[],
    plotType: "line" | "scatter"
  ) {
    this.name = name;
    this.color = color;
    this.xs = xs;
    this.ys = ys;
    this.plotType = plotType;
  }
}

type TickData = {
  text: string;
  bottom: number;
  left: number;
};

type ExistingTick = {
  dom: HTMLDivElement,
  data: TickData
};

class RichPlot {
  type: "rich_plot";
  key: number;
  container: HTMLElement;
  from_top_left: number[];
  size: number[];
  maxX: number;
  minX: number;
  maxY: number;
  minY: number;
  lines: Map<string, Line>;

  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;

  bodyContainer: HTMLElement;
  title: HTMLDivElement;
  titleTriangle: HTMLDivElement;
  yAxisLabel: HTMLDivElement;
  xAxisLabel: HTMLDivElement;
  legendContainer: HTMLDivElement;
  legendContainerLastHTML: string;

  existingTicks: ExistingTick[];

  show: boolean;

  constructor(
    container: HTMLElement,
    key: number,
    from_top_left: number[],
    size: number[],
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
    title: string,
    xAxisLabel: string,
    yAxisLabel: string
  ) {
    this.key = key;
    this.type = "rich_plot";
    this.container = container;
    this.from_top_left = from_top_left;
    this.size = size;
    this.maxX = maxX;
    this.minX = minX;
    this.maxY = maxY;
    this.minY = minY;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "DARTWindow-plot-canvas";
    this.container.appendChild(this.canvas);
    this.canvas.style.width = size[0] + 'px';
    this.canvas.style.height = size[1] + 'px';
    this.canvas.width = this.container.getBoundingClientRect().width * 2;
    this.canvas.height = this.container.getBoundingClientRect().height * 2;

    this.lines = new Map();

    this.title = document.createElement("div");
    this.title.style.position = 'absolute';
    this.title.style.top = '-30px';
    this.title.style.left = '0px';
    this.title.style.width = '100%';
    this.title.style.textAlign = 'center';
    this.title.innerText = title;
    this.title.className = "NimbleStandalone-plot-title";
    this.container.appendChild(this.title);

    this.titleTriangle = document.createElement("div");
    this.titleTriangle.className = "NimbleStandalone-triangle NimbleStandalone-triangle-shown";
    this.title.appendChild(this.titleTriangle);

    this.bodyContainer = document.createElement("div");
    this.bodyContainer.style.height = '100%';
    this.bodyContainer.style.width = '100%';
    this.bodyContainer.style.position = 'absolute';
    this.bodyContainer.style.top = '0';
    this.bodyContainer.style.left = '0';
    this.container.appendChild(this.bodyContainer);

    this.bodyContainer.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d") as CanvasRenderingContext2D;

    let labelHeight = 20;

    this.xAxisLabel = document.createElement("div");
    this.xAxisLabel.style.position = 'absolute';
    this.xAxisLabel.style.bottom = '-40px';
    this.xAxisLabel.style.left = '0px';
    this.xAxisLabel.style.width = '100%';
    this.xAxisLabel.style.textAlign = 'center';
    this.xAxisLabel.innerText = xAxisLabel;
    this.bodyContainer.appendChild(this.xAxisLabel);

    this.yAxisLabel = document.createElement("div");
    this.yAxisLabel.style.position = 'absolute';
    this.yAxisLabel.style.width = size[1] + 'px';
    this.yAxisLabel.style.left = ((-size[1] / 2) - 60 + (labelHeight / 2)) + 'px';
    this.yAxisLabel.style.top = ((size[1] / 2) - (labelHeight / 2)) + 'px';
    this.yAxisLabel.style.transform = 'rotate(-90deg)';
    this.yAxisLabel.style.textAlign = 'center';
    this.yAxisLabel.innerText = yAxisLabel;
    this.bodyContainer.appendChild(this.yAxisLabel);

    this.legendContainer = document.createElement("div");
    this.legendContainer.style.position = 'absolute';
    this.legendContainer.style.top = '10px';
    this.legendContainer.style.right = '10px';
    this.legendContainer.style.width = '60px';
    this.legendContainer.style.backgroundColor = 'white';
    this.legendContainer.style.border = '1px solid #ddd';
    this.legendContainer.style.fontSize = '12px';
    this.bodyContainer.appendChild(this.legendContainer);
    this.legendContainerLastHTML = '';

    this.existingTicks = [];
    this.show = true;

    // Make the title show/hide
    this.title.addEventListener("click", (e: MouseEvent) => {
      e.preventDefault();
      e.stopImmediatePropagation();

      this.show = !this.show;
      if (this.show) {
        this.bodyContainer.style.display = "block";
        this.titleTriangle.className = "NimbleStandalone-triangle NimbleStandalone-triangle-shown";
      }
      else {
        this.bodyContainer.style.display = "none";
        this.titleTriangle.className = "NimbleStandalone-triangle NimbleStandalone-triangle-hidden";
      }
    });

    // Make the container draggable
    this.container.addEventListener("mousedown", (e: MouseEvent) => {
      e.preventDefault();
      let startMouseX = e.clientX;
      let startMouseY = e.clientY;
      let startXPos = parseInt(this.container.style.left.replace('px', ''));
      let startYPos = parseInt(this.container.style.top.replace('px', ''));

      const onDrag = (e2: MouseEvent) => {
        let newMouseX = e2.clientX;
        let newMouseY = e2.clientY;

        this.container.style.left = (startXPos + (newMouseX - startMouseX)) + 'px';
        this.container.style.top = (startYPos + (newMouseY - startMouseY)) + 'px';
      };

      const onRelease = (e2: MouseEvent) => {
        window.removeEventListener("mousemove", onDrag);
        window.removeEventListener("mouseup", onRelease);
      }

      window.addEventListener("mousemove", onDrag);
      window.addEventListener("mouseup", onRelease);
    });

    this.redraw();
  }

  /**
   * This creates or overwrites a line of data on the plot with name `name`. If a line already exists with this name, it will overwrite the contents.
   * 
   * @param name 
   * @param xs 
   * @param ys 
   * @param color 
   * @param plotType 
   */
  setLineData = (
    name: string,
    xs: number[],
    ys: number[],
    color: string,
    plotType: "line" | "scatter") => {
    this.lines.set(name, new Line(name, color, xs, ys, plotType));
    this.updateLineAxis();
    this.redraw();
  }

  /**
   * This sets the bounds that the plot can operate in
   * 
   * @param minX 
   * @param maxX 
   * @param minY 
   * @param maxY 
   */
  setBounds = (
    minX: number,
    maxX: number,
    minY: number,
    maxY: number
  ) => {
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.redraw();
  };

  /**
   * This creates tick text on the DOM, and does its best to avoid unnecessary DOM re-renders by cacheing ticks from previous calls.
   * 
   * @param ticks The ticks labels to display
   */
  setTicks = (ticks: TickData[]) => {
    // 1. Go through the tick data
    for (let i = 0; i < ticks.length; i++) {
      let tick = ticks[i];
      let existingTick: ExistingTick | null = null;
      if (this.existingTicks.length > 0) {
        existingTick = this.existingTicks[i];
      }
      // 1.1. Create any ticks that don't exist yet
      if (existingTick == null) {
        existingTick = {
          dom: document.createElement("div"),
          data: ticks[i]
        };
        existingTick.dom.style.position = 'absolute';
        existingTick.dom.style.bottom = ticks[i].bottom + 'px';
        existingTick.dom.style.left = ticks[i].left + 'px';
        existingTick.dom.style.fontSize = '12px';
        existingTick.dom.innerText = ticks[i].text;
        this.existingTicks.push(existingTick);
        this.bodyContainer.appendChild(existingTick.dom);
      }

      // 1.2. Update any ticks that changed
      if (tick.bottom != existingTick.data.bottom) {
        existingTick.dom.style.bottom = tick.bottom + 'px';
        existingTick.data.bottom = tick.bottom;
      }
      if (tick.left != existingTick.data.left) {
        existingTick.dom.style.left = tick.left + 'px';
        existingTick.data.left = tick.left;
      }
      if (tick.text != existingTick.data.text) {
        existingTick.dom.innerText = tick.text;
        existingTick.data.text = tick.text;
      }
    }

    // 2. Remove any dom elements we're not using
    if (this.existingTicks.length > ticks.length) {
      let remaining = this.existingTicks.splice(ticks.length, this.existingTicks.length - ticks.length);
      remaining.forEach((e) => e.dom.remove());
    }
  };

  /**
   * This creates the axis labels. It does its best not to force a DOM re-render if it doesn't have to.
   */
  updateLineAxis = () => {
    let linesArray: Line[] = [];
    this.lines.forEach((line) => {
      linesArray.push(line);
    })
    linesArray.sort((a: Line, b: Line) => {
      return a.name.localeCompare(b.name);
    });

    let html = '<div>';
    for (let i = 0; i < linesArray.length; i++) {
      html += '<div>';
      html += '<div style="display: inline-block; background-color: ' + linesArray[i].color + '; width: 10px; height: 2px; margin-left: 3px; margin-right: 1px; margin-bottom: 3px;"></div>';
      html += '<span>' + linesArray[i].name + '</span>';
      html += '</div>';
    }
    html += '</div>';
    if (this.legendContainerLastHTML !== html) {
      console.log("Updating legend DOM");
      this.legendContainerLastHTML = html;
      this.legendContainer.innerHTML = html;
    }
  };

  redraw = () => {
    const width = this.container.getBoundingClientRect().width * 2;
    const height = this.container.getBoundingClientRect().height * 2;

    this.canvas.width = width;
    this.canvas.height = height;

    this.ctx.clearRect(0, 0, width, height);

    let ticksPadding = 10;

    const getY = (value: number) => {
      const percentage = (value - this.minY) / (this.maxY - this.minY);
      const pixel = (height - ticksPadding) * (1.0 - percentage);
      return pixel;
    };

    const getX = (value: number) => {
      const percentage = (value - this.minX) / (this.maxX - this.minX);
      return percentage * (width - ticksPadding) + ticksPadding;
    };

    // 1. Draw the ticks

    // 1.0. Draw axis bars
    this.ctx.beginPath();
    this.ctx.moveTo(getX(this.minX), 0);
    this.ctx.lineTo(getX(this.minX), height);
    this.ctx.moveTo(0, getY(this.minY));
    this.ctx.lineTo(width, getY(this.minY));
    this.ctx.strokeStyle = "black";
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    let maxNumTicks = 6;
    let ticks: TickData[] = [];

    // 1.1. Draw the X ticks
    let xSpan = this.maxX - this.minX;
    let xIncrement = xSpan / maxNumTicks;
    // Example, xSpan = 10, maxNumTicks = 6, 10/6 = 1.25ish. Let increments be 2.0
    let xPowerOfTen = Math.round(Math.log10(xIncrement));
    let xBase = Math.pow(10, xPowerOfTen);
    let xTick = Math.ceil(xIncrement / xBase) * xBase;
    let xFirstTick = Math.floor(this.minX / xTick) * xTick;
    let xLastTick = Math.ceil(this.maxX / xTick) * xTick;
    let xNumTicks = Math.round((xLastTick - xFirstTick) / xTick);

    this.ctx.lineWidth = 1.0;
    for (let i = 0; i < xNumTicks; i++) {
      let currentTick = xFirstTick + i * xTick;

      // Draw large tick
      this.ctx.beginPath();
      this.ctx.moveTo(getX(currentTick), getY(this.minY));
      this.ctx.lineTo(getX(currentTick), height);
      this.ctx.lineWidth = 2;
      this.ctx.strokeStyle = "black";
      this.ctx.stroke();

      // Only display text for ticks within bounds
      if (currentTick >= this.minX && currentTick <= this.maxX) {
        ticks.push({
          text: currentTick.toExponential(0),
          bottom: -15,
          left: getX(currentTick) / 2
        });
      }

      // Draw line guides
      this.ctx.beginPath();
      this.ctx.moveTo(getX(currentTick), getY(this.minY));
      this.ctx.lineTo(getX(currentTick), getY(this.maxY));
      this.ctx.strokeStyle = "#555";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();

      // Draw sub-ticks
      for (let j = 0; j < 4; j++) {
        let smallTick = currentTick + (j + 1) * (xTick / 5);
        this.ctx.beginPath();
        this.ctx.moveTo(getX(smallTick), getY(this.minY));
        this.ctx.lineTo(getX(smallTick), height - (ticksPadding / 2));
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = "black";
        this.ctx.stroke();
      }
    }

    // 1.2. Draw the Y ticks
    let ySpan = this.maxY - this.minY;
    let yIncrement = ySpan / maxNumTicks;
    // Example, xSpan = 10, maxNumTicks = 6, 10/6 = 1.25ish. Let increments be 2.0
    let yPowerOfTen = Math.round(Math.log10(yIncrement));
    let yBase = Math.pow(10, yPowerOfTen);
    let yTick = Math.ceil(yIncrement / yBase) * yBase;
    let yFirstTick = Math.floor(this.minY / yTick) * yTick;
    let yLastTick = Math.ceil(this.maxY / yTick) * yTick;
    let yNumTicks = Math.round((yLastTick - yFirstTick) / yTick);

    this.ctx.lineWidth = 1.0;
    for (let i = 0; i < yNumTicks; i++) {
      let currentTick = yFirstTick + i * yTick;

      // Draw large tick
      this.ctx.beginPath();
      this.ctx.moveTo(getX(this.minX), getY(currentTick));
      this.ctx.lineTo(0, getY(currentTick));
      this.ctx.lineWidth = 2;
      this.ctx.strokeStyle = "black";
      this.ctx.stroke();

      // Only display text for ticks within bounds
      if (currentTick >= this.minY && currentTick <= this.maxY) {
        ticks.push({
          text: currentTick.toExponential(0),
          bottom: (height - getY(currentTick)) / 2,
          left: -30
        });
      }

      // Draw line guides
      this.ctx.beginPath();
      this.ctx.moveTo(getX(this.minX), getY(currentTick));
      this.ctx.lineTo(getX(this.maxX), getY(currentTick));
      this.ctx.strokeStyle = "#555";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();

      for (let j = 0; j < 4; j++) {
        let smallTick = currentTick + (j + 1) * (yTick / 5);
        this.ctx.beginPath();
        this.ctx.moveTo(getX(this.minX), getY(smallTick));
        this.ctx.lineTo(ticksPadding / 2, getY(smallTick));
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = "black";
        this.ctx.stroke();
      }
    }

    this.setTicks(ticks);

    // Draw zero-lines
    if (this.minY < 0) {
      this.ctx.beginPath();
      this.ctx.moveTo(getX(this.minX), getY(0));
      this.ctx.lineTo(getX(this.maxX), getY(0));
      this.ctx.strokeStyle = "black";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }
    if (this.minX < 0) {
      this.ctx.beginPath();
      this.ctx.moveTo(getX(0), getY(this.minY));
      this.ctx.lineTo(getX(0), getY(this.maxY));
      this.ctx.strokeStyle = "black";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }

    // Draw the data
    this.lines.forEach((line: Line, key: string) => {
      this.ctx.strokeStyle = line.color;
      this.ctx.beginPath();
      this.ctx.moveTo(getX(line.xs[0]), getY(line.ys[0]));
      for (let i = 0; i < line.xs.length; i++) {
        this.ctx.lineTo(getX(line.xs[i]), getY(line.ys[i]));
      }
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    });
  };
}

export default RichPlot;
