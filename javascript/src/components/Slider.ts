class Slider {
  type: "slider";
  key: string;
  container: HTMLElement;
  from_top_left: number[];
  size: number[];
  min: number;
  max: number;
  value: number;
  onlyInts: boolean;
  horizontal: boolean;
  onChange: (number) => void;

  backgroundElem: HTMLDivElement;
  sliderElem: HTMLDivElement;

  constructor(
    container: HTMLElement,
    key: string,
    from_top_left: number[],
    size: number[],
    min: number,
    max: number,
    value: number,
    onlyInts: boolean,
    horizontal: boolean,
    onChange: (number) => void
  ) {
    this.type = "slider";
    this.key = key;
    this.container = container;
    this.from_top_left = from_top_left;
    this.size = size;
    this.min = min;
    this.max = max;
    this.value = value;
    this.onlyInts = onlyInts;
    this.horizontal = horizontal;
    this.onChange = onChange;

    this.backgroundElem = document.createElement("div");
    this.backgroundElem.className = "DARTWindow-slider-bg";
    this.sliderElem = document.createElement("div");
    this.container.appendChild(this.backgroundElem);
    this.backgroundElem.appendChild(this.sliderElem);

    this.sliderElem.style.position = "absolute";
    if (this.horizontal) {
      this.sliderElem.className = "DARTWindow-slider-horizontal";
      let percentage = (this.value - this.min) / (this.max - this.min);
      this.sliderElem.style.left = 100 * percentage + "%";
      this.sliderElem.style.top = "0";
    } else {
      this.sliderElem.className = "DARTWindow-slider-vertical";
      let percentage = (this.value - this.min) / (this.max - this.min);
      this.sliderElem.style.left = "0";
      this.sliderElem.style.bottom = 100 * percentage + "%";
    }

    this.backgroundElem.addEventListener("mousedown", (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const mouseMoveListener = (e: MouseEvent) => {
        const rect = this.backgroundElem.getBoundingClientRect();
        let percentage = 0;
        if (this.horizontal) {
          percentage = (e.clientX - rect.left) / rect.width;
        } else {
          percentage = (rect.bottom - e.clientY) / rect.height;
        }
        if (percentage < 0) percentage = 0;
        if (percentage > 1.0) percentage = 1.0;
        let newValue = percentage * (this.max - this.min) + this.min;
        if (this.onlyInts) {
          newValue = Math.round(newValue);
          percentage = (newValue - this.min) / (this.max - this.min);
        }
        if (newValue != this.value) {
          this.value = newValue;
          this.onChange(newValue);
        }
        this._refresh();
      };

      const mouseUpListener = () => {
        document.removeEventListener("mousemove", mouseMoveListener);
        document.removeEventListener("mouseup", mouseUpListener);
      };

      document.addEventListener("mousemove", mouseMoveListener);
      document.addEventListener("mouseup", mouseUpListener);
    });
  }

  _refresh = () => {
    let percentage = (this.value - this.min) / (this.max - this.min);
    if (this.horizontal) {
      this.sliderElem.style.left = 100 * percentage + "%";
    } else {
      this.sliderElem.style.bottom = 100 * percentage + "%";
    }
  };

  setValue = (value: number) => {
    this.value = value;
    this._refresh();
  };

  setMin = (value: number) => {
    this.min = value;
    this._refresh();
  };

  setMax = (value: number) => {
    this.max = value;
    this._refresh();
  };
}

export default Slider;
