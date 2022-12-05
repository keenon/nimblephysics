import NimbleStandalone from "./NimbleStandalone";
import NimbleStandaloneReact from "./NimbleStandaloneReact";
import React, { useState } from 'react';
import ReactDOM from 'react-dom';
// import rawBinary from '!!arraybuffer-loader!./data/movement2.bin';
// import rawBinary from '!!arraybuffer-loader!./data/spring_spine_3_35cm_0N.bin';
import rawBinary from '!!arraybuffer-loader!./data/sprint_zero_residuals.bin';
// import rawBinary from '!!arraybuffer-loader!./data/constant_curve.bin';
// import rawBinary from '!!arraybuffer-loader!./data/sprint_with_spine.bin';
// import rawBinary from '!!arraybuffer-loader!./data/sprint_3.1cm_44N.bin';
// import rawBinary from '!!arraybuffer-loader!./data/walk_1.2cm_1.4N.bin';
// import rawBinary from '!!arraybuffer-loader!./data/marker_trace.bin';

const rawArray = new Uint8Array(rawBinary);
console.log(rawArray);

const ReactTestBed = () => {
  const [show, setShow] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0.0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  let children = [];
  children.push(React.createElement("button", {
    onClick: () => {
      setShow(!show);
    },
    key: 'show'
  }, show ? "Hide" : "Show"));

  children.push(React.createElement("button", {
    onClick: () => {
      setPlaying(!playing);
    },
    key: 'play'
  }, playing ? "Pause" : "Play"));

  children.push(React.createElement("input", {
    type: 'number',
    value: frame,
    onChange: (e) => {
      setFrame(parseInt(e.target.value));
    },
    key: 'frame'
  }));

  if (show) {
    children.push(React.createElement("button", {
      onClick: () => {
        setLoading(!loading);
      },
      key: 'loading'
    }, loading ? "Set Not Loading" : "Set Loading"));
    children.push(React.createElement("button", {
      onClick: () => {
        setLoaded(!loaded);
      },
      key: 'loaded'
    }, loaded ? "Set Not Loaded" : "Set Loaded"));
    if (loading) {
      children.push(React.createElement("input", {
        type: 'range',
        min: '0',
        max: '1',
        value: loadingProgress,
        step: 'any',
        onChange: (e: any) => {
          setLoadingProgress(e.target.value);
        },
        key: 'loadingProgress'
      }));
    }

    children.push(React.createElement(NimbleStandaloneReact, {
      loading,
      loadingProgress,
      recording: loaded ? rawArray : null,
      style: {
        width: "800px",
        height: "500px"
      },
      key: 'gui',
      playing: playing,
      onPlayPause: (play) => {
        setPlaying(play);
      },
      frame: frame,
      onFrameChange: (frame) => {
        setFrame(frame);
      }
    }));
  }

  return React.createElement(
    "div",
    {
    },
    children
  );
};

const container = document.createElement("div");
document.body.appendChild(container);
ReactDOM.render(React.createElement(ReactTestBed), container);