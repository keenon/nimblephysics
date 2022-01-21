import NimbleStandalone from "./NimbleStandalone";
import NimbleStandaloneReact from "./NimbleStandaloneReact";
import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import previewJson from './data/preview.json';

const ReactTestBed = () => {
  const [show, setShow] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0.0);

  let children = [];
  children.push(React.createElement("button", {
    onClick: () => {
      setShow(!show);
    },
    key: 'show'
  }, show ? "Hide" : "Show"));

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
      recording: loaded ? previewJson : null,
      style: {
        width: "800px",
        height: "500px"
      },
      key: 'gui'
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