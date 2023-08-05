import NimbleStandaloneReact from "./NimbleStandaloneReact";
import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom';

const embedNimbleVisualizer = (div: HTMLElement, rawURL: string, defaultPlaybackSpeed: number = 1.0) => {
  console.log("Embedding Nimble Visualizer for "+rawURL+" into page.");
  console.log(div);
  // Clear the pre-existing contents of the div
  div.innerHTML = "";

  const Embedded = () => {
    const [frame, setFrame] = useState(0);
    const [playing, setPlaying] = useState(false);
    
    const [sizeX, setSizeX] = useState(div.getBoundingClientRect().width);
    const [sizeY, setSizeY] = useState(div.getBoundingClientRect().height);

    const onWindowResize = () => {
      setSizeX(div.getBoundingClientRect().width);
      setSizeY(div.getBoundingClientRect().height);
    };

    useEffect(() => {
      window.addEventListener('resize', onWindowResize);
      return () => {
        window.removeEventListener('resize', onWindowResize);
      }
    });

    let children = [];
    children.push(React.createElement(NimbleStandaloneReact, {
      loadUrl: rawURL,
      style: {
        width: sizeX+"px",
        height: sizeY+"px"
      },
      key: 'gui',
      playing: playing,
      onPlayPause: (play) => {
        setPlaying(play);
      },
      frame: frame,
      onFrameChange: (frame) => {
        setFrame(frame);
      },
      defaultPlaybackSpeed: defaultPlaybackSpeed
    }));

    return React.createElement(
      "div",
      {
      },
      children
    );
  };
  ReactDOM.render(React.createElement(Embedded), div);
}

if (window != null) {
  (window as any).embedNimbleVisualizer = embedNimbleVisualizer;
}

export default embedNimbleVisualizer;