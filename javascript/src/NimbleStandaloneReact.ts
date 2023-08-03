import React, { useRef, useEffect, useCallback } from "react";
import NimbleStandalone from "./NimbleStandalone";

type NimbleStandaloneReactProps = {
  loadUrl: string;
  style?: any;
  className?: any;
  // Making the controls accessible from the mounted component
  playing?: boolean;
  onPlayPause?: (playing: boolean) => void;
  frame?: number;
  onFrameChange?: (frame: number) => void;
}

const NimbleStandaloneReact: ((props: NimbleStandaloneReactProps) => React.ReactElement) = (props: NimbleStandaloneReactProps) => {
  // This is responsible for calling the imperitive methods on the GUI to reflect what's currently going on in the props.
  const setLoadingPropsOnStandalone = (gui: null | NimbleStandalone, pr: NimbleStandaloneReactProps) => {
    if (gui != null) {
      gui.loadRecording(pr.loadUrl);
    }
  };

  let standalone = useRef(null as NimbleStandalone | null);
  let standaloneNode = useRef(null as HTMLDivElement | null);
  let standaloneRef = useCallback((node?: HTMLDivElement) => {
    // Only call the update when the node ref actually changes out from under us
    if (node != null && standaloneNode.current !== node) {
      standaloneNode.current = node;
      // Clean up the old standalone GUI, if we still had one
      if (standalone.current != null) {
        standalone.current.dispose();
        standalone.current = null;
      }
      // Create the standalone GUI
      if (node != null) {
        let newStandalone = new NimbleStandalone(node);
        setLoadingPropsOnStandalone(newStandalone, props);
        // This doesn't cause a re-render
        standalone.current = newStandalone;
      }
    }
  }, []);

  // Clean up on dismount
  useEffect(() => {
    return () => {
      if (standalone.current != null) {
        console.log("Cleaning up the NimbleStandalone...");
        standalone.current.dispose();
      }
    };
  }, []);

  // Handle the props changes
  useEffect(() => {
    setLoadingPropsOnStandalone(standalone.current, props);
  }, [props.loadUrl]);

  useEffect(() => {
    const pr = props;
    const gui = standalone.current;
    if (gui != null) {
      if (pr.playing != null && pr.playing != gui.getPlaying()) {
        gui.setPlaying(pr.playing);
      }
      gui.registerPlayPauseListener(pr.onPlayPause);
      gui.registerFrameChangeListener(pr.onFrameChange);
    }
  }, [props.playing, props.onPlayPause, props.onFrameChange]);

  useEffect(() => {
    const pr = props;
    const gui = standalone.current;
    if (gui != null) {
      if (pr.frame != null && pr.frame != gui.getFrame()) {
        gui.setFrame(pr.frame);
      }
    }
  }, [props.frame]);

  return React.createElement(
    "div",
    {
      style: props.style,
      className: props.className,
      ref: standaloneRef,
    },
    ""
  );
};

export default NimbleStandaloneReact;
