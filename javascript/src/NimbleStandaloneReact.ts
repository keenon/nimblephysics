import React, { useRef, useEffect, useCallback } from "react";
import NimbleStandalone from "./NimbleStandalone";

type NimbleStandaloneReactProps = {
  loading: boolean;
  loadingProgress: number;
  recording: any;
  style?: any;
  className?: any;
}

const NimbleStandaloneReact: ((props: NimbleStandaloneReactProps) => React.ReactElement) = (props: NimbleStandaloneReactProps) => {
  // This is responsible for calling the imperitive methods on the GUI to reflect what's currently going on in the props.
  const setPropsOnStandalone = (gui: null | NimbleStandalone, pr: NimbleStandaloneReactProps) => {
    if (gui != null) {
      if (pr.loading) {
        gui.setLoadingProgress(pr.loadingProgress);
      }
      else {
        gui.hideLoadingBar();
        if (pr.recording != null) {
          gui.setRecording(pr.recording);
        }
      }
    }
  };

  let standalone = useRef(null as NimbleStandalone | null);
  let standaloneNode = useRef(null as HTMLDivElement | null);
  let standaloneRef = useCallback((node?: HTMLDivElement) => {
    // Only call the update when the node ref actually changes out from under us
    if (node != null && standaloneNode.current !== node) {
      console.log("New NODE object created!");
      console.log(node);
      standaloneNode.current = node;
      // Clean up the old standalone GUI, if we still had one
      if (standalone.current != null) {
        standalone.current.dispose();
        standalone.current = null;
      }
      // Create the standalone GUI
      if (node != null) {
        let newStandalone = new NimbleStandalone(node);
        setPropsOnStandalone(newStandalone, props);
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
    setPropsOnStandalone(standalone.current, props);
  }, [props.loading, props.loadingProgress, props.recording]);

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
