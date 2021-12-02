import React, { useEffect, useRef } from "react";
import NimbleStandalone from "./NimbleStandalone";

const NimbleStandaloneReact = (props: any) => {
  let [standalone, setStandalone] = useRef(null as null | NimbleStandalone);

  useEffect(() => {
    return () => {
      // Clean up
      // console.log("Cleaning up the NimbleStandalone...");
      if (standalone != null) {
        standalone.dispose();
      }
    };
  }, []);

  return React.createElement(
    "div",
    {
      ...props,
      ref: (r: HTMLDivElement) => {
        setStandalone(new NimbleStandalone(r));
        if (props.ref != null) {
          props.ref(r);
        }
      },
    },
    ""
  );
};

export default NimbleStandaloneReact;
