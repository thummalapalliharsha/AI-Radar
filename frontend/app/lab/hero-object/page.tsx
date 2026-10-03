"use client";

import { EvolvedArcsCanvas } from "./EvolvedArcsCanvas";

export default function HeroObjectLab() {
  return (
    <main
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100dvh",
        overflow: "hidden",
        background: "#05070B",
      }}
    >
      <EvolvedArcsCanvas interactive />
    </main>
  );
}