"use client";

import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import { MathUtils, Vector3 } from "three";
import type { JSX } from "react";

function GlassMaterial({ color = "#A9E8EA", transmission = 0.78, roughness = 0.1 }: { color?: string; transmission?: number; roughness?: number }) {
  return <meshPhysicalMaterial color={color} transmission={transmission} roughness={roughness} thickness={0.18} ior={1.38} metalness={0.02} />;
}

function LayeredCore() {
  return (
    <group rotation={[0.12, -0.18, 0.18]}>
      <mesh scale={[1.35, 1.35, 0.42]}>
        <icosahedronGeometry args={[1, 2]} />
        <GlassMaterial color="#83D9DF" transmission={0.86} roughness={0.08} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0.18, 0.1]} scale={[1.05, 1.05, 0.34]}>
        <torusGeometry args={[0.82, 0.08, 24, 96]} />
        <GlassMaterial color="#D5F7F3" transmission={0.72} roughness={0.12} />
      </mesh>
      <mesh scale={[0.42, 0.42, 0.42]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#4DE7E1" emissive="#194C55" emissiveIntensity={0.65} roughness={0.3} />
      </mesh>
    </group>
  );
}

function SignalArchitecture() {
  const bands = useMemo(() => [
    { radius: 0.72, tube: 0.055, rotation: [0.1, 0.1, 0.2] as [number, number, number], color: "#DDF7F8" },
    { radius: 1.05, tube: 0.065, rotation: [0.3, -0.2, -0.45] as [number, number, number], color: "#A9DDE9" },
    { radius: 1.4, tube: 0.075, rotation: [-0.18, 0.25, 0.25] as [number, number, number], color: "#9DBFE8" },
  ], []);

  return (
    <group rotation={[0.2, -0.1, 0.12]}>
      {bands.map((band, index) => (
        <mesh key={band.radius} rotation={band.rotation} position={[0, 0, (index - 1) * 0.14]}>
          <torusGeometry args={[band.radius, band.tube, 24, 128, MathUtils.degToRad(230 + index * 12)]} />
          <GlassMaterial color={band.color} transmission={0.82} roughness={0.09} />
        </mesh>
      ))}
      <mesh scale={[0.32, 0.32, 0.32]}>
        <sphereGeometry args={[1, 24, 16]} />
        <meshStandardMaterial color="#EAFDFC" emissive="#236E75" emissiveIntensity={0.55} roughness={0.25} />
      </mesh>
    </group>
  );
}

function ModularCore() {
  const modules = useMemo(() => [
    new Vector3(-0.82, 0.45, 0.06),
    new Vector3(0.82, 0.45, 0.06),
    new Vector3(-0.82, -0.45, 0.06),
    new Vector3(0.82, -0.45, 0.06),
  ], []);

  return (
    <group rotation={[0.18, -0.26, 0.12]}>
      <mesh scale={[0.44, 0.44, 0.62]}>
        <cylinderGeometry args={[0.55, 0.65, 1.5, 32]} />
        <meshStandardMaterial color="#B9C3CE" metalness={0.58} roughness={0.24} />
      </mesh>
      {modules.map((position, index) => (
        <mesh key={index} position={position} rotation={[0.12, 0.2, index % 2 ? -0.16 : 0.16]}>
          <boxGeometry args={[0.52, 0.52, 0.32]} />
          <meshPhysicalMaterial color={index % 2 ? "#A9DDE9" : "#CBEFED"} transmission={0.45} roughness={0.18} metalness={0.18} />
        </mesh>
      ))}
      <mesh position={[0, 0, 0.54]} scale={[0.2, 0.2, 0.2]}>
        <sphereGeometry args={[1, 20, 12]} />
        <meshStandardMaterial color="#4DE7E1" emissive="#1C5F67" emissiveIntensity={0.6} roughness={0.28} />
      </mesh>
    </group>
  );
}

function ConceptScene({ concept }: { concept: 0 | 1 | 2 }) {
  return (
    <>
      <color attach="background" args={["#0A0F17"]} />
      <ambientLight intensity={0.28} color="#D7F7F6" />
      <directionalLight position={[-3, 4, 5]} intensity={2.5} color="#F5FFFF" />
      <directionalLight position={[3, -1, 3]} intensity={0.65} color="#9CCFFF" />
      <pointLight position={[-2, 1.6, 2]} intensity={2.5} color="#4DE7E1" />
      {concept === 0 ? <LayeredCore /> : concept === 1 ? <SignalArchitecture /> : <ModularCore />}
    </>
  );
}

export function HeroConceptsCanvas({ concept }: { concept: 0 | 1 | 2 }): JSX.Element {
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.5]}
      camera={{ position: [0, 0, 4.8], fov: 42, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: false }}
    >
      <ConceptScene concept={concept} />
    </Canvas>
  );
}
