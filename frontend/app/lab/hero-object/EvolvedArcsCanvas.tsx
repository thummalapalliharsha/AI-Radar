"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import type { ReactNode } from "react";
import { Component, useMemo, useRef } from "react";
import { MathUtils, TorusGeometry, Vector3 } from "three";

type ArcLayer = {
  name: string;
  label: string;
  radius: number;
  spanDegrees: number;
  tubeRadius: number;
  depth: number;
  color: string;
  materialColor: string;
};

const arcLayers: ArcLayer[] = [
  { name: "Inner arc", label: "Now", radius: 1, spanDegrees: 220, tubeRadius: 0.075, depth: -0.12, color: "#E8FBFA", materialColor: "#BFEFED" },
  { name: "Middle arc", label: "24h", radius: 1.35, spanDegrees: 230, tubeRadius: 0.085, depth: 0, color: "#DDF7F8", materialColor: "#A9DDE9" },
  { name: "Outer arc", label: "7d", radius: 1.7, spanDegrees: 240, tubeRadius: 0.095, depth: 0.12, color: "#CFEAF5", materialColor: "#9DBFE8" },
];

const radialSegments = 40;
const tubularSegments = 192;
const openingRotation = MathUtils.degToRad(-25);
const rotationAxis = new Vector3(0, 0, 1);
const labelDirections = [
  { degrees: 35, anchor: "left" },
  { degrees: 60, anchor: "left" },
  { degrees: 145, anchor: "right" },
] as const;

function EvolvedArc({ arc }: { arc: ArcLayer }) {
  const geometry = useMemo(
    () => new TorusGeometry(arc.radius, arc.tubeRadius, radialSegments, tubularSegments, MathUtils.degToRad(arc.spanDegrees)),
    [arc],
  );

  return (
    <mesh
      name={arc.name}
      geometry={geometry}
      position={[0, 0, arc.depth]}
      rotation={[0, 0, openingRotation]}
    >
      <meshPhysicalMaterial
        color={arc.materialColor}
        transmission={0.8}
        roughness={0.07}
        thickness={arc.tubeRadius}
        ior={1.38}
        metalness={0}
      />
    </mesh>
  );
}

function EvolvedArcs({ scale }: { scale: number }) {
  return (
    <group name="Evolved Arcs" scale={scale}>
      {arcLayers.map((arc) => <EvolvedArc key={arc.name} arc={arc} />)}
    </group>
  );
}

function LabelProjection({ labels, scale }: { labels: React.RefObject<Array<HTMLSpanElement | null>>; scale: number }) {
  const endpoint = useMemo(() => new Vector3(), []);
  const tubePoint = useMemo(() => new Vector3(), []);

  useFrame(({ camera, size }) => {
    const clearance = size.width <= 400 ? 8.5 : 14.5;

    arcLayers.forEach((arc, index) => {
      const label = labels.current[index];
      if (!label) return;

      const angle = MathUtils.degToRad(arc.spanDegrees);
      endpoint
        .set(arc.radius * Math.cos(angle), arc.radius * Math.sin(angle), arc.depth)
        .applyAxisAngle(rotationAxis, openingRotation)
        .multiplyScalar(scale);
      const endpointNdc = endpoint.clone().project(camera);
      let projectedTubeRadius = 0;

      for (let segment = 0; segment <= radialSegments; segment += 1) {
        const tubeAngle = (segment / radialSegments) * Math.PI * 2;
        tubePoint
          .set(
            (arc.radius + arc.tubeRadius * Math.cos(tubeAngle)) * Math.cos(angle),
            (arc.radius + arc.tubeRadius * Math.cos(tubeAngle)) * Math.sin(angle),
            arc.depth + arc.tubeRadius * Math.sin(tubeAngle),
          )
          .applyAxisAngle(rotationAxis, openingRotation)
          .multiplyScalar(scale);
        const tubeNdc = tubePoint.project(camera);
        const tubeX = (tubeNdc.x - endpointNdc.x) * size.width * 0.5;
        const tubeY = -(tubeNdc.y - endpointNdc.y) * size.height * 0.5;
        projectedTubeRadius = Math.max(projectedTubeRadius, Math.hypot(tubeX, tubeY));
      }

      const direction = labelDirections[index];
      const directionRadians = MathUtils.degToRad(direction.degrees);
      const offset = projectedTubeRadius + clearance;
      const endpointX = (endpointNdc.x * 0.5 + 0.5) * size.width;
      const endpointY = (0.5 - endpointNdc.y * 0.5) * size.height;

      label.style.left = `${endpointX + Math.cos(directionRadians) * offset}px`;
      label.style.top = `${endpointY + Math.sin(directionRadians) * offset}px`;
      label.style.transform = direction.anchor === "right" ? "translateX(-100%)" : "none";
    });
  });

  return null;
}

function Scene({
  interactive,
  showLabels,
  transparent,
  labels,
  objectScale,
}: {
  interactive: boolean;
  showLabels: boolean;
  transparent: boolean;
  labels: React.RefObject<Array<HTMLSpanElement | null>>;
  objectScale: number;
}) {
  return (
    <>
      {!transparent && <color attach="background" args={["#05070B"]} />}
      <hemisphereLight args={["#F1FBFB", "#17232B", 0.28]} />
      <directionalLight position={[-3, 4, 5]} intensity={3} color="#F6FFFF" />
      <directionalLight position={[3, 1.5, 4]} intensity={0.5} color="#E8F7F8" />
      <pointLight position={[-2.8, 2.8, 1.5]} intensity={4.2} color="#F1FBFB" />
      <pointLight position={[2.8, -1.5, 2.4]} intensity={2.7} color="#DFEFF2" />
      <EvolvedArcs scale={objectScale} />
      {showLabels && <LabelProjection labels={labels} scale={objectScale} />}
      {interactive && (
        <OrbitControls
          makeDefault
          autoRotate={false}
          enableDamping={false}
          enableRotate
          enableZoom
          enablePan={false}
          target={[0, 0, 0]}
          minDistance={3.4}
          maxDistance={8}
        />
      )}
    </>
  );
}

type EvolvedArcsCanvasProps = {
  interactive?: boolean;
  showLabels?: boolean;
  transparent?: boolean;
  fallback?: ReactNode;
  pointerEvents?: "auto" | "none";
  objectScale?: number;
};

class CanvasErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("Evolved arcs canvas failed to render.", error);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? <div role="alert">The 3D visual could not be rendered.</div>;
    }
    return this.props.children;
  }
}

export function EvolvedArcsCanvas({
  interactive = true,
  showLabels = false,
  transparent = false,
  fallback,
  pointerEvents,
  objectScale = 1,
}: EvolvedArcsCanvasProps) {
  const labels = useRef<Array<HTMLSpanElement | null>>([]);

  return (
    <CanvasErrorBoundary fallback={fallback}>
      <div style={{ position: "relative", width: "100%", height: "100%", pointerEvents }}>
        <Canvas
          frameloop="demand"
          dpr={[1, 2]}
          camera={{ position: [0.5, 0.4, 6.2], fov: 70, near: 0.1, far: 100 }}
          gl={{ antialias: true, alpha: transparent }}
          onCreated={({ gl }) => {
            gl.toneMappingExposure = 1.15;
          }}
          fallback={fallback}
          style={{ display: "block", width: "100%", height: "100%", pointerEvents }}
        >
          <Scene
            interactive={interactive}
            showLabels={showLabels}
            transparent={transparent}
            labels={labels}
            objectScale={objectScale}
          />
        </Canvas>
        {showLabels && (
          <div aria-hidden="true" style={{ position: "absolute", inset: 0, pointerEvents: "none", userSelect: "none" }}>
            {arcLayers.map((arc, index) => (
              <span
                key={`${arc.name}-label`}
                ref={(element) => {
                  labels.current[index] = element;
                }}
                style={{
                  position: "absolute",
                  color: arc.color,
                  fontFamily: "var(--font-ui), sans-serif",
                  fontSize: "0.8rem",
                  letterSpacing: "0.08em",
                  lineHeight: 1,
                  whiteSpace: "nowrap",
                }}
              >
                {arc.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </CanvasErrorBoundary>
  );
}