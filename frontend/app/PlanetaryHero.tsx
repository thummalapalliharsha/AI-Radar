"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { JSX } from "react";
import {
  SRGBColorSpace,
  TextureLoader,
  type Group,
  type Texture,
} from "three";

const rotationPeriodSeconds = 105;
const reducedMotionQuery = "(prefers-reduced-motion: reduce)";
// Earth surface/cloud maps: Solar System Scope's NASA Blue Marble-derived 2K textures.

function subscribeToMotionPreference(onChange: () => void) {
  const mediaQuery = window.matchMedia(reducedMotionQuery);
  mediaQuery.addEventListener("change", onChange);
  return () => mediaQuery.removeEventListener("change", onChange);
}

function getMotionPreference() {
  return window.matchMedia(reducedMotionQuery).matches;
}

function getServerMotionPreference() {
  return true;
}

function EarthSurface({ textures }: { textures: [Texture, Texture] }): JSX.Element {
  const earth = useRef<Group>(null);

  useFrame((_, delta) => {
    if (earth.current && !document.hidden) {
      earth.current.rotation.y += Math.min(delta, 0.1) * ((Math.PI * 2) / rotationPeriodSeconds);
    }
  });

  return (
    <group ref={earth} rotation={[-0.12, -0.52, 0.025]}>
      <mesh name="earth-surface">
        <sphereGeometry args={[1.42, 96, 64]} />
        <meshStandardMaterial
          map={textures[0]}
          roughness={0.92}
          metalness={0}
          transparent
          onBeforeCompile={(shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
              "#include <opaque_fragment>",
              `
                float viewFacing = abs(dot(normalize(normal), normalize(vViewPosition)));
                float atmosphereFresnel = pow(1.0 - viewFacing, 1.05);
                float atmosphereFacing = smoothstep(
                  -0.35,
                  0.75,
                  dot(normalize(normal), normalize(vec3(-0.48, 0.42, 0.77)))
                );
                vec3 atmosphereColor = mix(
                  vec3(0.42, 0.72, 0.94),
                  vec3(0.88, 0.7, 0.44),
                  atmosphereFacing
                );
                float atmosphereEdgeFade = smoothstep(0.28, 0.96, atmosphereFresnel);
                diffuseColor.a *= 1.0 - atmosphereEdgeFade * 0.94;
                outgoingLight += atmosphereColor * atmosphereFresnel * 0.006;
                #include <opaque_fragment>
              `,
            );
          }}
        />
      </mesh>
      <mesh name="earth-clouds" scale={1.006} renderOrder={1}>
        <sphereGeometry args={[1.42, 96, 64]} />
        <meshStandardMaterial
          color="#F2EEE5"
          alphaMap={textures[1]}
          transparent
          opacity={0.28}
          roughness={1}
          depthWrite={false}
          onBeforeCompile={(shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
              "#include <opaque_fragment>",
              `
                float cloudViewFacing = abs(dot(normalize(normal), normalize(vViewPosition)));
                float cloudEdgeFade = smoothstep(0.28, 0.96, 1.0 - cloudViewFacing);
                diffuseColor.a *= 1.0 - cloudEdgeFade * 0.9;
                #include <opaque_fragment>
              `,
            );
          }}
        />
      </mesh>
    </group>
  );
}

function PlanetScene({ textures }: { textures: [Texture, Texture] }): JSX.Element {
  return (
    <>
      <ambientLight intensity={0.85} color="#D5DEE3" />
      <directionalLight position={[-3.4, 2.3, 4]} intensity={2} color="#F2EEE5" />
      <directionalLight position={[-3, 2.4, -3]} intensity={1.7} color="#D9B879" />
      <directionalLight position={[3.5, -1.5, 2]} intensity={0.18} color="#B96F46" />
      <EarthSurface textures={textures} />
    </>
  );
}

export function PlanetaryHero(): JSX.Element {
  const prefersReducedMotion = useSyncExternalStore(
    subscribeToMotionPreference,
    getMotionPreference,
    getServerMotionPreference,
  );
  const visual = useRef<HTMLDivElement>(null);
  const loadedTextures = useRef<Texture[]>([]);
  const [textures, setTextures] = useState<[Texture, Texture] | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let failed = false;
    const loader = new TextureLoader();
    const loadTexture = (path: string) => new Promise<Texture>((resolve, reject) => {
      loader.load(
        path,
        (texture) => {
          if (cancelled || failed) {
            texture.dispose();
            resolve(texture);
            return;
          }
          loadedTextures.current.push(texture);
          resolve(texture);
        },
        undefined,
        reject,
      );
    });

    Promise.all([
      loadTexture("/textures/earth-daymap.jpg"),
      loadTexture("/textures/earth-clouds.jpg"),
    ]).then(([surface, clouds]) => {
      if (cancelled) {
        surface.dispose();
        clouds.dispose();
        return;
      }
      surface.colorSpace = SRGBColorSpace;
      surface.anisotropy = 4;
      clouds.anisotropy = 4;
      setTextures([surface, clouds]);
    }).catch((error: unknown) => {
      failed = true;
      loadedTextures.current.forEach((texture) => texture.dispose());
      loadedTextures.current = [];
      if (!cancelled) {
        console.error("Unable to load the Earth surface and cloud textures.", error);
        setLoadFailed(true);
      }
    });

    return () => {
      cancelled = true;
      loadedTextures.current.forEach((texture) => texture.dispose());
      loadedTextures.current = [];
    };
  }, []);

  useEffect(() => {
    const element = visual.current;
    if (!element) {
      return;
    }

    const observer = new IntersectionObserver(([entry]) => {
      setIsVisible(entry.isIntersecting);
    }, { rootMargin: "80px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className="planetary-visual"
      ref={visual}
      role="img"
      aria-label="Rotating Earth with blue oceans, natural continents, and white cloud systems"
      data-rotation-period-seconds={rotationPeriodSeconds}
    >
      {!loadFailed && (
        <Canvas
          className="planetary-canvas"
          frameloop={isVisible && !prefersReducedMotion ? "always" : "demand"}
          dpr={[1, 1.35]}
          camera={{ position: [0, 0, 4.25], fov: 40, near: 0.1, far: 100 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
          fallback={<div className="planetary-fallback" aria-hidden="true" />}
        >
          {textures && <PlanetScene textures={textures} />}
        </Canvas>
      )}
      {(!textures || loadFailed) && <div className="planetary-fallback" aria-hidden="true" />}
    </div>
  );
}
