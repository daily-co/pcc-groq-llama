import PlasmaShader, {
  type PlasmaConfig,
  type PlasmaShaderRef,
} from "@/components/Plasma";
import { useRTVIClientMediaTrack } from "@pipecat-ai/client-react";
import { useEffect, useRef } from "react";

const defaultConfig: PlasmaConfig = {
  effectScale: 0.55,
  effectCenter: { x: 0, y: 0 },
  ringDistance: 0,
  ringBounce: 0.0,
  ringVariance: 0.35,
  ringAmplitude: 0.03,
  ringVisibility: 0.32,
  ringSegments: 6,
  ringThickness: 8,
  ringSpread: 0.08,
  colorCycleSpeed: 0.5,
  intensity: 1.95,
  radius: 1.65,
  glowFalloff: 1,
  glowThreshold: 0,
  plasmaSpeed: 0.3,
  rayLength: 1,
  color1: "#F55036",
  color2: "#fdba74",
  color3: "#22d3ee",
};

const thinkingConfig: PlasmaConfig = {
  ringDistance: 0.05,
  ringBounce: 0.25,
  ringVariance: 0.0,
  ringAmplitude: 0,
  ringVisibility: 0.3,
  ringThickness: 18,
  colorCycleSpeed: 3,
  intensity: 2,
  radius: 2.0,
  glowThreshold: 0.0,
  glowFalloff: 0.5,
  plasmaSpeed: 3,
  rayLength: 1,
};

const connectedConfig: PlasmaConfig = {
  ...defaultConfig,
  effectCenter: { x: 0, y: 0.1 },
};

export const AgentVisualization = ({
  state,
}: {
  state: "idle" | "connecting" | "connected" | "disconnected";
}) => {
  const audioTrack = useRTVIClientMediaTrack("audio", "bot");

  const shaderRef = useRef<PlasmaShaderRef>(null);

  useEffect(() => {
    switch (state) {
      case "connecting":
        shaderRef.current?.updateConfig(thinkingConfig);
        break;
      case "connected":
        shaderRef.current?.updateConfig(connectedConfig);
        break;
      default:
        shaderRef.current?.updateConfig(defaultConfig);
        break;
    }
  }, [state]);

  return (
    <PlasmaShader
      ref={shaderRef}
      initialConfig={defaultConfig}
      className="absolute inset-0 pointer-events-none animate-fade-in z-0"
      audioTrack={audioTrack}
    />
  );
};
