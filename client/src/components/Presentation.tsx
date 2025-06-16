import GroqLogoSVG from "@/assets/groq.svg";
import PipecatSVG from "@/components/PipecatSVG";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { XIcon } from "@/icons";
import {
  LLMHelper,
  RTVIClient,
  type RTVIClientParams,
  RTVIMessage,
  type TransportState,
} from "@pipecat-ai/client-js";
import { RTVIClientAudio, RTVIClientProvider } from "@pipecat-ai/client-react";
import { DailyTransport } from "@pipecat-ai/daily-transport";
import { useEffect, useRef, useState } from "react";
import { AgentVisualization } from "./AgentVisualization";
import { Controls } from "./Controls";

export interface PresentationProps {
  onConnect?: () => Promise<Response>;
}

export type AppState = "idle" | "connecting" | "connected" | "disconnected";

export const Presentation = ({ onConnect }: PresentationProps) => {
  const [client, setClient] = useState<RTVIClient | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<AppState>("idle");

  const isMounted = useRef(false);

  useEffect(() => {
    if (isMounted.current) return;

    isMounted.current = true;

    const transport = new DailyTransport();

    const client = new RTVIClient({
      transport,
      enableCam: false,
      enableMic: true,
      params: {
        baseUrl: "noop",
      },
      callbacks: {
        onTransportStateChanged: (state: TransportState) => {
          switch (state) {
            case "connecting":
            case "authenticating":
            case "connected":
              setState("connecting");
              break;
            case "ready":
              setState("connected");
              break;
            case "disconnected":
            case "disconnecting":
              setState("disconnected");
              break;
            default:
              setState("idle");
              break;
          }
        },
        onError: (message: RTVIMessage) => {
          setError(message.data as string);
        },
      },
      customConnectHandler: (async (_params, timeout) => {
        if (!onConnect) {
          return Promise.reject(new Error("No onConnect function provided"));
        }
        try {
          const response = await onConnect?.();
          clearTimeout(timeout);
          if (response.ok) {
            return response.json();
          }
          const errorData = await response.text();
          setError(`Connection failed: ${response.status} ${errorData}`);
          return Promise.reject(
            new Error(`Connection failed: ${response.status}`),
          );
        } catch (err) {
          setError(
            `Connection error: ${err instanceof Error ? err.message : String(err)}`,
          );
          return Promise.reject(err);
        }
      }) as (
        params: RTVIClientParams,
        timeout: NodeJS.Timeout | undefined,
        abortController: AbortController,
      ) => Promise<void>,
    });

    const llmHelper = new LLMHelper({});
    client.registerHelper("llm", llmHelper);

    setClient(client);
  }, [onConnect]);

  const handleEndSession = async () => {
    await client?.disconnect();
    setState("idle");
  };

  const handleStartSession = async () => {
    if (
      !client ||
      !["initialized", "disconnected", "error"].includes(client.state)
    ) {
      return;
    }
    setError(null);

    try {
      await client.connect();
    } catch (err) {
      console.error("Connection error:", err);
      setError(
        `Failed to start session: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  useEffect(() => {
    if (isMounted.current) {
      return;
    }
  }, []);

  if (error) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <Card className="shadow-long">
          <CardContent>
            <div className="bg-destructive text-background font-semibold text-center p-3 rounded-lg flex flex-col gap-2">
              An error occured connecting to agent.
              <p className="text-sm font-medium text-balanced text-background/80">
                It may be that the agent is at capacity. Please try again later.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
  return (
    <RTVIClientProvider client={client!}>
      <div className="w-full h-screen">
        <div className="flex flex-col h-full">
          <div className="relative bg-background overflow-hidden flex-1 shadow-long/[0.02]">
            <main className="flex flex-col gap-0 h-full relative justify-end items-center">
              <AgentVisualization state={state} />
              {["idle", "connecting", "disconnected"].includes(state) && (
                <div className="absolute w-full h-full flex items-center justify-center">
                  <Button
                    size="xl"
                    variant={state !== "idle" ? "secondary" : "default"}
                    onClick={handleStartSession}
                    disabled={state !== "idle"}
                    isLoading={state !== "idle"}
                  >
                    {state !== "idle" ? "Connecting..." : "Start session"}
                  </Button>
                </div>
              )}
              {state === "connected" && (
                <Controls onEndSession={handleEndSession} />
              )}
            </main>
          </div>
          <footer className="p-5 md:p-7">
            <div className="flex flex-row gap-3 items-center justify-center opacity-40">
              <a href="https://github.com/pipecat-ai" target="_blank">
                <PipecatSVG className="h-[24px] w-auto text-black" />
              </a>
              <XIcon className="size-5 text-subtle/80" />
              <a href="https://groq.com/" target="_blank">
                <img src={GroqLogoSVG} className="h-[24px] w-auto" />
              </a>
            </div>
          </footer>
        </div>
      </div>
      <RTVIClientAudio />
    </RTVIClientProvider>
  );
};

export default Presentation;
