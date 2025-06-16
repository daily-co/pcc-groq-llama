import { Presentation } from "@/components";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

//@ts-expect-error - fontsource-variable/geist is not typed
import "@fontsource-variable/geist";
//@ts-expect-error - fontsource-variable/geist is not typed
import "@fontsource-variable/geist-mono";

import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="w-full h-screen bg-zinc-50">
      <Presentation
        // Fake a server-side endpoint here that hits the PCC API and returns the auth bundle
        onConnect={async () => {
          const response = await fetch(import.meta.env.VITE_PCC_API_URL, {
            method: "POST",
            mode: "cors",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${import.meta.env.VITE_PCC_API_KEY}`,
            },
            body: JSON.stringify({
              createDailyRoom: true,
            }),
          });

          if (!response.ok) {
            throw new Error("Failed to connect to Pipecat");
          }
          const data = await response.json();
          if (data.error) {
            throw new Error(data.error);
          }

          return new Response(
            JSON.stringify({
              room_url: data.dailyRoom,
              token: data.dailyToken,
            }),
            { status: 200 },
          );
        }}
      />
    </div>
  </StrictMode>,
);
