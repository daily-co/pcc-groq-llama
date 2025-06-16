"use client";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/buttongroup";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownIcon, MicIcon, MicOffIcon } from "@/icons";
import {
  RTVIClientMicToggle,
  useRTVIClient,
  useRTVIClientMediaDevices,
  VoiceVisualizer,
} from "@pipecat-ai/client-react";
import { memo, useEffect } from "react";

const UserAudio: React.FC = () => {
  const client = useRTVIClient();
  const { availableMics, selectedMic, updateMic } = useRTVIClientMediaDevices();

  // @ts-expect-error _options is protected, but can be totally accessed in JS
  const hasAudio = client?._options?.enableMic;

  useEffect(() => {
    if (!client) return;

    if (["idle", "disconnected"].includes(client.state)) {
      client.initDevices();
    }
  }, [client]);

  if (!hasAudio) {
    return (
      <div className="flex items-center gap-2 bg-muted rounded-md p-2 text-muted-foreground font-mono text-sm">
        <MicOffIcon size={16} />
        Audio disabled
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <ButtonGroup className="w-full">
        <RTVIClientMicToggle>
          {({ isMicEnabled, onClick }) => (
            <Button
              onClick={onClick}
              variant={isMicEnabled ? "outline" : "destructive"}
              size="xl"
            >
              {isMicEnabled ? <MicIcon /> : <MicOffIcon />}
              <VoiceVisualizer
                participantType="local"
                backgroundColor="transparent"
                barColor={isMicEnabled ? "#00BC7D" : "#FFFFFF"}
                barCount={8}
                barWidth={3}
                barMaxHeight={38}
                barGap={4}
                barOrigin="center"
              />
            </Button>
          )}
        </RTVIClientMicToggle>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              className="border-s border-border p-2! flex-none"
              variant="outline"
              isIcon
              size="xl"
            >
              <ChevronDownIcon size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            {availableMics.map((mic) => (
              <DropdownMenuCheckboxItem
                key={mic.deviceId}
                checked={selectedMic?.deviceId === mic.deviceId}
                onCheckedChange={() => updateMic(mic.deviceId)}
              >
                {mic.label || `Mic ${mic.deviceId.slice(0, 5)}`}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </ButtonGroup>
    </div>
  );
};

export default memo(UserAudio);
