import UserAudio from "@/components/UserAudio";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { LogoutIcon } from "@/icons";

export const Controls = ({ onEndSession }: { onEndSession: () => void }) => {
  return (
    <div className="relative z-10 h-1/2 flex flex-col w-full items-center justify-center animate-fade-in-bottom">
      <Card className="shadow-long">
        <CardContent className="flex flex-row gap-4 p-4">
          <UserAudio />
          <div className="w-[1px] flex-1 bg-border/60 mx-2" />
          <Button onClick={onEndSession} size="xl" variant="outline" isIcon>
            <LogoutIcon className="text-destructive" />
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
