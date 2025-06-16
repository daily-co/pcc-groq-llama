import * as React from "react";

import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";

function Panel({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel"
      className={cn(
        "@container/panel bg-card text-card-foreground flex flex-col rounded-panel border",
        className,
      )}
      {...props}
    />
  );
}

const PanelHeaderVariants = cva("@container/panel-header", {
  variants: {
    variant: {
      default:
        "border-b flex items-center justify-center h-panel-header text-card-foreground p-2 @xs/panel:p-3 @md/panel:p-4",
      inline: "items-start text-foreground p-2 @xs/panel:p-3 @md/panel:p-4",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

function PanelHeader({
  variant,
  className,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof PanelHeaderVariants>) {
  return (
    <div
      data-slot="panel-header"
      className={cn(PanelHeaderVariants({ variant, className }), className)}
      {...props}
    />
  );
}

function PanelTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel-title"
      className={cn("text-mono-upper", className)}
      {...props}
    />
  );
}

function PanelContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel-content"
      className={cn(
        "text-foreground flex flex-col p-2 gap-2 @xs/panel:p-3 @xs/panel:gap-3 @md/panel:p-4 @md/panel:gap-4",
        className,
      )}
      {...props}
    />
  );
}

function PanelFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel-footer"
      className={cn(
        "flex items-center p-2 @xs/panel:p-3 @md/panel:p-4",
        className,
      )}
      {...props}
    />
  );
}

function PanelActions({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="panel-actions"
      className={cn("flex items-center gap-1 @xs/panel:gap-2", className)}
      {...props}
    />
  );
}

export {
  Panel,
  PanelActions,
  PanelContent,
  PanelFooter,
  PanelHeader,
  PanelTitle,
};
