import { LoaderIcon } from "@/icons";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive border border-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "text-foreground border bg-background hover:bg-accent dark:bg-input/30 dark:border-input dark:hover:bg-input/50",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost:
          "text-foreground hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",
        muted: "bg-mute text-mute-foreground hover:text-mute",
      },
      size: {
        default:
          "h-8 px-4 py-2 has-[>svg]:px-3 [&_svg:not([class*='size-'])]:size-5",
        sm: "h-7 rounded-lg gap-1.5 px-3 has-[>svg]:px-2.5 [&_svg:not([class*='size-'])]:size-4",
        lg: "h-10 rounded-lg px-6 has-[>svg]:px-4 [&_svg:not([class*='size-'])]:size-5",
        xl: "h-12 rounded-lg px-8 has-[>svg]:px-6 [&_svg:not([class*='size-'])]:size-5 text-lg rounded-xl",
      },
      isIcon: {
        true: "",
        false: "",
      },
    },
    compoundVariants: [
      {
        size: "default",
        isIcon: true,
        className:
          "size-8 @max-xs/panel:size-7 rounded-lg [&_svg:not([class*='size-'])]:size-5 @max-xs/panel:[&_svg:not([class*='size-'])]:size-4",
      },
      {
        size: "sm",
        isIcon: true,
        className:
          "size-7 @max-xs/panel:size-6 rounded-lg [&_svg:not([class*='size-'])]:size-4 @max-xs/panel:[&_svg:not([class*='size-'])]:size-3",
      },
      {
        size: "lg",
        isIcon: true,
        className:
          "size-10 @max-xs/panel:size-9 rounded-lg [&_svg:not([class*='size-'])]:size-5 @max-xs/panel:[&_svg:not([class*='size-'])]:size-4",
      },
      {
        size: "xl",
        isIcon: true,
        className:
          "size-12 @max-xs/panel:size-11 rounded-xl [&_svg:not([class*='size-'])]:size-5 @max-xs/panel:[&_svg:not([class*='size-'])]:size-4",
      },
    ],
    defaultVariants: {
      variant: "default",
      size: "default",
      isIcon: false,
    },
  },
);

function Button({
  className,
  variant,
  size,
  isIcon,
  isLoading = false,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
    isLoading?: boolean;
  }) {
  const Comp = asChild ? Slot : "button";

  if (isLoading) {
    return (
      <Comp
        data-slot="button"
        className={cn(buttonVariants({ variant, size, isIcon, className }))}
        {...props}
        disabled
      >
        <LoaderIcon className="animate-spin" />
        {props.children}
      </Comp>
    );
  }

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, isIcon, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
