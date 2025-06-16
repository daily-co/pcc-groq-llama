import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";

const buttonGroupVariants = cva("flex items-center *:rounded-none", {
  variants: {
    orientation: {
      horizontal:
        "flex-row *:first:rounded-s-xl *:last:rounded-e-xl *:-ml-[1px] *:first:ml-0",
      vertical:
        "flex-col *:first:rounded-t-xl *:last:rounded-b-xl *:-mt-[1px] *:first:mt-0",
    },
  },
  defaultVariants: {
    orientation: "horizontal",
  },
});

export const ButtonGroup = ({
  className,
  orientation = "horizontal",
  children,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof buttonGroupVariants>) => {
  return (
    <div
      className={cn("flex", buttonGroupVariants({ orientation }), className)}
      {...props}
    >
      {children}
    </div>
  );
};
