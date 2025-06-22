import { cn } from "@/lib/utils";

function LayoutSection({
  key,
  className,
  ...props
}: React.ComponentProps<"section"> & { key: string }) {
  return <section data-section={key} className={cn(className)} {...props} />;
}

export { LayoutSection };
