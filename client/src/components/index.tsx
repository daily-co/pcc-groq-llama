import { lazy, Suspense } from "react";
import type { PresentationProps } from "./Presentation";

const Presentation = lazy(() =>
  import("./Presentation").then((module) => ({ default: module.Presentation })),
);

const PresentationLazy = (props: PresentationProps) => {
  return (
    <div className="pipecat-ui w-full h-screen" id="pipecat-ai">
      <Suspense fallback={null}>
        <Presentation {...props} />
      </Suspense>
    </div>
  );
};

export { PresentationLazy as Presentation };
