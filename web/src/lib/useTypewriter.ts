"use client";

import { useEffect, useState } from "react";

/**
 * Reveals `total` characters over time, like a streamed reply.
 *
 * The API returns the whole answer at once (no server streaming in this
 * project), so this is purely presentational: it never changes what is shown,
 * only when. Speed adapts so long answers still finish in about 2.5 s, and it
 * jumps straight to the end when the user prefers reduced motion.
 */
export function useTypewriter(total: number, active: boolean): { shown: number; done: boolean } {
  const [shown, setShown] = useState(active ? 0 : total);

  useEffect(() => {
    if (!active) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const id = requestAnimationFrame(() => setShown(total));
      return () => cancelAnimationFrame(id);
    }
    // ~60 chars/s minimum, scaled up so nothing takes longer than ~150 frames.
    const step = Math.max(1, Math.ceil(total / 150));
    let frame = 0;
    let count = 0;
    const tick = () => {
      count = Math.min(total, count + step);
      setShown(count);
      if (count < total) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [total, active]);

  return { shown, done: shown >= total };
}
