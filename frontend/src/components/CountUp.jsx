import { useMotionValue, useSpring } from 'motion/react';
import { useEffect, useRef } from 'react';

// Adapted from ReactBits CountUp (https://reactbits.dev) — triggers on mount/value change
export default function CountUp({ to, from = 0, delay = 0, duration = 0.9, className = '' }) {
  const ref = useRef(null);
  const motionValue = useMotionValue(from);
  const damping = 20 + 40 * (1 / duration);
  const stiffness = 100 * (1 / duration);
  const springValue = useSpring(motionValue, { damping, stiffness });

  // Kick off animation after delay
  useEffect(() => {
    if (ref.current) ref.current.textContent = String(from);
    const t = setTimeout(() => motionValue.set(to), delay * 1000);
    return () => clearTimeout(t);
  }, [to, from, delay, motionValue]);

  // Write spring output to DOM
  useEffect(() => {
    return springValue.on('change', (v) => {
      if (ref.current) ref.current.textContent = String(Math.round(v));
    });
  }, [springValue]);

  return <span className={className} ref={ref} />;
}
