import React, { useRef, useEffect } from 'react';

const LensingBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -1000, y: -1000 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    
    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    window.addEventListener('resize', resize);
    resize();

    // Star generation
    const stars: { x: number; y: number; size: number; brightness: number }[] = [];
    const starCount = 800;
    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 1.5 + 0.5,
        brightness: Math.random(),
      });
    }

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Animation Loop
    let animationFrameId: number;

    // Lensing parameters
    const thetaE = 120; // Einstein Radius in pixels

    const render = () => {
      ctx.fillStyle = '#030712'; // Deep space background
      ctx.fillRect(0, 0, width, height);

      // Draw grid lines (optional, faint) to show warping better
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      const gridSize = 100;
      
      // We will draw stars instead of a grid for better performance and aesthetics
      ctx.fillStyle = 'white';

      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      stars.forEach(star => {
        // Simple Point Mass Lens Equation approximation for visual effect
        // Beta = Theta - Alpha(Theta)
        // Here we act as if the star is at 'Beta' (source plane) and we want to find 'Theta' (image plane).
        // However, for a real-time UI effect, it's computationally cheaper to treat the star 
        // as the image plane pixel and displace it *away* from the lens center if we are "pushing" pixels,
        // OR move the star *towards* the source position.
        
        // Let's implement a "push" effect: Stars near the mouse get pushed outwards (Einstein ring effect)
        const dx = star.x - mx;
        const dy = star.y - my;
        const distSq = dx * dx + dy * dy;
        const dist = Math.sqrt(distSq);

        let finalX = star.x;
        let finalY = star.y;

        // Avoid division by zero
        if (dist > 10) {
           // Calculate deflection magnitude alpha = thetaE^2 / dist (SIS approximation for visual speed)
           // We displace the star OUTWARD from the center
           const deflection = (thetaE * thetaE) / dist;
           const factor = deflection / dist; // Normalize direction
           
           finalX = star.x + dx * factor * 0.8;
           finalY = star.y + dy * factor * 0.8;
        }

        const alpha = star.brightness * (Math.random() * 0.2 + 0.8);
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(finalX, finalY, star.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw the "Lens" (The Mouse)
      // Gradient for the Dark Matter Halo
      const gradient = ctx.createRadialGradient(mx, my, 0, mx, my, thetaE);
      gradient.addColorStop(0, 'rgba(0, 0, 0, 0.8)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(mx, my, thetaE, 0, Math.PI * 2);
      ctx.fill();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="fixed top-0 left-0 w-full h-full z-0 pointer-events-none opacity-60"
    />
  );
};

export default LensingBackground;