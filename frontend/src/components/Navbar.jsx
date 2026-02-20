import { Clapperboard, Cpu, Zap } from 'lucide-react'

export default function Navbar() {
  return (
    <nav className="w-full border-b border-white/5 backdrop-blur-md bg-dark-900/60 sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <Clapperboard className="w-7 h-7 text-neon-cyan" />
            <div className="absolute inset-0 w-7 h-7 bg-neon-cyan/20 blur-lg rounded-full" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              <span className="neon-text">Content</span>
              <span className="text-white">Creator</span>
            </h1>
          </div>
        </div>

        {/* Badge */}
        <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400">
          <Cpu className="w-3.5 h-3.5" />
          <span>Local AI</span>
          <span className="text-slate-600">•</span>
          <Zap className="w-3.5 h-3.5 text-neon-cyan" />
          <span>GPU Accelerated</span>
        </div>
      </div>
    </nav>
  )
}
