import { useState } from 'react'
import {
  Download, RotateCcw, Play, Pause,
  Volume2, VolumeX, Maximize, CheckCircle2,
  Film, Clock, Layers,
} from 'lucide-react'

export default function VideoResult({ jobId, onReset }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const videoUrl = `/api/video/${jobId}`

  const handleDownload = async () => {
    try {
      const a = document.createElement('a')
      a.href = videoUrl
      a.download = `contentcreator_${jobId}.mp4`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    } catch (err) {
      // Fallback: open in new tab
      window.open(videoUrl, '_blank')
    }
  }

  return (
    <div className="mt-10 sm:mt-16 animate-fade-in">
      {/* Success header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-4 animate-slide-up">
          <CheckCircle2 className="w-4 h-4" />
          Video Ready
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold">
          <span className="text-white">Your Video is </span>
          <span className="neon-text">Complete</span>
        </h2>
      </div>

      {/* Video player */}
      <div className="glass gradient-border p-1 rounded-2xl max-w-3xl mx-auto">
        <div className="bg-dark-900/80 rounded-[0.9rem] overflow-hidden">
          <div className="relative aspect-video bg-black/50">
            <video
              src={videoUrl}
              controls
              autoPlay={false}
              playsInline
              className="w-full h-full object-contain"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            >
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      </div>

      {/* Info row */}
      <div className="flex flex-wrap items-center justify-center gap-6 mt-6 text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <Film className="w-3.5 h-3.5 text-neon-cyan" />
          MP4 / H.264
        </span>
        <span className="flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-neon-purple" />
          HD Quality
        </span>
        <span className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-neon-pink" />
          Job: {jobId}
        </span>
      </div>

      {/* Action buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8">
        <button
          onClick={handleDownload}
          className="btn-glow text-base px-10 py-3.5 rounded-2xl flex items-center gap-2.5"
        >
          <Download className="w-5 h-5" />
          Download Video
        </button>

        <button
          onClick={onReset}
          className="flex items-center gap-2 px-8 py-3.5 rounded-2xl border border-white/10 text-slate-300 hover:text-white hover:border-white/20 transition-all text-base"
        >
          <RotateCcw className="w-4 h-4" />
          Create Another
        </button>
      </div>
    </div>
  )
}
