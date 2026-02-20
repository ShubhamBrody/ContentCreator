import {
  FileText, Palette, Mic, Image, Film,
  Music, MessageSquare, Clapperboard,
  Check, Loader2, Circle, SkipForward,
  AlertTriangle, Clock,
} from 'lucide-react'

const STAGE_META = {
  script_parse:     { label: 'Script Parsing',       Icon: FileText },
  character_design: { label: 'Character Design',     Icon: Palette },
  tts:              { label: 'Voice Generation',     Icon: Mic },
  image_gen:        { label: 'Image Generation',     Icon: Image },
  video_gen:        { label: 'Video Generation',     Icon: Film },
  music_gen:        { label: 'Music Generation',     Icon: Music },
  subtitles:        { label: 'Subtitle Generation',  Icon: MessageSquare },
  assemble:         { label: 'Final Assembly',       Icon: Clapperboard },
}

const STAGE_ORDER = [
  'script_parse', 'character_design', 'tts', 'image_gen',
  'video_gen', 'music_gen', 'subtitles', 'assemble',
]

/** Format seconds into human-readable string like "1m 23s" or "45s" */
function formatTime(seconds) {
  if (!seconds || seconds <= 0) return null
  const s = Math.round(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  return rem > 0 ? `${m}m ${rem}s` : `${m}m`
}

function StatusIcon({ status }) {
  if (status === 'completed') return <Check className="w-4 h-4 text-emerald-400" />
  if (status === 'running')   return <Loader2 className="w-4 h-4 text-neon-cyan animate-spin" />
  if (status === 'skipped')   return <SkipForward className="w-3.5 h-3.5 text-slate-600" />
  if (status === 'failed')    return <AlertTriangle className="w-4 h-4 text-red-400" />
  return <Circle className="w-3 h-3 text-slate-600" />
}

/** Per-stage timing badge */
function TimingBadge({ status, timing }) {
  if (!timing) return null

  const elapsed = formatTime(timing.elapsed)
  const eta = formatTime(timing.eta)

  if (status === 'completed' && elapsed) {
    return (
      <span className="text-[10px] font-mono text-emerald-400/70 bg-emerald-500/10 px-1.5 py-0.5 rounded">
        {elapsed}
      </span>
    )
  }

  if (status === 'running') {
    return (
      <span className="flex items-center gap-1 text-[10px] font-mono text-neon-cyan/70 bg-neon-cyan/[0.06] px-1.5 py-0.5 rounded">
        <Clock className="w-2.5 h-2.5" />
        {elapsed || '0s'}
        {eta ? <span className="text-slate-500 ml-0.5">/ ~{eta}</span> : null}
      </span>
    )
  }

  // Pending — show estimate
  if (eta) {
    return (
      <span className="text-[10px] font-mono text-slate-600 px-1.5 py-0.5">
        ~{eta}
      </span>
    )
  }

  return null
}

export default function ProgressScreen({
  stages, progress, message, stageTimings, totalElapsed, totalEta,
}) {
  return (
    <div className="mt-16 sm:mt-24 flex flex-col items-center animate-fade-in">
      {/* Title */}
      <h2 className="text-2xl sm:text-3xl font-bold text-center mb-2">
        <span className="neon-text">Creating</span>{' '}
        <span className="text-white">Your Video</span>
      </h2>
      <p className="text-slate-400 text-sm mb-10">{message || 'Preparing pipeline...'}</p>

      {/* Stages list */}
      <div className="w-full max-w-lg glass gradient-border p-1 rounded-2xl">
        <div className="bg-dark-900/80 rounded-[0.9rem] p-6 space-y-1 stagger-children">
          {STAGE_ORDER.map((stageId) => {
            const status = stages[stageId] || 'pending'
            if (status === 'skipped') return null
            const meta = STAGE_META[stageId] || { label: stageId, Icon: Circle }
            const { Icon } = meta
            const timing = stageTimings?.[stageId]

            return (
              <div
                key={stageId}
                className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-500 animate-fade-in ${
                  status === 'running'
                    ? 'bg-neon-cyan/[0.06] border border-neon-cyan/10'
                    : status === 'completed'
                    ? 'bg-emerald-500/[0.04]'
                    : ''
                }`}
              >
                {/* Stage icon */}
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-all duration-500 ${
                  status === 'running'
                    ? 'bg-neon-cyan/10 text-neon-cyan shadow-[0_0_12px_rgba(0,229,255,0.15)]'
                    : status === 'completed'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-white/5 text-slate-500'
                }`}>
                  <Icon className="w-4 h-4" />
                </div>

                {/* Label */}
                <span className={`flex-1 text-sm font-medium transition-colors duration-300 ${
                  status === 'running' ? 'text-white' :
                  status === 'completed' ? 'text-slate-300' :
                  'text-slate-500'
                }`}>
                  {meta.label}
                </span>

                {/* Timing badge */}
                <TimingBadge status={status} timing={timing} />

                {/* Status */}
                <StatusIcon status={status} />
              </div>
            )
          })}
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-lg mt-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-500">Overall Progress</span>
          <div className="flex items-center gap-3">
            {totalElapsed > 0 && (
              <span className="text-[10px] font-mono text-slate-500">
                {formatTime(totalElapsed)}
                {totalEta > 0 ? ` / ~${formatTime(totalElapsed + totalEta)} total` : ''}
              </span>
            )}
            <span className="text-xs font-mono text-neon-cyan">{Math.round(progress)}%</span>
          </div>
        </div>
        <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #00e5ff, #7c3aed, #ec4899)',
              backgroundSize: '300% 100%',
              animation: 'progress-bar 2s ease-in-out infinite',
            }}
          />
        </div>
      </div>

      {/* Ambient animation */}
      <div className="mt-16 relative w-24 h-24">
        <div className="absolute inset-0 rounded-full bg-neon-cyan/10 animate-ping" style={{ animationDuration: '3s' }} />
        <div className="absolute inset-2 rounded-full bg-neon-purple/10 animate-ping" style={{ animationDuration: '3s', animationDelay: '1s' }} />
        <div className="absolute inset-4 rounded-full bg-neon-pink/10 animate-ping" style={{ animationDuration: '3s', animationDelay: '2s' }} />
        <div className="absolute inset-0 flex items-center justify-center">
          <Clapperboard className="w-8 h-8 text-neon-cyan/40 animate-float" />
        </div>
      </div>
    </div>
  )
}
