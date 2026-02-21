import { useState } from 'react'
import {
  Sparkles, MonitorPlay, Smartphone, Film,
  Mic, Music, MessageSquare, Palette,
  ChevronDown, ChevronUp, Plus, Trash2,
  AlertCircle, Wand2, RotateCcw, Clock,
} from 'lucide-react'

const PLATFORMS = [
  { id: 'reels', label: 'Reels', icon: Smartphone, desc: '9:16 vertical' },
  { id: 'shorts', label: 'Shorts', icon: Smartphone, desc: '9:16 vertical' },
  { id: 'youtube', label: 'YouTube', icon: MonitorPlay, desc: '16:9 landscape' },
]

const VOICES = [
  { id: 'en-US-GuyNeural', label: 'Male (US)' },
  { id: 'en-US-JennyNeural', label: 'Female (US)' },
  { id: 'en-US-AriaNeural', label: 'Narrator' },
  { id: 'en-GB-RyanNeural', label: 'Male (UK)' },
  { id: 'en-GB-SoniaNeural', label: 'Female (UK)' },
]

const QUALITIES = [
  { id: 'standard', label: 'Standard', desc: 'Fast, good quality' },
  { id: 'high', label: 'High', desc: 'Slower, refined output' },
  { id: 'ultra', label: 'Ultra', desc: 'Best quality, slowest' },
]

const CHAR_STYLES = [
  { id: 'stick_figure', label: 'Stick Figure' },
  { id: 'cartoon', label: 'Cartoon' },
  { id: 'manga', label: 'Manga' },
  { id: 'doodle', label: 'Doodle' },
  { id: 'whiteboard', label: 'Whiteboard' },
]

function Toggle({ enabled, onChange, label }) {
  return (
    <label className="flex items-center justify-between cursor-pointer group">
      <span className="text-sm text-slate-300 group-hover:text-white transition-colors">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        onClick={() => onChange(!enabled)}
        className={`toggle ${enabled ? 'active' : ''}`}
      >
        <span className="toggle-knob" />
      </button>
    </label>
  )
}

export default function CreateView({ onGenerate, onResume, resumable = [], isSubmitting, error }) {
  const [script, setScript] = useState('')
  const [platform, setPlatform] = useState('reels')
  const [numScenes, setNumScenes] = useState(7)
  const [voice, setVoice] = useState('en-US-GuyNeural')
  const [quality, setQuality] = useState('standard')
  const [musicEnabled, setMusicEnabled] = useState(true)
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true)
  const [characterStyle, setCharacterStyle] = useState('cartoon')
  const [characters, setCharacters] = useState([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  const addCharacter = () => {
    setCharacters([...characters, { name: '', description: '', traits: '' }])
  }
  const removeCharacter = (idx) => {
    setCharacters(characters.filter((_, i) => i !== idx))
  }
  const updateCharacter = (idx, field, value) => {
    const updated = [...characters]
    updated[idx] = { ...updated[idx], [field]: value }
    setCharacters(updated)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!script.trim()) return

    const fd = new FormData()
    fd.append('script', script)
    fd.append('platform', platform)
    fd.append('num_scenes', numScenes)
    fd.append('voice', voice)
    fd.append('quality', quality)
    fd.append('music_enabled', musicEnabled)
    fd.append('subtitles_enabled', subtitlesEnabled)
    fd.append('character_style', characterStyle)

    if (characters.length > 0) {
      const cleaned = characters
        .filter(c => c.name.trim())
        .map(c => ({
          name: c.name.trim(),
          description: c.description.trim(),
          traits: c.traits ? c.traits.split(',').map(t => t.trim()).filter(Boolean) : [],
        }))
      if (cleaned.length > 0) {
        fd.append('characters', JSON.stringify(cleaned))
      }
    }

    onGenerate(fd)
  }

  return (
    <form onSubmit={handleSubmit} className="mt-8 sm:mt-12 animate-fade-in space-y-6">
      {/* Hero text */}
      <div className="text-center mb-8">
        <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight">
          <span className="neon-text">Create</span>{' '}
          <span className="text-white">AI Videos</span>
        </h2>
        <p className="mt-3 text-slate-400 max-w-xl mx-auto text-sm sm:text-base">
          Describe your idea, tune the settings, and let local AI models craft your video.
          No cloud, no cost — powered by your GPU.
        </p>
      </div>

      {/* Error banner */}
      {error && (
        <div className="glass border-red-500/30 !border flex items-center gap-3 p-4 rounded-xl animate-slide-up">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Resume banner — shown when interrupted runs are available */}
      {resumable.length > 0 && (
        <div className="glass gradient-border p-1 rounded-2xl animate-slide-up">
          <div className="bg-dark-900/90 rounded-[0.9rem] p-4 sm:p-5 space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-amber-400 uppercase tracking-wider">
              <RotateCcw className="w-3.5 h-3.5" />
              Interrupted Runs
            </div>
            {resumable.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center gap-4 bg-white/[0.03] border border-white/5 rounded-xl p-3 sm:p-4"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white font-medium truncate">
                    {item.params?.script?.slice(0, 60) || item.project_name}
                    {(item.params?.script?.length || 0) > 60 ? '...' : ''}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-0.5 flex items-center gap-2">
                    <Clock className="w-3 h-3 inline" />
                    {item.updated_at ? new Date(item.updated_at).toLocaleString() : 'Unknown time'}
                    <span className="text-slate-600">·</span>
                    <span className="text-emerald-400/70">
                      {item.completed_stages?.length || 0}/{item.active_stages?.length || 0} stages done
                    </span>
                    <span className="text-slate-600">·</span>
                    <span className="text-neon-cyan/70">
                      {item.remaining_stages?.length || 0} remaining
                    </span>
                  </p>
                </div>
                <button
                  type="button"
                  disabled={isSubmitting}
                  onClick={() => onResume(item.project_dir)}
                  className="shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-lg
                    border border-amber-500/40 bg-amber-500/10 text-amber-400
                    hover:bg-amber-500/20 hover:border-amber-500/60
                    transition-all text-xs font-semibold disabled:opacity-40"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Resume
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Script Input */}
      <div className="glass gradient-border p-1 rounded-2xl">
        <div className="bg-dark-900/80 rounded-[0.9rem] p-4 sm:p-6">
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            <Wand2 className="w-3.5 h-3.5 inline mr-1.5 -mt-0.5" />
            Script / Video Idea
          </label>
          <textarea
            value={script}
            onChange={(e) => setScript(e.target.value)}
            rows={5}
            required
            placeholder="E.g., &quot;5 tips for better sleep — explain each with calming visuals, soothing narration, and smooth transitions...&quot;"
            className="w-full bg-transparent border-none outline-none resize-none text-white placeholder-slate-600 text-base sm:text-lg leading-relaxed"
          />
          <div className="flex justify-between items-center mt-2 text-xs text-slate-600">
            <span>{script.length} characters</span>
            <span>The more detail, the better the result</span>
          </div>
        </div>
      </div>

      {/* Platform & Quality */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Platform */}
        <div className="glass p-5 rounded-xl space-y-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Film className="w-3.5 h-3.5" /> Platform
          </label>
          <div className="flex gap-2">
            {PLATFORMS.map(p => {
              const Icon = p.icon
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setPlatform(p.id)}
                  className={`flex-1 flex flex-col items-center gap-1 py-3 px-2 rounded-lg border text-xs font-medium transition-all duration-200 ${
                    platform === p.id
                      ? 'border-neon-cyan/50 bg-neon-cyan/10 text-neon-cyan shadow-[0_0_15px_rgba(0,229,255,0.1)]'
                      : 'border-white/5 text-slate-400 hover:border-white/10 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {p.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Quality */}
        <div className="glass p-5 rounded-xl space-y-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" /> Quality
          </label>
          <div className="flex gap-2">
            {QUALITIES.map(q => (
              <button
                key={q.id}
                type="button"
                onClick={() => setQuality(q.id)}
                className={`flex-1 py-3 px-2 rounded-lg border text-xs font-medium transition-all duration-200 ${
                  quality === q.id
                    ? 'border-neon-purple/50 bg-neon-purple/10 text-neon-purple shadow-[0_0_15px_rgba(124,58,237,0.1)]'
                    : 'border-white/5 text-slate-400 hover:border-white/10 hover:text-white'
                }`}
              >
                {q.label}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-slate-600">{QUALITIES.find(q => q.id === quality)?.desc}</p>
        </div>
      </div>

      {/* Scenes & Voice */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Scene count */}
        <div className="glass p-5 rounded-xl space-y-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Scenes: <span className="text-white font-bold text-sm">{numScenes}</span>
          </label>
          <input
            type="range"
            min={3}
            max={20}
            value={numScenes}
            onChange={(e) => setNumScenes(Number(e.target.value))}
            className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-neon-cyan
              [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,229,255,0.5)] [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-600">
            <span>Short (3)</span><span>Long (20)</span>
          </div>
        </div>

        {/* Voice */}
        <div className="glass p-5 rounded-xl space-y-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Mic className="w-3.5 h-3.5" /> Voice
          </label>
          <div className="relative">
            <select
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-white appearance-none cursor-pointer outline-none focus:border-neon-cyan/40 transition-colors"
            >
              {VOICES.map(v => (
                <option key={v.id} value={v.id} className="bg-dark-800">{v.label}</option>
              ))}
            </select>
            <ChevronDown className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Toggles row */}
      <div className="glass p-5 rounded-xl">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Toggle label="Background Music" enabled={musicEnabled} onChange={setMusicEnabled} icon={Music} />
          <Toggle label="Subtitles / Captions" enabled={subtitlesEnabled} onChange={setSubtitlesEnabled} icon={MessageSquare} />
        </div>
      </div>

      {/* Advanced: Characters */}
      <div className="glass rounded-xl overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between p-5 text-left hover:bg-white/[0.02] transition-colors"
        >
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Palette className="w-3.5 h-3.5" /> Characters & Style
          </span>
          {showAdvanced ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </button>

        {showAdvanced && (
          <div className="px-5 pb-5 space-y-4 animate-slide-up">
            {/* Style select */}
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Character Art Style</label>
              <div className="flex flex-wrap gap-2">
                {CHAR_STYLES.map(s => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setCharacterStyle(s.id)}
                    className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                      characterStyle === s.id
                        ? 'border-neon-pink/50 bg-neon-pink/10 text-neon-pink'
                        : 'border-white/5 text-slate-400 hover:border-white/10'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Character cards */}
            {characters.map((char, idx) => (
              <div key={idx} className="bg-white/[0.03] rounded-lg p-4 space-y-3 border border-white/5 animate-fade-in">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400">Character {idx + 1}</span>
                  <button type="button" onClick={() => removeCharacter(idx)} className="text-slate-500 hover:text-red-400 transition-colors">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <input
                  type="text"
                  placeholder="Name (e.g., Alex)"
                  value={char.name}
                  onChange={(e) => updateCharacter(idx, 'name', e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 outline-none focus:border-neon-cyan/30 transition-colors"
                />
                <textarea
                  placeholder="Description (e.g., tall man with glasses and a beard)"
                  value={char.description}
                  onChange={(e) => updateCharacter(idx, 'description', e.target.value)}
                  rows={2}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 outline-none resize-none focus:border-neon-cyan/30 transition-colors"
                />
                <input
                  type="text"
                  placeholder="Traits (comma-separated: wears hat, curly hair)"
                  value={char.traits}
                  onChange={(e) => updateCharacter(idx, 'traits', e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 outline-none focus:border-neon-cyan/30 transition-colors"
                />
              </div>
            ))}

            <button
              type="button"
              onClick={addCharacter}
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-neon-cyan transition-colors"
            >
              <Plus className="w-4 h-4" /> Add Character
            </button>
          </div>
        )}
      </div>

      {/* Generate button */}
      <div className="pt-2 flex justify-center">
        <button
          type="submit"
          disabled={!script.trim() || isSubmitting}
          className="btn-glow text-base sm:text-lg px-12 py-4 rounded-2xl flex items-center gap-3"
        >
          {isSubmitting ? (
            <>
              <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Starting...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              Generate Video
            </>
          )}
        </button>
      </div>
    </form>
  )
}
