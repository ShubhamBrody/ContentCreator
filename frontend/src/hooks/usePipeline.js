import { useState, useCallback, useRef } from 'react'

const STAGE_ORDER = [
  'script_parse',
  'character_design',
  'tts',
  'image_gen',
  'video_gen',
  'music_gen',
  'subtitles',
  'assemble',
]

export default function usePipeline() {
  const [view, setView] = useState('create')       // 'create' | 'progress' | 'result'
  const [jobId, setJobId] = useState(null)
  const [stages, setStages] = useState({})
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const [error, setError] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const eventSourceRef = useRef(null)

  const generate = useCallback(async (formData) => {
    setIsSubmitting(true)
    setError(null)

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Server error (${response.status})`)
      }

      const { job_id } = await response.json()
      setJobId(job_id)
      setView('progress')
      setIsSubmitting(false)

      // Open SSE stream
      const es = new EventSource(`/api/progress/${job_id}`)
      eventSourceRef.current = es

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setStages(data.stages || {})
          setProgress(data.progress || 0)
          setMessage(data.message || '')

          if (data.status === 'completed') {
            setView('result')
            es.close()
          }

          if (data.status === 'failed') {
            setError(data.error || 'Generation failed')
            setView('create')
            es.close()
          }
        } catch {
          // ignore parse errors
        }
      }

      es.onerror = () => {
        // Retry a few times, then give up
        setTimeout(() => {
          if (es.readyState === EventSource.CLOSED) {
            setError('Connection lost. Check if the server is running.')
            setView('create')
          }
        }, 3000)
      }

    } catch (err) {
      setError(err.message)
      setIsSubmitting(false)
    }
  }, [])

  const reset = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }
    setView('create')
    setJobId(null)
    setStages({})
    setProgress(0)
    setMessage('')
    setError(null)
    setIsSubmitting(false)
  }, [])

  return {
    view,
    jobId,
    stages,
    progress,
    message,
    error,
    isSubmitting,
    generate,
    reset,
  }
}
