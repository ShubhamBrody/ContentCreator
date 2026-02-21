import { useState } from 'react'
import Navbar from './components/Navbar'
import CreateView from './components/CreateView'
import ProgressScreen from './components/ProgressScreen'
import VideoResult from './components/VideoResult'
import usePipeline from './hooks/usePipeline'

export default function App() {
  const pipeline = usePipeline()

  return (
    <div className="min-h-screen flex flex-col relative">
      {/* Decorative orbs */}
      <div className="orb w-72 h-72 bg-neon-cyan/5 top-[-5%] left-[-5%]" />
      <div className="orb w-96 h-96 bg-neon-purple/5 bottom-[10%] right-[-10%]" style={{ animationDelay: '3s' }} />

      <Navbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-16 relative z-10">
        {pipeline.view === 'create' && (
          <CreateView
            onGenerate={pipeline.generate}
            onResume={pipeline.resume}
            resumable={pipeline.resumable}
            isSubmitting={pipeline.isSubmitting}
            error={pipeline.error}
          />
        )}

        {pipeline.view === 'progress' && (
          <ProgressScreen
            stages={pipeline.stages}
            progress={pipeline.progress}
            message={pipeline.message}
            stageTimings={pipeline.stageTimings}
            totalElapsed={pipeline.totalElapsed}
            totalEta={pipeline.totalEta}
          />
        )}

        {pipeline.view === 'result' && (
          <VideoResult
            jobId={pipeline.jobId}
            onReset={pipeline.reset}
          />
        )}
      </main>
    </div>
  )
}
