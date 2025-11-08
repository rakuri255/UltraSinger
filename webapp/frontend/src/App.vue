<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
    <!-- Header -->
    <header class="glass border-b border-gray-700/50 sticky top-0 z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <svg class="h-10 w-10 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <div>
              <h1 class="text-2xl font-bold text-white">UltraSinger Web</h1>
              <p class="text-sm text-gray-400">AI-Powered Karaoke Generator</p>
            </div>
          </div>

          <div class="flex items-center gap-4">
            <a
              href="https://github.com/rakuri255/UltraSinger"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-400 hover:text-white transition-colors"
            >
              <svg class="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="grid lg:grid-cols-2 gap-8">
        <!-- Left Column: Upload Form -->
        <div>
          <UploadForm @job-created="handleJobCreated" />

          <!-- Info Cards -->
          <div class="mt-6 space-y-4">
            <div class="card p-4">
              <h3 class="font-bold mb-2 flex items-center gap-2">
                <svg class="h-5 w-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                How It Works
              </h3>
              <ol class="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                <li>Provide a YouTube URL or upload an audio file</li>
                <li>Select your language and quality preset</li>
                <li>AI separates vocals, transcribes lyrics, and detects pitch</li>
                <li>Download your ready-to-play UltraStar .txt file</li>
              </ol>
            </div>

            <div class="card p-4">
              <h3 class="font-bold mb-2 flex items-center gap-2">
                <svg class="h-5 w-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Tips for Best Results
              </h3>
              <ul class="text-sm text-gray-400 space-y-1 list-disc list-inside">
                <li>Use high-quality audio for better transcription</li>
                <li>Choose "Accurate" preset for professional results</li>
                <li>Processing time varies: 5-20 minutes per song</li>
                <li>GPU acceleration recommended for faster processing</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- Right Column: Job Queue -->
        <div>
          <JobQueue ref="jobQueueRef" />
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-gray-700/50 mt-12">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div class="text-center text-sm text-gray-400">
          <p>
            Built with
            <span class="text-red-500">♥</span>
            using
            <a href="https://github.com/rakuri255/UltraSinger" target="_blank" class="text-primary-400 hover:text-primary-300">
              UltraSinger
            </a>
          </p>
          <p class="mt-1">
            Powered by Whisper, Crepe, and Demucs AI models
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadForm from './components/UploadForm.vue'
import JobQueue from './components/JobQueue.vue'

const jobQueueRef = ref(null)

const handleJobCreated = (job) => {
  console.log('Job created:', job)
  // Refresh the job queue
  if (jobQueueRef.value) {
    jobQueueRef.value.refreshJobs()
  }
}
</script>
