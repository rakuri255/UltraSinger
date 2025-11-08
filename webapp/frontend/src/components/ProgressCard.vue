<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold truncate flex-1">
        {{ job.title || 'Processing...' }}
      </h3>
      <span
        :class="[
          'px-3 py-1 rounded-full text-xs font-bold',
          statusColors[job.status]
        ]"
      >
        {{ job.status.toUpperCase() }}
      </span>
    </div>

    <!-- Progress Bar -->
    <div v-if="job.status === 'processing' || job.status === 'queued'" class="mb-4">
      <div class="flex justify-between text-sm mb-2">
        <span class="text-gray-400">{{ currentStepMessage }}</span>
        <span class="text-primary-400 font-bold">{{ progressPercentage }}%</span>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
        <div
          class="bg-gradient-to-r from-primary-500 to-primary-400 h-full transition-all duration-300 ease-out"
          :style="{ width: progressPercentage + '%' }"
        >
          <div class="w-full h-full animate-pulse-slow"></div>
        </div>
      </div>
    </div>

    <!-- Step Indicators -->
    <div v-if="job.status === 'processing'" class="mb-4">
      <div class="space-y-2">
        <div
          v-for="step in steps"
          :key="step.key"
          class="flex items-center gap-3 text-sm"
        >
          <!-- Icon -->
          <div
            :class="[
              'w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0',
              getStepStatus(step.key) === 'completed' ? 'bg-green-500' :
              getStepStatus(step.key) === 'active' ? 'bg-primary-500 animate-pulse' :
              'bg-gray-700'
            ]"
          >
            <svg
              v-if="getStepStatus(step.key) === 'completed'"
              class="w-4 h-4 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            <svg
              v-else-if="getStepStatus(step.key) === 'active'"
              class="w-4 h-4 text-white animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <div v-else class="w-2 h-2 bg-gray-500 rounded-full"></div>
          </div>

          <!-- Label -->
          <span
            :class="[
              getStepStatus(step.key) === 'completed' ? 'text-green-400' :
              getStepStatus(step.key) === 'active' ? 'text-primary-400 font-medium' :
              'text-gray-500'
            ]"
          >
            {{ step.label }}
          </span>
        </div>
      </div>
    </div>

    <!-- Success State -->
    <div v-if="job.status === 'completed'" class="mb-4">
      <div class="flex items-center gap-3 text-green-400 mb-3">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span class="font-bold">Processing completed successfully!</span>
      </div>
      <a
        :href="downloadUrl"
        download
        class="btn btn-primary w-full"
      >
        <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        Download UltraStar File
      </a>
    </div>

    <!-- Error State -->
    <div v-if="job.status === 'failed'" class="mb-4">
      <div class="flex items-center gap-3 text-red-400 mb-2">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span class="font-bold">Processing failed</span>
      </div>
      <p class="text-sm text-gray-400 mb-3">{{ job.error_message || 'Unknown error occurred' }}</p>
    </div>

    <!-- Metadata -->
    <div class="grid grid-cols-2 gap-4 text-sm border-t border-gray-700 pt-4">
      <div>
        <span class="text-gray-400">Language:</span>
        <span class="ml-2 font-medium">{{ languageNames[job.language] }}</span>
      </div>
      <div>
        <span class="text-gray-400">Quality:</span>
        <span class="ml-2 font-medium capitalize">{{ job.quality }}</span>
      </div>
      <div>
        <span class="text-gray-400">Source:</span>
        <span class="ml-2 font-medium capitalize">{{ job.source }}</span>
      </div>
      <div v-if="elapsedTime">
        <span class="text-gray-400">Time:</span>
        <span class="ml-2 font-medium">{{ elapsedTime }}</span>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex gap-2 mt-4">
      <button
        v-if="job.status === 'processing' || job.status === 'queued'"
        @click="$emit('cancel', job.job_id)"
        class="btn btn-secondary flex-1"
      >
        Cancel
      </button>
      <button
        @click="$emit('delete', job.job_id)"
        class="btn bg-red-600 hover:bg-red-700 text-white flex-1"
      >
        Delete
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useWebSocket } from '@/composables/useWebSocket'
import { downloadResult } from '@/services/api'

const props = defineProps({
  job: {
    type: Object,
    required: true,
  },
})

defineEmits(['cancel', 'delete'])

const { progress } = useWebSocket(props.job.job_id)

const statusColors = {
  queued: 'bg-yellow-500/20 text-yellow-400',
  processing: 'bg-blue-500/20 text-blue-400',
  completed: 'bg-green-500/20 text-green-400',
  failed: 'bg-red-500/20 text-red-400',
  cancelled: 'bg-gray-500/20 text-gray-400',
}

const languageNames = {
  it: 'Italian',
  en: 'English',
  pl: 'Polish',
}

const steps = [
  { key: 'downloading', label: 'Downloading audio' },
  { key: 'separating', label: 'Separating vocals' },
  { key: 'transcribing', label: 'Transcribing lyrics' },
  { key: 'pitching', label: 'Detecting pitch' },
  { key: 'generating', label: 'Generating file' },
]

const currentStepMessage = computed(() => {
  if (progress.value?.message) {
    return progress.value.message
  }
  if (props.job.progress?.message) {
    return props.job.progress.message
  }
  return 'Initializing...'
})

const progressPercentage = computed(() => {
  if (progress.value?.percentage !== undefined) {
    return Math.round(progress.value.percentage)
  }
  if (props.job.progress?.percentage !== undefined) {
    return Math.round(props.job.progress.percentage)
  }
  return 0
})

const getStepStatus = (stepKey) => {
  const currentStep = progress.value?.step || props.job.progress?.step
  const stepIndex = steps.findIndex(s => s.key === stepKey)
  const currentIndex = steps.findIndex(s => s.key === currentStep)

  if (currentIndex === -1) return 'pending'
  if (stepIndex < currentIndex) return 'completed'
  if (stepIndex === currentIndex) return 'active'
  return 'pending'
}

const elapsedTime = computed(() => {
  const seconds = progress.value?.elapsed_seconds || props.job.progress?.elapsed_seconds
  if (!seconds) return null

  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
})

const downloadUrl = computed(() => {
  return downloadResult(props.job.job_id)
})
</script>
