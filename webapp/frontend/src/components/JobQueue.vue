<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-bold">Job Queue</h2>
      <button @click="refreshJobs" class="btn btn-secondary" :disabled="isLoading">
        <svg class="w-5 h-5" :class="{ 'animate-spin': isLoading }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>

    <!-- Loading State -->
    <div v-if="isLoading && jobs.length === 0" class="space-y-4">
      <div v-for="i in 3" :key="i" class="card p-6 animate-shimmer"></div>
    </div>

    <!-- Empty State -->
    <div v-else-if="jobs.length === 0" class="card p-12 text-center">
      <svg class="mx-auto h-16 w-16 text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <h3 class="text-xl font-bold mb-2">No jobs yet</h3>
      <p class="text-gray-400">Create your first karaoke file using the form above!</p>
    </div>

    <!-- Jobs List -->
    <div v-else class="space-y-4">
      <ProgressCard
        v-for="job in jobs"
        :key="job.job_id"
        :job="job"
        @cancel="handleCancel"
        @delete="handleDelete"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ProgressCard from './ProgressCard.vue'
import { listJobs, cancelJob, deleteJob } from '@/services/api'

const jobs = ref([])
const isLoading = ref(false)
let refreshInterval = null

const refreshJobs = async () => {
  isLoading.value = true
  try {
    const response = await listJobs()
    jobs.value = response.jobs
  } catch (error) {
    console.error('Failed to load jobs:', error)
  } finally {
    isLoading.value = false
  }
}

const handleCancel = async (jobId) => {
  try {
    await cancelJob(jobId)
    await refreshJobs()
  } catch (error) {
    console.error('Failed to cancel job:', error)
    alert('Failed to cancel job')
  }
}

const handleDelete = async (jobId) => {
  if (!confirm('Are you sure you want to delete this job?')) {
    return
  }

  try {
    await deleteJob(jobId)
    await refreshJobs()
  } catch (error) {
    console.error('Failed to delete job:', error)
    alert('Failed to delete job')
  }
}

onMounted(() => {
  refreshJobs()
  // Auto-refresh every 5 seconds
  refreshInterval = setInterval(refreshJobs, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})

defineExpose({
  refreshJobs,
})
</script>
