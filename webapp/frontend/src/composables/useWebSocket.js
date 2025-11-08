import { ref, onUnmounted } from 'vue'

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

export function useWebSocket(jobId) {
  const isConnected = ref(false)
  const progress = ref(null)
  const error = ref(null)

  let ws = null
  let reconnectTimeout = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 5

  const connect = () => {
    if (!jobId) return

    try {
      ws = new WebSocket(`${WS_BASE_URL}/api/ws/${jobId}`)

      ws.onopen = () => {
        console.log(`WebSocket connected for job ${jobId}`)
        isConnected.value = true
        reconnectAttempts = 0
        error.value = null
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          progress.value = data
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        error.value = 'Connection error'
      }

      ws.onclose = () => {
        console.log('WebSocket closed')
        isConnected.value = false

        // Attempt to reconnect if not max attempts
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000)
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`)

          reconnectTimeout = setTimeout(() => {
            connect()
          }, delay)
        } else {
          error.value = 'Failed to connect after multiple attempts'
        }
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      error.value = 'Failed to establish connection'
    }
  }

  const disconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }

    if (ws) {
      ws.close()
      ws = null
    }

    isConnected.value = false
  }

  // Auto-connect on creation
  connect()

  // Clean up on unmount
  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    progress,
    error,
    connect,
    disconnect,
  }
}
