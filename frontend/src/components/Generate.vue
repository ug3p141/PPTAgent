<template>
  <div class="generate-container">
    <p class="task-id">Task ID: {{ taskId }}</p>
    <progress :value="progress" max="100" class="progress-bar"></progress>
    <p class="status-message">{{ statusMessage }}</p>
    <a v-if="downloadLink" :href="downloadLink" download="pptagent.pptx"
      class="download-link">Download PPTX</a>
  </div>
</template>

<script>
export default {
  name: 'GenerateComponent',
  data() {
    return {
      progress: 0,
      statusMessage: 'Starting...',
      downloadLink: '',
      taskId: history.state.taskId,
      socketUrl: `ws://${this.$axios.defaults.baseURL.replace('http://', '')}/ws/${history.state.taskId}`
    }
  },
  created() {
    this.startGeneration()
  },
  methods: {
    async startGeneration() {
      console.log("Connecting to websocket", this.socketUrl)
      const socket = new WebSocket(this.socketUrl)

      socket.onmessage = (event) => {
        console.log("Socket Received message:", event.data)
        const data = JSON.parse(event.data)
        this.progress = data.progress
        this.statusMessage = data.status
        if (data.progress >= 100) {
          socket.close()
          this.fetchDownloadLink()
        }
      }
      socket.onerror = (error) => {
        console.error("WebSocket error:", error)
        this.statusMessage = 'WebSocket connection lost.'
      }
    },
    async fetchDownloadLink() {
      try {
        const downloadResponse = await this.$axios.get('/api/download', { params: { task_id: this.taskId }, responseType: 'blob' })
        this.downloadLink = URL.createObjectURL(downloadResponse.data)
      } catch (error) {
        console.error("Download error:", error)
        this.statusMessage += '\nFailed to continue the task.'
      }
    }
  }
}
</script>

<style scoped>
.generate-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

.progress-bar {
  width: 100%;
  height: 20px;
  margin-bottom: 10px;
  appearance: none;
  background-color: #f3f3f3;
}
.task-id {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #df8638;
}

.progress-bar::-webkit-progress-bar {
  background-color: #f3f3f3;
}

.progress-bar::-webkit-progress-value {
  background-color: #42b983;
}

.status-message {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #333;
}

.download-link {
  font-size: 1em;
  color: #42b983;
  text-decoration: none;
  border: 1px solid #42b983;
  padding: 8px 16px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.download-link:hover {
  background-color: #42b983;
  color: #fff;
}
</style>