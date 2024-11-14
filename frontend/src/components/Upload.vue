<template>
  <!-- Upload form -->
  <div class="upload-container">
    <div class="upload-options">
      <!-- Row 1: Upload Buttons -->
      <div class="upload-buttons">
        <div class="upload-section">
          <label for="pptx-upload" class="upload-label">
            Upload PPTX
            <span v-if="pptxFile" class="uploaded-symbol">✔️</span>
          </label>
          <input type="file" id="pptx-upload" @change="handleFileUpload($event, 'pptx')"
            accept=".pptx" />
        </div>
        <div class="upload-section">
          <label for="pdf-upload" class="upload-label">
            Upload PDF
            <span v-if="pdfFile" class="uploaded-symbol">✔️</span>
          </label>
          <input type="file" id="pdf-upload" @change="handleFileUpload($event, 'pdf')"
            accept=".pdf" />
        </div>
      </div>

      <!-- Row 2: Selectors -->
      <div class="selectors">
        <div class="model-selection">
          <select v-model="selectedModel">
            <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
          </select>
        </div>
        <div class="pages-selection">
          <select v-model="selectedPages">
            <option v-for="page in pagesOptions" :key="page" :value="page">{{ page }} 页</option>
          </select>
        </div>
      </div>
    </div>

    <!-- New Topic Input -->
    <div class="topic-input">
      <input type="text" v-model="topic" placeholder="Enter topic" />
      <button @click="downloadPdf" class="download-button">Generate PDF</button>
    </div>

    <button @click="goToGenerate" class="next-button">Next</button>
  </div>
</template>

<script>
export default {
  name: 'UploadComponent',
  data() {
    return {
      pptxFile: null,
      pdfFile: null,
      selectedModel: 'Qwen2.5-72B-Instruct',
      models: ['Qwen2.5-72B-Instruct'],
      selectedPages: 4,
      pagesOptions: Array.from({ length: 24 }, (_, i) => i + 3),
      topic: 'Large Language Models'
    }
  },
  methods: {
    handleFileUpload(event, fileType) {
      console.log("file uploaded :", fileType)
      const file = event.target.files[0]
      if (fileType === 'pptx') {
        this.pptxFile = file
      } else if (fileType === 'pdf') {
        this.pdfFile = file
      }
    },
    async goToGenerate() {
      this.$axios.get('/')
        .then(response => {
          console.log("Backend is running", response.data);
        })
        .catch(error => {
          console.error(error);
          alert('Backend is not running or too busy, your task will not be processed');
          return;
        });
      if (!this.pptxFile || !this.pdfFile) {
        alert('Please upload both PPTX and PDF files.');
        return;
      }
      const formData = new FormData()
      formData.append('pptxFile', this.pptxFile)
      formData.append('pdfFile', this.pdfFile)
      formData.append('numberOfPages', this.selectedPages)

      try {
        const uploadResponse = await this.$axios.post('/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        const taskId = uploadResponse.data.task_id
        console.log("Task ID:", taskId)
        this.$router.push({ name: 'Generate', state: { taskId: taskId } })
      } catch (error) {
        console.error("Upload error:", error)
        this.statusMessage = 'Failed to upload files.'
      }
    },
    async downloadPdf() {
      if (!this.topic) {
        alert('Please enter a topic.');
        return;
      }
      try {
        const response = await this.$axios.get(`/api/get_pdf?topic=${encodeURIComponent(this.topic)}`, { responseType: 'blob' });
        const url = URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'topic.pdf');
        document.body.appendChild(link);
        link.click();
        link.remove();
      } catch (error) {
        console.error("Download error:", error);
        alert('Failed to download PDF.');
      }
    }
  }
}
</script>

<style scoped>
.upload-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  background-color: #f0f8ff;
  padding: 40px;
  box-sizing: border-box;
}

.upload-options {
  display: flex;
  flex-direction: column;
  gap: 30px;
  width: 100%;
  max-width: 600px;
}

.upload-buttons,
.selectors {
  display: flex;
  justify-content: space-between;
  gap: 20px;
}

.upload-section,
.model-selection,
.pages-selection {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.upload-section {
  margin-left: 2em;
  margin-right: 2em;
}

.model-selection,
.pages-selection {
  margin-left: 3em;
  margin-right: 3em;
}

.upload-label {
  position: relative;
  background-color: #42b983;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  width: 100%;
  text-align: center;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}

.upload-label:hover {
  background-color: #369870;
}

.upload-section input[type="file"] {
  display: none;
}

.model-selection select,
.pages-selection select {
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ccc;
  width: 100%;
  height: 40px;
  box-sizing: border-box;
  font-size: 16px;
}

.next-button,
.download-button {
  background-color: #35495e;
  color: white;
  padding: 12px 0;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  width: 220px;
  margin-top: 30px;
  font-size: 20px;
  font-weight: 700;
  transition: background-color 0.3s, transform 0.2s;
}

.next-button:hover,
.download-button:hover {
  background-color: #2c3e50;
  transform: scale(1.05);
}

.uploaded-symbol {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: green;
  font-size: 18px;
}

.topic-input {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;
}

.topic-input input {
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ccc;
  width: 100%;
  max-width: 400px;
  margin-bottom: 10px;
  font-size: 16px;
}

@media (max-width: 600px) {
  .upload-buttons,
  .selectors {
    flex-direction: column;
    gap: 35px;
  }

  .next-button,
  .download-button {
    width: 100%;
  }
}
</style>