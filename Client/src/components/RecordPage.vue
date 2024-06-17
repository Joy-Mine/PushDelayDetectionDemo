<template>
  <div>
    <div class="text-center mt-5">
      <h1>PushDelayDetection - {{ model }}模型</h1>
      <button class="btn btn-primary mt-3" @click="startCapture" :disabled="isRecording">开始录制</button>
      <button class="btn btn-danger mt-3" @click="stopCapture" :disabled="!isRecording">停止录制</button>
    </div>
    <div class="row mt-3">
      <div class="col-md-6">
        <h3>预览</h3>
        <video ref="video" width="100%" height="400" autoplay></video>
      </div>
      <div class="col-md-6">
        <h3>识别结果</h3>
        <textarea v-model="results" class="form-control" rows="10" readonly></textarea>
      </div>
    </div>
    <div v-if="images.length > 0" class="mt-3">
      <h3>处理结果图片</h3>
      <div class="image-gallery">
        <div v-for="image in images" :key="image" class="image-container">
          <img :src="`${this.$httpUrl}/results/${image}`" class="result-image" @click="showImage(image)" />
        </div>
      </div>
    </div>

    <!-- 模态框，用于显示放大的图片 -->
    <div v-if="selectedImage" class="modal" @click="closeModal">
      <div class="modal-content">
        <span class="close" @click="closeModal">&times;</span>
        <img :src="`${this.$httpUrl}/results/${selectedImage}`" class="modal-image" />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      mediaRecorder: null,
      chunks: [],
      results: '',
      model: this.$route.params.model,
      isRecording: false,
      images: [],
      selectedImage: null
    };
  },
  methods: {
    async startCapture() {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: { cursor: "always" },
          audio: false
        });
        this.$refs.video.srcObject = stream;

        this.mediaRecorder = new MediaRecorder(stream);
        this.mediaRecorder.ondataavailable = (event) => {
          this.chunks.push(event.data);
        };

        this.mediaRecorder.onstop = () => {
          const blob = new Blob(this.chunks, { 'type': 'video/mp4;' });
          this.chunks = [];
          this.sendVideo(blob);
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        this.images = [];  // 清空之前的结果图片
      } catch (err) {
        console.error("Error: " + err);
      }
    },
    stopCapture() {
      if (this.mediaRecorder && this.isRecording) {
        this.mediaRecorder.stop();
        this.isRecording = false;
      }
    },
    async sendVideo(blob) {
      const formData = new FormData();
      formData.append('file', blob, 'recording.mp4');
      formData.append('model', this.model);

      try {
        const response = await this.$axios.post('/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        this.results = response.data;
        this.loadImages();
      } catch (error) {
        console.error('Error uploading video:', error);
      }
    },
    async loadImages() {
      try {
        const response = await this.$axios.get('/results');
        this.images = response.data;
        console.log(this.images);
      } catch (error) {
        console.error('Error loading images:', error);
      }
    },
    showImage(image) {
      this.selectedImage = image;
    },
    closeModal() {
      this.selectedImage = null;
    }
  }
};
</script>

<style scoped>
.image-gallery {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  max-height: 400px;
  overflow-y: scroll;
}

.image-container {
  flex: 1 0 21%;
  box-sizing: border-box;
}

.result-image {
  width: 100%;
  height: auto;
  cursor: pointer;
}

/* 模态框样式 */
.modal {
  display: flex;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.8);
  justify-content: center;
  align-items: center;
}

.modal-content {
  position: relative;
  background-color: #fefefe;
  margin: auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
}

.modal-image {
  width: 100%;
  height: auto;
}

.close {
  position: absolute;
  top: 10px;
  right: 25px;
  color: #aaa;
  font-size: 28px;
  font-weight: bold;
  cursor: pointer;
}

.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
}
</style>
