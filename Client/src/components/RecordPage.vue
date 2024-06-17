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
          <img :src="image" class="result-image" />
        </div>
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
      images: []
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
        this.loadImages();
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
      } catch (error) {
        console.error('Error uploading video:', error);
      }
    },
    async loadImages() {
      try {
        const response = await this.$axios.get('/api/results');
        this.images = response.data.map(image => `/Datas/results/${image}`);
      } catch (error) {
        console.error('Error loading images:', error);
      }
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
}
</style>
