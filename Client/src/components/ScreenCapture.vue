<template>
  <div>
    <button @click="startCapture">Start Capture</button>
    <video ref="video" autoplay></video>
  </div>
</template>

<script>
export default {
  methods: {
    async startCapture() {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: { cursor: "always" },
          audio: false
        });
        this.$refs.video.srcObject = stream;
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = async (event) => {
          const formData = new FormData();
          formData.append('file', event.data);

          console.log(formData)
          console.log(event)
          console.log(event.data)

          try {
            const response = await this.$axios.post('/upload', formData, {
              headers: {
                'Content-Type': 'multipart/form-data'
              }
            });
            console.log(response.data);
          } catch (error) {
            console.error(error);
          }
        };
        mediaRecorder.start();
      } catch (err) {
        console.error("Error: " + err);
      }
    }
  }
};
</script>

<style scoped>
/* Add your styles here */
</style>
