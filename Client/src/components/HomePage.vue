<template>
  <div class="text-center mt-5">
    <h1>PushDelayDetection</h1>
    <button class="btn btn-primary mt-3" @click="selectModel('yolov5')">YOLOv5模型</button>
    <button class="btn btn-secondary mt-3" @click="selectModel('自训练')">自训练模型</button>
  </div>
</template>

<script>
export default {
  methods: {
    selectModel(model) {
      this.$router.push({ name: 'record', params: { model } });
    }
  }
};
</script>

<style scoped>
/* Add your styles here */
</style>
