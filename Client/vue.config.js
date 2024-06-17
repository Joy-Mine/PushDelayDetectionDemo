const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer:{
    open: true,
    host: "localhost",
    port: 8099,
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:8088',
    //     changeOrigin: true
    //   }
    // }
  }
})
