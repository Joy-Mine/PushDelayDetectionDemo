import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import axios from 'axios';

const app = createApp(App);

const httpUrl = 'http://localhost:8088/api';
const axiosInstance = axios.create({
    baseURL: 'http://localhost:8088/api',
    headers: {
        'Content-Type': 'application/json',
    }
});

app.config.globalProperties.$httpUrl = httpUrl;
app.config.globalProperties.$axios = axiosInstance;

app.use(router);
app.mount('#app');
