import { createRouter, createWebHistory } from 'vue-router';
import HomePage from './components/HomePage.vue';
import RecordPage from './components/RecordPage.vue';

const routes = [
    { path: '/', name: 'home', component: HomePage },
    { path: '/record/:model', name: 'record', component: RecordPage }
];

const router = createRouter({
    history: createWebHistory(),
    routes
});

export default router;
