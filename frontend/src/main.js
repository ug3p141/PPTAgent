import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import axios from 'axios'
import './global.css';

const app = createApp(App)

axios.defaults.baseURL = 'http://124.16.138.144:9297'
app.config.globalProperties.$axios = axios
app.use(router)
app.mount('#app')
