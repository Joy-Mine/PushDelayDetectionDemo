import { createVuetify } from 'vuetify';
import 'vuetify/styles';
import '@mdi/font/css/materialdesignicons.css';

const vuetify = createVuetify({
    theme: {
        defaultTheme: 'light',
    },
    icons: {
        iconfont: 'mdi',
    },
});

export default vuetify;
