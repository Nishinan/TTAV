import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 80,
        proxy: {
            '/syncSession': 'http://127.0.0.1:5050',
            '/updateFocusContext': 'http://127.0.0.1:5050',
            '/startVisualizing': 'http://127.0.0.1:5050',
            '/getTrainingProcessInfo': 'http://127.0.0.1:5050',
            '/updateProjection': 'http://127.0.0.1:5050',
            '/getAllText': 'http://127.0.0.1:5050',
            '/getAlignment': 'http://127.0.0.1:5050',
            '/getAttributes': 'http://127.0.0.1:5050',
            '/getSimpleFilterResult': 'http://127.0.0.1:5050',
            '/getBackground': 'http://127.0.0.1:5050',
            '/getImageData': 'http://127.0.0.1:5050',
            '/getTextData': 'http://127.0.0.1:5050',
            '/getOriginalNeighbors': 'http://127.0.0.1:5050',
            '/getProjectionNeighbors': 'http://127.0.0.1:5050',
            '/getVisualizeMetrics': 'http://127.0.0.1:5050',
            '/getInfluenceSamples': 'http://127.0.0.1:5050',
            '/calculateTrainingEvents': 'http://127.0.0.1:5050',
            '/registerEIFBundle': 'http://127.0.0.1:5050',
        }
    }
})
