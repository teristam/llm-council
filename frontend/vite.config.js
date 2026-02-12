import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      'llmcouncil.sancgarden.uk'
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        // Long timeout for streaming SSE connections (10 minutes)
        timeout: 600000,
        proxyTimeout: 600000,
      }
    },
    // Disable HMR completely - it doesn't work well over reverse proxies
    // and causes full page reloads when websocket connection fails
    hmr: false,
  }
})
