import { defineConfig } from 'vite';

export default defineConfig({
  publicDir: 'public',
  server: {
    port: 5173,
    open: true,
  },
  build: {
    target: 'esnext',
  },
});
