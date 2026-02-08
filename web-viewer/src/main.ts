import { Viewer4D } from './Viewer4D';

const viewer = new Viewer4D();
(window as any).__viewer = viewer;
viewer.init().catch((err) => {
  console.error('Viewer init failed:', err);
  const overlay = document.getElementById('loading-text');
  if (overlay) overlay.textContent = `エラー: ${err.message}`;
});
