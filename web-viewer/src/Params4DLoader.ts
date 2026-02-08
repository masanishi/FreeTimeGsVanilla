/**
 * Params4DLoader — params_4d.bin バイナリ解析
 *
 * Binary layout (little-endian):
 *   Header (16 bytes):
 *     uint32  numSplats
 *     uint32  totalFrames
 *     uint32  texWidth
 *     uint32  texHeight
 *   Data:
 *     float32[numSplats * 3]  velocities  (vx, vy, vz per splat, C-order)
 *     float32[numSplats]      times       (mu_t per splat)
 *     float32[numSplats]      durations   (exp-transformed & clipped, min=0.02)
 */

export interface Params4D {
  numSplats: number;
  totalFrames: number;
  texWidth: number;
  texHeight: number;
  /** RGBA32F — (vx, vy, vz, 0) per texel, row-major */
  velocityData: Float32Array;
  /** RGBA32F — (mu_t, duration, 0, 0) per texel, row-major */
  timingData: Float32Array;
}

export async function loadParams4D(url: string): Promise<Params4D> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const buffer = await response.arrayBuffer();

  const headerView = new DataView(buffer, 0, 16);
  const numSplats = headerView.getUint32(0, true);
  const totalFrames = headerView.getUint32(4, true);
  const texWidth = headerView.getUint32(8, true);
  const texHeight = headerView.getUint32(12, true);

  const HEADER_SIZE = 16;
  const velOffset = HEADER_SIZE;
  const velBytes = numSplats * 3 * 4;
  const timeOffset = velOffset + velBytes;
  const timeBytes = numSplats * 4;
  const durOffset = timeOffset + timeBytes;

  const velocitiesRaw = new Float32Array(buffer, velOffset, numSplats * 3);
  const timesRaw = new Float32Array(buffer, timeOffset, numSplats);
  const durationsRaw = new Float32Array(buffer, durOffset, numSplats);

  // Pack into RGBA32F texel layout (row-major, texWidth × texHeight)
  const texelCount = texWidth * texHeight;
  const velocityData = new Float32Array(texelCount * 4);
  const timingData = new Float32Array(texelCount * 4);

  for (let i = 0; i < numSplats; i++) {
    const base4 = i * 4;
    velocityData[base4 + 0] = velocitiesRaw[i * 3 + 0];
    velocityData[base4 + 1] = velocitiesRaw[i * 3 + 1];
    velocityData[base4 + 2] = velocitiesRaw[i * 3 + 2];
    velocityData[base4 + 3] = 0;

    timingData[base4 + 0] = timesRaw[i];
    timingData[base4 + 1] = durationsRaw[i];
    timingData[base4 + 2] = 0;
    timingData[base4 + 3] = 0;
  }

  console.log(
    `[Params4D] Loaded ${numSplats.toLocaleString()} splats, ` +
    `${totalFrames} frames, texture ${texWidth}×${texHeight}`
  );

  return { numSplats, totalFrames, texWidth, texHeight, velocityData, timingData };
}
