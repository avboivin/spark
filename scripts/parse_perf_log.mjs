// Parse Spark viewer performance logs and compute summary statistics.
// Usage: node scripts/parse_perf_log.mjs <logfile>
//
// Extracts: startup time, traversal RPC times, frame times, huffman decode
// times, and user-marker events. Outputs avg/p50/p95/p99/max/min/count.

import fs from 'fs';

const file = process.argv[2];
if (!file) { console.error('Usage: node scripts/parse_perf_log.mjs <logfile>'); process.exit(1); }

const text = fs.readFileSync(file, 'utf-8');
const lines = text.split('\n');

const traverseRpc = [];
const frameDt = [];
const huffmanDecode = [];
const wasmReconstruct = [];
const totalDecode = [];
const freeable = [];
const markers = [];
let startupMs = null;
let traverseModeSwitch = null;
let frameCount = 0;
let firstTraverseTime = null;
let lastTraverseTime = null;
let _repairCount = 0;
let _fullCount = 0;

for (const line of lines) {
  // traverseLodTrees RPC
  let m = line.match(/(repair|full) traverse RPC = ([\d.]+)ms/);
  if (m) {
    const v = parseFloat(m[2]);
    traverseRpc.push(v);
    if (m[1] === 'repair') _repairCount++; else _fullCount++;
    if (!firstTraverseTime) {
      const tm = line.match(/^[\d-]+\s([\d:.]+)\sUTC/);
      if (tm) firstTraverseTime = tm[1];
    }
    const tml = line.match(/^[\d-]+\s([\d:.]+)\sUTC/);
    if (tml) lastTraverseTime = tml[1];
    continue;
  }

  // FRAME stall
  m = line.match(/FRAME = ([\d.]+)ms/);
  if (m) {
    frameDt.push(parseFloat(m[1]));
    frameCount++;
    continue;
  }

  // huffman-decode phase
  m = line.match(/huffman-decode phase = ([\d.]+)ms/);
  if (m) { huffmanDecode.push(parseFloat(m[1])); continue; }

  // WASM reconstruct
  m = line.match(/WASM reconstruct = ([\d.]+)ms/);
  if (m) { wasmReconstruct.push(parseFloat(m[1])); continue; }

  // total decode
  m = line.match(/total decode = ([\d.]+)ms/);
  if (m) { totalDecode.push(parseFloat(m[1])); continue; }

  // freeable
  m = line.match(/freeable=(\d+) pages/);
  if (m) { freeable.push(parseInt(m[1])); continue; }

  // startup metric
  m = line.match(/STARTUP METRIC: drop-to-first-bbox = (\d+)ms/);
  if (m) { startupMs = parseInt(m[1]); continue; }

  // traverse-mode auto-switch
  m = line.match(/\[traverse-mode\] auto-switch: (.+)/);
  if (m) { traverseModeSwitch = m[1]; continue; }

  // user markers
  m = line.match(/\[user-marker\] (.+)/);
  if (m) { markers.push({ time: line.substring(0, 23).trim(), event: m[1] }); continue; }
}

function stats(arr, label) {
  if (arr.length === 0) return '';
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((a, b) => a + b, 0);
  const avg = sum / n;
  const p50 = sorted[Math.floor(n * 0.50)];
  const p95 = sorted[Math.floor(n * 0.95)];
  const p99 = sorted[Math.floor(n * 0.99)];
  const min = sorted[0];
  const max = sorted[n - 1];
  return `${label.padEnd(22)} n=${String(n).padStart(5)} avg=${avg.toFixed(1).padStart(6)}ms p50=${p50.toFixed(1).padStart(6)}ms p95=${p95.toFixed(1).padStart(6)}ms p99=${p99.toFixed(1).padStart(6)}ms min=${min.toFixed(0).padStart(4)}ms max=${max.toFixed(0).padStart(5)}ms`;
}

console.log('=== SPARK PERFORMANCE LOG ANALYSIS ===');
console.log(`File: ${file}`);
console.log(`Lines: ${lines.length}`);
console.log(`Frame count: ${frameCount}`);
console.log('');

if (startupMs !== null) {
  console.log(`Startup (drop → first bbox): ${startupMs}ms`);
}
if (traverseModeSwitch) {
  console.log(`Traverse mode switch: ${traverseModeSwitch}`);
}
if (firstTraverseTime && lastTraverseTime) {
  console.log(`First traversal: ${firstTraverseTime} UTC`);
  console.log(`Last traversal:  ${lastTraverseTime} UTC`);
}
console.log('');

console.log('--- Traversal RPC ---');
console.log(stats(traverseRpc, '  Traversal RPC'));
const tPeriod = traverseRpc.length >= 2 ? (traverseRpc.length > 0 ? '  (continuous cadence)' : '') : '';
if (traverseRpc.length > 5) {
  const intervals = [];
  for (let i = 1; i < Math.min(traverseRpc.length, 30); i++) {
    intervals.push(Math.round(traverseRpc[i] - traverseRpc[i - 1]));
  }
  // Count run length of consecutive non-zero values (end-of-log thrash detection)
  const last10 = traverseRpc.slice(-10);
  const last10avg = last10.reduce((a,b)=>a+b,0) / last10.length;
  console.log(`  Last 10 avg: ${last10avg.toFixed(0)}ms  (thrash candidate if similar values repeat)`);
}
console.log('');

console.log('--- Frame Time ---');
console.log(stats(frameDt, '  Frame DT'));

if (huffmanDecode.length > 0) {
  console.log('');
  console.log('--- Huffman Decode ---');
  console.log(stats(huffmanDecode, '  Huffman decode'));
  console.log(stats(wasmReconstruct, '  WASM reconstruct'));
  console.log(stats(totalDecode, '  Total decode'));
}

if (freeable.length > 0) {
  console.log('');
  console.log('--- Page Eviction ---');
  const fmax = Math.max(...freeable);
  const favg = (freeable.reduce((a, b) => a + b, 0) / freeable.length).toFixed(1);
  console.log(`  freeable pages: avg=${favg} max=${fmax} samples=${freeable.length}`);
}

if (markers.length > 0) {
  console.log('');
  console.log('--- User Markers ---');
  for (const m of markers) {
    console.log(`  ${m.time}  ${m.event}`);
  }
}

console.log('');
console.log('=== SUMMARY ===');
const tAvg = traverseRpc.length > 0 ? (traverseRpc.reduce((a,b)=>a+b,0)/traverseRpc.length).toFixed(0) : 'N/A';
const tCount = traverseRpc.length;
const settledTraversals = traverseRpc.slice(-20);
const settledAvg = settledTraversals.length > 0
  ? (settledTraversals.reduce((a,b)=>a+b,0)/settledTraversals.length).toFixed(0)
  : 'N/A';
const settledRange = settledTraversals.length >= 2
  ? `${Math.min(...settledTraversals).toFixed(0)}-${Math.max(...settledTraversals).toFixed(0)}ms`
  : 'N/A';

console.log(`Startup:               ${startupMs ?? 'N/A'}ms`);
console.log(`Traversal count:       ${tCount} (${_repairCount} repair, ${_fullCount} full)`);
console.log(`Traversal avg (all):   ${tAvg}ms`);
console.log(`Traversal settled avg: ${settledAvg}ms (last 20)`);
console.log(`Traversal settled range: ${settledRange}`);
console.log(`Huffman avg:           ${huffmanDecode.length > 0 ? (huffmanDecode.reduce((a,b)=>a+b,0)/huffmanDecode.length).toFixed(0) : 'N/A'}ms`);
console.log(`Frames recorded:       ${frameCount}`);
console.log(`Frame avg:             ${frameDt.length > 0 ? (frameDt.reduce((a,b)=>a+b,0)/frameDt.length).toFixed(1) : 'N/A'}ms`);
