// Browser check for the dashboard's 3D engine view (needs Node 22+ and Google Chrome).
//
// With the dashboard running (uv run --all-extras streamlit run dashboard/app.py):
//   node scripts/check_engine3d.mjs [url]
// Chrome path: $CHROME, else the default macOS location.
//
// Selects the knock scenario, plays ~12 s of replay and checks, from the page's
// window.__engine3d counters, that the Three.js scene was created once and kept
// receiving updates across Streamlit reruns (no remounts or disposals), that the view
// flew into the engine, and that the crank turned and fired. Exit code 1 on failure.

import { spawn } from 'node:child_process';

const url = process.argv[2] || 'http://localhost:8501';
const chromePath = process.env.CHROME
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9400 + Math.floor(Math.random() * 400);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const chrome = spawn(chromePath, ['--headless=new', `--remote-debugging-port=${port}`,
  `--user-data-dir=${process.env.TMPDIR || '/tmp'}/engine3d-check-${port}`,
  '--enable-unsafe-swiftshader', 'about:blank']);
let failed = false;
try {
  let targets;
  for (let i = 0; i < 60 && !targets; i += 1) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); }
    catch { await sleep(250); }
  }
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve) => { ws.onopen = resolve; });
  let id = 0;
  const pending = {};
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (pending[message.id]) { pending[message.id](message.result); delete pending[message.id]; }
  };
  const send = (method, params = {}) => new Promise((resolve) => {
    id += 1;
    pending[id] = resolve;
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => (await send('Runtime.evaluate',
    { expression, returnByValue: true }))?.result?.value;
  const click = (label) => evaluate(`(() => {
    const b = [...document.querySelectorAll('button')].find((x) => x.innerText.trim().endsWith(${JSON.stringify(label)}));
    if (!b) return false; b.click(); return true; })()`);
  const snapshot = () => evaluate('JSON.stringify(window.__engine3d || null)').then(JSON.parse);

  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900,
                                                     deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });
  await sleep(8000);
  if (!(await click('Knock'))) throw new Error('scenario card "Knock" not found');
  await sleep(4000);
  if (!(await click('Play'))) throw new Error('"Play" button not found');
  await sleep(2000);
  const early = await snapshot();
  await sleep(10000);
  const late = await snapshot();
  ws.close();

  const checks = [
    ['scene created', early && early.mounts >= 1],
    ['scene created exactly once across reruns', late && late.mounts === 1],
    ['scene never disposed', late && late.disposed === 0],
    ['updates kept arriving (>= 10 more)', late && early && late.updates - early.updates >= 10],
    ['view moved into the engine on Play', late && late.view === 1],
    ['crank turned', late && late.crank > 5],
    ['combustion events fired', late && late.events?.fired > 0],
  ];
  for (const [name, ok] of checks) {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
    failed ||= !ok;
  }
  console.log(`counters: ${JSON.stringify(late)}`);
} catch (error) {
  console.log(`FAIL  ${error.message}`);
  failed = true;
} finally {
  chrome.kill();
}
process.exit(failed ? 1 : 0);
