// Browser check for the dashboard's 3D engine view (needs Node 22+ and Google Chrome).
//
// With the dashboard running (uv run --all-extras streamlit run dashboard/app.py):
//   node scripts/check_engine3d.mjs [url]
// Chrome path: $CHROME, else the default macOS location.
//
// 1. Knock scenario: plays ~12 s of replay and checks, from the page's window.__engine3d
//    counters, that the Three.js scene was created once and kept receiving updates
//    across Streamlit reruns (no remounts or disposals), that the engine view is
//    shown, and that the crank turned and fired.
// 2. Multi-fault preset 9 (knock on cylinder 1, misfire on cylinder 4), played to the
//    end: whenever two ALERT banners show, the Decision tile names both faults; the 3D
//    view marks both cylinders; after the replay ends the engine keeps running.
// 3. Recovery: drops the 3D view's WebGL context (as a GPU reset would) and checks the
//    notice appears, the renderer is rebuilt on a fresh canvas, and the engine animates.
// Throughout: no WebGL context is lost or evicted ("Too many active WebGL contexts"),
// and charts create no WebGL contexts during playback (they are SVG; the 3D view needs
// the page's context budget). Exit code 1 on failure.

import { spawn } from 'node:child_process';

const url = process.argv[2] || 'http://localhost:8501';
const chromePath = process.env.CHROME
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9400 + Math.floor(Math.random() * 400);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const PRESET = '9.';
const ROOT_3D = "(document.querySelector('.engine3d-root') || [...document.querySelectorAll('*')].map((e) => e.shadowRoot).filter(Boolean).map((r) => r.querySelector('.engine3d-root')).find(Boolean))";  // "9. Misfire and knock starting together (multi-fault v2)"

// Counts canvases that get a WebGL context, before any page script runs.
const COUNT_WEBGL = `(() => {
  const original = HTMLCanvasElement.prototype.getContext;
  const seen = new WeakSet();
  window.__webgl = { contexts: 0 };
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
    const context = original.call(this, type, ...rest);
    if (context && /webgl/i.test(type) && !seen.has(this)) {
      seen.add(this);
      window.__webgl.contexts += 1;
    }
    return context;
  };
})();`;

const chrome = spawn(chromePath, ['--headless=new', `--remote-debugging-port=${port}`,
  `--user-data-dir=${process.env.TMPDIR || '/tmp'}/engine3d-check-${port}`,
  '--enable-unsafe-swiftshader', 'about:blank']);
let failed = false;
const checks = [];
const check = (name, ok, detail = '') => {
  checks.push([name, Boolean(ok), detail]);
};
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
  let evictions = 0;
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.method === 'Log.entryAdded'
        && /too many active webgl contexts/i.test(message.params.entry.text)) evictions += 1;
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
  const mouse = async ([x, y]) => {
    for (const type of ['mousePressed', 'mouseReleased']) {
      await send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 });
    }
  };
  const center = (selector) => evaluate(`(() => {
    const el = ${selector}; if (!el) return null; el.scrollIntoView({ block: 'center' });
    const r = el.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()`);
  const snapshot = () => evaluate('JSON.stringify(window.__engine3d || null)').then(JSON.parse);
  const webgl = () => evaluate('window.__webgl ? window.__webgl.contexts : null');
  // Alert banners and the Decision tile after a finished rerun. While Streamlit reruns,
  // elements the new run has not reached yet (e.g. a banner that is about to go) stay in
  // the page for a moment, so a sample is only taken while no rerun is in progress.
  const readout = () => evaluate(`JSON.stringify({
    running: Boolean(document.querySelector('[data-testid=stStatusWidget]')
      ?.innerText.match(/running/i)) || document.querySelector('[data-stale=true]') !== null,
    time: [...document.querySelectorAll('[data-testid=stMetric]')]
      .find((m) => m.querySelector('[data-testid=stMetricLabel]')?.innerText.trim() === 'Replay time')
      ?.querySelector('[data-testid=stMetricValue]')?.innerText.trim() || '',
    playing: [...document.querySelectorAll('button')].some((b) => b.innerText.trim() === 'Pause'),
    alerts: [...document.querySelectorAll('.st-key-alerts [data-testid=stAlert]')]
      .filter((a) => a.innerText.includes('ALERT')).length,
    decision: [...document.querySelectorAll('[data-testid=stMetric]')]
      .find((m) => m.querySelector('[data-testid=stMetricLabel]')?.innerText.trim() === 'Decision')
      ?.querySelector('[data-testid=stMetricValue]')?.innerText.trim() || '' })`).then(JSON.parse);

  await send('Log.enable');
  await send('Page.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', { source: COUNT_WEBGL });
  // Tall viewport: the 3D view stays on screen below the evidence panels, so it animates.
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 2600,
                                                     deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url });
  await sleep(8000);

  // 1. Knock scenario.
  if (!(await click('Knock'))) throw new Error('scenario card "Knock" not found');
  await sleep(4000);
  if (!(await click('Play'))) throw new Error('"Play" button not found');
  await sleep(2000);
  const early = await snapshot();
  const contextsAtPlay = await webgl();
  await sleep(10000);
  const late = await snapshot();
  check('scene created', early && early.mounts >= 1);
  check('scene created exactly once across reruns', late && late.mounts === 1);
  check('scene never disposed', late && late.disposed === 0);
  check('updates kept arriving (>= 10 more)', late && early && late.updates - early.updates >= 10);
  check('engine view shown', late && late.view === 1);
  check('crank turned', late && late.crank > 5);
  check('combustion events fired', late && late.events?.fired > 0);
  await click('Pause');
  await sleep(1500);

  // 2. Multi-fault preset, played to the end.
  const box = await center('document.querySelector(\'input[aria-label="Demo preset"]\')');
  if (!box) throw new Error('"Demo preset" select not found');
  await mouse(box);
  await sleep(1200);
  const option = await center(`[...document.querySelectorAll('[role=option]')]
    .find((o) => o.innerText.trim().startsWith(${JSON.stringify(PRESET)}))`);
  if (!option) throw new Error(`preset "${PRESET}" not found`);
  await mouse(option);
  await sleep(6000);
  if (!(await click('Play'))) throw new Error('"Play" button not found for the preset');
  const contextsAtPreset = await webgl();
  let twoAlertSamples = 0;
  const badDecisions = new Set();
  const cylinders = new Set();
  let ended = false;
  let sawPlaying = false;
  let lastTime = '';
  for (let i = 0; i < 90 && !ended; i += 1) {
    await sleep(500);
    let view = await readout();
    for (let k = 0; k < 10 && view.running; k += 1) {
      await sleep(40);
      view = await readout();
    }
    lastTime = view.time;
    sawPlaying ||= view.playing;
    // Two banners and a tile naming fewer faults, twice in a row (not a rerun in flight).
    const mismatch = (v) => v.alerts >= 2
      && !(v.decision.includes('Knock-like') && v.decision.includes('Misfire'));
    if (view.alerts >= 2) twoAlertSamples += 1;
    if (mismatch(view)) {
      await sleep(250);
      const again = await readout();
      if (!again.running && mismatch(again)) badDecisions.add(`${again.time}: ${again.decision}`);
    }
    ((await snapshot())?.modelCylinders || []).forEach((c) => cylinders.add(c));
    ended = sawPlaying && !view.playing;
  }
  check('preset replay reached the end', ended, `replay time ${lastTime}`);
  check('two alerts were shown together', twoAlertSamples > 0, `${twoAlertSamples} samples`);
  check('Decision tile named both faults whenever two alerts showed', badDecisions.size === 0,
        [...badDecisions].join(' | '));
  check('3D view marked both localized cylinders (1 and 4)',
        cylinders.has(1) && cylinders.has(4), JSON.stringify([...cylinders]));
  await sleep(1500);
  const afterEnd = await snapshot();
  await sleep(4000);
  const later = await snapshot();
  check('engine keeps turning after the replay ended', later.crank - afterEnd.crank > 1,
        `${afterEnd.crank.toFixed(1)} -> ${later.crank.toFixed(1)} rad`);
  check('combustion keeps firing after the replay ended',
        later.events.fired + later.events.skipped
        > afterEnd.events.fired + afterEnd.events.skipped);
  check('scene still created once and never disposed',
        later.mounts === 1 && later.disposed === 0);
  const contextsAtEnd = await webgl();
  check('no WebGL context created by charts during playback',
        contextsAtEnd === contextsAtPlay && contextsAtEnd === contextsAtPreset,
        `${contextsAtPlay} at play, ${contextsAtEnd} at end`);
  check('browser evicted no WebGL context', evictions === 0, `${evictions} warnings`);
  check('3D view never lost its context', later.contextLost === 0,
        `lost ${later.contextLost}, restored ${later.contextRestored}, rebuilt ${later.rebuilds}`);

  // 3. Recovery: drop the 3D context the way a GPU reset does (never handed back), and
  //    check that the view rebuilds its renderer on a fresh canvas and keeps animating.
  // The component renders in a shadow root (Streamlit components v2).
  const ROOT = ROOT_3D;
  const dropped = await evaluate(`(() => {
    const canvas = ${ROOT}?.querySelector('canvas');
    const gl = canvas && (canvas.getContext('webgl2') || canvas.getContext('webgl'));
    const lose = gl && gl.getExtension('WEBGL_lose_context');
    if (!lose) return false; lose.loseContext(); return true; })()`);
  check('3D context could be dropped for the recovery check', dropped);
  await sleep(1500);
  const noticeShown = await evaluate(`!${ROOT}.querySelector('.engine3d-fallback').hidden`);
  await sleep(4500);
  const recovered = await snapshot();
  const noticeAfter = await evaluate(`!${ROOT}.querySelector('.engine3d-fallback').hidden`);
  await sleep(2000);
  const afterRecovery = await snapshot();
  check('notice shown while the context was lost', noticeShown);
  check('renderer rebuilt after the loss and notice cleared',
        recovered.rebuilds === 1 && !noticeAfter,
        `lost ${recovered.contextLost}, rebuilt ${recovered.rebuilds}`);
  check('engine animates again after the rebuild', afterRecovery.crank - recovered.crank > 0.5,
        `${recovered.crank.toFixed(1)} -> ${afterRecovery.crank.toFixed(1)} rad`);
  check('one canvas after the rebuild (old one removed)',
        await evaluate(`${ROOT}.querySelectorAll('canvas').length === 1`));
  ws.close();

  for (const [name, ok, detail] of checks) {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
    failed ||= !ok;
  }
  console.log(`counters: ${JSON.stringify(later)}`);
} catch (error) {
  console.log(`FAIL  ${error.message}`);
  failed = true;
} finally {
  chrome.kill();
}
process.exit(failed ? 1 : 0);
