// Engine view for the replay dashboard (Three.js, bundled locally).
//
// Streamlit calls render(root, data) on every rerun. The scene is created once per
// root element and kept in a WeakMap; later calls only replace the target state, and
// the requestAnimationFrame loop eases toward it. Geometry is built once.
//
// What drives what (all from `data`, sent by dashboard/engine3d.py):
//   rpm            crank speed (shown slowed by VISUAL_SPEED, labelled on screen)
//   firing_order   which cylinder fires at each 180 deg of the 720 deg cycle
//   animate        keep the engine running: while playing, and after the replay has
//                  finished (on the final decision). `playing` alone (a mid-replay
//                  pause) freezes it. Older payloads without it fall back to `playing`.
//   faults         every fault the model reports now (active alerts plus this window's
//                  detections): {fault, status, cylinder, intensity, color}.
//                  `cylinder` is the one the localizer names (sensor data + cam phase
//                  reference): the fault's effect is confined to it and it is marked
//                  with its piston head glowing in that fault's color (blue misfire,
//                  lavender knock, green valve-train impacts). A fault without
//                  a cylinder shows on every cylinder. Several faults compose: a
//                  misfire weakens or skips the firing, knock adds sparks and jolts,
//                  valve-train impacts add seat ticks. `intensity` is the model's
//                  confidence for that fault (or evidence level / 3, or 0.45 held).
//   effect         first reported fault (block outline color comes in edge_color)
//   vibration_ratio  current vibration RMS / normal-reference RMS (real sensor data)
//   truth_cylinder only when the evaluation-truth overlay is on (marked as truth)
//   egt            per-cylinder exhaust gas temperature scaled 0..1 (data contract 1.2
//                  sensors); colors each exhaust runner on the inferno scale, the
//                  same scale as the dashboard's thermal-camera panel. null: grey.
//   egt_c          the same temperatures in degC, for the HUD
//   crank_cylinder, thermal_cylinder  cylinders named by the independent crank-speed
//                  and EGT/CHT paths (HUD text only)
// Nothing here picks a cylinder on its own.

import * as THREE from '../vendor/three/three.module.min.js';

export const VISUAL_SPEED = 1 / 25;
const views = new WeakMap();
// Instrumentation for the persistence check (read by scripts/check_engine3d.mjs).
const stats = (window.__engine3d = window.__engine3d || { mounts: 0, updates: 0, disposed: 0 });
stats.events = stats.events || { fired: 0, skipped: 0, knock: 0, valveTicks: 0 };
stats.contextLost = stats.contextLost || 0;
stats.contextRestored = stats.contextRestored || 0;
stats.rebuilds = stats.rebuilds || 0;
const FAULT_KINDS = ['misfire', 'knock_like', 'valve_clearance_like'];

export function render(root, data) {
  let view = views.get(root);
  if (!view) {
    view = createView(root);
    views.set(root, view);
    stats.mounts += 1;
  }
  view.update(data || {});
  stats.updates += 1;
}

const COLORS = {
  cyan: 0x00e5e5,
  magenta: 0xf472b6,
  violet: 0xa78bfa,
  amber: 0xf59e0b,
  rose: 0xf05252,
  green: 0x22c55e,
  slate: 0x8b9bb4,
  steel: 0x1b2a3f,
  steelLight: 0x3a5272,
  piston: 0x9fb3c8,
  flame: 0xff7a1a, // combustion fireball
  flameCore: 0xfff1b0, // white-hot core
  ember: 0xffa53d, // glowing particles thrown out by the burn
  plug: 0xe8f9ff, // spark-plug arc
  alarm: 0xff2d2d, // fallback piston highlight
};
const EMBERS = 26; // ember particles per cylinder
// Fire colors per model state: fireball, hot core, embers and spark-plug arc.
// Each scenario has its own color so it is recognizable at a glance.
const FIRE = {
  normal: { flame: 0xff7a1a, core: 0xfff1b0, ember: 0xffa53d, plug: 0xfff8e1 }, // orange
  misfire: { flame: 0x2f8bff, core: 0xd2e8ff, ember: 0x6cb6ff, plug: 0xe8f4ff }, // blue
  knock_like: { flame: 0xb48cff, core: 0xf1e8ff, ember: 0xcdb4ff, plug: 0xf7f2ff }, // lavender
  valve_clearance_like: { flame: 0x22c55e, core: 0xd9ffe6, ember: 0x5ee38f, plug: 0xeafff1 }, // green
};

// Geometry (arbitrary units; one cylinder pitch = 1).
const PITCH = 1.0;
const CRANK_R = 0.42;
const ROD_L = 1.4;
const PISTON_H = 0.42;
const DECK_Y = 2.5;

// Car view: a transverse engine at BAY.scale in the engine bay; detail view: full size.
// The engine's crank axis is at its local origin; its oil pan reaches about -0.83.
const BAY = { scale: 0.24, y: 0.62 + 0.83 * 0.24 + 0.02, z: 2.75 };
const DETAIL_Y = 1.6;
const TRANSITION_S = 1.3;
// Reduced-motion stand-ins for the animated effects (steady glow per model state).
const STEADY = {
  normal: { flash: 0.35, color: 0xff7a1a },
  misfire: { flash: 0.3, color: 0x2f8bff },
  knock_like: { flash: 0.6, color: 0xb48cff },
  valve_clearance_like: { flash: 0.35, color: 0x22c55e },
};
const CAMERA = {
  car: { radius: 17, target: new THREE.Vector3(0, 1.0, 0.3) },
  engine: { radius: 11.5, target: new THREE.Vector3(0, DETAIL_Y + 0.95, 0) },
};
const CHAMBER_Y = 2.32;
// WebGL context loss: browsers keep only a few contexts per page and drop the oldest
// when more are made (the dashboard's charts are SVG so the 3D view is the only one),
// and a GPU reset drops every context. Three.js restores a lost context when the
// browser allows it; if that has not happened after REBUILD_MS, the renderer is rebuilt
// on a fresh canvas with the same scene. The notice only shows if a loss lasts.
const NOTICE_DELAY_MS = 750;
const REBUILD_MS = 4000;
const MAX_REBUILDS = 3; // per minute; after that the notice asks for a page reload

function createView(root) {
  const hud = root.querySelector('.engine3d-hud');
  const notice = root.querySelector('.engine3d-fallback');
  let renderer = makeRenderer();
  if (!renderer) return fallbackView(root);
  let disposed = false;
  let noticeTimer = 0;
  let rebuildTimer = 0;
  const rebuildTimes = [];

  function makeRenderer() {
    let next;
    try {
      next = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      return null;
    }
    next.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    next.domElement.addEventListener('webglcontextlost', onContextLost);
    next.domElement.addEventListener('webglcontextrestored', onContextRestored);
    root.prepend(next.domElement);
    return next;
  }

  function showNotice(text) {
    renderer.domElement.style.visibility = 'hidden';
    notice.hidden = false;
    notice.textContent = text;
  }

  function hideNotice() {
    renderer.domElement.style.visibility = '';
    notice.hidden = true;
  }

  function clearContextTimers() {
    clearTimeout(noticeTimer);
    clearTimeout(rebuildTimer);
  }

  function onContextLost(event) {
    // preventDefault tells the browser we can restore, so it may hand the context back.
    event.preventDefault();
    if (disposed || event.target !== renderer.domElement) return;
    stats.contextLost += 1;
    clearContextTimers();
    noticeTimer = setTimeout(() => showNotice(
      '3D view paused: the browser reclaimed its graphics context. Restoring…'), NOTICE_DELAY_MS);
    rebuildTimer = setTimeout(rebuild, REBUILD_MS);
  }

  function onContextRestored(event) {
    if (disposed || event.target !== renderer.domElement) return;
    stats.contextRestored += 1;
    clearContextTimers();
    hideNotice();
  }

  // New renderer and canvas for the same scene: Three.js uploads geometry, materials
  // and textures again for the new context on its first render.
  function rebuild() {
    if (disposed) return;
    const now = performance.now();
    while (rebuildTimes.length && now - rebuildTimes[0] > 60000) rebuildTimes.shift();
    const next = rebuildTimes.length < MAX_REBUILDS ? makeRenderer() : null;
    if (!next) {
      showNotice('3D view unavailable: the browser could not restore its graphics context. ' +
                 'Reload the page to try again.');
      return;
    }
    rebuildTimes.push(now);
    const old = renderer;
    old.setAnimationLoop(null);
    old.domElement.removeEventListener('webglcontextlost', onContextLost);
    old.domElement.removeEventListener('webglcontextrestored', onContextRestored);
    old.domElement.remove();
    old.dispose();
    renderer = next;
    resize();
    hideNotice();
    renderer.setAnimationLoop(frame);
    stats.rebuilds += 1;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
  scene.add(new THREE.HemisphereLight(0x9fd8ff, 0x0a0f1c, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(4, 7, 6);
  scene.add(key);
  const rim = new THREE.PointLight(COLORS.cyan, 22, 25);
  rim.position.set(-5, 3, -4);
  scene.add(rim);

  // The engine sits in `assembly`, whose pose blends from the car's engine bay (small,
  // transverse) to the full-size detail pose as the view moves from car to engine.
  const engine = buildEngine();
  const assembly = new THREE.Group();
  assembly.add(engine.group);
  scene.add(assembly);
  const car = buildCar();
  scene.add(car.group);
  scene.add(buildFloorGrid());

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const START = { theta: 0.72, phi: 1.12, zoom: 1 };
  const orbit = createOrbit(root, camera, START);
  const viewButtons = createViewControls(root, (mode) => setMode(mode));
  const transition = { target: 0, raw: 0 }; // 0 = car exterior, 1 = engine detail
  function setMode(mode) {
    transition.target = mode === 'engine' ? 1 : 0;
    viewButtons.show(mode);
  }
  viewButtons.onReset(() => Object.assign(orbit.view, START));
  setMode('car');
  const timer = new THREE.Timer();
  timer.connect(document);
  const sim = {
    target: {},
    crank: 0, // crank angle (rad), continuous
    lastCycle: 0,
    flash: [0, 0, 0, 0], // per-cylinder combustion glow 0..1
    flashColor: new THREE.Color(COLORS.flame),
    sparks: [0, 0, 0, 0],
    plug: [0, 0, 0, 0], // spark-plug arc 0..1, just before the fireball
    embers: [0, 0, 0, 0], // ember life 0..1
    ticks: [0, 0, 0, 0],
    alarm: [0, 0, 0, 0], // fault-colored highlight on the localized piston head, 0..1
    alarmFault: [null, null, null, null], // fault whose color the highlight shows (kept while fading)
    egt: [null, null, null, null], // eased exhaust runner temperature 0..1 (null: no sensor)
    layout: layoutFaults([]), // reported faults per cylinder (set on each update)
    jolt: 0,
    edgeColor: new THREE.Color(COLORS.cyan),
    stutter: 0,
  };
  const tmpColor = new THREE.Color();

  function resize() {
    const width = root.clientWidth || 1;
    const height = root.clientHeight || 1;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(root);
  resize();

  let visible = true;
  const intersection = new IntersectionObserver((entries) => {
    visible = entries[0]?.isIntersecting ?? true;
  });
  intersection.observe(root);

  function frame() {
    if (!root.isConnected) {
      dispose();
      return;
    }
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.1);
    if (!visible || document.hidden) return;
    step(dt);
    const m = stepTransition(dt);
    orbit.apply(lerp(CAMERA.car.radius, CAMERA.engine.radius, m),
                tmpTarget.lerpVectors(CAMERA.car.target, CAMERA.engine.target, m));
    renderer.render(scene, camera);
  }

  // Car -> engine: hood lifts, car fades, engine grows out of the bay, camera dollies in.
  const tmpTarget = new THREE.Vector3();
  function stepTransition(dt) {
    if (reduceMotion.matches) transition.raw = transition.target;
    else {
      const direction = Math.sign(transition.target - transition.raw);
      transition.raw = clamp01(transition.raw + direction * dt / TRANSITION_S);
    }
    const m = easeInOutCubic(transition.raw);
    const pose = smoothstep(clamp01((m - 0.15) / 0.85));
    assembly.scale.setScalar(lerp(BAY.scale, 1, pose));
    assembly.position.set(0, lerp(BAY.y, DETAIL_Y, pose), lerp(BAY.z, 0, pose));
    car.hood.rotation.z = 0.95 * clamp01(m / 0.35);
    car.setOpacity(1 - clamp01((m - 0.35) / 0.45));
    stats.view = m;
    return m;
  }

  function step(dt) {
    const t = sim.target;
    const still = reduceMotion.matches;
    const rpm = Number(t.rpm) || 0;
    const running = isRunning(t);
    const { perCylinder, located, misfire } = sim.layout;

    // Crank: real RPM scaled to a watchable speed. A misfire makes the crank stutter.
    if (running && !still) {
      let omega = (rpm / 60) * 2 * Math.PI * VISUAL_SPEED;
      if (misfire) omega *= 1 - 0.25 * sim.stutter;
      sim.crank += omega * dt;
    }
    sim.stutter = Math.max(0, sim.stutter - dt * 2.5);
    const theta = sim.crank;
    stats.crank = theta;
    const cycle = mod(theta, 4 * Math.PI);

    // Combustion events at each cylinder's compression TDC (plus a small offset). Each
    // cylinder shows its own localized faults plus any fault without a cylinder.
    engine.tdc.forEach((tdc, k) => {
      const fire = mod(tdc + (8 * Math.PI) / 180, 4 * Math.PI);
      if (crossed(sim.lastCycle, cycle, fire) && running) ignite(k, perCylinder[k]);
    });
    sim.lastCycle = cycle;

    // Pistons, rods and crank (slider-crank kinematics).
    engine.crankshaft.rotation.x = theta;
    engine.camshafts.forEach((cam) => { cam.rotation.x = theta / 2; });
    engine.cylinders.forEach((cyl, k) => {
      const psi = theta + engine.pins[k];
      const pinY = CRANK_R * Math.cos(psi);
      const pinZ = CRANK_R * Math.sin(psi);
      const wristY = pinY + Math.sqrt(ROD_L * ROD_L - pinZ * pinZ);
      cyl.piston.position.y = wristY + PISTON_H / 2 - 0.05;
      cyl.rod.position.set(cyl.x, (pinY + wristY) / 2, pinZ / 2);
      cyl.rod.rotation.x = Math.atan2(-pinZ, wristY - pinY);

      // Valves: exhaust opens late in the power stroke, intake around the exhaust TDC.
      const a = mod(cycle - engine.tdc[k], 4 * Math.PI);
      const exhaust = lift(a, 0.85 * Math.PI, 2.08 * Math.PI);
      const intake = lift(a, 1.92 * Math.PI, 3.15 * Math.PI);
      if (exhaust === 0 && cyl.exhaustOpen && running) seatImpact(k, perCylinder[k]);
      if (intake === 0 && cyl.intakeOpen && running) seatImpact(k, perCylinder[k]);
      cyl.exhaustOpen = exhaust > 0;
      cyl.intakeOpen = intake > 0;
      cyl.exhaust.position.y = cyl.valveY - 0.13 * exhaust;
      cyl.intake.position.y = cyl.valveY - 0.13 * intake;

      if (still) {
        // Reduced motion: no firing animation, so show the model state as a steady glow
        // (same colors as the animated effects) instead of hiding it.
        const own = perCylinder[k];
        const steady = STEADY[own[0]?.fault] || STEADY.normal;
        const strength = Math.max(0, ...own.map((f) => f.intensity));
        sim.flash[k] = running || own.length
          ? steady.flash * (own.length ? 0.5 + 0.5 * strength : 1) : 0;
        cyl.flameColor.set(steady.color);
        sim.sparks[k] = has(own, 'knock_like') ? 0.7 : 0;
        sim.ticks[k] = has(own, 'valve_clearance_like') ? 0.7 : 0;
      } else {
        // Effects decay.
        sim.flash[k] = Math.max(0, sim.flash[k] - dt * 1.7); // ~0.6 s burn
        sim.sparks[k] = Math.max(0, sim.sparks[k] - dt * 5);
        sim.plug[k] = Math.max(0, sim.plug[k] - dt * 14);
        sim.embers[k] = Math.max(0, sim.embers[k] - dt * 1.1);
        if (sim.embers[k] > 0) driftEmbers(cyl, dt);
        sim.ticks[k] = Math.max(0, sim.ticks[k] - dt * 6);
      }
      cyl.flame.material.opacity = sim.flash[k];
      cyl.flame.material.color.copy(cyl.flameColor);
      // Fireball: bright and tight at ignition, then expands as it fades.
      // It fills the chamber between the head and the descending piston crown.
      const burn = sim.flash[k];
      const crown = cyl.piston.position.y + PISTON_H / 2;
      const gap = Math.max(DECK_Y - crown, 0.12);
      const fireY = still ? CHAMBER_Y : crown + gap / 2;
      cyl.flame.position.y = fireY;
      cyl.core.position.y = fireY + 0.04;
      cyl.embers.position.y = fireY;
      cyl.flame.scale.set(still ? 0.8 : 0.75 + 0.55 * (1 - burn),
                          still ? 0.8 : Math.max(0.6, gap * 1.6) + 0.3 * (1 - burn), 1);
      cyl.core.material.opacity = still ? 0 : Math.min(1, burn * 1.4) * (cyl.burning ? 1 : 0);
      cyl.core.scale.setScalar(0.3 + 0.35 * burn);
      cyl.plug.material.opacity = still ? 0 : sim.plug[k];
      cyl.plug.scale.setScalar(0.12 + 0.18 * sim.plug[k]);
      cyl.embers.material.opacity = still ? 0 : sim.embers[k];
      cyl.sparks.material.opacity = sim.sparks[k];
      if (!still) cyl.sparks.rotation.y += dt * 6;
      cyl.tick.material.opacity = sim.ticks[k];
    });

    // Chamber light follows the most recent firing cylinder.
    let brightest = 0;
    sim.flash.forEach((value, k) => { if (value > sim.flash[brightest]) brightest = k; });
    engine.chamberLight.position.set(engine.cylinders[brightest].x, CHAMBER_Y, 0);
    engine.chamberLight.intensity = 22 * sim.flash[brightest];
    engine.chamberLight.color.copy(engine.cylinders[brightest].flameColor);

    // Shake: the measured vibration level relative to normal, plus knock jolts.
    const ratio = Number(t.vibration_ratio) || 1;
    const base = still ? 0 : Math.min(Math.max(ratio - 1, 0), 3) * 0.006;
    sim.jolt = Math.max(0, sim.jolt - dt * 9);
    const amp = still ? 0 : base + sim.jolt * 0.035;
    engine.shaker.position.set((Math.random() - 0.5) * amp, (Math.random() - 0.5) * amp,
                               (Math.random() - 0.5) * amp);

    // Block outline color follows the model state; an active alert pulses gently.
    tmpColor.set(t.edge_color || '#00e5e5');
    sim.edgeColor.lerp(tmpColor, Math.min(1, dt * 4));
    const pulse = t.alert && !still ? 0.75 + 0.25 * Math.sin(timer.getElapsed() * 4) : 0.85;
    engine.edgeMaterials.forEach((material) => {
      material.color.copy(sim.edgeColor);
      material.opacity = material.userData.base * pulse;
    });
    // Faulty-cylinder highlight: every piston head the localizer names glows and pulses
    // in its fault's color (the same colors as that fault's flame); each fades in when
    // its cylinder is named and out when the fault is no longer reported.
    const beat = still ? 1 : 0.65 + 0.35 * Math.sin(timer.getElapsed() * 5);
    engine.cylinders.forEach((cyl, k) => {
      const goal = located[k] ? 1 : 0;
      if (located[k]) sim.alarmFault[k] = located[k].fault;
      const tint = FIRE[sim.alarmFault[k]]?.flame ?? COLORS.alarm;
      cyl.piston.material.emissive.setHex(tint);
      cyl.alarm.material.color.setHex(tint);
      cyl.alarmLight.color.setHex(tint);
      sim.alarm[k] += (goal - sim.alarm[k]) * Math.min(1, dt * (still ? 60 : 4));
      const level = sim.alarm[k] * beat;
      cyl.piston.material.emissiveIntensity = 1.6 * level;
      cyl.alarm.material.opacity = 0.85 * level;
      cyl.alarmLight.position.y = cyl.piston.position.y + 0.35;
      cyl.alarmLight.intensity = 9 * level;
    });
    stats.modelCylinders = located.flatMap((f, k) => (f ? [k + 1] : [])); // browser checks
    // Exhaust runners glow with their measured exhaust gas temperature.
    const egt = Array.isArray(t.egt) && t.egt.length === 4 ? t.egt : null;
    engine.cylinders.forEach((cyl, k) => {
      const goal = egt ? Number(egt[k]) : null;
      if (goal === null || !Number.isFinite(goal)) {
        sim.egt[k] = null;
        cyl.runner.material.color.set(COLORS.steelLight);
        cyl.runner.material.emissiveIntensity = 0;
        return;
      }
      const now = sim.egt[k] === null ? goal : sim.egt[k];
      sim.egt[k] = now + (goal - now) * Math.min(1, dt * (still ? 60 : 2));
      inferno(sim.egt[k], cyl.runner.material.color);
      cyl.runner.material.emissive.copy(cyl.runner.material.color);
      cyl.runner.material.emissiveIntensity = 0.15 + 0.85 * sim.egt[k];
    });
    stats.egt = sim.egt.map((v) => (v === null ? null : Math.round(v * 100) / 100));
    engine.truth.visible = Number.isInteger(t.truth_cylinder);
    if (engine.truth.visible) {
      engine.truth.position.x = engine.cylinders[t.truth_cylinder - 1].x;
      engine.truth.rotation.y += still ? 0 : dt * 0.8;
    }
  }

  // `faults`: the faults shown on cylinder k (its localized ones, then engine-wide ones).
  function ignite(k, faults) {
    const cyl = engine.cylinders[k];
    const misfire = find(faults, 'misfire');
    if (misfire && Math.random() < 0.35 + 0.5 * misfire.intensity) {
      // Model says misfire (on the localized cylinder, or every cylinder when the
      // cylinder is not known): firing pulses become weak or missing.
      // A skipped firing stays dark; a weak one sputters: faint blue flicker, plug arc.
      const weak = Math.random() >= 0.5;
      sim.flash[k] = weak ? 0.25 : 0;
      sim.plug[k] = weak ? 0.6 : 0;
      stats.events.skipped += 1;
      sim.stutter = 1;
      cyl.flameColor.set(FIRE.misfire.flame);
      cyl.plug.material.color.set(FIRE.misfire.plug);
      cyl.burning = false; // no hot core or embers
      return;
    }
    // Flame color: the cylinder's first fault other than a misfire (a misfire that did
    // not skip this time burns blue).
    const lead = faults.find((f) => f.fault !== 'misfire') || misfire;
    const fire = FIRE[lead?.fault] || FIRE.normal;
    const knock = find(faults, 'knock_like');
    sim.flash[k] = 1;
    sim.plug[k] = 1;
    sim.embers[k] = 1;
    cyl.burning = true;
    stats.events.fired += 1;
    cyl.flameColor.set(fire.flame);
    cyl.core.material.color.set(fire.core);
    cyl.plug.material.color.set(fire.plug);
    launchEmbers(cyl, Boolean(knock), fire.ember);
    if (knock && Math.random() < 0.4 + 0.6 * knock.intensity) {
      sim.sparks[k] = 1;
      stats.events.knock += 1;
      sim.jolt = Math.max(sim.jolt, 0.6 + 0.4 * knock.intensity);
      scatter(cyl.sparks.geometry);
    }
  }

  function launchEmbers(cyl, violent, color) {
    // Embers start at the spark plug and burst outward, mostly sideways and down
    // (the burning charge pushes the piston), faster and pinker for knock.
    const positions = cyl.embers.geometry.attributes.position.array;
    const speed = violent ? 2.6 : 1.5;
    for (let i = 0; i < EMBERS; i += 1) {
      const angle = Math.random() * Math.PI * 2;
      const out = speed * (0.35 + Math.random() * 0.65);
      positions[i * 3] = (Math.random() - 0.5) * 0.06;
      positions[i * 3 + 1] = 0.08;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 0.06;
      cyl.emberVel[i * 3] = Math.cos(angle) * out * 0.5;
      cyl.emberVel[i * 3 + 1] = -Math.random() * out * 0.6 + 0.25;
      cyl.emberVel[i * 3 + 2] = Math.sin(angle) * out * 0.4;
    }
    cyl.embers.material.color.set(color);
    cyl.embers.geometry.attributes.position.needsUpdate = true;
  }

  function driftEmbers(cyl, dt) {
    const positions = cyl.embers.geometry.attributes.position.array;
    const drag = Math.exp(-dt * 3);
    for (let i = 0; i < EMBERS * 3; i += 1) {
      cyl.emberVel[i] *= drag;
      positions[i] += cyl.emberVel[i] * dt;
    }
    cyl.embers.geometry.attributes.position.needsUpdate = true;
  }

  function seatImpact(k, faults) {
    const valve = find(faults, 'valve_clearance_like');
    if (!valve) return;
    sim.ticks[k] = 0.5 + 0.5 * valve.intensity;
    stats.events.valveTicks += 1;
    sim.jolt = Math.max(sim.jolt, 0.25 * valve.intensity);
  }

  function dispose() {
    disposed = true;
    clearContextTimers();
    renderer.setAnimationLoop(null);
    timer.dispose();
    resizeObserver.disconnect();
    intersection.disconnect();
    orbit.dispose();
    scene.traverse((object) => {
      object.geometry?.dispose?.();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => {
        material?.map?.dispose?.();
        material?.dispose?.();
      });
    });
    renderer.dispose();
    // Hand the context back now rather than at garbage collection, so a remounted view
    // does not count against the browser's limit of live WebGL contexts.
    renderer.forceContextLoss();
    stats.disposed += 1;
  }

  renderer.setAnimationLoop(frame);

  return {
    update(data) {
      const previous = sim.target;
      const firstOrder = !previous.firing_order;
      // Fly into the engine when playback starts or a new alert is raised. The
      // Car / Engine buttons override this at any time.
      if ((data.playing && !previous.playing) || (data.alert && data.alert !== previous.alert)) {
        setMode('engine');
      }
      sim.target = data;
      sim.layout = layoutFaults(data.faults);
      if (firstOrder && Array.isArray(data.firing_order)) engine.setFiringOrder(data.firing_order);
      renderHud(hud, data);
      describe(root, data);
    },
  };
}

function renderHud(hud, data) {
  const rpm = Number.isFinite(data.rpm) ? `${Math.round(data.rpm).toLocaleString('en-US')} rpm`
                                        : 'rpm —';
  const order = (data.firing_order || [1, 3, 4, 2]).join('-');
  const speed = `shown at ×1/${Math.round(1 / VISUAL_SPEED)} speed`;
  hud.replaceChildren(
    line(`Inline-4 engine · firing order ${order} · ${rpm} · ${speed}`),
    line(data.status_text || '', data.edge_color),
  );
  if (data.finished) hud.append(line(finishedText(data)));
  if (Array.isArray(data.egt_c)) hud.append(line(egtText(data)));
  if (Number.isInteger(data.truth_cylinder)) {
    hud.append(line(`Evaluation truth overlay: fault in cylinder ${data.truth_cylinder} ` +
                    '(not used by the model)', '#fbbf24'));
  }
}

function finishedText(data) {
  const at = Number.isFinite(data.t) ? ` at ${data.t.toFixed(1)} s` : '';
  return `Replay ended${at} · engine shown running on the final decision`;
}

function egtText(data) {
  const values = data.egt_c.map((v, k) => `${k + 1}: ${v}`).join(' · ');
  const named = [];
  if (Number.isInteger(data.crank_cylinder)) named.push(`crank speed → cyl ${data.crank_cylinder}`);
  if (Number.isInteger(data.thermal_cylinder)) named.push(`EGT/CHT → cyl ${data.thermal_cylinder}`);
  return `Exhaust gas °C ${values}${named.length ? ` · ${named.join(' · ')}` : ''}`;
}

// Inferno colormap (matplotlib/Plotly), sampled at 9 stops; t in 0..1.
const INFERNO = [0x000004, 0x1b0c41, 0x4a0c6b, 0x781c6d, 0xa52c60, 0xcf4446, 0xed6925,
                 0xfb9b06, 0xfcffa4];
const infernoStops = INFERNO.map((hex) => new THREE.Color(hex));
function inferno(t, target) {
  const x = Math.min(Math.max(t, 0), 1) * (infernoStops.length - 1);
  const k = Math.min(Math.floor(x), infernoStops.length - 2);
  return target.copy(infernoStops[k]).lerp(infernoStops[k + 1], x - k);
}

// Accessible name of the view tracks the model state shown in it.
function describe(root, data) {
  root.setAttribute('aria-label', `Engine view. ${data.status_text || ''}. Drag or use arrow ` +
                    'keys to orbit, plus and minus to zoom.');
}

function line(text, color) {
  const element = document.createElement('div');
  element.textContent = text;
  if (color) element.style.color = color;
  return element;
}

// Without WebGL: a 2D inline-4 schematic (SVG) driven by the same state, with the same
// slider-crank motion and state colors, so the view still shows what the model says.
function fallbackView(root) {
  const fallback = root.querySelector('.engine3d-fallback');
  root.querySelector('.engine3d-controls')?.setAttribute('hidden', '');
  fallback.hidden = false;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (name, attrs, parent) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    parent?.appendChild(node);
    return node;
  };
  const svg = el('svg', { viewBox: '0 0 400 230', width: '100%', height: '78%',
                          'aria-hidden': 'true' });
  const block = el('rect', { x: 40, y: 30, width: 320, height: 150, rx: 8, fill: 'none',
                             'stroke-width': 2 }, svg);
  const pistons = [0, 1, 2, 3].map((k) => {
    const x = 70 + k * 75;
    el('rect', { x, y: 40, width: 50, height: 120, fill: 'rgba(148,163,184,0.06)',
                 stroke: 'rgba(148,163,184,0.35)' }, svg);
    el('text', { x: x + 25, y: 205, 'text-anchor': 'middle', fill: '#94a3b8',
                 'font-size': 12, 'font-family': 'Inter, sans-serif' }, svg)
      .textContent = `Cylinder ${k + 1}`;
    // Exhaust port above each cylinder, colored by its exhaust gas temperature.
    const port = el('rect', { x: x + 10, y: 14, width: 30, height: 10, rx: 3,
                              fill: 'rgba(148,163,184,0.25)' }, svg);
    return { port, glow: el('rect', { x, y: 40, width: 50, height: 16, opacity: 0 }, svg),
             piston: el('rect', { x: x + 3, y: 80, width: 44, height: 26, rx: 3,
                                  fill: '#9fb3c8' }, svg) };
  });
  const status = document.createElement('div');
  status.className = 'engine3d-fallback-status';
  fallback.replaceChildren(svg, status);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let target = {};
  let crank = 0;
  let last = performance.now();
  const tdc = [0, 3 * Math.PI, Math.PI, 2 * Math.PI];
  function frame(now) {
    if (!root.isConnected) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    const running = isRunning(target);
    if (running && !reduceMotion.matches) {
      crank += ((Number(target.rpm) || 0) / 60) * 2 * Math.PI * VISUAL_SPEED * dt;
    }
    pistons.forEach(({ piston, glow }, k) => {
      const psi = crank - tdc[k];
      const height = CRANK_R * Math.cos(psi) + Math.sqrt(ROD_L ** 2 - (CRANK_R * Math.sin(psi)) ** 2);
      piston.setAttribute('y', String(56 + (CRANK_R + ROD_L - height) * 90));
      glow.setAttribute('opacity', String(Math.max(0, Math.cos(mod(crank - tdc[k], 4 * Math.PI)))
                                          * (running ? 0.6 : 0)));
    });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return {
    update(data) {
      target = data;
      root.setAttribute('aria-label', `Engine schematic (2D). ${data.status_text || ''}.`);
      const color = data.edge_color || '#00e5e5';
      block.setAttribute('stroke', color);
      const { located, perCylinder } = layoutFaults(data.faults);
      pistons.forEach(({ glow, port, piston }, k) => {
        // Each cylinder glows in its own fault's color (else the state color); a
        // localized fault also turns that piston red.
        glow.setAttribute('fill', perCylinder[k][0]?.color || color);
        piston.setAttribute('fill', located[k]
          ? `#${FIRE[located[k].fault].flame.toString(16).padStart(6, '0')}` : '#9fb3c8');
        const egt = Array.isArray(data.egt) ? Number(data.egt[k]) : NaN;
        port.setAttribute('fill', Number.isFinite(egt)
          ? `#${inferno(egt, new THREE.Color()).getHexString()}` : 'rgba(148,163,184,0.25)');
      });
      status.replaceChildren(
        line('WebGL is unavailable, so a 2D schematic replaces the 3D view.'),
        line(data.status_text || '', color),
      );
      if (data.finished) status.append(line(finishedText(data)));
      if (Array.isArray(data.egt_c)) status.append(line(egtText(data)));
      if (Number.isInteger(data.truth_cylinder)) {
        status.append(line(`Evaluation truth overlay: fault in cylinder ${data.truth_cylinder} ` +
                           '(not used by the model)', '#fbbf24'));
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Geometry (built once)

function buildEngine() {
  const group = new THREE.Group();
  const shaker = new THREE.Group();
  group.add(shaker);
  const edgeMaterials = [];

  const edged = (mesh, opacity = 0.6) => {
    const material = new THREE.LineBasicMaterial({ color: COLORS.cyan, transparent: true,
                                                   opacity });
    material.userData.base = opacity;
    edgeMaterials.push(material);
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 25), material));
    return mesh;
  };
  const glass = (opacity) => new THREE.MeshStandardMaterial({
    color: COLORS.steel, transparent: true, opacity, metalness: 0.5, roughness: 0.4,
    depthWrite: false, side: THREE.DoubleSide,
  });
  const metal = (color, roughness = 0.3) => new THREE.MeshStandardMaterial({
    color, metalness: 0.85, roughness,
  });

  // Block, head, cam cover and oil pan: translucent shells with neon edges.
  const width = PITCH * 4 + 0.6;
  const block = edged(new THREE.Mesh(new THREE.BoxGeometry(width, 1.7, 1.5), glass(0.16)));
  block.position.y = 1.65;
  const head = edged(new THREE.Mesh(new THREE.BoxGeometry(width, 0.55, 1.35), glass(0.12)), 0.45);
  head.position.y = DECK_Y + 0.28;
  const cover = edged(new THREE.Mesh(new THREE.BoxGeometry(width - 0.2, 0.35, 1.1), glass(0.1)),
                      0.35);
  cover.position.y = DECK_Y + 0.73;
  const pan = edged(new THREE.Mesh(new THREE.BoxGeometry(width - 0.3, 0.55, 1.3), glass(0.1)),
                    0.35);
  pan.position.y = -0.55;
  shaker.add(block, head, cover, pan);

  // Crankshaft: main journal, webs and pins (pin angles set by the firing order).
  const crankshaft = new THREE.Group();
  const journal = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, width + 0.8, 24),
                                 metal(COLORS.steelLight));
  journal.rotation.z = Math.PI / 2;
  crankshaft.add(journal);
  const flywheel = edged(new THREE.Mesh(new THREE.CylinderGeometry(0.85, 0.85, 0.12, 48),
                                        metal(COLORS.steelLight, 0.5)), 0.5);
  flywheel.rotation.z = Math.PI / 2;
  flywheel.position.x = -width / 2 - 0.35;
  const mark = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.12, 0.3),
                              new THREE.MeshBasicMaterial({ color: COLORS.cyan }));
  mark.position.set(-width / 2 - 0.35, 0.7, 0);
  crankshaft.add(flywheel, mark);
  shaker.add(crankshaft);

  const webGeometry = new THREE.BoxGeometry(0.1, 0.75, 0.42);
  const pinGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.32, 20);
  const pistonGeometry = new THREE.CylinderGeometry(0.36, 0.36, PISTON_H, 36);
  const rodGeometry = new THREE.BoxGeometry(0.1, ROD_L, 0.14);
  const boreGeometry = new THREE.CylinderGeometry(0.4, 0.4, 1.35, 36, 1, true);
  const valveGeometry = new THREE.CylinderGeometry(0.11, 0.07, 0.04, 20);
  const stemGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.45, 8);
  const flameTexture = glowTexture();
  const pistonMaterial = metal(COLORS.piston, 0.25);
  const rodMaterial = metal(COLORS.steelLight, 0.35);
  const valveMaterial = metal(0xcbd5e1, 0.3);

  const cylinders = [];
  for (let k = 0; k < 4; k += 1) {
    const x = (k - 1.5) * PITCH;
    const crankPin = new THREE.Group(); // rotated per cylinder in setFiringOrder
    crankPin.position.x = x;
    const webA = new THREE.Mesh(webGeometry, rodMaterial);
    const webB = webA.clone();
    webA.position.set(-0.2, CRANK_R / 2, 0);
    webB.position.set(0.2, CRANK_R / 2, 0);
    const pin = new THREE.Mesh(pinGeometry, metal(COLORS.piston));
    pin.rotation.z = Math.PI / 2;
    pin.position.y = CRANK_R;
    crankPin.add(webA, webB, pin);
    crankshaft.add(crankPin);

    const bore = new THREE.Mesh(boreGeometry, glass(0.08));
    bore.position.set(x, 1.85, 0);
    // Own material per piston so only the localized one glows red.
    const piston = edged(new THREE.Mesh(pistonGeometry, pistonMaterial.clone()), 0.35);
    piston.material.emissive = new THREE.Color(COLORS.alarm);
    piston.material.emissiveIntensity = 0;
    // Red halo that rides on the piston (child, so it follows the stroke).
    const alarm = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: COLORS.alarm, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    alarm.scale.set(1.3, 0.8, 1);
    alarm.position.y = PISTON_H / 2 + 0.06; // on the crown, visible through the bore
    piston.add(alarm);
    piston.position.x = x;
    const rod = new THREE.Mesh(rodGeometry, rodMaterial);

    const valveY = DECK_Y + 0.02;
    const valve = (z) => {
      const head = new THREE.Mesh(valveGeometry, valveMaterial);
      const stem = new THREE.Mesh(stemGeometry, valveMaterial);
      stem.position.y = 0.24;
      head.add(stem);
      head.position.set(x, valveY, z);
      return head;
    };
    const intake = valve(0.2);
    const exhaust = valve(-0.2);

    const flame = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: COLORS.flame, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    flame.position.set(x, CHAMBER_Y, 0);
    flame.scale.setScalar(0.9);

    const sparks = new THREE.Points(sparkGeometry(), new THREE.PointsMaterial({
      color: FIRE.knock_like.ember, size: 0.06, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    sparks.position.set(x, CHAMBER_Y, 0);

    const core = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: COLORS.flameCore, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    core.position.set(x, CHAMBER_Y + 0.02, 0);
    const plug = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: COLORS.plug, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    plug.position.set(x, DECK_Y - 0.06, 0);
    const emberGeometry = new THREE.BufferGeometry();
    emberGeometry.setAttribute('position',
                               new THREE.BufferAttribute(new Float32Array(EMBERS * 3), 3));
    const embers = new THREE.Points(emberGeometry, new THREE.PointsMaterial({
      color: COLORS.ember, size: 0.12, map: flameTexture, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    embers.position.set(x, CHAMBER_Y, 0);

    const tick = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: FIRE.valve_clearance_like.ember, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    tick.position.set(x, DECK_Y + 0.12, 0);
    tick.scale.set(0.9, 0.45, 1);

    // Exhaust runner: from the port on the exhaust side of the head, out and down to
    // a collector below the block. Own material, colored by this cylinder's EGT.
    const runnerPath = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, DECK_Y + 0.14, -0.62),
      new THREE.Vector3(x, DECK_Y + 0.1, -1.0),
      new THREE.Vector3(x * 0.75, DECK_Y - 0.4, -1.22),
      new THREE.Vector3(x * 0.3, DECK_Y - 1.0, -1.3),
      new THREE.Vector3(x * 0.08, DECK_Y - 1.45, -1.3),
    ]);
    const runner = new THREE.Mesh(
      new THREE.TubeGeometry(runnerPath, 40, 0.085, 14, false),
      new THREE.MeshStandardMaterial({ color: COLORS.steelLight, metalness: 0.35,
                                       roughness: 0.45, emissive: new THREE.Color(0x000000),
                                       emissiveIntensity: 0 }),
    );

    shaker.add(bore, piston, rod, intake, exhaust, flame, core, plug, embers, sparks, tick,
               runner);
    cylinders.push({ x, piston, alarm, rod, intake, exhaust, valveY, flame, core, plug, embers,
                     runner,
                     emberVel: new Float32Array(EMBERS * 3), burning: false, sparks, tick,
                     crankPin,
                     flameColor: new THREE.Color(COLORS.flame), intakeOpen: false,
                     exhaustOpen: false });
  }

  // Exhaust collector below the runners.
  const collector = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.11, 1.1, 18),
                                   metal(COLORS.steelLight, 0.45));
  collector.position.set(0, DECK_Y - 1.95, -1.3);
  shaker.add(collector);

  // Overhead camshafts (turn at half crank speed) with one lobe per valve.
  const camshafts = [0.2, -0.2].map((z) => {
    const cam = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, width - 0.3, 12),
                                 metal(COLORS.steelLight));
    shaft.rotation.z = Math.PI / 2;
    cam.add(shaft);
    cylinders.forEach(({ x }) => {
      const lobe = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.16, 0.1), metal(COLORS.piston));
      lobe.position.set(x, 0.06, 0);
      cam.add(lobe);
    });
    cam.position.set(0, DECK_Y + 0.62, z);
    shaker.add(cam);
    return cam;
  });

  const chamberLight = new THREE.PointLight(COLORS.flame, 0, 4, 2);
  shaker.add(chamberLight);
  // Per cylinder (several faults can be localized at once): a red light that follows
  // the piston and tints the parts around it, in the fault's color. Built once with
  // zero intensity, so the light count never changes and no shader is recompiled when a
  // fault appears.
  cylinders.forEach((cyl) => {
    cyl.alarmLight = new THREE.PointLight(COLORS.alarm, 0, 1.4, 2);
    cyl.alarmLight.position.set(cyl.x, CHAMBER_Y, 0.45);
    shaker.add(cyl.alarmLight);
  });

  // Evaluation-truth marker (only shown when the overlay is on).
  const truth = new THREE.Mesh(
    new THREE.TorusGeometry(0.55, 0.018, 8, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.amber, transparent: true, opacity: 0.9 }),
  );
  truth.rotation.x = Math.PI / 2;
  truth.position.y = DECK_Y + 1.05;
  truth.visible = false;
  shaker.add(truth);

  const pins = [0, Math.PI, Math.PI, 0];
  const tdc = [0, 3 * Math.PI, Math.PI, 2 * Math.PI];
  const engine = {
    group, shaker, crankshaft, camshafts, cylinders, chamberLight, truth, edgeMaterials,
    pins, tdc,
    setFiringOrder(order) {
      // Compression TDC of the i-th cylinder in the firing order is at i * 180 deg of
      // the 720 deg cycle; its crank pin sits at minus that angle (mod 360 deg).
      order.forEach((cylinder, i) => {
        const k = cylinder - 1;
        if (k < 0 || k > 3) return;
        engine.tdc[k] = i * Math.PI;
        engine.pins[k] = mod(-i * Math.PI, 2 * Math.PI);
        cylinders[k].crankPin.rotation.x = engine.pins[k];
      });
    },
  };
  engine.setFiringOrder([1, 3, 4, 2]);
  return engine;
}

function buildFloorGrid() {
  const grid = new THREE.GridHelper(24, 48, COLORS.cyan, 0x1d3a40);
  grid.material.transparent = true;
  grid.material.opacity = 0.22;
  return grid;
}

// Stylized low-poly coupe: side profile extruded across its width, neon edges,
// translucent shell so the engine shows through. Built with length along X and turned
// so the nose points to +Z (transverse engine across the car, crank along X).
function buildCar() {
  const group = new THREE.Group();
  group.rotation.y = -Math.PI / 2;
  const fades = [];
  const shell = (color, opacity) => {
    const material = new THREE.MeshStandardMaterial({
      color, transparent: true, opacity, metalness: 0.6, roughness: 0.35, depthWrite: false,
      side: THREE.DoubleSide,
    });
    material.userData.base = opacity;
    fades.push(material);
    return material;
  };
  const neon = (mesh, color, opacity) => {
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    material.userData.base = opacity;
    fades.push(material);
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 18), material));
    return mesh;
  };
  const extrude = (shape, depth, bevel) => {
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.8,
      bevelSegments: 2, curveSegments: 18,
    });
    geometry.translate(0, 0, -depth / 2);
    return geometry;
  };

  const WHEEL_Y = 0.62;
  const body = new THREE.Shape();
  body.moveTo(-4.45, WHEEL_Y);
  body.lineTo(-3.78, WHEEL_Y);
  body.absarc(-3.0, WHEEL_Y, 0.78, Math.PI, 0, true);
  body.lineTo(2.22, WHEEL_Y);
  body.absarc(3.0, WHEEL_Y, 0.78, Math.PI, 0, true);
  body.lineTo(4.35, WHEEL_Y);
  body.lineTo(4.62, 0.95);
  body.lineTo(4.45, 1.2);
  body.lineTo(1.5, 1.45);
  body.lineTo(-2.75, 1.5);
  body.lineTo(-4.35, 1.4);
  body.lineTo(-4.62, 1.0);
  body.closePath();
  group.add(neon(new THREE.Mesh(extrude(body, 2.9, 0.12), shell(COLORS.steel, 0.2)),
                 COLORS.cyan, 0.75));

  const cabin = new THREE.Shape();
  cabin.moveTo(1.5, 1.45);
  cabin.lineTo(-0.15, 2.3);
  cabin.lineTo(-2.05, 2.32);
  cabin.lineTo(-3.25, 1.5);
  cabin.closePath();
  group.add(neon(new THREE.Mesh(extrude(cabin, 2.45, 0.08), shell(0x0e7490, 0.22)),
                 COLORS.cyan, 0.55));

  // Hood hinged at the windshield base; rotation.z > 0 lifts it.
  const hood = new THREE.Group();
  hood.position.set(1.5, 1.49, 0);
  const panel = new THREE.Mesh(new THREE.BoxGeometry(2.98, 0.05, 2.85), shell(COLORS.steel, 0.3));
  panel.position.x = 1.49;
  panel.rotation.z = Math.atan2(1.2 - 1.45, 2.95);
  hood.add(neon(panel, COLORS.magenta, 0.85));
  group.add(hood);

  const tire = new THREE.TorusGeometry(0.47, 0.16, 12, 36);
  const rim = new THREE.CylinderGeometry(0.33, 0.33, 0.22, 24);
  rim.rotateX(Math.PI / 2);
  for (const x of [-3.0, 3.0]) {
    for (const z of [-1.55, 1.55]) {
      const wheel = new THREE.Group();
      wheel.position.set(x, WHEEL_Y, z);
      wheel.add(neon(new THREE.Mesh(tire, shell(0x0b1220, 0.9)), COLORS.cyan, 0.35));
      wheel.add(neon(new THREE.Mesh(rim, shell(COLORS.steelLight, 0.7)), COLORS.cyan, 0.6));
      group.add(wheel);
    }
  }

  const strip = (x, y, color) => {
    const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 });
    material.userData.base = 1;
    fades.push(material);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.07, 2.5), material);
    mesh.position.set(x, y, 0);
    return mesh;
  };
  group.add(strip(4.56, 1.06, COLORS.cyan), strip(-4.6, 1.18, COLORS.rose));

  const glow = new THREE.MeshBasicMaterial({
    map: glowTexture(), color: COLORS.cyan, transparent: true, opacity: 0.35,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  glow.userData.base = 0.35;
  fades.push(glow);
  const underglow = new THREE.Mesh(new THREE.PlaneGeometry(10.5, 4.2), glow);
  underglow.rotation.x = -Math.PI / 2;
  underglow.position.y = 0.02;
  group.add(underglow);

  return {
    group,
    hood,
    setOpacity(alpha) {
      fades.forEach((material) => { material.opacity = material.userData.base * alpha; });
      group.visible = alpha > 0.01;
    },
  };
}

// Car / Engine / Reset buttons rendered in the component HTML.
function createViewControls(root, onSelect) {
  const buttons = [...root.querySelectorAll('.engine3d-controls button')];
  let reset = () => {};
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.view) onSelect(button.dataset.view);
      else if (button.dataset.action === 'reset') reset();
    });
  });
  return {
    show(mode) {
      buttons.forEach((button) => {
        if (button.dataset.view) {
          button.setAttribute('aria-pressed', String(button.dataset.view === mode));
        }
      });
    },
    onReset(callback) { reset = callback; },
  };
}

function glowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2,
                                                size / 2);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.25, 'rgba(255,255,255,0.65)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function sparkGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(28 * 3), 3));
  scatter(geometry);
  return geometry;
}

function scatter(geometry) {
  // Mutates the existing buffer in place (no per-event allocation).
  const positions = geometry.attributes.position.array;
  for (let i = 0; i < positions.length; i += 3) {
    const radius = 0.08 + Math.random() * 0.3;
    const angle = Math.random() * Math.PI * 2;
    positions[i] = Math.cos(angle) * radius;
    positions[i + 1] = (Math.random() - 0.3) * 0.25;
    positions[i + 2] = Math.sin(angle) * radius * 0.8;
  }
  geometry.attributes.position.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Helpers

// The engine turns while the replay plays and after it has finished; a pause freezes it.
function isRunning(data) {
  return Boolean(data.animate ?? data.playing);
}

// Reported faults per cylinder. perCylinder[k]: faults shown on cylinder k (the ones
// localized to it, then those without a cylinder, which show on every cylinder).
// located[k]: the first fault localized to cylinder k, or null. Unknown entries are
// ignored; intensity is clamped to 0..1.
function layoutFaults(faults) {
  const list = (Array.isArray(faults) ? faults : [])
    .filter((f) => f && FAULT_KINDS.includes(f.fault))
    .map((f) => ({ ...f, intensity: clamp01(Number(f.intensity) || 0) }));
  const cylinderOf = (f) => (Number.isInteger(f.cylinder) && f.cylinder >= 1 && f.cylinder <= 4
    ? f.cylinder - 1 : null);
  const wide = list.filter((f) => cylinderOf(f) === null);
  const own = [0, 1, 2, 3].map((k) => list.filter((f) => cylinderOf(f) === k));
  return {
    perCylinder: own.map((faults) => faults.concat(wide)),
    located: own.map((faults) => faults[0] || null),
    misfire: list.some((f) => f.fault === 'misfire'),
  };
}

function find(faults, kind) {
  return faults.find((f) => f.fault === kind) || null;
}

function has(faults, kind) {
  return faults.some((f) => f.fault === kind);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp01(value) {
  return Math.min(Math.max(value, 0), 1);
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

function mod(value, period) {
  return ((value % period) + period) % period;
}

// True when the cyclic angle moved past `mark` between two frames.
function crossed(previous, current, mark) {
  return previous <= current ? previous < mark && mark <= current
                             : previous < mark || mark <= current;
}

// Smooth 0..1..0 valve lift between open and close angles of the 720 deg cycle.
function lift(angle, open, close) {
  if (angle <= open || angle >= close) return 0;
  return Math.sin((Math.PI * (angle - open)) / (close - open)) ** 2;
}

// Minimal orbit control: drag or arrow keys rotate, wheel or +/- zoom. The base
// radius and look-at target come from the car/engine transition each frame.
function createOrbit(root, camera, start) {
  const view = { ...start };
  let drag = null;
  const clampPhi = (phi) => Math.min(Math.max(phi, 0.25), 1.5);
  const clampZoom = (zoom) => Math.min(Math.max(zoom, 0.5), 2.0);
  const onDown = (event) => {
    if (event.target.closest?.('button')) return;
    drag = { x: event.clientX, y: event.clientY, theta: view.theta, phi: view.phi };
    root.setPointerCapture?.(event.pointerId);
  };
  const onMove = (event) => {
    if (!drag) return;
    view.theta = drag.theta - (event.clientX - drag.x) * 0.008;
    view.phi = clampPhi(drag.phi - (event.clientY - drag.y) * 0.006);
  };
  const onUp = () => { drag = null; };
  const onWheel = (event) => {
    event.preventDefault();
    view.zoom = clampZoom(view.zoom * (event.deltaY > 0 ? 1.08 : 0.92));
  };
  const onKey = (event) => {
    if (event.target.closest?.('button')) return;
    const steps = { ArrowLeft: [0.12, 0, 1], ArrowRight: [-0.12, 0, 1], ArrowUp: [0, -0.08, 1],
                    ArrowDown: [0, 0.08, 1], '+': [0, 0, 0.9], '=': [0, 0, 0.9], '-': [0, 0, 1.1] };
    const step = steps[event.key];
    if (!step) return;
    event.preventDefault();
    view.theta += step[0];
    view.phi = clampPhi(view.phi + step[1]);
    view.zoom = clampZoom(view.zoom * step[2]);
  };
  root.addEventListener('pointerdown', onDown);
  root.addEventListener('pointermove', onMove);
  root.addEventListener('pointerup', onUp);
  root.addEventListener('pointercancel', onUp);
  root.addEventListener('wheel', onWheel, { passive: false });
  root.addEventListener('keydown', onKey);
  return {
    view,
    apply(radius, target) {
      const r = radius * view.zoom;
      camera.position.set(
        target.x + r * Math.sin(view.phi) * Math.sin(view.theta),
        target.y + r * Math.cos(view.phi),
        target.z + r * Math.sin(view.phi) * Math.cos(view.theta),
      );
      camera.lookAt(target);
    },
    dispose() {
      root.removeEventListener('pointerdown', onDown);
      root.removeEventListener('pointermove', onMove);
      root.removeEventListener('pointerup', onUp);
      root.removeEventListener('pointercancel', onUp);
      root.removeEventListener('wheel', onWheel);
      root.removeEventListener('keydown', onKey);
    },
  };
}
