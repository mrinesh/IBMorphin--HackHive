// Engine view for the DRIVEX front-end prototype.
// Ported from dashboard/static/engine3d/engine3d.js: same pistons, valves, crankshaft,
// spark colors and red faulty-piston highlight; engine only (the car view and the
// car-to-engine transition are removed). Plain script: THREE comes from the CDN global.
// Nothing here talks to a backend; the page feeds it generated sample state.

const VISUAL_SPEED = 1 / 25;
const stats = (window.__engine3d = window.__engine3d || { mounts: 0, updates: 0, disposed: 0 });
stats.events = stats.events || { fired: 0, skipped: 0, knock: 0, valveTicks: 0 };

function mountEngine(root) {
  const view = createView(root);
  stats.mounts += 1;
  return { update(data) { view.update(data || {}); stats.updates += 1; } };
}

const COLORS = {
  cyan: 0x64d2ff,
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
  alarm: 0xff2d2d, // localized faulty piston
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
const DETAIL_Y = 1.6;
// Reduced-motion stand-ins for the animated effects (steady glow per model state).
const STEADY = {
  normal: { flash: 0.35, color: 0xff7a1a },
  misfire: { flash: 0.3, color: 0x2f8bff },
  knock_like: { flash: 0.6, color: 0xb48cff },
  valve_clearance_like: { flash: 0.35, color: 0x22c55e },
};
const CAMERA = { radius: 11, target: null }; // target is set once THREE exists (createView)
const CHAMBER_Y = 2.32;

function createView(root) {
  // The overlay text may live outside the canvas box (data-engine-hud) so it never covers the model.
  const hud = root.querySelector('.engine3d-hud') || document.querySelector('[data-engine-hud]');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (error) {
    return fallbackView(root);
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  root.prepend(renderer.domElement);

  const canvas = renderer.domElement;
  const notice = root.querySelector('.engine3d-fallback');
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    canvas.style.visibility = 'hidden';
    notice.hidden = false;
    notice.textContent = '3D view paused: the graphics context was reset. Restoring…';
    stats.contextLost = (stats.contextLost || 0) + 1;
  });
  canvas.addEventListener('webglcontextrestored', () => {
    canvas.style.visibility = '';
    notice.hidden = true;
  });

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
  scene.add(new THREE.HemisphereLight(0x9fd8ff, 0x0a0f1c, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(4, 7, 6);
  scene.add(key);
  const rim = new THREE.PointLight(COLORS.cyan, 12, 25);
  rim.position.set(-5, 3, -4);
  scene.add(rim);

  CAMERA.target = new THREE.Vector3(0, DETAIL_Y + 1.45, 0);
  const engine = buildEngine();
  const assembly = new THREE.Group();
  assembly.add(engine.group);
  assembly.position.set(0, DETAIL_Y, 0);
  scene.add(assembly);
  scene.add(buildFloorGrid());

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const START = { theta: 0.72, phi: 1.12, zoom: 1 };
  const orbit = createOrbit(root, camera, START);
  const viewButtons = createViewControls(root, () => {});
  viewButtons.onReset(() => Object.assign(orbit.view, START));
  const clock = new THREE.Clock();
  const timer = { getDelta: () => clock.getDelta(), getElapsed: () => clock.elapsedTime, dispose() {} };
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
    alarm: [0, 0, 0, 0], // red highlight on the localized cylinder's piston, 0..1
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
    const dt = Math.min(timer.getDelta(), 0.1);
    if (!visible || document.hidden) return;
    step(dt);
    orbit.apply(CAMERA.radius, CAMERA.target);
    renderer.render(scene, camera);
  }

  function step(dt) {
    const t = sim.target;
    const still = reduceMotion.matches;
    const rpm = Number(t.rpm) || 0;
    const effect = t.effect || null;
    const intensity = Math.min(Math.max(Number(t.intensity) || 0, 0), 1);

    // Crank: real RPM scaled to a watchable speed. A misfire makes the crank stutter.
    if (t.playing && !still) {
      let omega = (rpm / 60) * 2 * Math.PI * VISUAL_SPEED;
      if (effect === 'misfire') omega *= 1 - 0.25 * sim.stutter;
      sim.crank += omega * dt;
    }
    sim.stutter = Math.max(0, sim.stutter - dt * 2.5);
    const theta = sim.crank;
    stats.crank = theta;
    const cycle = mod(theta, 4 * Math.PI);

    // A localized fault shows only on its cylinder; otherwise on every cylinder.
    const locSet = Array.isArray(t.model_cylinders) ? t.model_cylinders.map((c) => c - 1)
      : Number.isInteger(t.model_cylinder) ? [t.model_cylinder - 1] : [];
    const located = locSet.length ? locSet[0] : null;
    // Per-cylinder effects (several faults at once), else the one effect on the named cylinders.
    const effectOn = (k) => (t.cyl_effects && t.cyl_effects[k] !== undefined ? t.cyl_effects[k]
      : locSet.length === 0 || locSet.includes(k) ? effect : null);

    // Combustion events at each cylinder's compression TDC (plus a small offset).
    engine.tdc.forEach((tdc, k) => {
      const fire = mod(tdc + (8 * Math.PI) / 180, 4 * Math.PI);
      if (crossed(sim.lastCycle, cycle, fire) && t.playing) ignite(k, effectOn(k), intensity);
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
      if (exhaust === 0 && cyl.exhaustOpen && t.playing) seatImpact(k, effectOn(k), intensity);
      if (intake === 0 && cyl.intakeOpen && t.playing) seatImpact(k, effectOn(k), intensity);
      cyl.exhaustOpen = exhaust > 0;
      cyl.intakeOpen = intake > 0;
      cyl.exhaust.position.y = cyl.valveY - 0.13 * exhaust;
      cyl.intake.position.y = cyl.valveY - 0.13 * intake;

      if (still) {
        // Reduced motion: no firing animation, so show the model state as a steady glow
        // (same colors as the animated effects) instead of hiding it.
        const own = effectOn(k);
        const steady = STEADY[own] || STEADY.normal;
        sim.flash[k] = t.playing || own ? steady.flash * (own ? 0.5 + 0.5 * intensity : 1) : 0;
        cyl.flameColor.set(steady.color);
        sim.sparks[k] = own === 'knock_like' ? 0.7 : 0;
        sim.ticks[k] = own === 'valve_clearance_like' ? 0.7 : 0;
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
    tmpColor.set(t.edge_color || '#64d2ff');
    sim.edgeColor.lerp(tmpColor, Math.min(1, dt * 4));
    const pulse = t.alert && !still ? 0.75 + 0.25 * Math.sin(timer.getElapsed() * 4) : 0.85;
    engine.edgeMaterials.forEach((material) => {
      material.color.copy(sim.edgeColor);
      material.opacity = material.userData.base * pulse;
    });
    // Faulty-cylinder highlight: the piston the localizer names glows red and pulses;
    // it fades in when a cylinder is named and out when the alert clears.
    const beat = still ? 1 : 0.65 + 0.35 * Math.sin(timer.getElapsed() * 5);
    engine.cylinders.forEach((cyl, k) => {
      const goal = locSet.includes(k) ? 1 : 0;
      sim.alarm[k] += (goal - sim.alarm[k]) * Math.min(1, dt * (still ? 60 : 4));
      const level = sim.alarm[k] * beat;
      cyl.piston.material.emissiveIntensity = 1.6 * level;
      cyl.alarm.material.opacity = 0.85 * level;
    });
    if (located !== null) {
      const hot = engine.cylinders[located];
      engine.alarmLight.position.set(hot.x, hot.piston.position.y + 0.35, 0.45);
    }
    engine.alarmLight.intensity = located === null ? 0 : 9 * sim.alarm[located] * beat;
    engine.model.visible = located !== null;
    stats.modelCylinder = located === null ? null : located + 1; // for browser checks
    if (engine.model.visible) {
      engine.model.position.x = engine.cylinders[located].x;
      engine.model.material.color.copy(sim.edgeColor);
      engine.model.material.opacity = 0.95 * pulse;
    }
    engine.truth.visible = Number.isInteger(t.truth_cylinder);
    if (engine.truth.visible) {
      engine.truth.position.x = engine.cylinders[t.truth_cylinder - 1].x;
      engine.truth.rotation.y += still ? 0 : dt * 0.8;
    }
  }

  function ignite(k, effect, intensity) {
    const cyl = engine.cylinders[k];
    if (effect === 'misfire' && Math.random() < 0.35 + 0.5 * intensity) {
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
    const fire = FIRE[effect] || FIRE.normal;
    sim.flash[k] = 1;
    sim.plug[k] = 1;
    sim.embers[k] = 1;
    cyl.burning = true;
    stats.events.fired += 1;
    cyl.flameColor.set(fire.flame);
    cyl.core.material.color.set(fire.core);
    cyl.plug.material.color.set(fire.plug);
    launchEmbers(cyl, effect === 'knock_like', fire.ember);
    if (effect === 'knock_like' && Math.random() < 0.4 + 0.6 * intensity) {
      sim.sparks[k] = 1;
      stats.events.knock += 1;
      sim.jolt = Math.max(sim.jolt, 0.6 + 0.4 * intensity);
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

  function seatImpact(k, effect, intensity) {
    if (effect !== 'valve_clearance_like') return;
    sim.ticks[k] = 0.5 + 0.5 * intensity;
    stats.events.valveTicks += 1;
    sim.jolt = Math.max(sim.jolt, 0.25 * intensity);
  }

  function dispose() {
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
    stats.disposed += 1;
  }

  renderer.setAnimationLoop(frame);

  return {
    update(data) {
      const previous = sim.target;
      const firstOrder = !previous.firing_order;
      sim.target = data;
      if (firstOrder && Array.isArray(data.firing_order)) engine.setFiringOrder(data.firing_order);
      renderHud(hud, data);
      root.setAttribute('aria-label', `Engine view. ${data.status_text || ''}. Drag or use arrow keys to orbit, plus and minus to zoom.`);
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
  if (data.truth_text) {
    hud.append(line(data.truth_text, '#fbbf24'));
  } else if (Number.isInteger(data.truth_cylinder)) {
    hud.append(line(`Evaluation truth overlay: fault in cylinder ${data.truth_cylinder} ` +
                    '(not used by the model)', '#fbbf24'));
  }
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
    return { glow: el('rect', { x, y: 40, width: 50, height: 16, opacity: 0 }, svg),
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
    if (target.playing && !reduceMotion.matches) {
      crank += ((Number(target.rpm) || 0) / 60) * 2 * Math.PI * VISUAL_SPEED * dt;
    }
    pistons.forEach(({ piston, glow }, k) => {
      const psi = crank - tdc[k];
      const height = CRANK_R * Math.cos(psi) + Math.sqrt(ROD_L ** 2 - (CRANK_R * Math.sin(psi)) ** 2);
      piston.setAttribute('y', String(56 + (CRANK_R + ROD_L - height) * 90));
      glow.setAttribute('opacity', String(Math.max(0, Math.cos(mod(crank - tdc[k], 4 * Math.PI)))
                                          * (target.playing ? 0.6 : 0)));
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
      pistons.forEach(({ glow }) => glow.setAttribute('fill', color));
      status.replaceChildren(
        line('WebGL is unavailable, so a 2D schematic replaces the 3D view.'),
        line(data.status_text || '', color),
      );
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

    shaker.add(bore, piston, rod, intake, exhaust, flame, core, plug, embers, sparks, tick);
    cylinders.push({ x, piston, alarm, rod, intake, exhaust, valveY, flame, core, plug, embers,
                     emberVel: new Float32Array(EMBERS * 3), burning: false, sparks, tick,
                     crankPin,
                     flameColor: new THREE.Color(COLORS.flame), intakeOpen: false,
                     exhaustOpen: false });
  }

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
  // Red light that follows the localized piston and tints the parts around it.
  const alarmLight = new THREE.PointLight(COLORS.alarm, 0, 1.4, 2);
  shaker.add(alarmLight);

  // Evaluation-truth marker (only shown when the overlay is on).
  const truth = new THREE.Mesh(
    new THREE.TorusGeometry(0.55, 0.018, 8, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.amber, transparent: true, opacity: 0.9 }),
  );
  truth.rotation.x = Math.PI / 2;
  truth.position.y = DECK_Y + 1.05;
  truth.visible = false;
  shaker.add(truth);

  // Model localization marker: the cylinder the localizer names, in the state color.
  const model = new THREE.Mesh(
    new THREE.TorusGeometry(0.46, 0.03, 8, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.amber, transparent: true, opacity: 0.95 }),
  );
  model.rotation.x = Math.PI / 2;
  model.position.y = DECK_Y + 0.95;
  model.visible = false;
  shaker.add(model);

  const pins = [0, Math.PI, Math.PI, 0];
  const tdc = [0, 3 * Math.PI, Math.PI, 2 * Math.PI];
  const engine = {
    group, shaker, crankshaft, camshafts, cylinders, chamberLight, alarmLight, truth, model,
    edgeMaterials,
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
  grid.material.opacity = 0.16;
  return grid;
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
