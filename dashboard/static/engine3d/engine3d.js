// Engine view for the replay dashboard (Three.js, bundled locally).
//
// Streamlit calls render(root, data) on every rerun. The scene is created once per
// root element and kept in a WeakMap; later calls only replace the target state, and
// the requestAnimationFrame loop eases toward it. Geometry is built once.
//
// What drives what (all from `data`, sent by dashboard/engine3d.py):
//   rpm            crank speed (shown slowed by VISUAL_SPEED, labelled on screen)
//   firing_order   which cylinder fires at each 180 deg of the 720 deg cycle
//   effect         fault effect to show: the alert class, or an accepted fault class
//   intensity      model confidence for that class (or evidence level / 3)
//   vibration_ratio  current vibration RMS / normal-reference RMS (real sensor data)
//   truth_cylinder only when the evaluation-truth overlay is on (marked as truth)
// The model does not localize faults, so effects are engine-wide; nothing here
// picks a cylinder from model output.

import * as THREE from '../vendor/three/three.module.min.js';

export const VISUAL_SPEED = 1 / 25;
const views = new WeakMap();
// Instrumentation for the persistence check (read by scripts/check_engine3d.mjs).
const stats = (window.__engine3d = window.__engine3d || { mounts: 0, updates: 0, disposed: 0 });
stats.events = stats.events || { fired: 0, skipped: 0, knock: 0, valveTicks: 0 };

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
  cyan: 0x22d3ee,
  magenta: 0xf472b6,
  violet: 0xa78bfa,
  amber: 0xfbbf24,
  rose: 0xfb7185,
  green: 0x4ade80,
  slate: 0x64748b,
  steel: 0x1b2a3f,
  steelLight: 0x3a5272,
  piston: 0x9fb3c8,
  flame: 0x7dd3fc,
};

// Geometry (arbitrary units; one cylinder pitch = 1).
const PITCH = 1.0;
const CRANK_R = 0.42;
const ROD_L = 1.4;
const PISTON_H = 0.42;
const DECK_Y = 2.5;
const CHAMBER_Y = 2.32;

function createView(root) {
  const hud = root.querySelector('.engine3d-hud');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (error) {
    return fallbackView(root);
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  root.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
  scene.add(new THREE.HemisphereLight(0x9fd8ff, 0x0a0f1c, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(4, 7, 6);
  scene.add(key);
  const rim = new THREE.PointLight(COLORS.cyan, 22, 25);
  rim.position.set(-5, 3, -4);
  scene.add(rim);

  const engine = buildEngine();
  scene.add(engine.group);
  scene.add(buildFloorGrid());

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const orbit = createOrbit(root, camera, { radius: 10.5, theta: 0.72, phi: 1.12 });
  const timer = new THREE.Timer();
  timer.connect(document);
  const sim = {
    target: {},
    crank: 0, // crank angle (rad), continuous
    lastCycle: 0,
    flash: [0, 0, 0, 0], // per-cylinder combustion glow 0..1
    flashColor: new THREE.Color(COLORS.flame),
    sparks: [0, 0, 0, 0],
    ticks: [0, 0, 0, 0],
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
    orbit.apply();
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

    // Combustion events at each cylinder's compression TDC (plus a small offset).
    engine.tdc.forEach((tdc, k) => {
      const fire = mod(tdc + (8 * Math.PI) / 180, 4 * Math.PI);
      if (crossed(sim.lastCycle, cycle, fire) && t.playing) ignite(k, effect, intensity);
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
      if (exhaust === 0 && cyl.exhaustOpen && t.playing) seatImpact(k, effect, intensity);
      if (intake === 0 && cyl.intakeOpen && t.playing) seatImpact(k, effect, intensity);
      cyl.exhaustOpen = exhaust > 0;
      cyl.intakeOpen = intake > 0;
      cyl.exhaust.position.y = cyl.valveY - 0.13 * exhaust;
      cyl.intake.position.y = cyl.valveY - 0.13 * intake;

      // Effects decay.
      sim.flash[k] = Math.max(0, sim.flash[k] - dt * (still ? 0 : 3.2));
      sim.sparks[k] = Math.max(0, sim.sparks[k] - dt * 5);
      sim.ticks[k] = Math.max(0, sim.ticks[k] - dt * 6);
      cyl.flame.material.opacity = sim.flash[k];
      cyl.flame.material.color.copy(cyl.flameColor);
      cyl.flame.scale.setScalar(0.55 + 0.45 * sim.flash[k]);
      cyl.sparks.material.opacity = sim.sparks[k];
      cyl.sparks.rotation.y += dt * 6;
      cyl.tick.material.opacity = sim.ticks[k];
    });

    // Chamber light follows the most recent firing cylinder.
    let brightest = 0;
    sim.flash.forEach((value, k) => { if (value > sim.flash[brightest]) brightest = k; });
    engine.chamberLight.position.set(engine.cylinders[brightest].x, CHAMBER_Y, 0);
    engine.chamberLight.intensity = 14 * sim.flash[brightest];
    engine.chamberLight.color.copy(engine.cylinders[brightest].flameColor);

    // Shake: the measured vibration level relative to normal, plus knock jolts.
    const ratio = Number(t.vibration_ratio) || 1;
    const base = still ? 0 : Math.min(Math.max(ratio - 1, 0), 3) * 0.006;
    sim.jolt = Math.max(0, sim.jolt - dt * 9);
    const amp = still ? 0 : base + sim.jolt * 0.035;
    engine.shaker.position.set((Math.random() - 0.5) * amp, (Math.random() - 0.5) * amp,
                               (Math.random() - 0.5) * amp);

    // Block outline color follows the model state; an active alert pulses gently.
    tmpColor.set(t.edge_color || '#22d3ee');
    sim.edgeColor.lerp(tmpColor, Math.min(1, dt * 4));
    const pulse = t.alert && !still ? 0.75 + 0.25 * Math.sin(timer.getElapsed() * 4) : 0.85;
    engine.edgeMaterials.forEach((material) => {
      material.color.copy(sim.edgeColor);
      material.opacity = material.userData.base * pulse;
    });
    engine.truth.visible = Number.isInteger(t.truth_cylinder);
    if (engine.truth.visible) {
      engine.truth.position.x = engine.cylinders[t.truth_cylinder - 1].x;
      engine.truth.rotation.y += still ? 0 : dt * 0.8;
    }
  }

  function ignite(k, effect, intensity) {
    const cyl = engine.cylinders[k];
    if (effect === 'misfire' && Math.random() < 0.35 + 0.5 * intensity) {
      // Model says misfire (cylinder unknown): firing pulses become weak or missing.
      sim.flash[k] = Math.random() < 0.5 ? 0 : 0.25;
      stats.events.skipped += 1;
      sim.stutter = 1;
      cyl.flameColor.set(COLORS.amber);
      return;
    }
    sim.flash[k] = 1;
    stats.events.fired += 1;
    cyl.flameColor.set(effect === 'knock_like' ? COLORS.magenta : COLORS.flame);
    if (effect === 'knock_like' && Math.random() < 0.4 + 0.6 * intensity) {
      sim.sparks[k] = 1;
      stats.events.knock += 1;
      sim.jolt = Math.max(sim.jolt, 0.6 + 0.4 * intensity);
      scatter(cyl.sparks.geometry);
    }
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
      const firstOrder = !sim.target.firing_order;
      sim.target = data;
      if (data.height) root.style.setProperty('--engine3d-height', `${data.height}px`);
      if (firstOrder && Array.isArray(data.firing_order)) engine.setFiringOrder(data.firing_order);
      renderHud(hud, data);
    },
  };
}

function renderHud(hud, data) {
  const rpm = Number.isFinite(data.rpm) ? `${Math.round(data.rpm).toLocaleString('en-US')} RPM`
                                        : 'RPM —';
  const order = (data.firing_order || [1, 3, 4, 2]).join('-');
  const speed = `shown at ×1/${Math.round(1 / VISUAL_SPEED)} speed`;
  hud.replaceChildren(
    line(`Inline-4 · firing ${order} · ${rpm} · ${speed}`),
    line(data.status_text || '', data.edge_color),
  );
  if (Number.isInteger(data.truth_cylinder)) {
    hud.append(line(`Evaluation truth overlay: fault in cylinder ${data.truth_cylinder} ` +
                    '(not used by the model)', '#fbbf24'));
  }
}

function line(text, color) {
  const element = document.createElement('div');
  element.textContent = text;
  if (color) element.style.color = color;
  return element;
}

function fallbackView(root) {
  const fallback = root.querySelector('.engine3d-fallback');
  fallback.hidden = false;
  fallback.textContent = 'WebGL is not available in this browser, so the 3D engine view is ' +
    'off. The diagnosis panel and charts show the same model output.';
  return { update() {} };
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
    const piston = edged(new THREE.Mesh(pistonGeometry, pistonMaterial), 0.35);
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
      color: COLORS.magenta, size: 0.06, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    sparks.position.set(x, CHAMBER_Y, 0);

    const tick = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTexture, color: COLORS.violet, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    tick.position.set(x, DECK_Y + 0.12, 0);
    tick.scale.set(0.9, 0.45, 1);

    shaker.add(bore, piston, rod, intake, exhaust, flame, sparks, tick);
    cylinders.push({ x, piston, rod, intake, exhaust, valveY, flame, sparks, tick, crankPin,
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
  group.position.y = -1.0;
  return engine;
}

function buildFloorGrid() {
  const grid = new THREE.GridHelper(24, 48, COLORS.cyan, 0x12324a);
  grid.material.transparent = true;
  grid.material.opacity = 0.22;
  grid.position.y = -2.0;
  return grid;
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

// Minimal orbit control: drag or arrow keys rotate, wheel or +/- zoom.
function createOrbit(root, camera, start) {
  const view = { ...start };
  const target = new THREE.Vector3(0, 0.6, 0);
  let drag = null;
  const clampPhi = (phi) => Math.min(Math.max(phi, 0.25), 1.55);
  const clampRadius = (radius) => Math.min(Math.max(radius, 5), 22);
  const onDown = (event) => {
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
    view.radius = clampRadius(view.radius * (event.deltaY > 0 ? 1.08 : 0.92));
  };
  const onKey = (event) => {
    const steps = { ArrowLeft: [0.12, 0, 1], ArrowRight: [-0.12, 0, 1], ArrowUp: [0, -0.08, 1],
                    ArrowDown: [0, 0.08, 1], '+': [0, 0, 0.9], '=': [0, 0, 0.9], '-': [0, 0, 1.1] };
    const step = steps[event.key];
    if (!step) return;
    event.preventDefault();
    view.theta += step[0];
    view.phi = clampPhi(view.phi + step[1]);
    view.radius = clampRadius(view.radius * step[2]);
  };
  root.addEventListener('pointerdown', onDown);
  root.addEventListener('pointermove', onMove);
  root.addEventListener('pointerup', onUp);
  root.addEventListener('pointercancel', onUp);
  root.addEventListener('wheel', onWheel, { passive: false });
  root.addEventListener('keydown', onKey);
  return {
    view,
    target,
    apply() {
      camera.position.set(
        target.x + view.radius * Math.sin(view.phi) * Math.sin(view.theta),
        target.y + view.radius * Math.cos(view.phi),
        target.z + view.radius * Math.sin(view.phi) * Math.cos(view.theta),
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
