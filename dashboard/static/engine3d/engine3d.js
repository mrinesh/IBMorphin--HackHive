// Engine view for the replay dashboard (Three.js, bundled locally).
//
// Streamlit calls render(root, data) on every rerun. The scene is created once per
// root element and kept in a WeakMap; later calls only replace the target state,
// and the requestAnimationFrame loop eases toward it. Geometry is built once.

import * as THREE from '../vendor/three/three.module.min.js';

const views = new WeakMap();
// Instrumentation for the persistence check (read by tests and the browser console).
const stats = (window.__engine3d = window.__engine3d || { mounts: 0, updates: 0, disposed: 0 });

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
  steel: 0x1b2a3f,
  steelLight: 0x2b3f5c,
  piston: 0x8aa0b8,
};

function createView(root) {
  const hud = root.querySelector('.engine3d-hud');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (error) {
    const fallback = root.querySelector('.engine3d-fallback');
    fallback.hidden = false;
    fallback.textContent = 'WebGL is not available in this browser, so the 3D view is off.';
    return { update() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  root.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  scene.add(new THREE.HemisphereLight(0x9fd8ff, 0x0a0f1c, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(4, 6, 5);
  scene.add(key);
  const rim = new THREE.PointLight(COLORS.cyan, 18, 20);
  rim.position.set(-4, 2, -3);
  scene.add(rim);

  const engine = buildEngine();
  scene.add(engine.group);

  const orbit = createOrbit(root, camera, { radius: 9, theta: 0.75, phi: 1.1 });
  const state = { target: {}, crank: 0 };
  const timer = new THREE.Timer();
  timer.connect(document);
  const sceneId = Math.random().toString(36).slice(2, 8);
  root.dataset.sceneId = sceneId;

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
    const rpm = Number(state.target.rpm) || 0;
    if (state.target.playing) state.crank += dt * (rpm / 60) * 2 * Math.PI * 0.025;
    engine.crankshaft.rotation.x = state.crank;
    orbit.apply();
    renderer.render(scene, camera);
  }
  renderer.setAnimationLoop(frame);

  function dispose() {
    renderer.setAnimationLoop(null);
    timer.dispose();
    resizeObserver.disconnect();
    intersection.disconnect();
    orbit.dispose();
    scene.traverse((object) => {
      object.geometry?.dispose?.();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => material?.dispose?.());
    });
    renderer.dispose();
    stats.disposed += 1;
  }

  return {
    update(data) {
      state.target = data;
      if (data.height) root.style.setProperty('--engine3d-height', `${data.height}px`);
      hud.textContent = `scene ${sceneId} · ${Math.round(Number(data.rpm) || 0)} rpm`;
    },
  };
}

function edges(mesh, color = COLORS.cyan, opacity = 0.55) {
  const lines = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry, 25),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
  mesh.add(lines);
  return mesh;
}

function buildEngine() {
  const group = new THREE.Group();
  const block = edges(new THREE.Mesh(
    new THREE.BoxGeometry(4.6, 2.2, 1.6),
    new THREE.MeshStandardMaterial({ color: COLORS.steel, transparent: true, opacity: 0.25,
                                     metalness: 0.6, roughness: 0.4, depthWrite: false }),
  ));
  block.position.y = 1.4;
  group.add(block);

  const pistonGeometry = new THREE.CylinderGeometry(0.36, 0.36, 0.5, 32);
  const pistonMaterial = new THREE.MeshStandardMaterial({ color: COLORS.piston, metalness: 0.8,
                                                          roughness: 0.3 });
  const pistons = [];
  for (let k = 0; k < 4; k += 1) {
    const piston = edges(new THREE.Mesh(pistonGeometry, pistonMaterial), COLORS.cyan, 0.35);
    piston.position.set(-1.65 + k * 1.1, 1.6, 0);
    group.add(piston);
    pistons.push(piston);
  }

  const crankshaft = new THREE.Group();
  const journal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.12, 0.12, 4.8, 24),
    new THREE.MeshStandardMaterial({ color: COLORS.steelLight, metalness: 0.9, roughness: 0.25 }),
  );
  journal.rotation.z = Math.PI / 2;
  crankshaft.add(journal);
  crankshaft.position.y = 0;
  group.add(crankshaft);

  return { group, pistons, crankshaft };
}

// Minimal orbit control: drag or arrow keys rotate, wheel or +/- zoom.
function createOrbit(root, camera, start) {
  const view = { ...start };
  const target = new THREE.Vector3(0, 1, 0);
  let drag = null;
  const clampPhi = (phi) => Math.min(Math.max(phi, 0.25), 1.5);
  const clampRadius = (radius) => Math.min(Math.max(radius, 4.5), 16);
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
