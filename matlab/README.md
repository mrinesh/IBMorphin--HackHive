# MATLAB workspace

From the repository root in MATLAB:

```matlab
addpath('matlab');
project = initialize_project();          % inspect settings and installed products
runs = generate_integration_batch();     % write data/raw and evaluation_truth
report = validate_run_matlab('data/raw/run_00001', ...
    'evaluation_truth/run_00001_events.json');
plot_run_overview('data/raw/run_00001', ...
    'evaluation_truth/run_00001_events.json', ...
    'reports/generated/integration_v1_run_00001.png');   % review figure
```

`generate_integration_batch` reads `configs/mvp.json` (shared settings) and
`configs/integration_batch.json` (batch scenarios and generator parameters). It
refuses to overwrite existing runs unless called with `Overwrite=true`; use
`OutputRoot`/`TruthRoot` to write elsewhere and `Runs` to select scenarios.
Requires Signal Processing Toolbox (`butter`, `pwelch`, `hann`).

## Generator structure

| File | Role |
|---|---|
| `load_project_config.m` | Read `mvp.json` and check sampling limits |
| `generate_integration_batch.m` | Entry point: loop scenarios, generate, export |
| `export_run.m` | Write `waveforms.mat` (`-v7`), `telemetry.csv`, `metadata.json`, private truth |
| `validate_run_matlab.m` | Lengths, dtype, clipping, firing rate vs RPM, intended fault signature |
| `plot_run_overview.m` | Review figure (spectrograms, zooms, RPM) using evaluation truth |
| `sim/make_streams.m` | Independent `mrg32k3a` substreams per run and purpose |
| `sim/operating_profile.m` | RPM, load, throttle at 1 kHz control rate |
| `sim/integrate_crank.m`, `sim/crank_event_times.m` | Crank angle from RPM and its inverse |
| `sim/combustion_schedule.m` | Combustion and valve-closing events in firing order |
| `sim/apply_fault.m` | Misfire, knock-like, valve-clearance-like modifications |
| `sim/render_sensor.m` | Audio or vibration channel from events, orders, resonance, noise |
| `sim/render_telemetry.m` | Long-form telemetry at configured rates |
| `sim/generate_run.m` | One complete run, metadata, and truth |

Events are keyed by crank angle and converted to time through the integrated
RPM, so a misfire's RPM dip moves later events consistently in every stream.
Transients are damped sinusoids evaluated at exact sample times with a smooth
onset; the noise-free spectrum above 90% of Nyquist is 42-57 dB below the total
for the integration runs.

## Modeling status

The structure (firing order, crank-angle timing, shared events across sensors)
is physically motivated. All amplitudes, frequencies, decay times, sensor
transfer settings, and fault mappings are heuristic choices, not fitted to
engine measurements. Units are nominal. Do not export MATLAB tables/objects
inside the waveform file or use hidden fault truth as an inference input.
