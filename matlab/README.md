# MATLAB workspace

From the repository root in MATLAB:

```matlab
addpath('matlab');
project = initialize_project();
```

The initialization function loads `configs/mvp.json`, checks basic sampling limits,
and prints MATLAB/product information. It writes no sensor data. The full shared
schema and cross-field checks run through the Python `validate-config` command.

MATLAB access is assumed by the project. Simulink, Signal Processing Toolbox,
Powertrain Blockset, and AUTOSAR Blockset are separate availability checks. A
product inventory does not prove its license is usable; exercise any required
specialized function before making it a dependency.

The next milestone is a reusable event-and-response generator, then numeric
`single` arrays saved with `-v7`, telemetry CSV, and JSON metadata. See
`docs/DATA_CONTRACT.md`. Do not export MATLAB tables/objects inside the waveform
file or use hidden fault truth as an inference input.
