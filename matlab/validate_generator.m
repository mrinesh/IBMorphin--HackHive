function report = validate_generator(options)
%VALIDATE_GENERATOR Generator checks that do not depend on one exported run.
%   report = validate_generator() from the repository root writes
%   reports/generated/generator_validation/validation.json and
%   matlab_features.json (MATLAB reference values for Python feature checks);
%   other batches (Config=...) write to generator_validation/<batch_id>/.
%   Checks:
%     1. firing frequency from the audio envelope at fixed RPM (within 1%)
%     2. each fault's signature increases with severity
%     3. audio-vibration event lag equals the configured sensor delays
%     4. noise-free energy above 0.9 x Nyquist (aliasing control)
%     5. bit-identical regeneration
%     6. reference feature values on exported core_v1 runs (if present)
%     7. fault combinations: in every pair and the triple, each fault keeps at
%        least half of its single-fault signature effect (vs normal) when the
%        same run is compared with and without that fault (not masked)
%     8. contract 1.2 batches (crank and/or thermal generator_params): the
%        1.1 streams are bit-identical with those blocks removed, and a
%        misfire slows the crank on the misfiring stroke and cools its EGT

    arguments
        options.Config (1,1) string = ""
        options.ReferenceRuns (1,:) string = ["run_00003", "run_00005", "run_00007"]
    end
    root = string(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(root, "matlab", "sim"));
    if options.Config == ""
        options.Config = fullfile(root, "configs", "core_v1.json");
    end
    project = load_project_config(root);
    batch = jsondecode(fileread(options.Config));
    outDir = fullfile(root, "reports", "generated", "generator_validation");
    if batch.batch_id ~= "core_v1"  % keep the core_v1 reference files in place
        outDir = fullfile(outDir, batch.batch_id);
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    report = struct();
    fsA = project.streams.audio.sample_rate_hz;
    fsV = project.streams.vibration.sample_rate_hz;

    % 1. Firing frequency at fixed RPM.
    rows = struct('rpm', {}, 'expected_hz', {}, 'measured_hz', {}, 'error_pct', {});
    for rpm = [1000, 2000, 3000]
        run = generate_run(project, batch, scenario(rpm, 0.5, 'normal', NaN, 0));
        envelope = band_envelope(fillmissing(double(run.audio), 'constant', 0), fsA, [500 3000]);
        envelope = envelope - mean(envelope);
        n = numel(envelope);
        spectrum = abs(fft(envelope .* hann(n, 'periodic')));
        f = (0:n - 1)' * fsA / n;
        expected = rpm / 60 * project.engine.cylinders / (project.engine.strokes / 2);
        near = f > 0.7 * expected & f < 1.3 * expected;
        [~, k] = max(spectrum .* near);
        rows(end + 1) = struct('rpm', rpm, 'expected_hz', expected, 'measured_hz', f(k), ...
            'error_pct', 100 * abs(f(k) - expected) / expected); %#ok<AGROW>
    end
    report.firing_frequency = rows;
    report.firing_frequency_passed = all([rows.error_pct] < 1);

    % 2. Severity sweeps: signature after onset relative to before onset.
    severities = [0.2, 0.5, 0.8];
    sweep = struct();
    for condition = ["misfire", "knock_like", "valve_clearance_like"]
        values = zeros(size(severities));
        for k = 1:numel(severities)
            run = generate_run(project, batch, scenario(2000, 0.5, char(condition), 10, severities(k)));
            values(k) = signature(run, condition, 10, fsA, fsV, project);
        end
        sweep.(condition) = struct('severity', severities, 'after_over_before', values, ...
            'increasing', all(diff(values) > 0) && values(1) > 1);
    end
    report.severity_sweep = sweep;
    report.severity_sweep_passed = all(structfun(@(s) s.increasing, sweep));

    % 3. Cross-modal lag between event-locked high-band envelopes.
    run = generate_run(project, batch, scenario(1500, 0.5, 'normal', NaN, 0));
    a = band_envelope(double(run.audio), fsA, [2000 8000]);
    v = band_envelope(double(resample(double(run.vibration), fsA, fsV)), fsA, [2000 8000]);
    [c, lags] = xcorr(a - mean(a), v - mean(v), round(0.01 * fsA));
    [~, k] = max(c);
    sensors = batch.generator_params.sensors;
    expectedLag = sensors.audio.delay_s - sensors.vibration.delay_s;
    report.cross_modal_lag = struct('measured_s', lags(k) / fsA, 'expected_s', expectedLag);
    report.cross_modal_lag_passed = abs(lags(k) / fsA - expectedLag) < 2 / fsA + 1e-4;

    % 4. Aliasing control on a noise-free knock render.
    knock = scenario(3000, 0.7, 'knock_like', 5, 0.9);
    knock = normalize_scenario(knock, batch.duration_s);
    [streams, ~] = make_streams(project.dataset.master_seed, 9100, 1);
    params = batch.generator_params;
    profile = operating_profile(knock, batch.duration_s, batch.warmup_s, params, streams.profile);
    events = combustion_schedule(profile, project.engine, params, streams.events);
    engineParams = struct('knock_freq_hz', 8500);
    [events, profile] = apply_fault(events, profile, knock, params, engineParams, streams.fault);
    alias = struct();
    for name = ["audio", "vibration"]
        fs = project.streams.(name).sample_rate_hz;
        x = double(render_sensor(events, profile, params.sensors.(name), params.sources, fs, ...
            batch.duration_s, batch.warmup_s, streams.(name), NoiseFree=true));
        [pxx, f] = pwelch(x, hann(8192), 4096, 8192, fs);
        alias.(name + "_db") = 10 * log10(trapz(f(f >= 0.45 * fs), pxx(f >= 0.45 * fs)) / trapz(f, pxx));
    end
    report.aliasing = alias;
    report.aliasing_passed = alias.audio_db < -40 && alias.vibration_db < -40;

    % 5. Reproducibility.
    first = generate_run(project, batch, scenario(1800, 0.4, 'misfire', 8, 0.6));
    second = generate_run(project, batch, scenario(1800, 0.4, 'misfire', 8, 0.6));
    report.reproducible = isequaln(first.audio, second.audio) ...
        && isequaln(first.vibration, second.vibration) && isequaln(first.telemetry, second.telemetry) ...
        && isequaln(first.cam_sync_s, second.cam_sync_s);
    if isfield(first, 'crank_tooth_s')
        report.reproducible = report.reproducible && isequaln(first.crank_tooth_s, second.crank_tooth_s);
    end

    % 6. MATLAB reference features for the Python implementation.
    reference = struct('run_id', {}, 'window_end_s', {}, 'name', {}, 'value', {});
    rawRoot = fullfile(root, "data", "raw", batch.batch_id);
    analysis = project.analysis;
    for runId = options.ReferenceRuns
        file = fullfile(rawRoot, runId, "waveforms.mat");
        if ~isfile(file)
            continue
        end
        w = load(file);
        tEnd = 10;
        for name = ["audio", "vibration"]
            fs = w.(name + "_fs_hz");
            x = double(w.(name)(round((tEnd - analysis.window_s) * fs) + 1:round(tEnd * fs)));
            [pxx, f] = pwelch(x, hann(analysis.welch_segment_samples, 'periodic'), ...
                round(analysis.welch_segment_samples * analysis.welch_overlap_fraction), ...
                analysis.welch_segment_samples, fs);
            low = analysis.(name + "_low_hz");
            edges = [low, 500, 1500, 3000, 5000, 7000, analysis.high_hz];
            df = f(2) - f(1);
            power = arrayfun(@(k) sum(pxx(f >= edges(k) & f < edges(k + 1))) * df, 1:numel(edges) - 1);
            envelope = band_envelope(x, fs, [5000 10000]);
            values = {name + "_log_rms", log10(sqrt(mean(x .^ 2)) + 1e-12); ...
                name + "_band_frac_4", power(5) / sum(power); ...
                name + "_env_kurtosis_high", kurtosis(envelope) - 3};
            for k = 1:size(values, 1)
                reference(end + 1) = struct('run_id', char(runId), 'window_end_s', tEnd, ...
                    'name', char(values{k, 1}), 'value', values{k, 2}); %#ok<AGROW>
            end
        end
    end
    report.reference_feature_count = numel(reference);

    % 7. Combinations: each fault adds its own signature on top of the others.
    % Same operating profile and streams; faults on cylinders 1-3 from 10 s.
    names = ["misfire", "knock_like", "valve_clearance_like"];
    sets = {names([1 2]), names([1 3]), names([2 3]), names};
    normalRun = generate_run(project, batch, multi_scenario(2000, 0.5, strings(1, 0), 10, 0.6));
    single = struct();
    for f = names
        single.(f) = signature(generate_run(project, batch, multi_scenario(2000, 0.5, f, 10, 0.6)), ...
            f, 10, fsA, fsV, project) / signature(normalRun, f, 10, fsA, fsV, project);
    end
    combos = struct('faults', {}, 'fault', {}, 'with', {}, 'without', {}, 'ratio', {}, ...
        'single_ratio', {}, 'retained', {});
    for k = 1:numel(sets)
        members = sets{k};
        withRun = generate_run(project, batch, multi_scenario(2000, 0.5, members, 10, 0.6));
        for f = members
            withoutRun = generate_run(project, batch, ...
                multi_scenario(2000, 0.5, members(members ~= f), 10, 0.6));
            with = signature(withRun, f, 10, fsA, fsV, project);
            without = signature(withoutRun, f, 10, fsA, fsV, project);
            retained = (with / without - 1) / (single.(f) - 1);
            combos(end + 1) = struct('faults', char(join(members, "+")), 'fault', char(f), ...
                'with', with, 'without', without, 'ratio', with / without, ...
                'single_ratio', single.(f), 'retained', retained); %#ok<AGROW>
        end
    end
    report.combinations = combos;
    report.combinations_passed = all([combos.ratio] > 1) && all([combos.retained] >= 0.5);
    % 8. Contract 1.2 streams leave the 1.1 streams untouched.
    report.sensors_1_2_passed = true;
    if isfield(batch.generator_params, 'crank') || isfield(batch.generator_params, 'thermal')
        older = batch;
        older.generator_params = rmfield(older.generator_params, ...
            intersect(fieldnames(older.generator_params), {'crank', 'thermal'}));
        base = generate_run(project, older, scenario(1800, 0.4, 'misfire', 8, 0.6));
        thermalRows = startsWith(first.telemetry.signal, ["egt", "cht"]);
        unchanged = isequaln(first.audio, base.audio) && isequaln(first.vibration, base.vibration) ...
            && isequaln(first.cam_sync_s, base.cam_sync_s) ...
            && isequaln(first.telemetry(~thermalRows, :), base.telemetry);
        egt = @(c, t0, t1) mean(first.telemetry.value(first.telemetry.signal == "egt_cyl" + c ...
            & first.telemetry.t_s >= t0 & first.telemetry.t_s < t1));
        cooled = (egt(2, 2, 7) - egt(2, 25, 30)) - (egt(1, 2, 7) - egt(1, 25, 30));
        report.sensors_1_2 = struct('unchanged_1_1_streams', unchanged, ...
            'misfire_egt_cooling_c', cooled, ...
            'crank_misfire_strokes', first.render_info.crank.misfire_strokes, ...
            'crank_p95_phase_error_deg', first.render_info.crank.p95_phase_error_deg);
        report.sensors_1_2_passed = unchanged && cooled > 20 ...
            && first.render_info.crank.misfire_strokes > 0 ...
            && first.render_info.crank.p95_phase_error_deg < 3;
    end
    report.passed = report.firing_frequency_passed && report.severity_sweep_passed ...
        && report.cross_modal_lag_passed && report.aliasing_passed && report.reproducible ...
        && report.combinations_passed && report.sensors_1_2_passed;

    write_json(report, fullfile(outDir, "validation.json"));
    write_json(reference, fullfile(outDir, "matlab_features.json"));
    fprintf('firing frequency errors (%%): %s\n', mat2str([rows.error_pct], 3));
    for condition = string(fieldnames(sweep))'
        fprintf('%-21s after/before by severity: %s\n', condition, ...
            mat2str(sweep.(condition).after_over_before, 3));
    end
    fprintf('cross-modal lag %.5f s (expected %.5f s)\n', report.cross_modal_lag.measured_s, expectedLag);
    fprintf('aliasing: audio %.1f dB, vibration %.1f dB above 0.9 x Nyquist\n', ...
        alias.audio_db, alias.vibration_db);
    for c = combos
        fprintf('%-45s %-21s with/without %.3f (alone %.3f, retained %.0f%%)\n', ...
            c.faults, c.fault, c.ratio, c.single_ratio, 100 * c.retained);
    end
    if isfield(report, 'sensors_1_2')
        fprintf('1.2 sensors: 1.1 streams unchanged %d, misfire EGT cooling %.0f C, p95 phase %.2f deg\n', ...
            report.sensors_1_2.unchanged_1_1_streams, report.sensors_1_2.misfire_egt_cooling_c, ...
            report.sensors_1_2.crank_p95_phase_error_deg);
    end
    fprintf('reproducible %d; reference features %d; PASSED %d\n', report.reproducible, ...
        numel(reference), report.passed);
end

function s = scenario(rpm, load, condition, onset, severity)
    s = struct('run_id', 'run_09001', 'engine_id', 'engine_001', 'condition', condition, ...
        'onset_s', onset, 'affected_cylinder', 2, 'severity', severity, ...
        'rpm', [rpm, rpm], 'load', [load, load], 'ramp_s', [1, 29], 'profile_kind', 'steady');
end

function s = multi_scenario(rpm, load, conditions, onset, severity)
    % One fault per listed condition, on cylinders 1, 2, 3 in the order given
    % by misfire, knock_like, valve_clearance_like; an empty list is normal.
    cylinder = struct('misfire', 1, 'knock_like', 2, 'valve_clearance_like', 3);
    faults = struct('condition', {}, 'onset_s', {}, 'affected_cylinder', {}, 'severity', {});
    for c = conditions
        faults(end + 1) = struct('condition', char(c), 'onset_s', onset, ...
            'affected_cylinder', cylinder.(c), 'severity', severity); %#ok<AGROW>
    end
    s = struct('run_id', 'run_09002', 'engine_id', 'engine_001', 'faults', faults, ...
        'rpm', [rpm, rpm], 'load', [load, load], 'ramp_s', [1, 29], 'profile_kind', 'steady');
end

function value = signature(run, condition, onset, fsA, fsV, project)
    % Feature-free summaries of each fault's intended mechanism.
    before = @(x, fs) x(round(2 * fs):round((onset - 1) * fs));
    after = @(x, fs) x(round((onset + 1) * fs):end);
    switch condition
        case "misfire"
            metric = @(x, fs) peak_cv(band_envelope(x, fs, [500 3000]), fs, ...
                mean(run.profile.rpm) / 60 * project.engine.cylinders / 2);
            x = double(run.audio);
            value = metric(after(x, fsA), fsA) / metric(before(x, fsA), fsA);
        case "knock_like"
            x = double(run.vibration);
            value = band_power(after(x, fsV), fsV, [5000 9000]) / band_power(before(x, fsV), fsV, [5000 9000]);
        case "valve_clearance_like"
            % Per-cycle timing variability of the strongest high-band impacts.
            x = double(run.vibration);
            value = kurtosis(band_envelope(after(x, fsV), fsV, [2000 8000])) ...
                / kurtosis(band_envelope(before(x, fsV), fsV, [2000 8000]));
    end
end

function cv = peak_cv(envelope, fs, firingHz)
    width = max(1, round(0.002 * fs));
    smooth = conv(envelope, ones(width, 1) / width, 'same');
    [peaks, ~] = findpeaks(smooth, 'MinPeakDistance', round(0.7 * fs / firingHz));
    cv = std(peaks, 1) / mean(peaks);
end

function p = band_power(x, fs, band)
    [pxx, f] = pwelch(x, hann(8192), 4096, 8192, fs);
    p = trapz(f(f >= band(1) & f <= band(2)), pxx(f >= band(1) & f <= band(2)));
end

function envelope = band_envelope(x, fs, band)
    % Same window-local FFT band-pass and Hilbert envelope as the Python features.
    x = x(:);
    n = numel(x);
    spectrum = fft(x);
    f = (0:n - 1)' * fs / n;
    f = min(f, fs - f);  % two-sided frequency magnitude
    spectrum(f < band(1) | f >= min(band(2), fs / 2)) = 0;
    envelope = abs(hilbert(real(ifft(spectrum))));
end

function write_json(data, path)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(data, PrettyPrint=true), 'char');
end
