function report = validate_run_matlab(runDir, truthPath)
%VALIDATE_RUN_MATLAB Sanity checks on one exported run and its private truth.
%   report = validate_run_matlab('data/raw/run_00001', ...
%       'evaluation_truth/run_00001_events.json')
%   Checks lengths, dtype, finite values, clipping, firing-event rate against
%   observed RPM, fault events after onset, and the condition's intended
%   signature. Python validate-runs repeats the format checks independently.

    config = load_project_config();
    w = load(fullfile(runDir, 'waveforms.mat'));
    metadata = jsondecode(fileread(fullfile(runDir, 'metadata.json')));
    telemetry = readtable(fullfile(runDir, 'telemetry.csv'), 'TextType', 'string');
    truth = jsondecode(fileread(truthPath));
    duration = metadata.duration_s;
    issues = strings(0, 1);
    metrics = struct();

    for name = ["audio", "vibration"]
        x = w.(name);
        fs = w.(name + "_fs_hz");
        if ~isa(x, 'single') || ~iscolumn(x)
            issues(end + 1) = name + ": expected single column vector"; %#ok<AGROW>
        end
        if numel(x) ~= round(duration * fs) || fs ~= config.streams.(name).sample_rate_hz
            issues(end + 1) = name + ": sample count or rate mismatch"; %#ok<AGROW>
        end
        if ~all(isfinite(x))
            issues(end + 1) = name + ": non-finite samples"; %#ok<AGROW>
        end
        clipped = mean(abs(x) >= metadata.streams.(name).full_scale);
        if clipped > config.runtime.max_clipped_fraction
            issues(end + 1) = sprintf("%s: clipped fraction %.4f", name, clipped); %#ok<AGROW>
        end
        metrics.(name + "_rms") = rms(double(x));
        metrics.(name + "_peak") = max(abs(double(x)));
    end

    % Firing-event rate: a four-stroke engine fires cylinders/2 times per revolution.
    rpm = telemetry(telemetry.signal == "rpm", :);
    expected = trapz([rpm.t_s; duration], [rpm.value; rpm.value(end)]) / 60 ...
        * config.engine.cylinders / 2;
    kinds = string({truth.events.kind})';
    times = [truth.events.time_s]';
    faulted = [truth.events.faulted]';
    observed = nnz(kinds == "combustion");
    metrics.combustion_events = observed;
    metrics.expected_combustion_events = expected;
    if abs(observed - expected) / expected > 0.02
        issues(end + 1) = sprintf("combustion events %d vs expected %.1f from RPM", observed, expected);
    end

    if any(faulted & times < truth.onset_s)
        issues(end + 1) = "faulted events occur before onset";
    end
    metrics.faulted_events = nnz(faulted);
    switch truth.condition
        case 'misfire'
            ok = any(faulted & kinds == "combustion");
        case 'knock_like'
            fs = w.vibration_fs_hz;
            before = w.vibration(1:round((truth.onset_s - 0.5) * fs));
            after = w.vibration(round((truth.onset_s + 0.5) * fs):end);
            ratio = band_power(after, fs, [5000 9000]) / band_power(before, fs, [5000 9000]);
            metrics.knock_band_ratio = ratio;
            ok = any(kinds == "knock_burst") && ratio > 1.5;
        case 'valve_clearance_like'
            ok = any(faulted & kinds == "valve_impact");
        otherwise
            ok = ~any(faulted);
    end
    if ~ok
        issues(end + 1) = "intended " + truth.condition + " signature not found";
    end

    report = struct('run_id', metadata.run_id, 'passed', isempty(issues), ...
        'issues', issues, 'metrics', metrics);
    if report.passed
        fprintf('%s: passed\n', metadata.run_id);
    else
        fprintf('%s: FAILED\n', metadata.run_id);
        fprintf('  %s\n', issues);
    end
end

function p = band_power(x, fs, band)
    [pxx, f] = pwelch(double(x), hann(8192), 4096, 8192, fs);
    inBand = f >= band(1) & f <= band(2);
    p = trapz(f(inBand), pxx(inBand));
end
