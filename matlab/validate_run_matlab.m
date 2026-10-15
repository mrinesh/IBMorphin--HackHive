function report = validate_run_matlab(runDir, truthPath)
%VALIDATE_RUN_MATLAB Sanity checks on one exported run and its private truth.
%   report = validate_run_matlab('data/raw/run_00001', ...
%       'evaluation_truth/run_00001_events.json')
%   Checks lengths, dtype, finite values, clipping, firing-event rate against
%   observed RPM, fault events after onset, and the condition's intended
%   signature (knock: 5-9 kHz energy after knocked vs unknocked combustion
%   must exceed 1.1x; below 2x it is reported as faint).
%   Python validate-runs repeats the format checks independently.

    config = load_project_config();
    w = load(fullfile(runDir, 'waveforms.mat'));
    metadata = jsondecode(fileread(fullfile(runDir, 'metadata.json')));
    telemetry = readtable(fullfile(runDir, 'telemetry.csv'), 'TextType', 'string');
    truth = jsondecode(fileread(truthPath));
    duration = metadata.duration_s;
    % Declared data-quality problems (sampled datasets) are expected, not failures.
    declaredClip = isfield(truth, 'disturbances') && truth.disturbances.clip_gain > 1;
    declaredGap = "";
    if isfield(truth, 'disturbances') && ~isempty(truth.disturbances.missing_stream)
        declaredGap = string(truth.disturbances.missing_stream);
    end
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
        missing = mean(isnan(x));
        if any(isinf(x)) || (missing > 0 && declaredGap ~= name) || missing > 0.05
            issues(end + 1) = sprintf("%s: non-finite samples (%.4f missing)", name, missing); %#ok<AGROW>
        end
        clipped = mean(abs(x) >= metadata.streams.(name).full_scale);
        if clipped > config.runtime.max_clipped_fraction && ~declaredClip
            issues(end + 1) = sprintf("%s: clipped fraction %.4f", name, clipped); %#ok<AGROW>
        end
        metrics.(name + "_rms") = rms(double(x), 'omitnan');
        metrics.(name + "_peak") = max(abs(double(x)));
        metrics.(name + "_clipped") = clipped;
    end

    % Firing-event rate: a four-stroke engine fires cylinders/2 times per revolution.
    rpm = telemetry(telemetry.signal == "rpm" & telemetry.valid == 1, :);
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

    onset = truth.onset_s;
    if isempty(onset)
        onset = Inf;  % normal run: no fault onset
    end
    if any(faulted & times < onset)
        issues(end + 1) = "faulted events occur before onset";
    end
    metrics.faulted_events = nnz(faulted);
    switch truth.condition
        case 'misfire'
            ok = any(faulted & kinds == "combustion");
        case 'knock_like'
            % Compare 5-9 kHz energy just after knocked vs unknocked combustion
            % on the affected cylinder; robust to RPM/load drift over the run.
            cylinders = [truth.events.cylinder]';
            bursts = times(kinds == "knock_burst");
            fired = times(kinds == "combustion" & cylinders == truth.affected_cylinder ...
                & times >= onset);
            knocked = arrayfun(@(t) any(bursts > t & bursts <= t + 0.005), fired);
            fs = w.vibration_fs_hz;
            [b, a] = butter(4, [5000 9000] / (fs / 2), 'bandpass');
            energy = filtfilt(b, a, fillmissing(double(w.vibration), "constant", 0)) .^ 2;
            windowEnergy = arrayfun(@(t) sum(energy(max(1, round(t * fs)) ...
                + (0:round(0.005 * fs)))), fired(fired < duration - 0.006));
            knocked = knocked(fired < duration - 0.006);
            ratio = mean(windowEnergy(knocked)) / mean(windowEnergy(~knocked));
            metrics.knock_band_ratio = ratio;
            % The check confirms the intervention exists. Low-severity knock is
            % intentionally faint (ratio near 1.1-2); it is reported, not failed.
            metrics.knock_faint = ratio < 2;
            ok = any(knocked) && any(~knocked) && ratio > 1.1;
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
