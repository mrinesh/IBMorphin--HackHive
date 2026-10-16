function report = validate_run_matlab(runDir, truthPath)
%VALIDATE_RUN_MATLAB Sanity checks on one exported run and its private truth.
%   report = validate_run_matlab('data/raw/run_00001', ...
%       'evaluation_truth/run_00001_events.json')
%   Checks lengths, dtype, finite values, clipping, cam_sync phase lock to
%   cylinder 1 combustion (contract 1.1), firing-event rate against
%   observed RPM, and, for each fault in the run, that its events start only
%   after its own onset and that its intended signature is present (knock:
%   5-9 kHz energy after knocked vs unknocked combustion must exceed 1.1x;
%   below 2x it is reported as faint). Multi-fault runs also check that no
%   knock burst follows a misfired stroke on the same cylinder. Truth written
%   before per-fault fields existed is read as one fault.
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

    cylinders = [truth.events.cylinder]';
    if isfield(w, 'cam_sync_s')  % data contract 1.1 phase reference
        issues = [issues; check_cam_sync(w.cam_sync_s, metadata, config, rpm, duration, ...
            times(kinds == "combustion" & cylinders == 1))];
        metrics.cam_sync_pulses = numel(w.cam_sync_s);
    end
    faults = truth_faults(truth);
    if isfield(truth.events, 'fault')
        eventFault = string({truth.events.fault})';
    else  % older truth: every faulted event belongs to the run's one condition
        eventFault = repmat("", numel(kinds), 1);
        eventFault(faulted) = string(truth.condition);
    end
    metrics.faulted_events = nnz(faulted);
    if isempty(faults) && any(faulted)
        issues(end + 1) = "normal run has faulted events";
    end
    unexpected = setdiff(unique(eventFault(eventFault ~= "")), string({faults.condition}));
    if ~isempty(unexpected)
        issues(end + 1) = "events tagged with unlisted fault " + unexpected(1);
    end
    for k = 1:numel(faults)
        f = faults(k);
        condition = string(f.condition);
        mine = eventFault == condition;
        if any(mine & times < f.onset_s)
            issues(end + 1) = condition + ": faulted events occur before onset"; %#ok<AGROW>
        end
        switch condition
            case "misfire"
                ok = any(mine & kinds == "combustion");
            case "knock_like"
                % Compare 5-9 kHz energy just after knocked vs unknocked combustion
                % on the affected cylinder; robust to RPM/load drift over the run.
                bursts = times(kinds == "knock_burst" & mine);
                fired = times(kinds == "combustion" & cylinders == f.affected_cylinder ...
                    & times >= f.onset_s & eventFault ~= "misfire");
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
            case "valve_clearance_like"
                ok = any(mine & kinds == "valve_impact");
        end
        if ~ok
            issues(end + 1) = "intended " + condition + " signature not found"; %#ok<AGROW>
        end
    end
    % A misfired (weakened) stroke cannot also knock.
    misfired = find(eventFault == "misfire" & kinds == "combustion");
    burstIdx = find(kinds == "knock_burst");
    for m = misfired'
        if any(cylinders(burstIdx) == cylinders(m) & times(burstIdx) > times(m) ...
                & times(burstIdx) <= times(m) + 0.005)
            issues(end + 1) = "knock burst follows a misfired stroke"; %#ok<AGROW>
            break
        end
    end
    metrics.faults = strjoin(string({faults.condition}), "+");

    report = struct('run_id', metadata.run_id, 'passed', isempty(issues), ...
        'issues', issues, 'metrics', metrics);
    if report.passed
        fprintf('%s: passed\n', metadata.run_id);
    else
        fprintf('%s: FAILED\n', metadata.run_id);
        fprintf('  %s\n', issues);
    end
end

function issues = check_cam_sync(pulses, metadata, config, rpm, duration, cylinder1)
    % One pulse per 720 degrees, at a fixed crank phase before cylinder 1's
    % combustion: the phase from each pulse to the next cylinder-1 combustion
    % must be constant and match cam_sync_angle_deg (plus the small combustion
    % offset after TDC).
    issues = strings(0, 1);
    pulses = pulses(:);
    info = metadata.streams.cam_sync;
    if numel(pulses) ~= info.n_pulses || info.angle_deg ~= config.engine.cam_sync_angle_deg
        issues(end + 1) = "cam_sync: pulse count or angle disagrees with metadata/config";
        return
    end
    if any(diff(pulses) <= 0) || any(pulses < 0 | pulses >= duration)
        issues(end + 1) = "cam_sync: pulses not increasing inside [0, duration)";
        return
    end
    cycles = trapz([rpm.t_s; duration], [rpm.value; rpm.value(end)]) / 120;
    if abs(numel(pulses) - cycles) > 0.02 * cycles + 1
        issues(end + 1) = sprintf("cam_sync: %d pulses vs %.1f cycles from RPM", ...
            numel(pulses), cycles);
    end
    rpmAt = interp1(rpm.t_s, rpm.value, pulses, 'linear', 'extrap');
    phase = nan(size(pulses));
    for k = 1:numel(pulses)
        next = cylinder1(find(cylinder1 > pulses(k), 1));
        if ~isempty(next) && next - pulses(k) < 120 / rpmAt(k)
            phase(k) = 6 * rpmAt(k) * (next - pulses(k));
        end
    end
    phase = phase(~isnan(phase));
    expected = mod(-config.engine.cam_sync_angle_deg, 720);
    if isempty(phase) || std(phase) > 3 || abs(mean(phase) - expected) > 30
        issues(end + 1) = "cam_sync: pulses are not phase-locked to cylinder 1 combustion";
    end
end

function faults = truth_faults(truth)
    % Per-fault lifecycle list; older truth describes its one fault in flat fields.
    faults = struct('condition', {}, 'onset_s', {}, 'affected_cylinder', {});
    if isfield(truth, 'faults')
        raw = truth.faults;
        if iscell(raw)
            raw = [raw{:}];
        end
        for k = 1:numel(raw)
            faults(end + 1) = struct('condition', raw(k).condition, ...
                'onset_s', raw(k).onset_s, 'affected_cylinder', raw(k).affected_cylinder); %#ok<AGROW>
        end
    elseif ~strcmp(truth.condition, 'normal')
        faults = struct('condition', truth.condition, 'onset_s', truth.onset_s, ...
            'affected_cylinder', truth.affected_cylinder);
    end
end
