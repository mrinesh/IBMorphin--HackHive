function scenarios = sample_scenarios(projectConfig, batchConfig)
%SAMPLE_SCENARIOS Draw every run's scenario for a sampled dataset (e.g. core_v1).
%   Operating profile, disturbances, and data-quality problems are drawn from
%   the same distributions for every condition, so telemetry and context carry
%   no class shortcut. Each run uses its own substream, so adding or removing
%   runs never changes the others. Core runs come first (in engine order),
%   then challenge runs on the test engines.
%   Single-fault datasets (core_v1) list runs_per_condition. Multi-fault
%   datasets (core_v2) list fault_sets: each entry's faults (empty = normal)
%   and runs_per_engine; challenge.fault_sets lists one run per set per test
%   engine. Every fault in a set draws its own onset, cylinder, severity,
%   recovery, and degradation, so faults overlap with staggered onsets.

    durationS = batchConfig.duration_s;
    conditions = string(projectConfig.conditions);
    if isfield(batchConfig, 'runs_per_condition')
        perCondition = batchConfig.runs_per_condition;
    end
    splits = batchConfig.engine_splits;
    splitNames = string(fieldnames(splits));
    engines = strings(0, 1);
    engineSplit = strings(0, 1);
    for k = 1:numel(splitNames)
        ids = string(splits.(splitNames(k)));
        engines = [engines; ids(:)]; %#ok<AGROW>
        engineSplit = [engineSplit; repmat(splitNames(k), numel(ids), 1)]; %#ok<AGROW>
    end
    [engines, order] = sort(engines);
    engineSplit = engineSplit(order);
    assert(numel(engines) == projectConfig.dataset.engine_count, ...
        'Engine splits must list dataset.engine_count engines.');

    plan = struct('engine', {}, 'split', {}, 'condition', {}, 'faults', {}, 'challenge', {});
    testEngines = engines(engineSplit == "test");
    if isfield(batchConfig, 'fault_sets')
        sets = batchConfig.fault_sets;
        for e = 1:numel(engines)
            for k = 1:numel(sets)
                for r = 1:sets(k).runs_per_engine
                    plan(end + 1) = plan_entry(engines(e), engineSplit(e), ...
                        as_strings(sets(k).faults), false); %#ok<AGROW>
                end
            end
        end
        challengeSets = batchConfig.challenge.fault_sets;
        if ~iscell(challengeSets)
            challengeSets = num2cell(challengeSets, 2);
        end
        for e = 1:numel(testEngines)
            for k = 1:numel(challengeSets)
                plan(end + 1) = plan_entry(testEngines(e), "test", ...
                    as_strings(challengeSets{k}), true); %#ok<AGROW>
            end
        end
    else
        for e = 1:numel(engines)
            for c = conditions'
                for r = 1:perCondition.(c)
                    plan(end + 1) = plan_entry(engines(e), engineSplit(e), c, false); %#ok<AGROW>
                end
            end
        end
        for e = 1:numel(testEngines)
            for c = conditions'
                plan(end + 1) = plan_entry(testEngines(e), "test", c, true); %#ok<AGROW>
            end
        end
    end
    assert(nnz([plan.challenge]) == projectConfig.dataset.challenge_runs, ...
        'Challenge run count must match dataset.challenge_runs.');

    scenarios = struct([]);
    for k = 1:numel(plan)
        stream = RandStream('mrg32k3a', 'Seed', dataset_seed(projectConfig, batchConfig));
        stream.Substream = batchConfig.sampling.scenario_substream + k;
        s = draw_run(plan(k), k, durationS, batchConfig, projectConfig, stream);
        if isempty(scenarios)
            scenarios = s;
        else
            scenarios(end + 1) = s; %#ok<AGROW>
        end
    end
end

function entry = plan_entry(engine, split, faults, challenge)
    % faults: "normal" or a fault list (empty list = normal).
    faults = faults(faults ~= "normal");
    condition = "normal";
    if ~isempty(faults)
        condition = join(sort(faults), "+");
    end
    entry = struct('engine', engine, 'split', split, 'condition', condition, ...
        'faults', reshape(faults, 1, []), 'challenge', challenge);
end

function out = as_strings(x)
    % jsondecode gives [] for an empty list and a cell or char array otherwise.
    if isempty(x)
        out = strings(1, 0);
    elseif ischar(x)
        out = string(cellstr(x))';
    else
        out = reshape(string(x), 1, []);
    end
end

function s = draw_run(plan, runNumber, durationS, batch, project, stream)
    % Draw order (profile, then each fault, then disturbances) is fixed; a
    % single-fault run draws exactly the same numbers as before multi-fault
    % support, so single-fault datasets regenerate identically.
    u = @(range) range(1) + diff(range) * rand(stream);
    s = struct();
    s.run_id = sprintf('run_%05d', runNumber);
    s.engine_id = char(plan.engine);
    s.split = char(plan.split);
    s.condition = char(plan.condition);
    s.challenge = plan.challenge;
    s = draw_profile(s, plan, durationS, batch, project, stream, u);
    if isfield(batch, 'fault_sets')
        faults = struct('condition', {}, 'onset_s', {}, 'affected_cylinder', {}, ...
            'severity', {}, 'severity_end', {}, 'degradation', {}, 'recovery_s', {});
        for c = plan.faults
            faults(end + 1) = draw_fault(c, plan.challenge, durationS, batch, project, ...
                stream, u); %#ok<AGROW>
        end
        s.faults = faults;
    else
        f = draw_fault(plan.condition, plan.challenge, durationS, batch, project, stream, u);
        names = ["onset_s", "affected_cylinder", "severity", "severity_end", "degradation", ...
            "recovery_s"];
        for name = names
            s.(name) = f.(name);
        end
    end
    s = draw_disturbances(s, plan, durationS, batch, stream, u);
    if isfield(s, 'faults')
        s = normalize_scenario(s, durationS);  % fills the flat summary fields
    end
end

function s = draw_profile(s, plan, durationS, batch, project, stream, u)
    % Operating profile: identical distributions for every condition.
    sp = batch.sampling;
    clampRpm = @(r) min(max(r, project.engine.rpm_min), project.engine.rpm_max);
    clampLoad = @(l) min(max(l, project.engine.load_min), project.engine.load_max);
    kinds = string(sp.profile_kinds);
    s.profile_kind = char(kinds(find(rand(stream) <= cumsum(sp.profile_probabilities), 1)));
    rpmRange = sp.rpm_range;
    if plan.challenge
        ranges = batch.challenge.rpm_ranges;
        rpmRange = ranges(randi(stream, size(ranges, 1)), :);
    end
    rpm0 = u(rpmRange);
    load0 = u(sp.load_range);
    switch s.profile_kind
        case 'steady'
            s.rpm = [rpm0, rpm0];
            s.load = [load0, load0];
            s.ramp_s = [1, durationS - 1];
        case 'ramp'
            direction = sign(rand(stream) - 0.5);
            s.rpm = [rpm0, clampRpm(rpm0 + direction * u(sp.ramp_rpm_span))];
            s.load = [load0, clampLoad(load0 + direction * u(sp.ramp_load_span))];
            start = u([1, 8]);
            s.ramp_s = [start, min(start + u([8, 20]), durationS - 1)];
        case 'step'
            s.rpm = [rpm0, clampRpm(rpm0 + u([-100, 100]))];
            s.load = [load0, clampLoad(load0 + sign(rand(stream) - 0.5) * u(sp.step_load_span))];
            start = u([4, durationS - 6]);
            s.ramp_s = [start, start + sp.step_duration_s];
    end
    if plan.challenge  % keep challenge runs inside their reserved RPM band
        s.rpm = min(max(s.rpm, rpmRange(1)), rpmRange(2));
    end
end

function f = draw_fault(condition, challenge, durationS, batch, project, stream, u)
    % One fault's lifecycle; "normal" draws nothing.
    sp = batch.sampling;
    f = struct('condition', char(condition), 'onset_s', NaN, 'affected_cylinder', NaN, ...
        'severity', 0, 'severity_end', 0, 'degradation', false, 'recovery_s', [NaN, NaN]);
    if condition == "normal"
        return
    end
    f.onset_s = u(sp.onset_range_s);
    f.affected_cylinder = randi(stream, project.engine.cylinders);
    severityRange = sp.severity_range;
    if challenge
        severityRange = batch.challenge.severity_range;
    end
    f.severity = u(severityRange);
    f.severity_end = f.severity;
    if condition == "valve_clearance_like" && ~challenge ...
            && rand(stream) < sp.degradation_probability_valve
        f.degradation = true;  % accelerated, simulated deterioration
        f.onset_s = u([4, 8]);
        f.severity = sp.degradation_severity(1);
        f.severity_end = sp.degradation_severity(2);
    elseif rand(stream) < sp.recovery_probability
        start = f.onset_s + u([3, 8]);
        stop = start + u(sp.recovery_duration_s);
        if stop <= durationS - 3
            f.recovery_s = [start, stop];
        end
    end
end

function s = draw_disturbances(s, plan, durationS, batch, stream, u)
    % Disturbances and data-quality problems, independent of condition.
    rb = batch.robustness;
    s.noise_scale = 1;
    s.resonance_shift = 1;
    if plan.challenge
        s.noise_scale = batch.challenge.noise_scale;
        s.resonance_shift = batch.challenge.resonance_shift;
    end
    s.impact_rate_hz = u(rb.unrelated_impact_rate_hz);
    s.clip_gain = 1;
    if rand(stream) < rb.clipping_probability
        s.clip_gain = u(rb.clipping_gain);
    end
    s.missing_stream = '';
    s.missing_gap_s = [NaN, NaN];
    if rand(stream) < rb.missing_gap_probability
        streams = ["audio", "vibration"];
        s.missing_stream = char(streams(randi(stream, 2)));
        start = u([3, durationS - 3]);
        s.missing_gap_s = [start, start + u(rb.missing_gap_s)];
    end
    s.dropout_signal = '';
    s.dropout_s = [NaN, NaN];
    if rand(stream) < rb.telemetry_dropout_probability
        signals = ["rpm", "load", "throttle"];
        s.dropout_signal = char(signals(randi(stream, 3)));
        start = u([3, durationS - 3]);
        s.dropout_s = [start, start + u(rb.telemetry_dropout_s)];
    end
end
