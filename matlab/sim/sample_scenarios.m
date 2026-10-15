function scenarios = sample_scenarios(projectConfig, batchConfig)
%SAMPLE_SCENARIOS Draw every run's scenario for a sampled dataset (e.g. core_v1).
%   Operating profile, disturbances, and data-quality problems are drawn from
%   the same distributions for every condition, so telemetry and context carry
%   no class shortcut. Each run uses its own substream, so adding or removing
%   runs never changes the others. Core runs come first (runs_per_condition
%   per engine, in engine order), then challenge runs on the test engines.

    durationS = batchConfig.duration_s;
    conditions = string(projectConfig.conditions);
    perCondition = batchConfig.runs_per_condition;
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

    plan = struct('engine', {}, 'split', {}, 'condition', {}, 'challenge', {});
    for e = 1:numel(engines)
        for c = conditions'
            for r = 1:perCondition.(c)
                plan(end + 1) = struct('engine', engines(e), 'split', engineSplit(e), ...
                    'condition', c, 'challenge', false); %#ok<AGROW>
            end
        end
    end
    testEngines = engines(engineSplit == "test");
    for e = 1:numel(testEngines)
        for c = conditions'
            plan(end + 1) = struct('engine', testEngines(e), 'split', "test", ...
                'condition', c, 'challenge', true); %#ok<AGROW>
        end
    end
    assert(nnz([plan.challenge]) == projectConfig.dataset.challenge_runs, ...
        'Challenge run count must match dataset.challenge_runs.');

    scenarios = struct([]);
    for k = 1:numel(plan)
        stream = RandStream('mrg32k3a', 'Seed', projectConfig.dataset.master_seed);
        stream.Substream = batchConfig.sampling.scenario_substream + k;
        s = draw_run(plan(k), k, durationS, batchConfig, projectConfig, stream);
        if isempty(scenarios)
            scenarios = s;
        else
            scenarios(end + 1) = s; %#ok<AGROW>
        end
    end
end

function s = draw_run(plan, runNumber, durationS, batch, project, stream)
    sp = batch.sampling;
    rb = batch.robustness;
    u = @(range) range(1) + diff(range) * rand(stream);
    clampRpm = @(r) min(max(r, project.engine.rpm_min), project.engine.rpm_max);
    clampLoad = @(l) min(max(l, project.engine.load_min), project.engine.load_max);

    s = struct();
    s.run_id = sprintf('run_%05d', runNumber);
    s.engine_id = char(plan.engine);
    s.split = char(plan.split);
    s.condition = char(plan.condition);
    s.challenge = plan.challenge;

    % Operating profile: identical distributions for every condition.
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

    % Fault lifecycle.
    s.onset_s = NaN;
    s.affected_cylinder = NaN;
    s.severity = 0;
    s.severity_end = 0;
    s.degradation = false;
    s.recovery_s = [NaN, NaN];
    if plan.condition ~= "normal"
        s.onset_s = u(sp.onset_range_s);
        s.affected_cylinder = randi(stream, project.engine.cylinders);
        severityRange = sp.severity_range;
        if plan.challenge
            severityRange = batch.challenge.severity_range;
        end
        s.severity = u(severityRange);
        s.severity_end = s.severity;
        if plan.condition == "valve_clearance_like" && ~plan.challenge ...
                && rand(stream) < sp.degradation_probability_valve
            s.degradation = true;  % accelerated, simulated deterioration
            s.onset_s = u([4, 8]);
            s.severity = sp.degradation_severity(1);
            s.severity_end = sp.degradation_severity(2);
        elseif rand(stream) < sp.recovery_probability
            start = s.onset_s + u([3, 8]);
            stop = start + u(sp.recovery_duration_s);
            if stop <= durationS - 3
                s.recovery_s = [start, stop];
            end
        end
    end

    % Disturbances and data-quality problems, independent of condition.
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
