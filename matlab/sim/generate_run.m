function run = generate_run(projectConfig, batchConfig, scenario)
%GENERATE_RUN Synthesize one run: waveforms, telemetry, metadata, private truth.
%   run.audio and run.vibration are single column vectors at native rates.
%   run.metadata is public; run.truth holds labels and source events and must
%   only be written to evaluation_truth/.

    params = batchConfig.generator_params;
    durationS = batchConfig.duration_s;
    warmupS = batchConfig.warmup_s;
    scenario = normalize_scenario(scenario, durationS);
    runNumber = sscanf(scenario.run_id, 'run_%d');
    engineNumber = sscanf(scenario.engine_id, 'engine_%d');
    masterSeed = dataset_seed(projectConfig, batchConfig);
    [streams, streamIndices] = make_streams(masterSeed, runNumber, engineNumber);

    engineParams = draw_engine_params(params.engine_variation, streams.engine);
    profile = operating_profile(scenario, durationS, warmupS, params, streams.profile);
    events = combustion_schedule(profile, projectConfig.engine, params, streams.events);
    [events, profile] = apply_fault(events, profile, scenario, params, engineParams, streams.fault);
    events = add_unrelated_impacts(events, profile, scenario, params, durationS, warmupS, ...
        streams.disturbance);
    % Cam phase reference from the final crank profile (after misfire RPM dips).
    camSyncS = cam_sync_times(profile, projectConfig.engine.cam_sync_angle_deg, durationS);

    audioFs = projectConfig.streams.audio.sample_rate_hz;
    vibrationFs = projectConfig.streams.vibration.sample_rate_hz;
    sensors = params.sensors;
    [audio, audioInfo] = render_sensor(events, profile, sensors.audio, params.sources, ...
        audioFs, durationS, warmupS, streams.audio, ...
        GainFactor=engineParams.audio_gain, ...
        ResonanceFactor=engineParams.audio_resonance * scenario.resonance_shift, ...
        NoiseScale=scenario.noise_scale, ClipGain=scenario.clip_gain);
    [vibration, vibrationInfo] = render_sensor(events, profile, sensors.vibration, params.sources, ...
        vibrationFs, durationS, warmupS, streams.vibration, ...
        GainFactor=engineParams.vibration_gain, ...
        ResonanceFactor=engineParams.vibration_resonance * scenario.resonance_shift, ...
        NoiseScale=scenario.noise_scale);
    telemetry = render_telemetry(profile, projectConfig.streams.telemetry_rates_hz, params, ...
        engineParams, durationS, streams.telemetry);

    % Data-quality problems: a waveform gap (NaN) and a telemetry dropout (invalid rows).
    gap = scenario.missing_gap_s;
    if ~isempty(scenario.missing_stream)
        if strcmp(scenario.missing_stream, 'audio')
            audio(missing_mask(numel(audio), audioFs, gap)) = NaN;
        else
            vibration(missing_mask(numel(vibration), vibrationFs, gap)) = NaN;
        end
    end
    if ~isempty(scenario.dropout_signal)
        drop = telemetry.signal == string(scenario.dropout_signal) ...
            & telemetry.t_s >= scenario.dropout_s(1) & telemetry.t_s < scenario.dropout_s(2);
        telemetry.value(drop) = NaN;
        telemetry.valid(drop) = 0;
    end

    run = struct();
    run.audio = audio;
    run.vibration = vibration;
    run.audio_fs_hz = audioFs;
    run.vibration_fs_hz = vibrationFs;
    run.telemetry = telemetry;
    run.cam_sync_s = camSyncS;
    run.render_info = struct('audio', audioInfo, 'vibration', vibrationInfo);
    run.metadata = build_metadata(projectConfig, batchConfig, scenario, audio, vibration, ...
        camSyncS);
    run.truth = build_truth(batchConfig, scenario, events, durationS, masterSeed, streamIndices);
    run.profile = profile;
    run.engine_params = engineParams;
end

function mask = missing_mask(n, fs, gap)
    t = (0:n - 1)' / fs;
    mask = t >= gap(1) & t < gap(2);
end

function events = add_unrelated_impacts(events, profile, scenario, params, durationS, warmupS, stream)
    % Broadband transients unrelated to the engine cycle (tools, road, body), in every class.
    n = poissrnd_stream(stream, scenario.impact_rate_hz * (durationS + warmupS));
    if n == 0
        return
    end
    src = params.sources.unrelated_impact;
    amplitude = src.amplitude_range;
    extra = struct();
    extra.time_s = sort(-warmupS + (durationS + warmupS) * rand(stream, n, 1));
    extra.angle_deg = interp1(profile.t, profile.theta_deg, extra.time_s);
    extra.cylinder = zeros(n, 1);
    extra.kind = repmat("unrelated_impact", n, 1);
    extra.intensity = amplitude(1) + diff(amplitude) * rand(stream, n, 1);
    extra.faulted = false(n, 1);
    extra.fault = strings(n, 1);
    extra.freq_scale = ones(n, 1);
    extra.tau_scale = ones(n, 1);
    extra.mode_freq_hz = src.freq_range_hz(1) + diff(src.freq_range_hz) * rand(stream, n, 1);
    extra.mode_tau_s = src.tau_range_s(1) + diff(src.tau_range_s) * rand(stream, n, 1);
    names = fieldnames(events);
    for k = 1:numel(names)
        events.(names{k}) = [events.(names{k}); extra.(names{k})];
    end
    events = sort_events(events);
end

function n = poissrnd_stream(stream, lambda)
    % Poisson draw from a specific stream (inverse transform; lambda is small).
    n = 0;
    p = exp(-lambda);
    cumulative = p;
    u = rand(stream);
    while u > cumulative && n < 1000
        n = n + 1;
        p = p * lambda / n;
        cumulative = cumulative + p;
    end
end

function e = draw_engine_params(v, stream)
    % Per-engine variation, fixed for every run of the same virtual engine.
    e = struct();
    e.knock_freq_hz = v.knock_freq_range_hz(1) + diff(v.knock_freq_range_hz) * rand(stream);
    e.audio_gain = 1 + v.sensor_gain_cv * randn(stream);
    e.vibration_gain = 1 + v.sensor_gain_cv * randn(stream);
    e.audio_resonance = 1 + v.resonance_shift_cv * randn(stream);
    e.vibration_resonance = 1 + v.resonance_shift_cv * randn(stream);
    e.coolant_start_c = v.coolant_start_range_c(1) + diff(v.coolant_start_range_c) * rand(stream);
    e.ltft_offset_pct = v.ltft_offset_range_pct(1) + diff(v.ltft_offset_range_pct) * rand(stream);
end

function metadata = build_metadata(projectConfig, batchConfig, scenario, audio, vibration, camSyncS)
    streamsCfg = projectConfig.streams;
    sensors = batchConfig.generator_params.sensors;
    metadata = struct();
    metadata.schema_version = '1.1';  % 1.1 adds the cam_sync phase reference
    metadata.run_id = scenario.run_id;
    metadata.engine_id = scenario.engine_id;
    metadata.batch_id = batchConfig.batch_id;
    metadata.generator_version = batchConfig.generator_version;
    metadata.config_schema_version = projectConfig.schema_version;
    metadata.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
    metadata.synthetic = true;
    metadata.duration_s = batchConfig.duration_s;
    metadata.streams = struct( ...
        'audio', stream_info(streamsCfg.audio, sensors.audio, numel(audio)), ...
        'vibration', stream_info(streamsCfg.vibration, sensors.vibration, numel(vibration)), ...
        'cam_sync', cam_sync_info(projectConfig.engine, camSyncS));
    rates = streamsCfg.telemetry_rates_hz;
    units = struct('rpm', 'rpm', 'load', 'percent', 'throttle', 'percent', ...
        'coolant_temp', 'degC', 'short_term_fuel_trim', 'percent', 'long_term_fuel_trim', 'percent');
    names = fieldnames(rates);
    for k = 1:numel(names)
        metadata.telemetry_signals.(names{k}) = struct('rate_hz', rates.(names{k}), 'unit', units.(names{k}));
    end
    zeroClock = struct('offset_s', 0, 'drift_ppm', 0);
    metadata.clock = struct('audio', zeroClock, 'vibration', zeroClock, 'telemetry', zeroClock);
    metadata.waveform_summary = struct('audio', summarize(audio), 'vibration', summarize(vibration));
end

function info = stream_info(streamCfg, sensor, n)
    info = struct('fs_hz', streamCfg.sample_rate_hz, 't0_s', 0, 'unit', streamCfg.unit, ...
        'n_samples', n, 'full_scale', sensor.full_scale);
end

function info = cam_sync_info(engine, camSyncS)
    % Pulse timestamps share the vibration acquisition clock (same DAQ).
    info = struct('unit', 's', 'clock', 'vibration', 'reference', 'cylinder_1_firing_tdc', ...
        'angle_deg', engine.cam_sync_angle_deg, 'n_pulses', numel(camSyncS));
end

function s = summarize(x)
    % Missing samples (NaN) are excluded from sums and counted separately.
    x = double(x);
    s = struct('n_samples', numel(x), 'n_missing', nnz(isnan(x)), ...
        'sum', sum(x, 'omitnan'), 'sum_sq', sum(x .^ 2, 'omitnan'), ...
        'first', x(1), 'last', x(end));
end

function truth = build_truth(batchConfig, scenario, events, durationS, masterSeed, streamIndices)
    inRun = events.time_s >= 0 & events.time_s < durationS;
    eventList = struct('time_s', num2cell(events.time_s(inRun)), ...
        'cylinder', num2cell(events.cylinder(inRun)), ...
        'kind', cellstr(events.kind(inRun)), ...
        'intensity', num2cell(events.intensity(inRun)), ...
        'faulted', num2cell(events.faulted(inRun)), ...
        'fault', cellstr(events.fault(inRun)));

    % NaN fields are written as null by jsonencode ([] would be written as []).
    condition = scenario.condition;
    truth = struct();
    truth.schema_version = '1.0';
    truth.run_id = scenario.run_id;
    truth.engine_id = scenario.engine_id;
    truth.split = scenario.split;
    truth.challenge = scenario.challenge;
    truth.generator_version = batchConfig.generator_version;
    truth.condition = condition;
    truth.onset_s = scenario.onset_s;
    truth.affected_cylinder = scenario.affected_cylinder;
    truth.severity = scenario.severity;
    truth.severity_end = scenario.severity_end;
    truth.degradation = scenario.degradation;
    truth.recovery_s = scenario.recovery_s;
    truth.profile_kind = scenario.profile_kind;
    % Per-fault lifecycle and activity; written as a JSON list even for one fault.
    faults = scenario.faults;
    entries = cell(1, numel(faults));
    for k = 1:numel(faults)
        entry = faults(k);
        entry.active_intervals = num2cell(fault_intervals(entry, durationS), 2);
        entries{k} = entry;
    end
    truth.conditions = cellstr(string({faults.condition}));
    truth.faults = entries;
    truth.active_intervals = num2cell(union_intervals(faults, durationS), 2);
    truth.label_timeline = label_timeline(faults, durationS);
    truth.disturbances = struct('impact_rate_hz', scenario.impact_rate_hz, ...
        'clip_gain', scenario.clip_gain, 'noise_scale', scenario.noise_scale, ...
        'resonance_shift', scenario.resonance_shift, ...
        'missing_stream', scenario.missing_stream, 'missing_gap_s', scenario.missing_gap_s, ...
        'dropout_signal', scenario.dropout_signal, 'dropout_s', scenario.dropout_s);
    truth.events = eventList;
    truth.seed = struct('master_seed', masterSeed, 'substreams', streamIndices);
end

function intervals = fault_intervals(fault, durationS)
    if ~isnan(fault.recovery_s(1))
        intervals = [fault.onset_s, fault.recovery_s(1); fault.recovery_s(2), durationS];
    else
        intervals = [fault.onset_s, durationS];
    end
end

function merged = union_intervals(faults, durationS)
    % Times when at least one fault is active (equals the fault's own intervals
    % for a single fault).
    pieces = zeros(0, 2);
    for k = 1:numel(faults)
        pieces = [pieces; fault_intervals(faults(k), durationS)]; %#ok<AGROW>
    end
    pieces = sortrows(pieces);
    merged = zeros(0, 2);
    for k = 1:size(pieces, 1)
        if ~isempty(merged) && pieces(k, 1) <= merged(end, 2)
            merged(end, 2) = max(merged(end, 2), pieces(k, 2));
        else
            merged(end + 1, :) = pieces(k, :); %#ok<AGROW>
        end
    end
end

function timeline = label_timeline(faults, durationS)
    % Stable segments labeled with the active fault set ('normal' if none,
    % otherwise sorted names joined with '+'), separated by a 1 s transition
    % around each activity change. Changes closer than 1 s share one transition.
    changes = [];
    for k = 1:numel(faults)
        changes = [changes; reshape(fault_intervals(faults(k), durationS), [], 1)]; %#ok<AGROW>
    end
    changes = unique(changes(changes > 0 & changes < durationS))';
    spans = zeros(0, 2);
    for c = changes
        if ~isempty(spans) && c - 0.5 <= spans(end, 2)
            spans(end, 2) = c + 0.5;
        else
            spans(end + 1, :) = [c - 0.5, c + 0.5]; %#ok<AGROW>
        end
    end
    edges = [0, reshape(spans', 1, []), durationS];
    labels = cell(1, numel(edges) - 1);
    for k = 1:numel(labels)
        if mod(k, 2) == 0
            labels{k} = 'transition';
        else
            labels{k} = active_label(faults, (edges(k) + edges(k + 1)) / 2, durationS);
        end
    end
    timeline = struct('start_s', num2cell(edges(1:end - 1)), 'end_s', num2cell(edges(2:end)), ...
        'label', labels);
end

function label = active_label(faults, t, durationS)
    active = strings(0, 1);
    for k = 1:numel(faults)
        intervals = fault_intervals(faults(k), durationS);
        if any(intervals(:, 1) <= t & t < intervals(:, 2))
            active(end + 1) = string(faults(k).condition); %#ok<AGROW>
        end
    end
    label = 'normal';
    if ~isempty(active)
        label = char(join(sort(active), "+"));
    end
end
