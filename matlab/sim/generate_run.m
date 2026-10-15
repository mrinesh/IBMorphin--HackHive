function run = generate_run(projectConfig, batchConfig, scenario)
%GENERATE_RUN Synthesize one run: waveforms, telemetry, metadata, private truth.
%   run.audio and run.vibration are single column vectors at native rates.
%   run.metadata is public; run.truth holds labels and source events and must
%   only be written to evaluation_truth/.

    params = batchConfig.generator_params;
    durationS = batchConfig.duration_s;
    warmupS = batchConfig.warmup_s;
    runNumber = sscanf(scenario.run_id, 'run_%d');
    engineNumber = sscanf(scenario.engine_id, 'engine_%d');
    masterSeed = projectConfig.dataset.master_seed;
    [streams, streamIndices] = make_streams(masterSeed, runNumber, engineNumber);

    engineParams = draw_engine_params(params.engine_variation, streams.engine);
    profile = operating_profile(scenario, durationS, warmupS, params, streams.profile);
    events = combustion_schedule(profile, projectConfig.engine, params, streams.events);
    [events, profile] = apply_fault(events, profile, scenario, params, engineParams, streams.fault);

    audioFs = projectConfig.streams.audio.sample_rate_hz;
    vibrationFs = projectConfig.streams.vibration.sample_rate_hz;
    sensors = params.sensors;
    [audio, audioInfo] = render_sensor(events, profile, sensors.audio, params.sources, ...
        audioFs, durationS, warmupS, streams.audio, ...
        GainFactor=engineParams.audio_gain, ResonanceFactor=engineParams.audio_resonance);
    [vibration, vibrationInfo] = render_sensor(events, profile, sensors.vibration, params.sources, ...
        vibrationFs, durationS, warmupS, streams.vibration, ...
        GainFactor=engineParams.vibration_gain, ResonanceFactor=engineParams.vibration_resonance);
    telemetry = render_telemetry(profile, projectConfig.streams.telemetry_rates_hz, params, ...
        engineParams, durationS, streams.telemetry);

    run = struct();
    run.audio = audio;
    run.vibration = vibration;
    run.audio_fs_hz = audioFs;
    run.vibration_fs_hz = vibrationFs;
    run.telemetry = telemetry;
    run.render_info = struct('audio', audioInfo, 'vibration', vibrationInfo);
    run.metadata = build_metadata(projectConfig, batchConfig, scenario, audio, vibration);
    run.truth = build_truth(batchConfig, scenario, events, durationS, masterSeed, streamIndices);
    run.profile = profile;
    run.engine_params = engineParams;
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

function metadata = build_metadata(projectConfig, batchConfig, scenario, audio, vibration)
    streamsCfg = projectConfig.streams;
    sensors = batchConfig.generator_params.sensors;
    metadata = struct();
    metadata.schema_version = '1.0';
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
        'vibration', stream_info(streamsCfg.vibration, sensors.vibration, numel(vibration)));
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

function s = summarize(x)
    x = double(x);
    s = struct('n_samples', numel(x), 'sum', sum(x), 'sum_sq', sum(x .^ 2), ...
        'first', x(1), 'last', x(end));
end

function truth = build_truth(batchConfig, scenario, events, durationS, masterSeed, streamIndices)
    inRun = events.time_s >= 0 & events.time_s < durationS;
    eventList = struct('time_s', num2cell(events.time_s(inRun)), ...
        'cylinder', num2cell(events.cylinder(inRun)), ...
        'kind', cellstr(events.kind(inRun)), ...
        'intensity', num2cell(events.intensity(inRun)), ...
        'faulted', num2cell(events.faulted(inRun)));

    condition = scenario.condition;
    onset = scenario.onset_s;
    truth = struct();
    truth.schema_version = '1.0';
    truth.run_id = scenario.run_id;
    truth.engine_id = scenario.engine_id;
    truth.generator_version = batchConfig.generator_version;
    truth.condition = condition;
    truth.onset_s = onset;
    truth.affected_cylinder = scenario.affected_cylinder;
    truth.severity = scenario.severity;
    if strcmp(condition, 'normal')
        truth.onset_s = [];             % encoded as null
        truth.affected_cylinder = [];
        truth.severity = 0;
        truth.active_intervals = zeros(0, 2);
        truth.label_timeline = struct('start_s', 0, 'end_s', durationS, 'label', 'normal');
    else
        truth.active_intervals = {[onset, durationS]};
        truth.label_timeline = struct( ...
            'start_s', {0, onset - 0.5, onset + 0.5}, ...
            'end_s', {onset - 0.5, onset + 0.5, durationS}, ...
            'label', {'normal', 'transition', condition});
    end
    truth.events = eventList;
    truth.seed = struct('master_seed', masterSeed, 'substreams', streamIndices);
end
