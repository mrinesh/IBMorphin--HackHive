function scenario = normalize_scenario(scenario, durationS)
%NORMALIZE_SCENARIO Fill optional scenario fields with neutral defaults.
%   Hand-written integration scenarios omit the lifecycle, disturbance, and
%   data-quality fields that sample_scenarios draws; defaults disable them.

    defaults = struct('split', '', 'challenge', false, 'profile_kind', 'ramp', ...
        'severity_end', NaN, 'degradation', false, 'recovery_s', [NaN, NaN], ...
        'noise_scale', 1, 'resonance_shift', 1, 'impact_rate_hz', 0, 'clip_gain', 1, ...
        'missing_stream', '', 'missing_gap_s', [NaN, NaN], ...
        'dropout_signal', '', 'dropout_s', [NaN, NaN]);
    names = fieldnames(defaults);
    for k = 1:numel(names)
        if ~isfield(scenario, names{k})
            scenario.(names{k}) = defaults.(names{k});
        end
    end
    if isnan(scenario.severity_end)
        scenario.severity_end = scenario.severity;
    end
    if strcmp(scenario.condition, 'normal')
        scenario.onset_s = NaN;
        scenario.affected_cylinder = NaN;
        scenario.severity = 0;
        scenario.severity_end = 0;
    end
    scenario.recovery_s = reshape(scenario.recovery_s, 1, []);
    scenario.duration_s = durationS;
end
