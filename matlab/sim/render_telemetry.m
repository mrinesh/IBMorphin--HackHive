function telemetry = render_telemetry(profile, rates, params, engineParams, durationS, stream)
%RENDER_TELEMETRY Long-form telemetry table (t_s, signal, value, unit, valid).
%   Signals are sampled at their configured rates on the master clock.
%   Load and throttle are converted from fractions to percent. No condition
%   receives a unique telemetry value; fuel trims and coolant are generic.

    p = params.telemetry;
    v = params.engine_variation;
    specs = {
        "rpm",                  "rpm"
        "load",                 "percent"
        "throttle",             "percent"
        "coolant_temp",         "degC"
        "short_term_fuel_trim", "percent"
        "long_term_fuel_trim",  "percent"
    };

    parts = cell(size(specs, 1), 1);
    for k = 1:size(specs, 1)
        name = specs{k, 1};
        rate = rates.(name);
        tk = (0:round(durationS * rate) - 1)' / rate;
        n = numel(tk);
        at = @(field) interp1(profile.t, profile.(field), tk, 'linear');
        switch name
            case "rpm"
                value = round(at('rpm') + p.rpm_noise_rpm * randn(stream, n, 1));
            case "load"
                value = round(100 * at('load') + p.load_noise_pct * randn(stream, n, 1), 1);
            case "throttle"
                value = round(at('throttle_pct') + p.throttle_noise_pct * randn(stream, n, 1), 1);
            case "coolant_temp"
                warm = engineParams.coolant_start_c + (v.coolant_target_c - engineParams.coolant_start_c) ...
                    * (1 - exp(-tk / v.coolant_tau_s));
                value = round(warm + p.coolant_noise_c * randn(stream, n, 1), 1);
            case "short_term_fuel_trim"
                value = round(p.stft_noise_pct * randn(stream, n, 1), 1);
            case "long_term_fuel_trim"
                value = round(engineParams.ltft_offset_pct + p.ltft_drift_pct_per_s * tk, 1);
        end
        parts{k} = table(tk, repmat(name, n, 1), value, repmat(specs{k, 2}, n, 1), ones(n, 1), ...
            'VariableNames', {'t_s', 'signal', 'value', 'unit', 'valid'});
    end
    telemetry = sortrows(vertcat(parts{:}), {'t_s', 'signal'});
end
