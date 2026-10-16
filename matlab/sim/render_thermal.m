function telemetry = render_thermal(profile, events, engine, rates, params, offsets, durationS, stream)
%RENDER_THERMAL Per-cylinder exhaust gas (EGT) and cylinder head (CHT) temperatures.
%   Long-form rows egt_cyl<N> and cht_cyl<N> (degC) in the telemetry table
%   format. Each cylinder's temperature follows a first-order lag toward a
%   target set by its most recent strokes:
%     energy      each combustion stroke's intensity (about 1; a misfired
%                 stroke keeps only its residual), so a dead cylinder's EGT
%                 and CHT fall toward egt_dead_c and cht_dead_c;
%     knock       detonation raises heat flux into the head: CHT rises by
%                 knock_cht_rise_c x burst intensity while EGT falls slightly
%                 (knock_egt_drop_c); EGT alone is not a knock detector;
%     valve       a valve_clearance_like event adds valve_egt_rise_c x its
%                 severity to EGT (hot gas escaping a poorly seated valve);
%     operating   both baselines rise with load and rpm; each cylinder has a
%                 fixed per-engine offset (normal cylinder-to-cylinder spread).
%   EGT uses the faster egt_tau_s (thermocouple in the runner); CHT uses
%   cht_tau_s, shortened from real head time constants (minutes) so effects
%   appear within a 30 s run. Values are sampled at the configured rates
%   with sensor noise from stream. Heuristic mappings; not fitted to engines.

    p = params.thermal;
    valveScale = params.faults.valve_clearance_like.freq_scale_per_severity;
    t = profile.t;
    dt = t(2) - t(1);
    egtBase = p.egt_base_c + p.egt_per_load_c * profile.load + p.egt_per_krpm_c * profile.rpm / 1000;
    chtBase = p.cht_base_c + p.cht_per_load_c * profile.load + p.cht_per_krpm_c * profile.rpm / 1000;

    names = strings(0, 1);
    series = zeros(numel(t), 0);
    for c = 1:engine.cylinders
        mine = events.cylinder == c;
        strokes = find(mine & events.kind == "combustion");
        strokeT = events.time_s(strokes);
        energy = min(events.intensity(strokes), p.max_energy_ratio);
        % Knock intensity of the burst following each stroke (0 if none).
        bursts = find(mine & events.kind == "knock_burst");
        knock = zeros(size(strokes));
        for b = bursts'
            k = find(strokeT <= events.time_s(b), 1, 'last');
            if ~isempty(k)
                knock(k) = max(knock(k), events.intensity(b));
            end
        end
        valves = find(mine & events.kind == "valve_impact");
        severity = zeros(size(valves));
        faulted = events.fault(valves) == "valve_clearance_like";
        severity(faulted) = (events.freq_scale(valves(faulted)) - 1) / valveScale;

        energyNow = hold_last(t, strokeT, energy, 1);
        knockNow = hold_last(t, strokeT, knock, 0);
        valveNow = hold_last(t, events.time_s(valves), severity, 0);
        egtTarget = p.egt_dead_c + (egtBase + offsets.egt_c(c) - p.egt_dead_c) .* energyNow ...
            + p.valve_egt_rise_c * valveNow - p.knock_egt_drop_c * knockNow;
        chtTarget = p.cht_dead_c + (chtBase + offsets.cht_c(c) - p.cht_dead_c) .* energyNow ...
            + p.knock_cht_rise_c * knockNow;
        names = [names; "egt_cyl" + c; "cht_cyl" + c]; %#ok<AGROW>
        series = [series, lag(egtTarget, dt, p.egt_tau_s), lag(chtTarget, dt, p.cht_tau_s)]; %#ok<AGROW>
    end

    parts = cell(numel(names), 1);
    for k = 1:numel(names)
        isEgt = startsWith(names(k), "egt");
        if isEgt
            rate = rates.egt_rate_hz;
            noise = p.egt_noise_c;
        else
            rate = rates.cht_rate_hz;
            noise = p.cht_noise_c;
        end
        tk = (0:round(durationS * rate) - 1)' / rate;
        n = numel(tk);
        value = round(interp1(t, series(:, k), tk, 'linear') + noise * randn(stream, n, 1), 1);
        parts{k} = table(tk, repmat(names(k), n, 1), value, repmat("degC", n, 1), ones(n, 1), ...
            'VariableNames', {'t_s', 'signal', 'value', 'unit', 'valid'});
    end
    telemetry = vertcat(parts{:});
end

function y = hold_last(t, eventT, value, initial)
    % Zero-order hold of per-event values on the control grid.
    y = initial * ones(size(t));
    if isempty(eventT)
        return
    end
    [eventT, order] = sort(eventT);
    value = value(order);
    k = discretize(t, [eventT; inf]);
    have = ~isnan(k);
    y(have) = value(k(have));
end

function y = lag(x, dt, tau)
    % First-order lag starting in equilibrium with the first target value.
    a = 1 - exp(-dt / tau);
    y = filter(a, [1, a - 1], x, (1 - a) * x(1));
end
