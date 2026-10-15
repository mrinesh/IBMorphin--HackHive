function [events, profile] = apply_fault(events, profile, scenario, params, engineParams, stream)
%APPLY_FAULT Apply every fault in scenario.faults to the events, in physical order.
%   misfire:              intermittent weak combustion plus a brief RPM dip,
%                         which also shifts later event timing.
%   knock_like:           damped 5-9 kHz bursts shortly after combustion.
%   valve_clearance_like: shifted, sharper, more variable valve impacts whose
%                         amplitude may rise or fall.
%   Faults run in the order misfire -> knock_like -> valve_clearance_like, so
%   later events follow the misfire RPM dips, and a misfired (weakened) power
%   stroke on the same cylinder cannot also knock. Each fault has its own
%   onset, cylinder, and severity, and is inactive during its recovery_s. When
%   a fault's degradation is true, its severity ramps linearly from severity
%   to severity_end between onset and the end of the run (accelerated,
%   simulated deterioration). Heuristic parameter mappings; not fitted to
%   engine measurements. Every changed or added event has faulted = true and
%   fault = its condition, for the evaluation truth. scenario must already be
%   completed by normalize_scenario. A single fault draws from the stream
%   exactly as before multi-fault support, so single-fault runs are unchanged.

    order = ["misfire", "knock_like", "valve_clearance_like"];
    conditions = string({scenario.faults.condition});
    unknown = setdiff(conditions, order);
    if ~isempty(unknown)
        error('apply_fault:UnknownCondition', 'Unknown condition "%s".', unknown(1));
    end
    for condition = order
        k = find(conditions == condition, 1);
        if ~isempty(k)
            [events, profile] = apply_one(events, profile, scenario.faults(k), ...
                scenario.duration_s, params, engineParams, stream);
        end
    end
end

function [events, profile] = apply_one(events, profile, fault, durationS, params, engineParams, stream)
    condition = string(fault.condition);
    rec = fault.recovery_s;
    recovering = events.time_s >= rec(1) & events.time_s < rec(2);  % NaN bounds: never
    affected = events.time_s >= fault.onset_s & ~recovering ...
        & events.cylinder == fault.affected_cylinder;
    severityAt = @(t) severity_schedule(fault, durationS, t);
    f = params.faults.(condition);

    switch condition
        case "misfire"
            candidates = find(affected & events.kind == "combustion");
            probability = min(f.base_probability ...
                + f.probability_per_severity * severityAt(events.time_s(candidates)), 1);
            hit = candidates(rand(stream, numel(candidates), 1) < probability);
            residual = f.residual_range(1) + diff(f.residual_range) * rand(stream, numel(hit), 1);
            events.intensity(hit) = events.intensity(hit) .* residual;
            events.faulted(hit) = true;
            events.fault(hit) = condition;
            profile = add_rpm_dips(profile, events.time_s(hit), 1 - residual, f);
            events.time_s = crank_event_times(profile, events.angle_deg);

        case "knock_like"
            candidates = find(affected & events.kind == "combustion" ...
                & events.fault ~= "misfire");  % a misfired stroke cannot knock
            probability = min(f.base_probability ...
                + f.probability_per_severity * severityAt(events.time_s(candidates)), 1);
            hit = candidates(rand(stream, numel(candidates), 1) < probability);
            n = numel(hit);
            severity = severityAt(events.time_s(hit));
            src = params.sources.knock_burst;
            burst = struct();
            burst.angle_deg = events.angle_deg(hit) ...
                + f.delay_range_deg(1) + diff(f.delay_range_deg) * rand(stream, n, 1);
            burst.cylinder = events.cylinder(hit);
            burst.kind = repmat("knock_burst", n, 1);
            burst.intensity = severity .* (0.6 + 0.8 * rand(stream, n, 1)) .* events.intensity(hit);
            burst.faulted = true(n, 1);
            burst.fault = repmat(condition, n, 1);
            burst.freq_scale = ones(n, 1);
            burst.tau_scale = ones(n, 1);
            burst.mode_freq_hz = min(max(engineParams.knock_freq_hz ...
                + f.freq_jitter_hz * randn(stream, n, 1), src.freq_range_hz(1)), src.freq_range_hz(2));
            burst.mode_tau_s = src.tau_range_s(1) + diff(src.tau_range_s) * rand(stream, n, 1);
            burst.time_s = crank_event_times(profile, burst.angle_deg);
            events = append_events(events, burst);

        case "valve_clearance_like"
            hit = find(affected & events.kind == "valve_impact");
            n = numel(hit);
            severity = severityAt(events.time_s(hit));
            events.angle_deg(hit) = events.angle_deg(hit) + f.timing_shift_deg * severity ...
                + f.extra_jitter_deg * severity .* randn(stream, n, 1);
            events.freq_scale(hit) = 1 + f.freq_scale_per_severity .* severity;
            events.tau_scale(hit) = 1 + f.tau_scale_per_severity .* severity;
            events.intensity(hit) = events.intensity(hit) ...
                .* max(0.3, 1 + f.amplitude_spread * severity .* randn(stream, n, 1));
            events.faulted(hit) = true;
            events.fault(hit) = condition;
            events.time_s = crank_event_times(profile, events.angle_deg);
            events = sort_events(events);

        otherwise
            error('apply_fault:UnknownCondition', 'Unknown condition "%s".', condition);
    end
    % Shifted, delayed, or slowed events can fall beyond the simulated profile.
    events = drop_events(events, isnan(events.time_s));
end

function severity = severity_schedule(fault, durationS, t)
    severity = fault.severity * ones(size(t));
    if fault.degradation
        progress = min(max((t - fault.onset_s) / (durationS - fault.onset_s), 0), 1);
        severity = fault.severity + (fault.severity_end - fault.severity) * progress;
    end
end

function profile = add_rpm_dips(profile, times, depthFraction, f)
    % A missing power stroke briefly slows the crank; shift later crank angles.
    dip = zeros(size(profile.t));
    for k = 1:numel(times)
        dip = dip + depthFraction(k) * exp(-0.5 * ((profile.t - times(k) - 2 * f.rpm_dip_sigma_s) ...
            / f.rpm_dip_sigma_s) .^ 2);
    end
    profile.rpm = profile.rpm - f.rpm_dip_rpm * dip;
    profile = integrate_crank(profile);
end

function events = drop_events(events, remove)
    names = fieldnames(events);
    for k = 1:numel(names)
        events.(names{k})(remove) = [];
    end
end

function events = append_events(events, extra)
    names = fieldnames(events);
    for k = 1:numel(names)
        events.(names{k}) = [events.(names{k}); extra.(names{k})];
    end
    events = sort_events(events);
end
