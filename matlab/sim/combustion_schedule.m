function events = combustion_schedule(profile, engine, params, stream)
%COMBUSTION_SCHEDULE Crank-angle events for a four-stroke engine.
%   Combustion occurs combustion_offset_deg after each cylinder's firing TDC;
%   TDCs are spaced 720/cylinders apart in firing order. Each cylinder also
%   has intake- and exhaust-valve closing impacts at fixed crank offsets.
%   Events are keyed by crank angle; times come from crank_event_times, so a
%   later RPM change (such as a misfire dip) moves events consistently.

    p = params.events;
    cylinders = engine.cylinders;
    spacing = 720 / cylinders;
    tdc = zeros(cylinders, 1);
    tdc(engine.firing_order) = (0:cylinders - 1)' * spacing;

    kinds = ["combustion", "valve_impact", "valve_impact"];
    offsets = [p.combustion_offset_deg, p.intake_valve_close_deg, p.exhaust_valve_close_deg];
    thetaMin = profile.theta_deg(1);
    thetaMax = profile.theta_deg(end);

    angle = [];
    cylinder = [];
    kind = strings(0, 1);
    for c = 1:cylinders
        for k = 1:numel(kinds)
            base = tdc(c) + offsets(k);
            cycles = (ceil((thetaMin - base) / 720):floor((thetaMax - base) / 720))';
            angle = [angle; base + 720 * cycles]; %#ok<AGROW>
            cylinder = [cylinder; repmat(c, numel(cycles), 1)]; %#ok<AGROW>
            kind = [kind; repmat(kinds(k), numel(cycles), 1)]; %#ok<AGROW>
        end
    end

    n = numel(angle);
    isCombustion = kind == "combustion";
    intensity = exp(p.valve_intensity_cv * randn(stream, n, 1));
    intensity(isCombustion) = exp(p.combustion_intensity_cv * randn(stream, nnz(isCombustion), 1));
    jitter = zeros(n, 1);
    jitter(~isCombustion) = p.valve_jitter_deg * randn(stream, nnz(~isCombustion), 1);
    if isfield(p, 'combustion_jitter_deg')  % cycle-to-cycle combustion timing jitter
        jitter(isCombustion) = p.combustion_jitter_deg * randn(stream, nnz(isCombustion), 1);
    end

    events = struct();
    events.angle_deg = angle + jitter;
    events.cylinder = cylinder;
    events.kind = kind;
    events.intensity = intensity;
    events.faulted = false(n, 1);
    events.fault = strings(n, 1);  % condition that changed or added the event ("" if none)
    events.freq_scale = ones(n, 1);
    events.tau_scale = ones(n, 1);
    events.mode_freq_hz = nan(n, 1);
    events.mode_tau_s = nan(n, 1);
    events = sort_events(events);
    events.time_s = crank_event_times(profile, events.angle_deg);
end
