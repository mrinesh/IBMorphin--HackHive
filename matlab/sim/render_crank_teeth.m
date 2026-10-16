function [toothS, info] = render_crank_teeth(profile, events, wheel, params, toothErrorDeg, durationS, stream)
%RENDER_CRANK_TEETH Edge times of a 60-2 crank trigger wheel on the master clock.
%   The instantaneous crank speed seen by the wheel is the control-rate
%   profile plus stroke-resolved terms the 1 kHz profile cannot carry:
%     gas ripple   each power stroke adds its combustion energy (intensity)
%                  over the 180 degrees after TDC while the load and the next
%                  compression take a steady share, so a healthy stroke gives
%                  a zero-mean dip-then-rise of size gas_ripple_rpm_ref x
%                  (reference_rpm / rpm) x load factor;
%     misfire      a misfired stroke loses its energy, so speed falls through
%                  that stroke and recovers exponentially afterwards. The
%                  recovery time is chosen so the lost crank angle equals the
%                  smooth misfire dip already in profile.rpm, which this
%                  function replaces; the wheel therefore stays phase-locked
%                  to the acoustic events (residual reported in info);
%     inertia      reciprocating masses give a second-order term that grows
%                  with rpm (fast at TDC and BDC, slow at mid-stroke).
%   Edges are the tooth falling edges at first_tooth_angle_deg + 6 x k
%   (k = 0..teeth-missing-1) after cylinder 1 firing TDC, mod 360, plus a
%   fixed per-engine tooth error, timestamp jitter, and timer quantization.
%   Nothing is written back to profile, so audio, vibration, cam_sync and
%   telemetry are unchanged. Returns a column of times in [0, durationS).
%   Heuristic parameter mappings; not fitted to engine measurements.

    c = params.crank;
    offsetDeg = params.events.combustion_offset_deg;
    step = c.grid_deg;
    theta = (ceil(profile.theta_deg(1) / step) * step:step:profile.theta_deg(end))';
    t = interp1(profile.theta_deg, profile.t, theta, 'linear');
    [smoothDip, dips] = misfire_dips(profile, t);
    rpm = interp1(profile.t, profile.rpm, t, 'linear') + smoothDip;
    loadNow = interp1(profile.t, profile.load, t, 'linear');
    gain = c.gas_ripple_rpm_ref * c.reference_rpm ./ rpm ...
        .* (0.5 + loadNow) / (0.5 + c.reference_load);

    % Normalized cumulative stroke energy: f(x) = (1 - cos x)/2 - x/pi peaks at
    % +-0.105 over a healthy stroke, so dividing by that gives unit amplitude.
    shapePeak = 0.10549;
    isCombustion = events.kind == "combustion";
    tdc = events.angle_deg(isCombustion) - offsetDeg;
    energy = events.intensity(isCombustion);
    [tdc, order] = sort(tdc);
    energy = energy(order);
    misfired = false(size(tdc));
    if ~isempty(dips.angle_deg)
        [found, where] = ismembertol(dips.angle_deg - offsetDeg, tdc, 1e-6, 'DataScale', 1);
        misfired(where(found)) = true;
        energy(where(found)) = 1 - dips.depth(found);
    end
    stroke = discretize(theta, [tdc; inf]);
    inStroke = ~isnan(stroke);
    x = zeros(size(theta));
    x(inStroke) = pi * min((theta(inStroke) - tdc(stroke(inStroke))) / 180, 1);
    e = ones(size(theta));
    e(inStroke) = energy(stroke(inStroke));
    loss = e;  % healthy strokes scale the whole (zero-sum) stroke shape
    loss(inStroke & misfired(max(stroke, 1))) = 1;  % misfire: load still takes its share
    cumulative = e .* (1 - cos(x)) / 2 - loss .* x / pi;

    % Carry each misfire's deficit past its stroke with an exponential recovery
    % whose area matches the smooth dip it replaces.
    strokeEnd = interp1(theta, t, tdc(misfired) + 180, 'linear', 'extrap');
    misfireRows = find(misfired);
    tau = zeros(numel(misfireRows), 1);
    for k = 1:numel(misfireRows)
        depth = 1 - energy(misfireRows(k));
        endS = strokeEnd(k);
        g = interp1(t, gain, endS, 'linear', 'extrap');
        dropRpm = g * depth / shapePeak;
        strokeS = 30 / interp1(t, rpm, endS, 'linear', 'extrap');
        areaRpmS = dips.amplitude_rpm * depth * dips.sigma_s * sqrt(2 * pi);
        tau(k) = max(areaRpmS / dropRpm - strokeS / 2, c.min_recovery_s);
        after = find(t >= endS & t < endS + 10 * tau(k));
        cumulative(after) = cumulative(after) - depth * exp(-(t(after) - endS) / tau(k));
    end

    inertia = c.inertia_ripple_rpm_ref * rpm / c.reference_rpm .* cos(2 * deg2rad(theta));
    speed = rpm + gain .* cumulative / shapePeak + inertia;

    % Time at each grid angle, then remove slow drift against the profile clock
    % so the wheel and the event-driven streams share one crank phase.
    mid = (speed(1:end - 1) + speed(2:end)) / 2;
    tWheel = t(1) + [0; cumsum(diff(theta) ./ (6 * mid))];
    lag = tWheel - t;
    window = max(3, round(c.drift_window_s * 6 * mean(rpm) / step));
    lag = lag - movmean(lag, window);
    tWheel = t + lag;

    % Tooth falling edges.
    teeth = wheel.teeth_per_rev - wheel.missing_teeth;
    pitch = 360 / wheel.teeth_per_rev;
    first = wheel.first_tooth_angle_deg;
    revs = (floor((theta(1) - first) / 360):ceil((theta(end) - first) / 360))';
    nominal = first + 360 * revs + pitch * (0:teeth - 1);
    actual = nominal + reshape(toothErrorDeg, 1, []);
    edgeAngle = reshape(actual', [], 1);
    edgeAngle = edgeAngle(edgeAngle > theta(1) & edgeAngle < theta(end));
    toothS = interp1(theta, tWheel, edgeAngle, 'linear');
    toothS = toothS + c.edge_jitter_s * randn(stream, numel(toothS), 1);
    toothS = round(toothS / wheel.timer_resolution_s) * wheel.timer_resolution_s;
    toothS = toothS(toothS >= 0 & toothS < durationS);

    inRun = t >= 0 & t < durationS;
    info = struct();
    phaseError = abs(lag(inRun) .* 6 .* rpm(inRun));
    info.max_phase_error_deg = max(phaseError);
    info.p95_phase_error_deg = prctile(phaseError, 95);
    info.misfire_strokes = numel(misfireRows);
    info.recovery_s = tau;
    info.speed_rpm_range = [min(speed(inRun)), max(speed(inRun))];
end

function [dip, dips] = misfire_dips(profile, t)
    % The smooth misfire dips apply_fault subtracted from profile.rpm.
    dip = zeros(size(t));
    dips = struct('angle_deg', zeros(0, 1), 'depth', zeros(0, 1), ...
        'amplitude_rpm', 0, 'sigma_s', 1);
    if ~isfield(profile, 'misfire_dips')
        return
    end
    dips = profile.misfire_dips;
    for k = 1:numel(dips.time_s)
        dip = dip + dips.amplitude_rpm * dips.depth(k) ...
            * exp(-0.5 * ((t - dips.time_s(k) - 2 * dips.sigma_s) / dips.sigma_s) .^ 2);
    end
end
