function profile = operating_profile(scenario, durationS, warmupS, params, stream)
%OPERATING_PROFILE Control-rate RPM, load (0-1), throttle (%), and crank angle.
%   The profile spans [-warmupS, durationS + tail] so filters settle before
%   t = 0 and events near the end are fully rendered.

    rate = params.control_rate_hz;
    p = params.profile;
    tailS = 0.05;
    t = (-warmupS:1/rate:durationS + tailS)';

    rampS = scenario.ramp_s;
    fraction = min(max((t - rampS(1)) / (rampS(2) - rampS(1)), 0), 1);
    fraction = fraction .^ 2 .* (3 - 2 * fraction);  % smoothstep

    [b, a] = butter(1, p.wander_cutoff_hz / (rate / 2));
    rpm = scenario.rpm(1) + diff(scenario.rpm) * fraction ...
        + p.rpm_wander_rpm * wander(stream, numel(t), b, a);
    load = scenario.load(1) + diff(scenario.load) * fraction ...
        + p.load_wander * wander(stream, numel(t), b, a);
    load = min(max(load, 0), 1);

    profile = struct();
    profile.t = t;
    profile.rpm = rpm;
    profile.load = load;
    profile.throttle_pct = p.throttle_offset_pct + p.throttle_per_load_pct * load;
    profile.theta0_deg = 720 * rand(stream);
    profile = integrate_crank(profile);
end

function w = wander(stream, n, b, a)
    w = filter(b, a, randn(stream, n, 1));
    w = (w - mean(w)) / std(w);
end
