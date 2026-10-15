function [signal, info] = render_sensor(events, profile, sensor, sources, fs, durationS, warmupS, stream, options)
%RENDER_SENSOR Render one waveform channel at its native sample rate.
%   Each event excites damped sinusoidal modes evaluated analytically at the
%   exact sample times, with a raised-cosine onset so content stays below
%   Nyquist. Shaft-order tones, a sensor resonance, and colored plus white
%   noise are added. Gains, delays, mode weights, and resonance differ per
%   sensor, so audio and vibration are not scaled copies of each other.
%   Heuristic parameters; units are nominal, not calibrated.
%
%   options.NoiseFree = true omits noise (used for aliasing checks).

    arguments
        events struct
        profile struct
        sensor struct
        sources struct
        fs (1,1) double
        durationS (1,1) double
        warmupS (1,1) double
        stream
        options.NoiseFree (1,1) logical = false
        options.GainFactor (1,1) double = 1
        options.ResonanceFactor (1,1) double = 1
    end

    nOut = round(durationS * fs);
    nWarm = round(warmupS * fs);
    t = (-nWarm:nOut - 1)' / fs;
    x = zeros(size(t));

    loadAt = @(time) interp1(profile.t, profile.load, time, 'linear', 'extrap');
    assert(all(isfinite(events.time_s)), 'render_sensor:EventTime', 'Event times must be finite.');
    eventTime = events.time_s + sensor.delay_s;
    for k = 1:numel(eventTime)
        switch events.kind(k)
            case "combustion"
                src = sources.combustion;
                freq = src.freq_hz;
                tau = src.tau_s;
                gains = sensor.combustion_gains(:) * (0.5 + loadAt(events.time_s(k)));
            case "valve_impact"
                src = sources.valve_impact;
                freq = src.freq_hz * events.freq_scale(k);
                tau = src.tau_s * events.tau_scale(k);
                gains = sensor.valve_gains(:);
            case "knock_burst"
                src = sources.knock_burst;
                freq = events.mode_freq_hz(k);
                tau = events.mode_tau_s(k);
                gains = sensor.knock_gain;
        end
        x = add_transient(x, t, fs, eventTime(k), events.intensity(k) * gains, ...
            freq(:), tau(:), src.ramp_s);
    end

    % Shaft-order tones follow the delayed crank angle; scale with speed and load.
    theta = interp1(profile.t, profile.theta_deg, t - sensor.delay_s, 'linear', 'extrap');
    rpmAt = interp1(profile.t, profile.rpm, t, 'linear', 'extrap');
    toneScale = (rpmAt / 2000) .* (0.5 + loadAt(t));
    phase = 2 * pi * rand(stream, numel(sensor.orders), 1);
    for k = 1:numel(sensor.orders)
        x = x + sensor.order_amplitudes(k) * toneScale ...
            .* sin(2 * pi * sensor.orders(k) * theta / 360 + phase(k));
    end

    band = sensor.resonance_hz(:)' * options.ResonanceFactor;
    [b, a] = butter(2, band / (fs / 2), 'bandpass');
    x = x + sensor.resonance_mix * filter(b, a, x);
    y = sensor.gain * options.GainFactor * x;

    if ~options.NoiseFree
        [bl, al] = butter(2, sensor.noise_lowpass_hz / (fs / 2));
        colored = filter(bl, al, randn(stream, numel(t), 1));
        colored = colored / std(colored) * sensor.colored_noise_rms;
        y = y + colored + sensor.white_noise_rms * randn(stream, numel(t), 1);
    end

    y = y(nWarm + 1:end);
    assert(all(isfinite(y)), 'render_sensor:NonFinite', 'Rendered signal contains non-finite samples.');
    clipped = abs(y) >= sensor.full_scale;
    y = min(max(y, -sensor.full_scale), sensor.full_scale);
    signal = single(y);
    info = struct('clipped_fraction', mean(clipped), 'peak', max(abs(y)), 'rms', rms(y));
end

function x = add_transient(x, t, fs, t0, amplitude, freq, tau, rampS)
    % Sum of damped modes starting at t0 with a smooth raised-cosine onset.
    first = max(1, ceil((t0 - t(1)) * fs) + 1);
    last = min(numel(t), floor((t0 + 6 * max(tau) - t(1)) * fs) + 1);
    if first > last
        return
    end
    dt = t(first:last) - t0;
    onset = 0.5 * (1 - cos(pi * min(dt / rampS, 1)));
    burst = zeros(size(dt));
    for m = 1:numel(freq)
        burst = burst + amplitude(m) * exp(-dt / tau(m)) .* sin(2 * pi * freq(m) * dt);
    end
    x(first:last) = x(first:last) + onset .* burst;
end
