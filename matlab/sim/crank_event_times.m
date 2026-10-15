function timeS = crank_event_times(profile, angleDeg)
%CRANK_EVENT_TIMES Invert the monotonic crank angle to find event times.

    timeS = interp1(profile.theta_deg, profile.t, angleDeg, 'linear');
end
