function timeS = cam_sync_times(profile, angleDeg, durationS)
%CAM_SYNC_TIMES Cam position (phase reference) pulse times on the master clock.
%   One pulse per 720-degree cycle, when the crank passes angleDeg after
%   cylinder 1's firing TDC (crank angle 0 mod 720 in combustion_schedule).
%   Pulses follow the final crank profile, so misfire RPM dips move them
%   consistently with the combustion events. Deterministic: no random draws,
%   so adding the reference leaves every other stream unchanged. Returns a
%   column vector of times in [0, durationS).

    thetaStart = interp1(profile.t, profile.theta_deg, 0, 'linear');
    thetaEnd = interp1(profile.t, profile.theta_deg, durationS, 'linear');
    cycles = (ceil((thetaStart - angleDeg) / 720):floor((thetaEnd - angleDeg) / 720))';
    timeS = crank_event_times(profile, angleDeg + 720 * cycles);
    timeS = timeS(timeS >= 0 & timeS < durationS);
end
