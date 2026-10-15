function profile = integrate_crank(profile)
%INTEGRATE_CRANK Crank angle in degrees from RPM: theta = theta0 + integral(6*rpm).

    profile.theta_deg = profile.theta0_deg + 6 * cumtrapz(profile.t, profile.rpm);
end
