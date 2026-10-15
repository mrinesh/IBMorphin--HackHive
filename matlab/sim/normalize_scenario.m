function scenario = normalize_scenario(scenario, durationS)
%NORMALIZE_SCENARIO Fill optional scenario fields with neutral defaults.
%   Hand-written integration scenarios omit the lifecycle, disturbance, and
%   data-quality fields that sample_scenarios draws; defaults disable them.
%   Every scenario leaves with scenario.faults, one entry per active fault
%   (condition, onset_s, affected_cylinder, severity, severity_end,
%   degradation, recovery_s). A single-fault scenario's flat lifecycle fields
%   become a one-element list; a multi-fault scenario's flat fields are derived
%   from its list (condition joined with '+', earliest onset). Idempotent.

    defaults = struct('split', '', 'challenge', false, 'profile_kind', 'ramp', ...
        'severity_end', NaN, 'degradation', false, 'recovery_s', [NaN, NaN], ...
        'noise_scale', 1, 'resonance_shift', 1, 'impact_rate_hz', 0, 'clip_gain', 1, ...
        'missing_stream', '', 'missing_gap_s', [NaN, NaN], ...
        'dropout_signal', '', 'dropout_s', [NaN, NaN]);
    names = fieldnames(defaults);
    for k = 1:numel(names)
        if ~isfield(scenario, names{k})
            scenario.(names{k}) = defaults.(names{k});
        end
    end
    scenario.duration_s = durationS;
    if isfield(scenario, 'faults')
        scenario = from_fault_list(scenario);
        return
    end
    if isnan(scenario.severity_end)
        scenario.severity_end = scenario.severity;
    end
    if strcmp(scenario.condition, 'normal')
        scenario.onset_s = NaN;
        scenario.affected_cylinder = NaN;
        scenario.severity = 0;
        scenario.severity_end = 0;
    end
    scenario.recovery_s = reshape(scenario.recovery_s, 1, []);
    scenario.faults = empty_faults();
    if ~strcmp(scenario.condition, 'normal')
        scenario.faults = make_fault(scenario.condition, scenario.onset_s, ...
            scenario.affected_cylinder, scenario.severity, scenario.severity_end, ...
            scenario.degradation, scenario.recovery_s);
    end
end

function scenario = from_fault_list(scenario)
    raw = scenario.faults;
    if iscell(raw)  % jsondecode gives a cell array when entries differ in shape
        raw = [raw{:}];
    end
    faults = empty_faults();
    for k = 1:numel(raw)
        f = raw(k);
        severityEnd = f.severity;
        if isfield(f, 'severity_end') && ~isempty(f.severity_end) && ~isnan(f.severity_end)
            severityEnd = f.severity_end;
        end
        degradation = isfield(f, 'degradation') && logical(f.degradation);
        recovery = [NaN, NaN];
        if isfield(f, 'recovery_s') && ~isempty(f.recovery_s)
            recovery = reshape(double(f.recovery_s), 1, []);
        end
        faults(end + 1) = make_fault(f.condition, f.onset_s, f.affected_cylinder, ...
            f.severity, severityEnd, degradation, recovery); %#ok<AGROW>
    end
    conditions = sort(string({faults.condition}));
    assert(numel(unique(conditions)) == numel(conditions), ...
        'normalize_scenario:DuplicateFault', 'Each condition may appear once per run.');
    scenario.faults = faults;
    if isempty(faults)
        scenario.condition = 'normal';
        scenario.onset_s = NaN;
        scenario.affected_cylinder = NaN;
        scenario.severity = 0;
        scenario.severity_end = 0;
        return
    end
    % Flat fields summarize the list; per-fault values live in scenario.faults.
    scenario.condition = char(join(conditions, "+"));
    scenario.onset_s = min([faults.onset_s]);
    cylinders = unique([faults.affected_cylinder]);
    scenario.affected_cylinder = NaN;
    if isscalar(cylinders)
        scenario.affected_cylinder = cylinders;
    end
    scenario.severity = max([faults.severity]);
    scenario.severity_end = max([faults.severity_end]);
    scenario.degradation = any([faults.degradation]);
    scenario.recovery_s = [NaN, NaN];
    if isscalar(faults)
        scenario.recovery_s = faults.recovery_s;
    end
end

function f = make_fault(condition, onset, cylinder, severity, severityEnd, degradation, recovery)
    f = struct('condition', char(condition), 'onset_s', double(onset), ...
        'affected_cylinder', double(cylinder), 'severity', double(severity), ...
        'severity_end', double(severityEnd), 'degradation', logical(degradation), ...
        'recovery_s', reshape(double(recovery), 1, []));
end

function f = empty_faults()
    f = struct('condition', {}, 'onset_s', {}, 'affected_cylinder', {}, 'severity', {}, ...
        'severity_end', {}, 'degradation', {}, 'recovery_s', {});
end
