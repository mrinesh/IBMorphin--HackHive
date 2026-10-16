function config = load_project_config(repositoryRoot)
%LOAD_PROJECT_CONFIG Read configs/mvp.json and check basic sampling limits.
%   config = load_project_config() uses the repository containing this file.
%   The full schema and cross-field checks run in Python (validate-config).

    if nargin < 1
        repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
    end
    configPath = fullfile(repositoryRoot, 'configs', 'mvp.json');
    config = jsondecode(fileread(configPath));

    assert(strcmp(config.schema_version, '1.0'), ...
        'Unsupported configuration schema version.');
    assert(config.analysis.high_hz < config.streams.audio.sample_rate_hz / 2, ...
        'Acoustic analysis band must be below Nyquist.');
    assert(config.analysis.high_hz < config.streams.vibration.sample_rate_hz / 2, ...
        'Vibration analysis band must be below Nyquist.');
    if isfield(config.streams, 'crank_wheel')
        wheel = config.streams.crank_wheel;
        toothPeriodS = 60 / config.engine.rpm_max / wheel.teeth_per_rev;
        assert(wheel.missing_teeth < wheel.teeth_per_rev / 4, ...
            'Crank wheel gap must be a small part of the revolution.');
        assert(wheel.timer_resolution_s < toothPeriodS / 100, ...
            'Crank timer resolution must be far below the tooth period at rpm_max.');
    end
end
