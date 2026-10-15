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
end
