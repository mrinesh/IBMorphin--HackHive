function project = initialize_project()
%INITIALIZE_PROJECT Inspect MATLAB access and load shared project settings.
%   From the repository root: addpath('matlab'); project = initialize_project;
%   This function does not generate signals or require specialized toolboxes.

    matlabRoot = fileparts(mfilename('fullpath'));
    repositoryRoot = fileparts(matlabRoot);
    configPath = fullfile(repositoryRoot, 'configs', 'mvp.json');
    project = struct();
    project.root = repositoryRoot;
    project.config = jsondecode(fileread(configPath));
    project.matlab_version = version;
    project.products = ver;

    assert(strcmp(project.config.schema_version, '1.0'), ...
        'Unsupported configuration schema version.');
    assert(project.config.analysis.high_hz < ...
        project.config.streams.audio.sample_rate_hz / 2, ...
        'Acoustic analysis band must be below Nyquist.');
    assert(project.config.analysis.high_hz < ...
        project.config.streams.vibration.sample_rate_hz / 2, ...
        'Vibration analysis band must be below Nyquist.');

    fprintf('Engine diagnostics workspace: %s\n', repositoryRoot);
    fprintf('MATLAB: %s\n', project.matlab_version);
    fprintf('Shared configuration loaded: %s\n', configPath);
    fprintf('Installed products (license execution must be checked separately):\n');
    disp(struct2table(project.products));
    fprintf('Next milestone: small synchronized integration batch.\n');
end
