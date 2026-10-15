function project = initialize_project()
%INITIALIZE_PROJECT Inspect MATLAB access and load shared project settings.
%   From the repository root: addpath('matlab'); project = initialize_project;
%   This function does not generate signals or require specialized toolboxes.

    matlabRoot = fileparts(mfilename('fullpath'));
    repositoryRoot = fileparts(matlabRoot);
    configPath = fullfile(repositoryRoot, 'configs', 'mvp.json');
    project = struct();
    project.root = repositoryRoot;
    project.config = load_project_config(repositoryRoot);
    project.matlab_version = version;
    project.products = ver;

    fprintf('Engine diagnostics workspace: %s\n', repositoryRoot);
    fprintf('MATLAB: %s\n', project.matlab_version);
    fprintf('Shared configuration loaded: %s\n', configPath);
    fprintf('Installed products (license execution must be checked separately):\n');
    disp(struct2table(project.products));
    fprintf('Generate the integration batch with generate_integration_batch().\n');
end
