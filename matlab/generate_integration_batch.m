function runs = generate_integration_batch(options)
%GENERATE_INTEGRATION_BATCH Generate and export the runs in configs/integration_batch.json.
%   From the repository root:
%     addpath('matlab'); runs = generate_integration_batch();
%   Options:
%     OutputRoot  public run bundles (default data/raw)
%     TruthRoot   evaluation-only truth (default evaluation_truth)
%     Runs        run indices to generate (default all)
%     Overwrite   replace existing outputs (default false)

    arguments
        options.OutputRoot (1,1) string = ""
        options.TruthRoot (1,1) string = ""
        options.Runs double = []
        options.Overwrite (1,1) logical = false
    end
    repositoryRoot = string(fileparts(fileparts(mfilename('fullpath'))));
    if options.OutputRoot == ""
        options.OutputRoot = fullfile(repositoryRoot, "data", "raw");
    end
    if options.TruthRoot == ""
        options.TruthRoot = fullfile(repositoryRoot, "evaluation_truth");
    end
    addpath(fullfile(repositoryRoot, "matlab", "sim"));

    projectConfig = load_project_config(repositoryRoot);
    batchConfig = jsondecode(fileread(fullfile(repositoryRoot, "configs", "integration_batch.json")));
    scenarios = batchConfig.runs;
    assert(numel(scenarios) == projectConfig.dataset.integration_runs, ...
        'integration_batch.json run count must match dataset.integration_runs in mvp.json.');
    assert(batchConfig.duration_s == projectConfig.dataset.integration_duration_s, ...
        'integration_batch.json duration must match dataset.integration_duration_s in mvp.json.');
    indices = options.Runs;
    if isempty(indices)
        indices = 1:numel(scenarios);
    end

    runs = struct('run_id', {}, 'condition', {}, 'paths', {}, 'seconds', {}, ...
        'audio_peak', {}, 'vibration_peak', {}, 'clipped', {});
    for k = indices
        scenario = scenarios(k);
        assert(any(strcmp(scenario.condition, projectConfig.conditions)), ...
            'Unknown condition "%s".', scenario.condition);
        timer = tic;
        run = generate_run(projectConfig, batchConfig, scenario);
        paths = export_run(run, options.OutputRoot, options.TruthRoot, Overwrite=options.Overwrite);
        seconds = toc(timer);
        info = run.render_info;
        clipped = max(info.audio.clipped_fraction, info.vibration.clipped_fraction);
        runs(end + 1) = struct('run_id', scenario.run_id, 'condition', scenario.condition, ...
            'paths', paths, 'seconds', seconds, 'audio_peak', info.audio.peak, ...
            'vibration_peak', info.vibration.peak, 'clipped', clipped); %#ok<AGROW>
        fprintf('%s  %-22s  %5.1f s  audio peak %.3f  vibration peak %.1f  clipped %.4f\n', ...
            scenario.run_id, scenario.condition, seconds, info.audio.peak, ...
            info.vibration.peak, clipped);
    end
end
