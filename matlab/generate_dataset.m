function summary = generate_dataset(options)
%GENERATE_DATASET Generate a sampled dataset such as configs/core_v1.json.
%   From the repository root:
%     addpath('matlab'); generate_dataset();                    % every run
%     generate_dataset(Runs=1:80);                              % one worker's share
%   Options:
%     Config      dataset configuration (default configs/core_v1.json)
%     OutputRoot  public bundles (default data/raw/<batch_id>)
%     TruthRoot   evaluation truth (default evaluation_truth/<batch_id>)
%     Runs        run indices to generate (default all); disjoint ranges may run
%                 in separate MATLAB processes
%     Overwrite   replace existing outputs (default false: existing runs are skipped)
%   Failed runs are recorded in quarantine_<first>-<last>.json with the error
%   instead of stopping the batch.

    arguments
        options.Config (1,1) string = ""
        options.OutputRoot (1,1) string = ""
        options.TruthRoot (1,1) string = ""
        options.Runs double = []
        options.Overwrite (1,1) logical = false
    end
    repositoryRoot = string(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(repositoryRoot, "matlab", "sim"));
    if options.Config == ""
        options.Config = fullfile(repositoryRoot, "configs", "core_v1.json");
    end
    projectConfig = load_project_config(repositoryRoot);
    batchConfig = jsondecode(fileread(options.Config));
    assert(batchConfig.duration_s == projectConfig.dataset.core_duration_s, ...
        'Dataset duration must match dataset.core_duration_s in mvp.json.');
    if options.OutputRoot == ""
        options.OutputRoot = fullfile(repositoryRoot, "data", "raw", batchConfig.batch_id);
    end
    if options.TruthRoot == ""
        options.TruthRoot = fullfile(repositoryRoot, "evaluation_truth", batchConfig.batch_id);
    end
    if ~isfolder(options.TruthRoot)
        mkdir(options.TruthRoot);
    end

    scenarios = sample_scenarios(projectConfig, batchConfig);
    write_json(scenarios, fullfile(options.TruthRoot, "scenarios.json"));
    indices = options.Runs;
    if isempty(indices)
        indices = 1:numel(scenarios);
    end

    timer = tic;
    generated = 0;
    skipped = 0;
    failures = struct('run_id', {}, 'error', {});
    for k = indices
        scenario = scenarios(k);
        runDir = fullfile(options.OutputRoot, scenario.run_id);
        if ~options.Overwrite && isfolder(runDir)
            skipped = skipped + 1;
            continue
        end
        try
            run = generate_run(projectConfig, batchConfig, scenario);
            export_run(run, options.OutputRoot, options.TruthRoot, Overwrite=options.Overwrite);
            generated = generated + 1;
            fprintf('%s  %-8s %-21s %s  %.0f s elapsed\n', scenario.run_id, scenario.split, ...
                scenario.condition, ternary(scenario.challenge, 'challenge', 'core'), toc(timer));
        catch error
            failures(end + 1) = struct('run_id', scenario.run_id, 'error', error.message); %#ok<AGROW>
            fprintf(2, '%s  QUARANTINED: %s\n', scenario.run_id, error.message);
        end
    end
    if ~isempty(failures)
        write_json(failures, fullfile(options.TruthRoot, ...
            sprintf("quarantine_%d-%d.json", indices(1), indices(end))));
    end
    summary = struct('generated', generated, 'skipped', skipped, ...
        'quarantined', numel(failures), 'seconds', toc(timer));
    fprintf('generated %d, skipped %d, quarantined %d in %.0f s\n', generated, skipped, ...
        numel(failures), summary.seconds);
end

function write_json(data, path)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(data, PrettyPrint=true), 'char');
end

function out = ternary(condition, a, b)
    if condition
        out = a;
    else
        out = b;
    end
end
