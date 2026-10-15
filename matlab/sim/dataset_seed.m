function seed = dataset_seed(projectConfig, batchConfig)
%DATASET_SEED Master seed for a batch: its own master_seed if set, else the project's.
%   A separate seed keeps a new dataset (e.g. core_v2) from reusing the random
%   streams of an earlier one that shares run and engine numbers.

    seed = projectConfig.dataset.master_seed;
    if isfield(batchConfig, 'master_seed')
        seed = batchConfig.master_seed;
    end
end
