function paths = export_run(run, outputRoot, truthRoot, options)
%EXPORT_RUN Write one run bundle following docs/DATA_CONTRACT.md.
%   <outputRoot>/<run_id>/waveforms.mat   (-v7, single column vectors, cam_sync_s,
%                                          crank_tooth_s from contract 1.2)
%   <outputRoot>/<run_id>/telemetry.csv
%   <outputRoot>/<run_id>/metadata.json
%   <truthRoot>/<run_id>_events.json      (evaluation only)

    arguments
        run struct
        outputRoot (1,1) string
        truthRoot (1,1) string
        options.Overwrite (1,1) logical = false
    end

    runId = run.metadata.run_id;
    runDir = fullfile(outputRoot, runId);
    truthPath = fullfile(truthRoot, runId + "_events.json");
    if ~options.Overwrite && (isfolder(runDir) || isfile(truthPath))
        error('export_run:Exists', '%s already exists; pass Overwrite=true to replace it.', runId);
    end
    if ~isfolder(runDir)
        mkdir(runDir);
    end
    if ~isfolder(truthRoot)
        mkdir(truthRoot);
    end

    audio = single(run.audio(:));
    vibration = single(run.vibration(:));
    audio_fs_hz = double(run.audio_fs_hz);
    vibration_fs_hz = double(run.vibration_fs_hz);
    audio_t0_s = double(run.metadata.streams.audio.t0_s);
    vibration_t0_s = double(run.metadata.streams.vibration.t0_s);
    cam_sync_s = double(run.cam_sync_s(:));  % 0x1 when no pulse falls inside the run
    paths = struct();
    paths.waveforms = fullfile(runDir, 'waveforms.mat');
    names = {'audio', 'vibration', 'audio_fs_hz', 'vibration_fs_hz', 'audio_t0_s', ...
        'vibration_t0_s', 'cam_sync_s'};
    if isfield(run, 'crank_tooth_s')
        crank_tooth_s = double(run.crank_tooth_s(:)); %#ok<NASGU> saved by name
        names{end + 1} = 'crank_tooth_s';
    end
    save(paths.waveforms, names{:}, '-v7');

    paths.telemetry = fullfile(runDir, 'telemetry.csv');
    write_telemetry_csv(run.telemetry, paths.telemetry);
    paths.metadata = fullfile(runDir, 'metadata.json');
    write_json(run.metadata, paths.metadata);
    paths.truth = truthPath;
    write_json(run.truth, truthPath);
end

function write_telemetry_csv(telemetry, path)
    % Explicit formatting: missing values are written as empty fields.
    value = compose("%.10g", telemetry.value);
    value(isnan(telemetry.value)) = "";
    lines = compose("%.6f,%s,%s,%s,%d", telemetry.t_s, string(telemetry.signal), ...
        value, string(telemetry.unit), telemetry.valid);
    writelines(["t_s,signal,value,unit,valid"; lines], path);
end

function write_json(data, path)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(data, PrettyPrint=true), 'char');
end
