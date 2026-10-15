function fig = plot_run_overview(runDir, truthPath, outputPath)
%PLOT_RUN_OVERVIEW Review figure for one run: spectrograms, zooms, RPM.
%   Uses the evaluation truth to mark onset and faulted events, so it is a
%   generator review tool, not an inference display.
%   plot_run_overview('data/raw/run_00001', ...
%       'evaluation_truth/run_00001_events.json', 'reports/generated/run_00001.png')

    w = load(fullfile(runDir, 'waveforms.mat'));
    telemetry = readtable(fullfile(runDir, 'telemetry.csv'), 'TextType', 'string');
    truth = jsondecode(fileread(truthPath));
    onset = truth.onset_s;
    isNormal = isempty(onset);
    if isNormal
        onset = numel(w.audio) / w.audio_fs_hz / 2;  % mid-run reference for the zooms
    end
    times = [truth.events.time_s]';
    faulted = [truth.events.faulted]';

    fig = figure('Visible', 'off', 'Position', [0 0 1400 1000]);
    layout = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact');
    if isNormal
        heading = sprintf('%s  |  normal  |  zooms compare %.1f s and %.1f s', ...
            truth.run_id, onset - 1, onset + 2);
    else
        heading = sprintf('%s  |  %s  |  onset %.1f s, cylinder %d, severity %.2f', ...
            truth.run_id, truth.condition, onset, truth.affected_cylinder, truth.severity);
    end
    title(layout, heading, 'Interpreter', 'none');

    % Zoom after onset starts just before the first faulted event 2 s past onset.
    zoom = 0.15;
    later = times(faulted & times >= onset + 2);
    zoomStart = onset + 2;
    if ~isempty(later)
        zoomStart = later(1) - 0.02;
    end

    names = ["audio", "vibration"];
    for k = 1:2
        x = double(w.(names(k)));
        fs = w.(names(k) + "_fs_hz");
        ax = nexttile(layout, k);
        [s, f, t] = spectrogram(x, hann(2048), 2048 - 512, 2048, fs);
        imagesc(ax, t, f / 1000, 10 * log10(abs(s) .^ 2 + eps));
        axis(ax, 'xy');
        if ~isNormal
            xline(ax, onset, 'w--', 'LineWidth', 1.5);
        end
        ylabel(ax, 'kHz');
        title(ax, names(k) + " spectrogram");
        clim(ax, max(clim(ax)) - [70 0]);
        colorbar(ax);

        ax = nexttile(layout, k + 2);
        tAxis = (0:numel(x) - 1)' / fs;
        before = tAxis >= onset - 1 - zoom & tAxis < onset - 1;
        after = tAxis >= zoomStart & tAxis < zoomStart + zoom;
        plot(ax, tAxis(before) - tAxis(find(before, 1)), x(before), 'Color', [0.5 0.5 0.5]);
        hold(ax, 'on');
        plot(ax, tAxis(after) - tAxis(find(after, 1)), x(after), 'Color', [0.85 0.33 0.1]);
        marks = times(faulted & times >= zoomStart & times < zoomStart + zoom) - zoomStart;
        if ~isempty(marks)
            xline(ax, marks, ':', 'Color', [0.85 0.33 0.1]);
        end
        hold(ax, 'off');
        legend(ax, {'1 s before onset', 'after onset'}, 'Location', 'northeast');
        xlabel(ax, 's');
        title(ax, names(k) + " zoom (dotted: faulted events)");
    end

    ax = nexttile(layout, 5, [1 2]);
    rpm = telemetry(telemetry.signal == "rpm", :);
    plot(ax, rpm.t_s, rpm.value);
    if ~isNormal
        xline(ax, onset, 'k--');
    end
    xlabel(ax, 's');
    ylabel(ax, 'rpm');
    title(ax, 'RPM telemetry');

    if nargin >= 3
        exportgraphics(fig, outputPath, 'Resolution', 110);
    end
end
