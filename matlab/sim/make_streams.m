function [streams, indices] = make_streams(masterSeed, runNumber, engineNumber)
%MAKE_STREAMS Independent reproducible random streams for one run.
%   Each purpose uses its own mrg32k3a substream of the master seed, so
%   changing one stage (for example sensor noise) never shifts another.
%   Run substreams are runNumber*16 + k; engine substreams start at 2^40.
%   crank and thermal (data contract 1.2) and the per-engine sensor draws
%   (engine_sensors, from 2^41) were appended later, so earlier streams keep
%   their indices and every older stream is unchanged.

    purposes = ["profile", "events", "fault", "audio", "vibration", "telemetry", "disturbance", ...
        "crank", "thermal"];
    streams = struct();
    indices = struct();
    for k = 1:numel(purposes)
        index = runNumber * 16 + k;
        streams.(purposes(k)) = substream(masterSeed, index);
        indices.(purposes(k)) = index;
    end
    index = 2^40 + engineNumber;
    streams.engine = substream(masterSeed, index);
    indices.engine = index;
    index = 2^41 + engineNumber;
    streams.engine_sensors = substream(masterSeed, index);
    indices.engine_sensors = index;
end

function stream = substream(masterSeed, index)
    stream = RandStream('mrg32k3a', 'Seed', masterSeed);
    stream.Substream = index;
end
