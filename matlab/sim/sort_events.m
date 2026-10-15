function events = sort_events(events)
%SORT_EVENTS Order every per-event field by crank angle.

    [~, order] = sort(events.angle_deg);
    names = fieldnames(events);
    for k = 1:numel(names)
        events.(names{k}) = events.(names{k})(order);
    end
end
