//! Which queued request a lane host admits next: the head, or one behind a head that waits for pool room.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// The queued job to admit (its index), or null to hold; the caller holds the lock and a lane is free.
pub fn pick(h: *LaneHost) ?usize {
    const active = h.core.activeCount();
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    const at = for (h.queued.items, 0..) |job, i| {
        if (ready(job, active, now)) break i;
    } else return null;
    const a = h.info_.admit orelse return at;
    if (active == 0) return at; // alone: the whole pool is free (kept prefixes give their blocks back)
    const head = h.queued.items[at];
    if (fits(a, head)) return at;
    if (head.waits_since == null) head.waits_since = h.admissions;
    // it waits on the streams admitted before it waited: while one runs, later requests that fit go first; then none do
    for (h.admitted.items) |j| if (j.seq < head.waits_since.?) {
        for (h.queued.items[at + 1 ..], at + 1..) |j2, i| if (ready(j2, active, now) and fits(a, j2)) return i;
        return null;
    };
    return null;
}

fn ready(job: *const Job, active: usize, now: i96) bool {
    const n = job.full_wait orelse return true;
    return active == 0 or active < n or now >= job.full_retry;
}

fn fits(a: api.Admit, job: *const Job) bool {
    return a.fits(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}
