FAULT_MODES = {
    "none",
    "duplicate_effect",
    "dependency_change",
    "crash_before_commit",
    "crash_after_commit_before_ack",
    "delayed_ack",
    "compensation_failure",
}

PRIMARY_FAULT_MODES = frozenset(FAULT_MODES)

REPLAY_FAULTS = {
    "duplicate_effect",
    "crash_after_commit_before_ack",
    "delayed_ack",
}
