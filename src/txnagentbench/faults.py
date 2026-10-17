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

PRIMARY_BLOCK_FAULTS = {
    "concurrency": frozenset({"none"}),
    "dependency": frozenset({"dependency_change"}),
    "recovery": frozenset(
        {
            "duplicate_effect",
            "crash_before_commit",
            "crash_after_commit_before_ack",
            "delayed_ack",
            "compensation_failure",
        }
    ),
}

PRIMARY_BLOCK_POLICIES = {
    "concurrency": frozenset(
        {"serial", "naive", "occ", "postgres_serializable", "mtpo_reproduction", "txnguard"}
    ),
    "dependency": frozenset(
        {"serial", "naive", "occ", "postgres_serializable", "mtpo_reproduction", "txnguard"}
    ),
    "recovery": frozenset(
        {
            "serial",
            "naive",
            "occ",
            "postgres_serializable",
            "mtpo_reproduction",
            "serial_recovery",
            "recovery_only",
            "occ_recovery",
            "txnguard_no_recovery",
            "txnguard",
        }
    ),
}

REPLAY_FAULTS = {
    "duplicate_effect",
    "crash_after_commit_before_ack",
    "delayed_ack",
}
