"""Diagnose unresolved primary-impact reference convergence cases."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation import (
    ReferenceDiagnosticStatus,
    export_reference_diagnostics,
    make_smoke_adaptive_batch,
    run_adaptive_primary_batch,
    run_reference_convergence_diagnostics,
)


def main() -> None:
    cases, batch_config = make_smoke_adaptive_batch()
    batch = run_adaptive_primary_batch(cases, config=batch_config)
    dataset = run_reference_convergence_diagnostics(
        cases,
        checked_case_ids=batch.selected_reference_case_ids,
        batch_config=batch_config,
        add_extra_fine_for_unresolved=True,
    )
    paths = export_reference_diagnostics(dataset, Path("artifacts/reference_diagnostics"))
    summary = dataset.summary
    print(f"checked case count: {summary.checked_case_count}")
    print(
        "converged / near-converged / unresolved / invalid: "
        f"{summary.converged_case_count} / {summary.near_converged_case_count} / "
        f"{summary.unresolved_case_count} / {summary.invalid_case_count}"
    )
    print(f"not checked: {summary.not_checked_case_count}")
    print(f"extra-fine cases / newly converged: {summary.extra_fine_case_count} / {summary.extra_fine_converged_case_count}")
    print(f"reason counts: {dict(summary.reason_counts)}")
    print(f"blocking metric counts: {dict(summary.blocking_metric_counts)}")
    print("unresolved cases:")
    for item in dataset.diagnostics:
        if item.status not in {
            ReferenceDiagnosticStatus.NOT_CONVERGED,
            ReferenceDiagnosticStatus.NEAR_CONVERGED,
            ReferenceDiagnosticStatus.INVALID,
        }:
            continue
        print(
            f"  {item.case.case_id}: reason={item.primary_reason.value}, "
            f"blocking={list(item.blocking_metrics)}, "
            f"extra_fine={None if item.extra_fine_status is None else item.extra_fine_status.value}"
        )
    print(
        "sphere-plane / sphere-sphere unresolved: "
        f"{summary.sphere_plane_unresolved_count} / {summary.sphere_sphere_unresolved_count}"
    )
    print(f"high-impact-speed unresolved: {summary.high_speed_unresolved_count}")
    print(f"macro timestep 1/240 unresolved: {summary.dt240_unresolved_count}")
    print(f"exports: {dict(paths)}")


if __name__ == "__main__":
    main()

