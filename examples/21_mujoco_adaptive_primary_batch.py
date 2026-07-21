"""Run the unified adaptive primary-impact batch pipeline."""

from __future__ import annotations

import argparse

from physical_simulation.evaluation import (
    make_smoke_adaptive_batch,
    make_standard_adaptive_batch,
    run_adaptive_primary_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive primary-impact batch evaluation.")
    parser.add_argument("--mode", choices=("smoke", "standard"), default="smoke")
    args = parser.parse_args()

    if args.mode == "standard":
        cases, config = make_standard_adaptive_batch()
    else:
        cases, config = make_smoke_adaptive_batch()

    result = run_adaptive_primary_batch(cases, config=config)
    summary = result.summary
    print(f"batch mode: {args.mode}")
    print(f"case count: {summary.total_case_count}")
    print(f"completed/invalid: {summary.completed_case_count}/{summary.invalid_case_count}")
    print(f"primary matched: {summary.primary_matched_case_count}")
    print(f"reference checked/converged: {summary.reference_checked_case_count}/{summary.reference_converged_case_count}")
    print(f"primary case outcome counts: {dict(summary.primary_case_outcome_counts)}")
    print(f"failure reason counts: {dict(summary.primary_reason_counts)}")
    print(
        "restitution improvement: "
        f"{summary.primary_restitution_improvement_numerator}/"
        f"{summary.primary_restitution_improvement_denominator}/"
        f"{_fmt_rate(summary.primary_restitution_improvement_rate)}"
    )
    print(
        "penetration improvement: "
        f"{summary.primary_penetration_improvement_numerator}/"
        f"{summary.primary_penetration_improvement_denominator}/"
        f"{_fmt_rate(summary.primary_penetration_improvement_rate)}"
    )
    print(
        "duration improvement: "
        f"{summary.primary_duration_improvement_numerator}/"
        f"{summary.primary_duration_improvement_denominator}/"
        f"{_fmt_rate(summary.primary_duration_improvement_rate)}"
    )
    print(f"mean step saving: {_fmt(summary.mean_adaptive_step_saving)}")
    print(f"median step saving: {_fmt(summary.median_adaptive_step_saving)}")
    print(f"mean/median/maximum step ratio: {_fmt(summary.mean_adaptive_step_ratio)}/{_fmt(summary.median_adaptive_step_ratio)}/{_fmt(summary.maximum_adaptive_step_ratio)}")
    print(f"worst restitution case: {result.worst_cases.maximum_primary_restitution_error_case_id}")
    print(f"worst penetration case: {result.worst_cases.maximum_primary_penetration_error_case_id}")
    print(f"largest step ratio case: {result.worst_cases.maximum_adaptive_step_ratio_case_id}")
    print(f"Pareto nondominated cases: {[point.case_id for point in result.nondominated_accuracy_cost_points]}")
    print(f"exports: {dict(result.artifact_paths)}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()

