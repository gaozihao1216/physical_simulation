"""Run adaptive failure attribution with reference convergence checks."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation import (
    ConvergenceSelectionConfig,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    build_attribution_dataset,
    export_attribution_csv,
    export_convergence_csv,
    export_diagnostic_json,
    run_contact_benchmark,
    write_attribution_markdown_report,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def main() -> None:
    cases = (
        SpherePlaneBenchmarkCase("drop_h0.4_dt120_solref0.02_0.3", 0.4, 1.0 / 120.0, (0.02, 0.3)),
        SpherePlaneBenchmarkCase("drop_h0.7_dt240_solref0.02_0.5", 0.7, 1.0 / 240.0, (0.02, 0.5)),
        SpherePlaneBenchmarkCase("drop_h1.0_dt240_solref0.02_0.3", 1.0, 1.0 / 240.0, (0.02, 0.3)),
        SpherePlaneBenchmarkCase("drop_h1.3_dt240_solref0.01_0.3", 1.3, 1.0 / 240.0, (0.01, 0.3)),
        SpherePlaneBenchmarkCase("regression_h1_dt240_solref0.005_0.3", 1.0, 1.0 / 240.0, (0.005, 0.3)),
        SphereSphereBenchmarkCase("sphere_sphere_headon_dt240_solref0.01_0.3", 1.0 / 240.0, (0.01, 0.3)),
    )
    recommendation = SubstepRecommendationConfig(maximum_substeps=16)
    benchmark = run_contact_benchmark(cases, recommendation=recommendation)
    diagnostics = build_attribution_dataset(
        cases=cases,
        benchmark=benchmark,
        selection=ConvergenceSelectionConfig(
            top_k_restitution_error=3,
            top_k_penetration_error=3,
            include_all_not_improved=True,
            include_nonphysical_coarse_cases=True,
        ),
        recommendation=recommendation,
    )

    output_dir = Path("artifacts/contact_attribution")
    export_convergence_csv(diagnostics.convergence, output_dir / "reference_convergence.csv")
    export_attribution_csv(diagnostics.attributions, output_dir / "adaptive_attribution.csv")
    export_diagnostic_json(diagnostics, output_dir / "diagnostics.json")
    write_attribution_markdown_report(diagnostics, output_dir / "report.md")

    summary = diagnostics.summary
    mean_saving = _mean([comparison.adaptive_step_saving for comparison in benchmark.comparisons])
    print(f"benchmark case count: {summary.total_cases}")
    print(f"selected convergence case count: {summary.convergence_checked_cases}")
    print(f"reference converged count: {summary.converged_reference_cases}")
    print(f"reference unresolved count: {summary.unresolved_reference_cases}")
    print(f"failure reason counts: {dict(summary.failure_reason_counts)}")
    print(f"improvement outcome counts: {dict(summary.improvement_outcome_counts)}")
    print(f"worst restitution case: {summary.maximum_adaptive_restitution_error_case_id}")
    print(f"worst penetration case: {summary.maximum_adaptive_penetration_error_case_id}")
    print(f"shortest prediction lead case: {summary.lowest_prediction_lead_case_id}")
    print(f"mean adaptive step saving: {mean_saving:.6f}")
    print(
        "exports: "
        f"{output_dir / 'reference_convergence.csv'}, "
        f"{output_dir / 'adaptive_attribution.csv'}, "
        f"{output_dir / 'diagnostics.json'}, "
        f"{output_dir / 'report.md'}"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
