"""Run a medium adaptive-substepping contact benchmark and export artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from physical_simulation.evaluation import (
    BenchmarkMode,
    BenchmarkValidity,
    build_benchmark_markdown_report,
    export_benchmark_csv,
    export_benchmark_json,
    generate_default_benchmark_cases,
    improvement_rates,
    run_contact_benchmark,
    write_benchmark_markdown_report,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def main() -> None:
    cases = generate_default_benchmark_cases(include_regression_case=True)
    dataset = run_contact_benchmark(
        cases,
        recommendation=SubstepRecommendationConfig(maximum_substeps=16),
    )
    output_dir = Path("artifacts/contact_benchmark")
    export_benchmark_csv(dataset.results, output_dir / "benchmark.csv")
    export_benchmark_json(dataset, output_dir / "benchmark.json")
    write_benchmark_markdown_report(dataset, output_dir / "report.md")

    counts = Counter(result.validity.value for result in dataset.results)
    rates = improvement_rates(dataset.comparisons)
    adaptive_ratios = [comparison.adaptive_step_ratio for comparison in dataset.comparisons]
    adaptive_savings = [comparison.adaptive_step_saving for comparison in dataset.comparisons]
    adaptive_e_errors = [
        comparison.adaptive_restitution_error
        for comparison in dataset.comparisons
        if comparison.adaptive_restitution_error is not None
    ]
    adaptive_penetration_errors = [comparison.adaptive_penetration_error for comparison in dataset.comparisons]
    worst_coarse = max(
        dataset.comparisons,
        key=lambda comparison: -1.0 if comparison.coarse_restitution_error is None else comparison.coarse_restitution_error,
    )
    worst_adaptive = max(
        dataset.comparisons,
        key=lambda comparison: -1.0 if comparison.adaptive_restitution_error is None else comparison.adaptive_restitution_error,
    )
    nonphysical = [
        result.case_id
        for result in dataset.results
        if result.mode is BenchmarkMode.FIXED_COARSE and result.validity is BenchmarkValidity.NONPHYSICAL_REBOUND
    ]
    not_improved = [
        comparison.case_id
        for comparison in dataset.comparisons
        if comparison.adaptive_improves_restitution is False or not comparison.adaptive_improves_penetration
    ]

    print(f"case count: {len(dataset.cases)}")
    print(f"run count: {len(dataset.results)}")
    print(f"failure counts: {dict(sorted(counts.items()))}")
    print(f"adaptive restitution improvement rate: {rates['restitution']:.3f}")
    print(f"adaptive penetration improvement rate: {rates['penetration']:.3f}")
    print(f"adaptive restitution error mean/max: {_mean(adaptive_e_errors):.6f} / {max(adaptive_e_errors):.6f}")
    print(
        "adaptive penetration error mean/max: "
        f"{_mean(adaptive_penetration_errors):.6f} / {max(adaptive_penetration_errors):.6f}"
    )
    print(f"adaptive step ratio mean: {_mean(adaptive_ratios):.6f}")
    print(f"adaptive step saving mean: {_mean(adaptive_savings):.6f}")
    print(f"worst coarse case: {worst_coarse.case_id}")
    print(f"worst adaptive case: {worst_adaptive.case_id}")
    print(f"nonphysical e>1 coarse cases: {', '.join(nonphysical) or 'none'}")
    print(f"adaptive not improved cases: {', '.join(not_improved) or 'none'}")
    print(f"exports: {output_dir / 'benchmark.csv'}, {output_dir / 'benchmark.json'}, {output_dir / 'report.md'}")

    # Keep a compact report preview in terminal for quick inspection.
    preview = "\n".join(build_benchmark_markdown_report(dataset).splitlines()[:12])
    print()
    print(preview)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
