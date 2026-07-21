from pathlib import Path

import pytest

from physical_simulation.evaluation import (
    BenchmarkMode,
    BenchmarkValidationConfig,
    BenchmarkValidity,
    ContactBenchmarkDataset,
    ContactBenchmarkResult,
    RestitutionOutcome,
    build_benchmark_markdown_report,
    classify_benchmark_validity,
    compare_benchmark_results,
    export_benchmark_csv,
    export_benchmark_json,
    improvement_rates,
    write_benchmark_markdown_report,
)


def _result(
    mode: BenchmarkMode,
    *,
    restitution: float | None,
    penetration: float,
    steps: int,
    validity: BenchmarkValidity = BenchmarkValidity.VALID,
) -> ContactBenchmarkResult:
    return ContactBenchmarkResult(
        case_id="case",
        mode=mode,
        validity=validity,
        timestep=1.0 / 240.0,
        macro_timestep=1.0 / 240.0,
        total_simulation_time=1.0,
        outcome=RestitutionOutcome.REBOUNDED if restitution is not None else RestitutionOutcome.TIMEOUT,
        impact_speed=1.0 if restitution is not None else None,
        rebound_speed=restitution if restitution is not None else None,
        restitution=restitution,
        maximum_penetration=penetration,
        normalized_penetration=penetration / 0.1,
        contact_duration_seconds=1.0 / 240.0,
        final_position=(0.0, 0.0, 0.1),
        final_linear_velocity=(0.0, 0.0, 0.0),
        final_angular_velocity=(0.0, 0.0, 0.0),
        macro_step_count=240,
        physics_step_count=steps,
        wall_time_seconds=0.01,
        adaptive_substepped_macro_steps=10 if mode is BenchmarkMode.ADAPTIVE else None,
        adaptive_max_substep_count=16 if mode is BenchmarkMode.ADAPTIVE else None,
        normal_energy_ratio=None if restitution is None else restitution * restitution,
    )


def test_validity_classification_rules() -> None:
    config = BenchmarkValidationConfig(maximum_restitution=1.05, maximum_normalized_penetration=0.25)

    assert (
        classify_benchmark_validity(
            outcome=RestitutionOutcome.REBOUNDED,
            restitution=1.06,
            normalized_penetration=0.1,
            validation=config,
        )
        is BenchmarkValidity.NONPHYSICAL_REBOUND
    )
    assert (
        classify_benchmark_validity(
            outcome=RestitutionOutcome.REBOUNDED,
            restitution=0.2,
            normalized_penetration=0.26,
            validation=config,
        )
        is BenchmarkValidity.EXCESSIVE_PENETRATION
    )
    assert (
        classify_benchmark_validity(
            outcome=RestitutionOutcome.TIMEOUT,
            restitution=None,
            normalized_penetration=0.1,
            validation=config,
        )
        is BenchmarkValidity.TIMEOUT
    )
    assert (
        classify_benchmark_validity(
            outcome=RestitutionOutcome.REBOUNDED,
            restitution=float("inf"),
            normalized_penetration=0.1,
            validation=config,
        )
        is BenchmarkValidity.UNSTABLE
    )


def test_benchmark_comparison_and_step_saving() -> None:
    comparison = compare_benchmark_results(
        (
            _result(BenchmarkMode.FIXED_COARSE, restitution=2.0, penetration=0.012, steps=240),
            _result(BenchmarkMode.FIXED_FINE, restitution=0.4, penetration=0.004, steps=3840),
            _result(BenchmarkMode.ADAPTIVE, restitution=0.41, penetration=0.0042, steps=480),
        )
    )

    assert comparison.coarse_restitution_error == pytest.approx(1.6)
    assert comparison.adaptive_restitution_error == pytest.approx(0.01)
    assert comparison.adaptive_step_ratio == 0.125
    assert comparison.adaptive_step_saving == 0.875
    assert comparison.adaptive_improves_restitution is True
    assert comparison.adaptive_improves_penetration is True


def test_benchmark_exports(tmp_path: Path) -> None:
    results = (
        _result(BenchmarkMode.FIXED_COARSE, restitution=2.0, penetration=0.012, steps=240),
        _result(BenchmarkMode.FIXED_FINE, restitution=0.4, penetration=0.004, steps=3840),
        _result(BenchmarkMode.ADAPTIVE, restitution=0.41, penetration=0.0042, steps=480),
    )
    comparison = compare_benchmark_results(results)
    dataset = ContactBenchmarkDataset(
        config={"example": True},
        mujoco_version="test",
        cases=({"case_id": "case", "case_type": "sphere_plane"},),
        results=results,
        comparisons=(comparison,),
        units={"timestep": "s"},
    )

    csv_path = tmp_path / "benchmark.csv"
    json_path = tmp_path / "benchmark.json"
    md_path = tmp_path / "report.md"
    export_benchmark_csv(results, csv_path)
    export_benchmark_json(dataset, json_path)
    write_benchmark_markdown_report(dataset, md_path)

    assert "fixed_coarse" in csv_path.read_text(encoding="utf8")
    assert '"mujoco_version": "test"' in json_path.read_text(encoding="utf8")
    markdown = md_path.read_text(encoding="utf8")
    assert "total cases: 1" in markdown
    assert "e > 1 cases" in markdown
    assert build_benchmark_markdown_report(dataset) == markdown
    assert improvement_rates((comparison,)) == {"restitution": 1.0, "penetration": 1.0}
