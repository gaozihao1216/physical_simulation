"""Compare fixed coarse, fixed fine, and fixed-substepped MuJoCo sphere drops."""

from __future__ import annotations

from dataclasses import dataclass, replace

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import measure_restitution
from physical_simulation.mujoco import MuJoCoContactSolverParams, MuJoCoSubstepRunner
from physical_simulation.scene import AssetInstanceSpec, create_scene

RUNTIME_BODY_ID = "sphere_01/sphere"
SPHERE_RADIUS = 0.1


@dataclass(frozen=True)
class RunReport:
    label: str
    macro_timestep: float
    substep_count: int
    substep_timestep: float
    final_time: float
    final_position: tuple[float, float, float]
    final_velocity: tuple[float, float, float]
    measured_restitution: float | None
    maximum_penetration: float
    mj_step_count: int


def _body_with_contact_params(body, params):
    return replace(
        body,
        colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders),
    )


def _build_scene(*, timestep: float):
    params = MuJoCoContactSolverParams(
        solref=(0.02, 0.5),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    )
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_contact_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_contact_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0), params),
    )
    return create_scene(
        scene_id="mujoco_fixed_substepping",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )


def _run_fixed(label: str, *, timestep: float, steps: int) -> RunReport:
    backend = MuJoCoBackend()
    backend.load_scene(_build_scene(timestep=timestep))
    try:
        measurement = measure_restitution(backend, RUNTIME_BODY_ID, max_steps=steps, characteristic_length=SPHERE_RADIUS)
        result = backend.reset()
        for _ in range(steps):
            result = backend.step()
        state = result.get_body_state(RUNTIME_BODY_ID)
        return RunReport(
            label=label,
            macro_timestep=timestep,
            substep_count=1,
            substep_timestep=timestep,
            final_time=result.time,
            final_position=state.position,
            final_velocity=state.linear_velocity,
            measured_restitution=measurement.measured_restitution,
            maximum_penetration=measurement.maximum_penetration_depth,
            mj_step_count=steps,
        )
    finally:
        backend.close()


def _run_substepped(*, macro_timestep: float, substep_count: int, macro_steps: int) -> RunReport:
    backend = MuJoCoBackend()
    backend.load_scene(_build_scene(timestep=macro_timestep))
    try:
        runner = MuJoCoSubstepRunner(backend, macro_timestep=macro_timestep)
        result = runner.reset()
        samples = [result]
        for _ in range(macro_steps):
            advance = runner.step(substep_count=substep_count, substep_callback=samples.append)
            result = advance.simulation_result
        state = result.get_body_state(RUNTIME_BODY_ID)
        measurement = _measure_sampled_results(samples)
        return RunReport(
            label="Substepped",
            macro_timestep=macro_timestep,
            substep_count=substep_count,
            substep_timestep=macro_timestep / substep_count,
            final_time=result.time,
            final_position=state.position,
            final_velocity=state.linear_velocity,
            measured_restitution=measurement["measured_restitution"],
            maximum_penetration=measurement["maximum_penetration"],
            mj_step_count=runner.physics_step_count,
        )
    finally:
        backend.close()


def _measure_sampled_results(results):
    last_downward_speed = 0.0
    impact_speed = 0.0
    contact_seen = False
    contact_ended = False
    maximum_penetration = 0.0
    for result in results:
        state = result.get_body_state(RUNTIME_BODY_ID)
        if not contact_seen and state.linear_velocity[2] < 0.0:
            last_downward_speed = -state.linear_velocity[2]
        contacts = tuple(c for c in result.contacts if c.body_a == RUNTIME_BODY_ID or c.body_b == RUNTIME_BODY_ID)
        if contacts:
            if not contact_seen:
                impact_speed = last_downward_speed
            contact_seen = True
            maximum_penetration = max(maximum_penetration, max(c.penetration_depth for c in contacts))
        elif contact_seen:
            contact_ended = True
        if contact_ended and state.linear_velocity[2] > 1.0e-6:
            return {
                "measured_restitution": state.linear_velocity[2] / impact_speed if impact_speed > 0.0 else None,
                "maximum_penetration": maximum_penetration,
            }
    return {"measured_restitution": None, "maximum_penetration": maximum_penetration}


def main() -> None:
    macro_timestep = 1.0 / 240.0
    substep_count = 16
    macro_steps = 240
    reports = (
        _run_fixed("Fixed coarse", timestep=macro_timestep, steps=macro_steps),
        _run_fixed("Fixed fine", timestep=macro_timestep / substep_count, steps=macro_steps * substep_count),
        _run_substepped(macro_timestep=macro_timestep, substep_count=substep_count, macro_steps=macro_steps),
    )
    print("macro dt = 1/240, substeps = 16, substep dt = 1/3840")
    print(
        "label,macro_timestep,substep_count,substep_timestep,final_time,"
        "final_position,final_velocity,measured_restitution,maximum_penetration,mj_step_count"
    )
    for report in reports:
        print(
            f"{report.label},{report.macro_timestep:.9f},{report.substep_count},"
            f"{report.substep_timestep:.9f},{report.final_time:.6f},"
            f"{_format_vector(report.final_position)},"
            f"{_format_vector(report.final_velocity)},"
            f"{_format_optional(report.measured_restitution)},"
            f"{report.maximum_penetration:.6f},"
            f"{report.mj_step_count}"
        )


def _format_vector(values: tuple[float, float, float]) -> str:
    return "(" + " ".join(f"{value:.6f}" for value in values) + ")"


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
