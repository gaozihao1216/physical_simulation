# Architecture

This document sketches the intended layering for `physical_simulation`. Phase 1 implements a backend-independent, serializable Physics IR for parametric assets. Runtime simulation and backend integrations are intentionally out of scope for this phase.

## Core Boundaries

```text
Visual Geometry != Collision Geometry
Geometry Description != Runtime Simulation
Physics IR must remain independent of a specific backend
GeometrySpec
    local final physical dimensions

Transform.scale
    import/visual convenience only

Physics scale
    must be baked before entering RigidBodySpec or PhysicsSceneSpec

PhysicsAssetSpec
    reusable definition

AssetInstanceSpec
    scene placement of an asset

PhysicsSceneSpec
    immutable simulation input

SimulationStepResult
    runtime output, not part of asset definition
```

The project uses meters, kilograms, seconds, radians, newtons, and `N*m`. Coordinates are right-handed with `+Z` up. Internal rotations use quaternions in `(w, x, y, z)` order.

Visual transform may contain scale. RigidBody, Collider, and Scene instance transforms may not contain non-unit scale.

## Geometry Layer

Consumes reconstructed visual assets such as GLB files and prepares geometry references for physical authoring.

In Phase 1 this layer is represented only by analytic parametric geometry: box, sphere, cylinder, and capsule. GLB import and mesh geometry are not implemented.

## Physics Authoring Layer

Defines physical semantics for assets, including rigid bodies, collision shapes, materials, mass properties, joints, and actuators.

In Phase 1 this includes transform, material, mass properties, visual specs, collider specs, rigid body specs, and parametric builders. Joints and actuators remain future work.

## Physics IR Layer

Provides a backend-independent intermediate representation for scenes, bodies, colliders, dynamics, articulations, robot tasks, and evaluation settings.

The IR is stored as dataclasses with explicit validation and JSON round-trip support. Business code should depend on this IR rather than MuJoCo, Isaac Sim, or any other backend-specific schema.

Phase 1.5 separates four levels:

```text
GeometrySpec -> RigidBodySpec -> PhysicsAssetSpec -> PhysicsSceneSpec
```

`GeometrySpec` stores final local physical dimensions. `RigidBodySpec` describes one rigid body inside an asset. `PhysicsAssetSpec` groups reusable materials and bodies. `PhysicsSceneSpec` places asset instances into a simulation input scene.

## Backend Layer

Adapts the Physics IR to concrete physics engines such as MuJoCo or Isaac Sim through a common backend interface.

Phase 1 keeps only the abstract backend interface. No backend adapter is implemented yet.

## Runtime Layer

Owns simulation stepping, reset behavior, state queries, contact events, sensor updates, and deterministic execution settings.

This layer is not implemented in Phase 1.

Phase 1.5 adds runtime state value objects only: `RigidBodyState`, `JointState`, `ContactPoint`, and `SimulationStepResult`. These are runtime outputs, not asset definitions, and they do not implement stepping.

## Robot Task Layer

Defines robot-centered test procedures such as dropping, pushing, stability checks, joint motion tests, and grasping tasks.

This layer is not implemented in Phase 1.

## Evaluation Layer

Computes metrics, classifies failures, and produces evaluation reports from simulation traces and task outcomes.

This layer is not implemented in Phase 1.
