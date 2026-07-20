# Architecture

本文档简要说明 `physical_simulation` 的分层设计。核心原则是：业务代码依赖统一 Physics IR，具体物理后端只通过 compiler/backend 层接入。

## Geometry Layer

负责承接上游重建几何，例如 GLB、mesh 或参数化 primitive。

当前阶段只实现 box、sphere、cylinder、capsule 等参数化几何。`GeometrySpec` 中的尺寸是最终物理尺寸，不通过 `Transform.scale` 隐式改变质量、体积或惯量。

## Physics Authoring Layer

负责人工或自动补充物理语义，包括 visual、collider、material、mass properties、rigid body、joint 和 actuator。

当前支持 visual/collider 分离、基础材料、质量属性和单刚体资产。关节、执行器和 mesh collider 仍未实现。

## Physics IR Layer

提供后端无关的中间表示：

```text
GeometrySpec
-> RigidBodySpec
-> PhysicsAssetSpec
-> AssetInstanceSpec
-> PhysicsSceneSpec
```

`PhysicsSceneSpec` 是不可变模型输入。IR 中使用稳定的业务 ID 与 runtime body ID，不保存 MuJoCo numeric ID。

## Backend Layer

当前 MuJoCo 路径为：

```text
PhysicsSceneSpec
-> MuJoCoCompiler
-> MuJoCoCompilationResult
-> MJCF
-> mujoco.MjModel / mujoco.MjData
-> reset / step
-> SimulationStepResult
```

`MjModel` 是加载后的 MuJoCo 模型，`MjData` 是 backend 内部可变状态。MuJoCo numeric body ID 和 geom ID 只保存在 `MuJoCoBackend` 内部，用于状态查询和接触映射，不进入 Physics IR 或业务层公共接口。

## Collision Layering

Dynamic collision 使用 MuJoCo 原生碰撞过滤：

```text
collision_group -> contype
collision_mask  -> conaffinity
```

dynamic-dynamic、dynamic-static、dynamic-fixed 组合不生成 explicit pair，由 MuJoCo 自动碰撞机制处理。

Explicit pair 仅用于需要强制启用的 fixed-fixed collision：

```text
both bodies have no DoF
+ collision group/mask allows collision
+ different runtime bodies
-> <contact><pair ... /></contact>
```

Visual geom 永远不进入 explicit pair。同一 runtime body 内部的多个 collider 永远不互相生成 pair。pair 使用 canonical geom-name key 去重并稳定排序。

显式 pair 使用自身的 contact 参数：`condim=3`、`margin=0`、`gap=0`。friction 使用两个材质 `dynamic_friction` 的几何平均作为 sliding friction，并固定 torsional friction 为 `0.005`、rolling friction 为 `0.0001`。`static_friction` 和 `restitution` 暂未映射。`solref` / `solimp` 没有对应 Physics IR 参数，因此使用 MuJoCo 默认值。

## Runtime Layer

Phase 2C2 已支持 reset、单步 step、刚体世界位姿读取、世界线速度和角速度读取，并把结果封装为后端无关的 `SimulationStepResult` 快照。

Phase 2D1 已支持把 MuJoCo active contacts 映射为 `ContactPoint`，并随 `SimulationStepResult.contacts` 返回：

```text
MjData contacts
-> MuJoCoBackend contact adapter
-> ContactPoint
-> SimulationStepResult
```

`ContactPoint` 只包含 runtime body IDs、世界接触点、从 `body_a` 指向 `body_b` 的单位法向和非负穿透深度。接触力、摩擦力和冲量不在当前阶段读取。

## Robot Task Layer

负责机器人相关任务，例如下落、推动、稳定性、关节运动、抓取和夹爪控制。

当前仍未实现，后续会基于 Runtime Layer 提供的状态和控制接口构建。

## Evaluation Layer

负责从仿真轨迹和任务结果中计算指标、分类失败原因并生成评估报告。

当前仍未实现，后续会依赖可复现的仿真运行结果。
