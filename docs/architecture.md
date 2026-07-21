# Architecture

本文档简要说明 `physical_simulation` 的分层设计。核心原则是：业务代码依赖统一 Physics IR，具体物理后端只通过 compiler/backend 层接入。

## Geometry Layer

负责承接上游重建几何，例如 GLB、mesh 或参数化 primitive。

当前 `GeometrySpec` 已支持 box、sphere、cylinder、capsule、wedge/ramp、cone、frustum、ellipsoid、spherical cap 和 regular prism 等参数化几何。`GeometrySpec` 中的尺寸是最终物理尺寸，不通过 `Transform.scale` 隐式改变质量、体积或惯量。

MuJoCo 当前只直接编译 box、sphere、cylinder 和 capsule。wedge/ramp、cone、frustum、ellipsoid、spherical cap 和 regular prism 已作为 Physics IR 语义存在，并支持体积、序列化和 scale baking；后续需要通过 mesh / convex mesh fallback 或后端专用扩展接入真实仿真。

## Physics Authoring Layer

负责人工或自动补充物理语义，包括 visual、collider、material、mass properties、rigid body、joint 和 actuator。

当前支持 visual/collider 分离、基础材料、质量属性和单刚体资产。关节、执行器和 mesh collider 仍未实现。

`dynamics.compound_inertia` 已支持由多个 primitive 组件计算组合刚体的总质量、整体质心、完整 3x3 惯量张量、主惯量和主轴方向。该计算会处理子组件旋转产生的非对角惯量项，并使用平行轴定理把各组件惯量平移到整体质心。当前 MuJoCo 编译路径仍主要消费既有 `MassProperties.inertia_diagonal` 视图；principal-axis orientation 接入 Physics IR 与 MJCF `<inertial>` 是后续工作。

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
MjData.contact
-> MuJoCoBackend contact adapter
-> ContactPoint
-> SimulationStepResult
```

Phase 2D3A 在不修改 `SimulationStepResult` 的前提下，增加了单点 contact wrench 读取：

```text
MjData.contact
-> mapped ContactPoint
-> mj_contactForce
-> contact-frame wrench
-> world-frame ContactWrench
```

`ContactPoint` 只包含 runtime body IDs、世界接触点、从 `body_a` 指向 `body_b` 的单位法向和非负穿透深度；`normal_force` 与 `tangential_force` 当前保持为 `None`。`ContactWrench` 描述 MuJoCo 求解器在该接触点产生的作用力和纯接触力矩。contact torque 不包含 `(contact_position - body_center) x contact_force`，因此不是关于刚体质心的 net torque。

MuJoCo backend 内部用私有 mapped contact 同时保存 raw contact index、geom IDs、runtime body IDs 和公开 `ContactPoint`，`get_contacts()` 与 `get_contact_wrenches()` 复用同一提取路径。这样可以保证两者顺序一致，并避免从排序后的 `ContactPoint` 反向猜测原始 `data.contact`。

Phase 2D3A.5 增加了更有区分度的验证场景，但不新增生产聚合 API：

```text
ContactWrench sequence
-> test-local per-body aggregation
-> net force
-> torque about COM
-> translational / rotational response validation
```

当前关于 COM 的合力和合力矩只在测试与示例中局部计算。正式的 per-body / body-pair 聚合、冲量积分和公开分析 API 将在 Phase 2D3B 设计。

## Evaluation Layer

Phase 2D2 增加了轻量轨迹采样和 resting-contact 指标：

```text
MuJoCoBackend
-> SimulationStepResult sequence
-> trajectory sampling
-> RestingContactMetrics
```

Backend 负责产生物理状态；Evaluation 只解释已经采样的轨迹，不修改 MuJoCo 状态、不启动 GUI、不做 wall-clock sleep，也不访问 MuJoCo 原生对象。Phase 2D3A 之后，Evaluation 可以读取 backend-independent `ContactWrench`，但当前仍不进行冲量积分、跨接触点聚合或关于刚体质心的 net wrench 计算。

`simulate_body_trajectory()` 会对已加载的 backend 调用 `reset()`，记录 reset 后样本，然后推进固定步数并记录目标 body 的 `RigidBodyState` 与当前 contacts。`evaluate_resting_contact()` 使用最后窗口内的速度、位置漂移和四元数角距离判断 `settled`。

## Robot Task Layer

负责机器人相关任务，例如下落、推动、稳定性、关节运动、抓取和夹爪控制。

当前仍未实现，后续会基于 Runtime Layer 提供的状态和控制接口构建。

后续完整任务评估、失败分类、接触冲量分析、接触点聚合和机器人交互评估仍未实现。
