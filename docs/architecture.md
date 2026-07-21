# Architecture

本文档简要说明 `physical_simulation` 的分层设计。核心原则是：业务代码依赖统一 Physics IR，具体物理后端只通过 compiler/backend 层接入。

## Geometry Layer

负责承接上游重建几何，例如 GLB、mesh 或参数化 primitive。

当前 `GeometrySpec` 已支持 box、sphere、cylinder、capsule、wedge/ramp、cone、frustum、ellipsoid、spherical cap 和 regular prism 等参数化几何。`GeometrySpec` 中的尺寸是最终物理尺寸，不通过 `Transform.scale` 隐式改变质量、体积或惯量。

MuJoCo 当前直接编译 box、sphere、cylinder 和 capsule。wedge/ramp、cone、frustum 和 regular prism 通过 deterministic convex mesh fallback 编译为 MJCF `<asset><mesh>` 与 `type="mesh"` geom，并已通过真实 MuJoCo 加载测试。ellipsoid 和 spherical cap 仍只作为 Physics IR 语义存在，后续需要曲面采样 mesh fallback 或后端专用扩展。

## Physics Authoring Layer

负责人工或自动补充物理语义，包括 visual、collider、material、mass properties、rigid body、joint、actuator 和可选后端专用参数。

当前支持 visual/collider 分离、基础材料、质量属性、单刚体资产，以及 `ColliderSpec.mujoco_contact_params` 形式的 MuJoCo 专用接触 solver 参数。该参数不放入通用 `PhysicsMaterialSpec`，也不导入 MuJoCo Python 运行库。关节、执行器和 mesh collider 仍未实现。

`dynamics.compound_inertia` 已支持由多个 primitive 组件计算组合刚体的总质量、整体质心、完整 3x3 惯量张量、主惯量和主轴方向。该计算会处理子组件旋转产生的非对角惯量项，并使用平行轴定理把各组件惯量平移到整体质心。`MassProperties` 已能保存 `inertia_tensor` 和 `principal_axes`；MuJoCo 编译路径会在主轴不与 body frame 对齐时输出 `<inertial quat="..." diaginertia="...">`。

`dynamics.polyhedral_inertia` 已支持 wedge/ramp 与 regular prism 的闭合三角网格体积分解，并支持 circular frustum 的连续解析积分。输出包含整体质心、完整惯量张量、主惯量和主轴方向。

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

MuJoCo 没有标准 per-geom `restitution` 字段。`solref` / `solimp` 定义软约束接触行为；`ReferenceRestitutionTarget` 只是标定目标，不直接参与 MJCF 编译，也不会自动从 `PhysicsMaterialSpec.restitution` 推导 solver 参数。

dynamic contact 使用 geom 上的 `solref`、`solimp`、`margin`、`gap`、`priority` 和 `solmix`，由 MuJoCo 自身规则混合。explicit pair 使用 pair 自身参数，因此 compiler 必须解析最终 pair 参数：`condim=3`，friction 使用项目 explicit-pair policy，也就是两个材质 `dynamic_friction` 的几何平均；如果 collider 配置了 MuJoCo solver 参数，pair 会解析最终 `solref/solimp/margin/gap`，否则不显式写 `solref/solimp` 并使用 MuJoCo 默认。这个 explicit-pair friction policy 与 MuJoCo 默认 friction 混合规则是两回事。峰值接触力、最大穿透、测得恢复系数和 timeout/settled 判断都依赖 timestep 与 solver 参数。

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

Phase 2D3A.5 增加了更有区分度的验证场景，并在后续补充了生产级 contact wrench 聚合 API：

```text
ContactWrench sequence
-> per-body / body-pair aggregation
-> net force
-> torque about chosen center
-> discrete impulse integration
-> translational / rotational response validation
```

`BodyContactWrench` 聚合某个 runtime body 受到的总接触力和关于指定中心的总力矩。`BodyPairContactWrench` 聚合一对 runtime bodies 之间交换的接触作用。`BodyContactImpulse` 使用固定 timestep 对 body 聚合 wrench 做离散积分。该冲量是 rectangle-rule 近似，不是 MuJoCo 内部逐 contact impulse 直接读数。

Runtime control 已支持自由动态刚体的基础扰动接口：

```text
set_body_velocity
apply_force
apply_torque
clear_applied_forces
```

这些接口只作用于有 freejoint 的 dynamic body。`apply_force(point=...)` 接收世界系施力点，并转换为关于当前 body COM 的等效力矩。它们用于推动、撞击、扰动稳定性和后续机器人任务验证；`step(action=...)` 仍未作为通用控制入口。

Phase 2G1 增加了固定子步进 runner：

```text
MuJoCoBackend loaded with macro timestep
-> MuJoCoSubstepRunner.step(substep_count=N)
-> temporarily set model.opt.timestep = macro_timestep / N
-> run N internal mj_step calls
-> restore original model.opt.timestep
-> return macro-end SimulationStepResult
```

`MuJoCoBackend.step()` 仍然保持一次调用等于一次 `mj_step`。子步进作为独立 runner 存在，runner 自己维护 `macro_step_index` 和累计 `physics_step_count`；`SimulationStepResult.step_index` 仍表示实际 MuJoCo physics step 计数。缩小 timestep 只提高 MuJoCo soft-contact 模型的数值分辨率，不自动修改 `solref/solimp`，也不会把软接触变成硬碰撞。

Phase 2G3 把显式碰撞候选、解析预测和 solver recommendation 接入 adaptive runner：

```text
SimulationStepResult
+ explicit adaptive candidates
-> analytic prediction
-> solver timescale estimate
-> substep recommendation
-> contact motion state machine
-> MuJoCoSubstepRunner
```

`AdaptiveMuJoCoRunner` 当前支持调用者显式注册的 sphere-plane 与 sphere-sphere candidate。状态机使用 `FREE`、`APPROACHING`、`IMPACTING`、`RESTING` 和 `SEPARATING` 描述粗粒度接触阶段：普通自由运动使用 `substep_count=1`；预测到即将碰撞时进入 approaching 并使用 solver 推荐子步；active contact 阶段保持细子步；脱离接触后短暂保持 cached recommendation；持续接触且线速度、法向速度和角速度落入静止窗口后进入 resting，并恢复 macro timestep。

多候选同时命中时，runner 选择 `actual_substep_timestep` 最小的候选；若相同则用 candidate id 稳定排序。adaptive runner 不从所有 geom 自动枚举候选，不做 Hertz 接触时间估计，不做 rollback 或事件精确落点，也不会自动修改 `solref/solimp`。

Phase 2G4 在 Evaluation Layer 增加 benchmark 与失真诊断，不改变 Runtime Layer 的物理推进语义：

```text
Contact benchmark case
-> FIXED_COARSE / FIXED_FINE / ADAPTIVE
-> ContactBenchmarkResult
-> BenchmarkValidity
-> BenchmarkComparison against fixed fine
-> CSV / JSON / Markdown report
```

`FIXED_COARSE` 使用 macro timestep，`FIXED_FINE` 使用配置的最细 fixed grid，`ADAPTIVE` 只在预测或接触窗口使用更细 substeps。benchmark 关注 fixed coarse 是否发生非物理 `e > 1`、最大穿透是否过大、adaptive 是否接近 fixed fine、以及 adaptive 的 `physics_step_count` 是否显著低于 fixed fine。`wall_time_seconds` 只作为环境相关的观测值，正式回归以 MuJoCo `physics_step_count` 为主。

Phase 2G5 增加 reference convergence 和 adaptive failure attribution：

```text
baseline fixed-fine h
-> finer h/2
-> ultra-fine h/4
-> ReferenceConvergenceResult

AdaptiveStepDecision + substep SimulationStepResult
-> AdaptiveDiagnosticTrace
-> AdaptiveFailureAttribution
```

fixed-fine 只是 baseline reference，不自动等价于真实解。convergence 检查只改变 timestep，不改变 `solref`、`solimp`、积分器、solver iterations 或场景初始状态。每个 refinement level 都通过完整 reset 独立运行。指标收敛使用 `D1=|Q_h-Q_h/2|`、`D2=|Q_h/2-Q_h/4|` 和绝对/相对容差判断，不假设接触问题有固定阶数，也不做 Richardson extrapolation。

adaptive 未改善并不直接等于 adaptive 失败。归因层会区分预测提前量不足、达到 substep 上限、时间分辨率不足、过早退出 fine mode、多次 contact episode、reference 未收敛和指标采样敏感等情况。Phase 2G5 只生成诊断和报告，不自动修改 adaptive runner 配置，不做优化器或材料参数反演。

## Evaluation Layer

Phase 2D2 增加了轻量轨迹采样和 resting-contact 指标：

```text
MuJoCoBackend
-> SimulationStepResult sequence
-> trajectory sampling
-> RestingContactMetrics
```

Backend 负责产生物理状态；Evaluation 只解释已经采样的轨迹，不修改 MuJoCo 状态、不启动 GUI、不做 wall-clock sleep，也不访问 MuJoCo 原生对象。Phase 2D3A 之后，Evaluation 可以读取 backend-independent `ContactWrench`、`BodyContactWrench`、`BodyPairContactWrench` 和离散 `BodyContactImpulse`。Phase 2F1/2F1.5 增加了 `measure_restitution()`，通过标准 sphere-drop 测量接触前最后下降速度、脱离接触后首次明确上升速度、最大穿透、归一化穿透和接触持续时间，并计算 `rebound_speed / impact_speed` 作为观测到的恢复系数。

恢复系数测量使用显式 outcome：

```text
free fall
-> first effective contact
-> detached upward motion: REBOUNDED
-> continuous slow contact window: SETTLED_IN_CONTACT
-> max_steps without either condition: TIMEOUT
```

`SETTLED_IN_CONTACT` 表示持续接触并趋于静止，`measured_restitution=0`，但 contact duration 为 `None`，因为静止支撑不是一次超长碰撞。`TIMEOUT` 表示测量未收敛，`measured_restitution=None`，不能解释为完全非弹性碰撞。接触持续步数和物理持续时间不同：`contact_duration_seconds = contact_duration_steps * timestep`。

`simulate_body_trajectory()` 会对已加载的 backend 调用 `reset()`，记录 reset 后样本，然后推进固定步数并记录目标 body 的 `RigidBodyState` 与当前 contacts。`evaluate_resting_contact()` 使用最后窗口内的速度、位置漂移和四元数角距离判断 `settled`。

Phase 2G2 增加了 solver contact timescale 和解析碰撞预测：

```text
MuJoCoContactSolverParams
-> estimate_solver_contact_timescale()
-> SolverContactTimescale
-> recommend_solver_substeps()
-> SubstepRecommendation

Sphere / plane or sphere / sphere state
-> CollisionPrediction
-> SolverCollisionEstimate
```

该估计描述的是 MuJoCo soft-constraint 的数值时间尺度，而不是 Hertz、杨氏模量或材料弹性模型。第一版使用 `assumed_impedance = max(solimp[0], solimp[1])` 作为最快约束动力学的保守估计，并只支持恒速度 sphere-plane 与 sphere-sphere 解析预测。Phase 2G2 本身不自动执行 substeps，不调用 `MuJoCoSubstepRunner`，不修改 timestep，也不修改 `solref/solimp`。

Phase 2G3 在 Runtime Layer 中新增 `AdaptiveMuJoCoRunner` 后，Evaluation 可以对比 coarse、fixed fine 和 adaptive 三种推进方式。Phase 2G4 将这种对比固化为可导出的 benchmark 数据集：每个 case 运行三种模式，保存 validity、恢复系数误差、穿透误差、rebound velocity 误差、step ratio、saving 和 adaptive 状态统计。Phase 2G5 进一步检查 fixed-fine reference 是否 timestep 收敛，并对 adaptive 未改善 case 生成结构化 failure attribution。adaptive 的目标是接近收敛参考的接触精度，同时避免在普通运动或稳定支撑阶段持续使用小 timestep。当前仍属于显式候选驱动方案，不支持任意 geometry 预测、Hertz contact-time、rollback、自动候选生成、自动调参或 robot/task policy。

fixed coarse 下出现 `e > 1` 被视为数值失真诊断，不视为材料具有额外能量。近似法向能量比 `eta_E = e^2` 只用于 sphere-plane 法向碰撞诊断，不声称代表任意三维碰撞的完整能量守恒分析。

## Robot Task Layer

负责机器人相关任务，例如下落、推动、稳定性、关节运动、抓取和夹爪控制。

当前仍未实现，后续会基于 Runtime Layer 提供的状态和控制接口构建。

后续完整任务评估、失败分类、MuJoCo 内部 impulse 读取和机器人交互评估仍未实现。
