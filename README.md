# physical_simulation

`physical_simulation` 是一个独立 Python 子项目，用于把重建后的 3D 资产逐步转化为可验证、可复现、可接入物理后端的仿真资产与任务评估流程。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：在视觉几何重建完成之后，补充物理语义，构建后端无关的 Physics IR，并逐步接入碰撞体生成、刚体动力学、关节系统、机器人任务和动态评估。

当前已经支持参数化 Physics IR、场景表示、MJCF 编译、MuJoCo 模型加载、reset、单步 step、刚体世界状态读取、MuJoCo active contact 到 `ContactPoint` 的映射、单点 `ContactWrench` 读取、按 body/body-pair 的 contact wrench 聚合、离散 contact impulse 积分、MuJoCo 接触 solver 参数配置、基础 drop/resting-contact/restitution 标定评估、显式候选驱动的自适应 MuJoCo 子步进 runner、coarse/fine/adaptive 接触 benchmark 与失真诊断，以及 fixed-fine reference convergence 和 adaptive failure attribution。关节、机器人和完整任务框架仍未实现。

## 与 3D Reconstruction 模块的边界

上游 3D reconstruction 模块负责几何重建、视觉网格、纹理生成和 GLB 等视觉资产导出。本项目从重建结果之后开始工作：

- 输入可以是 `Reconstructed GLB` 或后续人工/自动生成的几何资产。
- 不负责神经重建、网格生成、纹理生成或视觉资产训练。
- 负责物理语义，包括刚体、碰撞体、材料、质量属性、场景实例、后端编译、运行时状态和评估接口。

## 整体数据流

```text
Reconstructed GLB
-> Physical Asset
-> Collision and Dynamics Authoring
-> Physics Backend
-> Robot Task
-> Evaluation Report
```

## 当前目录结构

```text
physical_simulation/
|-- README.md
|-- pyproject.toml
|-- configs/
|-- docs/
|   `-- architecture.md
|-- examples/
|-- scripts/
|-- src/
|   `-- physical_simulation/
|       |-- assets/
|       |-- articulation/
|       |-- backends/
|       |-- collision/
|       |-- compilers/
|       |-- dynamics/
|       |-- evaluation/
|       |-- robots/
|       |-- runtime/
|       |-- scene/
|       |-- serialization/
|       |-- tasks/
|       |-- utils/
|       |-- validation/
|       `-- visualization/
`-- tests/
    |-- integration/
    `-- unit/
```

## 各模块职责

- `assets`：物理资产、刚体、碰撞体、材料、基础几何、质量属性和物理资产构造器。
- `collision`：碰撞几何生成、primitive fitting、凸包和凸分解；当前仍是预留模块。
- `dynamics`：质量、重心、惯量、力、力矩和刚体动力学配置。
- `articulation`：关节、运动轴、自由度、限制和执行器；当前仍是预留模块。
- `compilers`：把后端无关的 `PhysicsSceneSpec` 编译为具体后端输入；当前支持 MJCF 生成。
- `backends`：MuJoCo、Isaac Sim 等物理后端适配器；当前支持 MuJoCo 模型加载、reset、单步推进、刚体状态读取、contact 映射和单点 contact wrench 读取。
- `runtime`：运行时状态对象，包括 `RigidBodyState`、`JointState`、`ContactPoint`、`ContactWrench` 和 `SimulationStepResult`。
- `robots`：机器人模型加载、控制器和夹爪控制；当前仍是预留模块。
- `tasks`：下落、推动、稳定性、关节运动和抓取等测试任务；当前仍是预留模块。
- `evaluation`：仿真指标、失败分类和评估报告；当前仍是预留模块。
- `visualization`：场景显示、轨迹回放和调试可视化；当前仍是预留模块。

## 开发路线

- Phase 1：Physics IR / Parametric Physics Asset Representation。
- Phase 1.5：Physics IR Semantic Hardening and Scene Representation。
- Phase 2A：Transform Composition。
- Phase 2B：PhysicsSceneSpec -> MJCF Compiler。
- Phase 2C1：MuJoCo Model Loading and ID Mapping。
- Phase 2C2：Reset, Step and Rigid-Body State。
- Phase 2D1：MuJoCo Contact Mapping。
- Phase 2D1.5：Explicit Contact Pair Semantics Audit。
- Phase 2D2：Drop and Resting Contact Validation。
- Phase 2D3A：MuJoCo Contact Wrench Extraction。
- Phase 2D3A.5：Multi-Directional Contact and Off-Center Impact Validation。
- Phase 2D3B：Contact Force Aggregation and Impulse。
- Phase 2F1：MuJoCo Contact Solver Parameters and Restitution Calibration。
- Phase 2F1.5：Restitution Measurement Robustness。
- Phase 2G1：Fixed Substepping Infrastructure。
- Phase 2G2：Solver Contact Timescale and Analytic Collision Prediction。
- Phase 2G3：Adaptive MuJoCo Runner and Contact State Machine。
- Phase 2G4：Adaptive Substepping Benchmark and Failure Diagnostics。
- Phase 2G5：Adaptive Failure Attribution and Reference Convergence。
- Phase 2G6：Episode-Level Contact Metrics and Event Matching。
- Phase 2G7：Primary-Impact Failure Attribution Integration。
- Phase 2G8：Unified Batch Primary-Impact Evaluation Pipeline。
- Phase 2G9：Reference Convergence Diagnostics。
- Phase 2H1：Automatic Adaptive Prediction Candidate Construction。
- Phase 3：MuJoCo Backend。
- Phase 4：Rigid Body Simulation。
- Phase 5：Articulation。
- Phase 6：Robot Tasks。
- Phase 7：Dynamic Evaluation。

## Phase 2D1 / 2D1.5 当前能力

已支持：

- MuJoCo active contact extraction：遍历 `data.contact[0:data.ncon]`。
- geom ID -> runtime body ID：公开结果只包含 `"{instance_id}/{body_id}"`，不泄漏 MuJoCo numeric ID 或 sanitized name。
- world-space contact position：`ContactPoint.position` 使用 MuJoCo 世界接触点。
- deterministic body ordering：`body_a/body_b` 按 runtime body ID 字典序稳定排序。
- body_a -> body_b contact normal：MuJoCo `contact.frame[:3]` 为 `geom1 -> geom2`，如果排序交换双方则同步翻转 normal。
- non-negative penetration depth：`penetration_depth = max(0.0, -contact.dist)`。
- multiple contact points：同一 body pair 的多个接触点全部保留，不合并、不平均。
- dynamic collision：dynamic-dynamic、dynamic-static、dynamic-fixed 使用 MuJoCo `contype/conaffinity` 自动碰撞过滤，不生成 explicit pair。
- explicit pair：仅用于双方都没有自由度、且 Physics IR collision group/mask 允许碰撞的 fixed-fixed collision。
- explicit pair 参数：`condim=3`、`margin=0`、`gap=0`。
- explicit pair friction：sliding friction 使用两个 `dynamic_friction` 的几何平均；torsional friction 为 `0.005`；rolling friction 为 `0.0001`。

Phase 2D1 / 2D1.5 尚未读取求解器接触力、冲量或任务成功指标；单点接触力读取从 Phase 2D3A 开始提供。

`static_friction` 和 `restitution` 当前暂未映射到 MJCF explicit pair。未配置 MuJoCo solver 参数时，`solref` 和 `solimp` 不显式设置，使用 MuJoCo 默认值。

## Phase 2D3A 当前能力

已支持：

- `MuJoCoBackend.get_contact_wrenches()`：为当前每个公开 `ContactPoint` 返回一个对应的 `ContactWrench`。
- `mj_contactForce`：读取 MuJoCo 求解器针对单个 contact index 输出的接触力和接触力矩。
- contact-frame -> world-frame：MuJoCo `contact.frame` 的三条轴按行存储，局部力和力矩通过 `frame.T @ vector` 转换到世界坐标。
- force on body_a/body_b：先按 MuJoCo geom1/geom2 原始方向解释受力，再映射到公开的 `body_a/body_b`，不通过 public normal 猜测原始方向。
- pure contact torque：`ContactWrench.contact_torque_on_body_*_world` 表示接触点处的纯接触力矩，不包含 `(contact_position - body_center) x contact_force`，因此不是关于刚体质心的总力矩。
- normal/tangential force magnitude：在公开 `ContactPoint.normal` 约定下分解施加到 `body_b` 的世界接触力，两个 magnitude 均保证非负。
- inactive contact：`efc_address < 0` 时仍返回一一对应的 `ContactWrench`，但力、力矩和 magnitude 为零。
- static support-force validation：1 kg box 和 sphere 的静置总支撑力接近 `9.81 N`。
- impact force extraction：下落碰撞瞬间可以读取正的法向接触力。

当前仍未支持：

- MuJoCo 内部逐 contact impulse 直接读数。
- 定量摩擦验证。
- automatic restitution mapping。
- joints、robots、meshes、GUI 和完整 task framework。

`ContactPoint` 仍只描述接触几何，`normal_force` 与 `tangential_force` 在当前阶段保持为 `None`。`ContactWrench` 描述求解器在该接触点产生的作用力，不修改 `SimulationStepResult` 字段。

## Phase 2D3A.5 当前能力

已验证：

- 一个动态刚体可以同时接触两个不同的 runtime body。
- V 形槽中，左右两个非平行接触力共同支撑 sphere。
- 左右接触力都具有水平分量和竖直分量，水平合力接近零，竖直合力接近 `mg`。
- V 形槽稳定状态下，基于测试局部聚合计算出的 sphere COM torque 接近零。
- 倾斜长方体偏心落地时，首次有效接触点相对 COM 有明显水平偏移。
- 当前 `condim=3` 下 pure contact torque 接近零，但 `r x F` 产生非零 COM torque。
- 偏心碰撞后 angular velocity 明显非零。
- `+20 deg` 与 `-20 deg` 镜像场景产生相反符号的 `torque_y` 和 `angular_velocity_y`。
- 多方向接触和偏心碰撞验证在 reset 后保持确定性。

此前 box-ground 支撑测试虽然有多个 contact point，但宏观运动和接触力方向高度对称，接近一维支撑。Phase 2D3A.5 专门验证多方向接触、多个外部刚体共同作用，以及偏心接触导致的旋转响应。

当前已新增公共聚合 API：`BodyContactWrench`、`BodyPairContactWrench` 和 `BodyContactImpulse`。它们支持按 body 或 body pair 聚合合力、关于指定中心的合力矩，并用固定 timestep 对 body 聚合 wrench 做离散冲量积分。

## Phase 2F1 / 2F1.5 当前能力

已支持 MuJoCo 专用接触 solver 参数，但它们不属于通用 `PhysicsMaterialSpec`：

- `MuJoCoContactSolverParams`：包含 `solref`、`solimp`、`margin`、`gap`、`priority` 和 `solmix`。
- `ColliderSpec.mujoco_contact_params`：允许单个 collision geom 可选携带 MuJoCo 参数；visual geom 不携带该参数。
- dynamic contact：普通 dynamic-dynamic、dynamic-static、dynamic-fixed contact 仍使用 MuJoCo geom 参数和 MuJoCo 自身 priority/solmix 混合规则。
- explicit fixed-fixed pair：由于 `<contact><pair>` 会覆盖 geom 参数，compiler 会为 pair 明确解析最终 `solref/solimp/margin/gap`；friction 仍使用本项目 explicit-pair policy，即两个 `dynamic_friction` 的几何平均。
- `ReferenceRestitutionTarget`：后端无关的标定目标，只描述目标恢复系数和参考撞击速度，不参与 MJCF 编译。
- `measure_restitution()`：标准 sphere-drop 测量入射速度、反弹速度、测得恢复系数、接触起止步、最大穿透深度、归一化穿透和 outcome。
- `RestitutionOutcome.REBOUNDED`：脱离接触后出现明确上升速度，`measured_restitution = rebound_speed / impact_speed`，contact duration 有有限值。
- `RestitutionOutcome.SETTLED_IN_CONTACT`：持续接触且速度窗口足够小，`measured_restitution = 0`，contact duration 为 `None`。
- `RestitutionOutcome.TIMEOUT`：达到 `max_steps` 但既未反弹也未稳定，`measured_restitution = None`，不能解释为完全非弹性碰撞。
- `measure_restitution_sweep()`：通过不同初始高度产生不同 impact speed，并按测得的 impact speed 排序返回。

MuJoCo 没有标准 per-geom `restitution` 字段。`solref` / `solimp` 定义软约束接触行为；当前项目不会把 `PhysicsMaterialSpec.restitution` 自动转换成 `solref/solimp`。峰值接触力、最大穿透和测得恢复系数都依赖 timestep 与 solver 参数。

接触持续步数不等于物理持续时间，需要乘以 timestep 才是 seconds。持续静止接触不是一次超长碰撞；恢复系数必须结合 outcome 使用。

## Phase 2G1 当前能力

已支持 MuJoCo 固定子步进基础设施：

- `MuJoCoSubstepRunner`：包裹一个已加载的 `MuJoCoBackend`，一次外部 macro step 内执行固定数量的内部 `mj_step`。
- `SubstepAdvanceResult`：记录 `macro_step_index`、累计 `physics_step_count`、`macro_timestep`、`substep_timestep`、`substep_count` 和宏步末端 `SimulationStepResult`。
- 时间语义：`substep_timestep = macro_timestep / substep_count`，一次 runner step 推进的总物理时间仍等于 `macro_timestep`。
- 计数语义：runner 维护外部 `macro_step_index` 和自身累计 `physics_step_count`；`SimulationStepResult.step_index` 继续表示 backend 实际 MuJoCo physics step 计数。
- `substep_callback`：默认关闭；需要精细标定时可在每个内部 substep 后观察 `SimulationStepResult`。

本阶段只是后续自适应碰撞 timestep 的基础设施，`substep_count` 仍需显式指定。缩小 timestep 只提高当前 MuJoCo soft-contact 模型的数值分辨率；不会自动改变 `solref/solimp`，也不会自动把软接触变成硬碰撞。

## Phase 2G2 当前能力

已支持 MuJoCo solver 时间尺度估计和简单解析碰撞预测：

- `estimate_solver_contact_timescale()`：从 `MuJoCoContactSolverParams.solref/solimp` 估计 soft-constraint 数值时间尺度。
- `DampingRegime`：区分 `UNDERDAMPED`、`CRITICAL` 和 `OVERDAMPED`。
- `recommend_solver_substeps()`：根据 characteristic timescale、macro timestep 和配置推荐固定 `substep_count`。
- `AnalyticPlane` 与 `predict_sphere_plane_collision()`：恒速度 sphere-plane 首次接触预测。
- `predict_sphere_sphere_collision()`：恒速度 sphere-sphere 首次接触预测。
- `SolverCollisionEstimate`：组合 collision prediction、solver timescale 和 substep recommendation。

Phase 2G2 基于 MuJoCo `solref/solimp` 的 soft-constraint 时间尺度，不是 Hertz 接触模型。它只给出推荐，不会自动调用 `MuJoCoSubstepRunner`，不会修改 timestep，不会修改 `solref/solimp`。

## Phase 2G3 当前能力

已支持显式候选驱动的自适应 MuJoCo 子步进 runner：

- `AdaptiveMuJoCoRunner`：包裹已加载的 `MuJoCoBackend`，并复用 `MuJoCoSubstepRunner` 执行内部子步。
- `SpherePlaneAdaptiveCandidate` 与 `SphereSphereAdaptiveCandidate`：调用者显式声明需要预测的简单接触候选，不从任意 MJCF geom 自动枚举。
- `ContactMotionState`：包含 `FREE`、`APPROACHING`、`IMPACTING`、`RESTING` 和 `SEPARATING`。
- approaching/impacting/separating：根据解析预测、active contact 和缓存的 solver recommendation 使用更细 substeps。
- resting/free：稳定接触或普通运动恢复为 `substep_count=1`，也就是使用 macro timestep。
- 多候选选择：对所有命中的候选计算推荐，选择 `actual_substep_timestep` 最小者；相同 timestep 用 candidate id 稳定打破平局。
- `AdaptiveStepDecision`：记录状态转移、候选 ID、prediction、solver estimate、substep 数、实际子步 timestep、是否观察到 contact 和决策原因。

Phase 2G3 不实现 Hertz 接触时间估计，不支持任意 geometry 自动预测，不做 rollback 或事件精确落点，不修改 `solref/solimp`，也不根据接触状态自动改变材料或 solver 参数。当前 sphere-plane candidate 的 active-contact 匹配以 sphere runtime body 为核心，适合受控标定场景；复杂多接触场景后续需要更精确的 geom/body pair 绑定。

## Phase 2G4 当前能力

已支持自适应子步进 benchmark 和失真诊断：

- `BenchmarkMode`：统一运行 `FIXED_COARSE`、`FIXED_FINE` 和 `ADAPTIVE` 三种模式。
- `BenchmarkValidity`：把结果分类为 `VALID`、`NONPHYSICAL_REBOUND`、`EXCESSIVE_PENETRATION`、`TIMEOUT` 或 `UNSTABLE`。
- `ContactBenchmarkResult`：记录 timestep、总仿真时间、outcome、入射/反弹速度、恢复系数、最大穿透、最终状态、macro/physics step 数、wall time 和 adaptive 统计。
- `BenchmarkComparison`：相对 fixed fine 计算 coarse/adaptive 的 restitution、penetration、rebound velocity 误差，以及 adaptive step ratio 和 saving。
- `AdaptiveRunStatistics`：记录状态持续 macro steps、substep count 分布、最大 substep count、首次 approaching/contact 时间、prediction lead time 和 substepped macro-step 占比。
- `export_benchmark_csv()`、`export_benchmark_json()`、`write_benchmark_markdown_report()`：导出表格、完整 JSON 数据集和 Markdown 报告。
- `examples/17_mujoco_adaptive_benchmark.py`：运行默认 benchmark，并把结果导出到 `artifacts/contact_benchmark/`。

adaptive substepping 提高的是 MuJoCo soft-contact 的离散数值分辨率，不改变材料语义，也不自动改变 `solref/solimp`。fixed coarse 可能因为 timestep 过粗产生 `e > 1` 的非物理能量增益；这类结果会标记为 `NONPHYSICAL_REBOUND`，不会解释为高弹材料。benchmark 必须同时比较恢复系数误差、穿透误差、稳定性和 MuJoCo physics step 数；`wall_time_seconds` 只记录，不作为严格回归指标。

## Phase 2G5 当前能力

已支持自适应失败归因和 reference convergence：

- `ReferenceConvergenceResult`：对 fixed-fine、finer、ultra-fine 三个 timestep refinement level 计算收敛状态。
- `ReferenceMetricConvergence`：记录 `D1=|Q_h-Q_h/2|`、`D2=|Q_h/2-Q_h/4|`、difference ratio、绝对/相对容差和状态。
- `AdaptiveDiagnosticTrace` 与 `AdaptiveContactEpisodeTrace`：记录 prediction lead、实际 contact 时间、contact episode、substep 分布、最大穿透发生时间、是否达到 substep 上限等结构化诊断。
- `AdaptiveFailureAttribution`：用确定性优先级给出 `LATE_PREDICTION`、`SHORT_PREDICTION_LEAD`、`MAX_SUBSTEPS_LIMITED`、`EARLY_FINE_EXIT`、`MULTIPLE_CONTACT_EPISODES`、`REFERENCE_NOT_CONVERGED` 等原因。
- `ImprovementOutcome`：区分 `IMPROVED`、`NOT_IMPROVED`、`BOTH_ACCEPTABLE`、`REFERENCE_UNRESOLVED` 和 `NOT_APPLICABLE`。
- `examples/18_mujoco_adaptive_failure_attribution.py`：运行中等规模 benchmark，选择未改善和误差较大的 case，执行 convergence 检查并导出 attribution 报告。

fixed-fine 是数值参考，不自动等于真实解。只有 reference convergence 为 `CONVERGED` 时，最细 refinement level 才可作为 converged reference；未收敛时仍可报告误差，但必须标记 `REFERENCE_NOT_CONVERGED`，不能把 adaptive 偏离 fixed-fine 自动解释为 adaptive 策略失败。

Phase 2G5 只做诊断和报告，不自动修改 adaptive 配置，不做自动调参，不改变 `solref/solimp`，不修改 `backend.step()`，也不引入 Hertz、rollback 或新碰撞几何预测。

## Phase 2G6 当前能力

已支持 episode-level contact metrics 和 event matching：

- `ContactEpisodeSample`：记录每个 physics step 的候选接触状态、penetration、法向相对速度、body 速度、substep 数和 adaptive state。
- `RawContactInterval`：从 `active_contact=False -> True -> False` 提取原始连续接触区间；仿真结束时仍 active 的区间会标记 `ended_while_contact_active`。
- `segment_contact_episodes()`：用 gap duration、gap steps 和 separation velocity 合并短暂 contact chatter，并输出独立 `ContactEpisodeMetrics`。
- `ContactEpisodeKind`：区分 `PRIMARY_IMPACT`、`SECONDARY_IMPACT`、`RESTING_CONTACT`、`CONTACT_CHATTER` 和 `UNCLASSIFIED`。
- `match_contact_episodes()`：按 candidate、kind、start time 和 impact speed 确定性匹配 coarse/fine/adaptive 的 episode。
- `PrimaryImpactBenchmarkComparison`：优先比较 Episode 0 / `PRIMARY_IMPACT`，避免把后续反弹混入首次冲击指标。
- `EpisodeReferenceConvergenceResult`：对 primary impact 执行 fine/finer/ultra-fine episode-level convergence，允许 primary impact 已收敛但 run-level 后续反弹仍 unresolved。
- `examples/19_mujoco_contact_episode_analysis.py`：导出 `artifacts/contact_episodes/episodes.csv`、`matches.csv`、`comparisons.csv`、`diagnostics.json` 和 `report.md`。

法向相对速度约定固定为：`normal_relative_velocity < 0` 表示沿接触法向接近，`> 0` 表示分离。sphere-plane 的法向为 `plane -> sphere`；sphere-sphere 的法向为 `body_a -> body_b`。episode restitution 使用该 episode 接触前/接触开始处的法向接近速度和接触结束后的首次法向分离速度，不跨越多个 episode。

一次 simulation run 可以包含多个 contact episodes；运行级 restitution、maximum penetration 和 contact duration 可能混合多次碰撞。Phase 2G6 将 episode segmentation 作为数值诊断层，不修改 MuJoCo contact solver、不修改 adaptive timestep 决策，也不会自动调整 `solref/solimp`。

## Phase 2G7 当前能力

已支持 primary-impact 优先的 adaptive 归因集成：

- `attribute_primary_impact_failure()`：优先使用匹配后的 `PRIMARY_IMPACT` episode、episode-level comparison 和 primary reference convergence 做归因。
- `AttributionScope`：区分 `PRIMARY_IMPACT`、`RUN_LEVEL`、`FALLBACK_RUN_LEVEL` 和 `UNAVAILABLE`，避免把运行级指标误读成首次冲击指标。
- `PrimaryImpactCaseOutcome`：区分 `IMPROVED`、`PARTIALLY_IMPROVED`、`NOT_IMPROVED`、`BOTH_ACCEPTABLE`、`REFERENCE_UNRESOLVED`、`EPISODE_UNMATCHED` 和 `INVALID_ADAPTIVE`。
- primary metric outcomes：分别比较 restitution、maximum penetration 和 contact duration，只有 primary reference 收敛且 episode 匹配后才判定 adaptive 是否改善。
- run-level fallback：仅在 primary episode 缺失或 unmatched 时作为退路；如果 primary reference 未收敛，不会用 run-level 结果覆盖。
- secondary episode diagnostics：后续反弹、多 episode、contact chatter、prediction lead 和 substep cap 会作为辅助证据；如果首次冲击指标已经改善，这些现象不会反向判定 primary impact 失败。
- `RunPrimaryAttributionDifference`：显式记录 run-level 和 primary-level 结论差异，例如 run-level unresolved 但 primary 已收敛，或 run-level not improved 但 primary improved。
- `examples/20_mujoco_primary_impact_attribution.py`：运行 primary-impact attribution 示例，并导出 CSV、JSON 和 Markdown 报告到 `artifacts/contact_primary_attribution/`。

primary-impact attribution 是当前 adaptive 碰撞精度诊断的首选路径。运行级 `attribute_adaptive_failure()` 仍保留用于长轨迹整体诊断，以及 primary episode 缺失或无法匹配时的 fallback。后续反弹不能覆盖首次冲击的评价；primary reference convergence 和 run-level reference convergence 是不同问题，必须分别报告。

## Phase 2G8 当前能力

已支持统一批量 primary-impact 评估管线：

- `AdaptiveBatchCase`：声明 sphere-plane / sphere-sphere batch case，包括 macro timestep、总时长、MuJoCo contact solver 参数、sphere 参数、初始状态和 metadata。
- `run_adaptive_primary_batch()`：统一执行 fixed coarse、fixed fine、adaptive、episode extraction、primary matching、selected reference convergence、primary attribution、summary 和 artifact export。
- `ReferenceEvaluationStatus`：在 batch 层区分 `NOT_CHECKED`、`CONVERGED`、`NOT_CONVERGED` 和 `INVALID`，不会把未检查 reference 记为未收敛。
- `ReferenceEvaluationMode`：支持 `ALL`、`SELECTED` 和 `NONE`；默认 selected 模式先用 fixed-fine 作为 provisional baseline，再确定性选择需要 finer / ultra-fine refinement 的 case。
- `generate_sphere_plane_batch_cases()` / `generate_sphere_sphere_batch_cases()`：提供确定性 case generator，默认不运行完整笛卡尔积，避免示例过慢。
- `make_smoke_adaptive_batch()`：生成 8 个快速 case，包含 sphere-plane 和 sphere-sphere，并至少选择 2 个 reference check。
- `make_standard_adaptive_batch()`：生成约 40 个标准 case，覆盖高度、macro timestep、solref、半径、质量和 sphere-sphere 碰撞类型。
- `AdaptiveBatchSummary`：报告明确分母的改善率、reference coverage、primary matched/unmatched、step ratio、step saving 和 primary metric errors。
- `AdaptiveBatchGroupSummary`：按 scene type、macro timestep、solref、impact-speed range、sphere radius 和 sphere mass 分组统计。
- `AccuracyCostPoint` 与 `find_nondominated_accuracy_cost_points()`：导出精度-成本点和 non-dominated case；组合 error 需要调用方显式提供权重。
- `examples/21_mujoco_adaptive_primary_batch.py`：一条命令从 case generation 运行到 CSV / JSON / Markdown report。

数据流：

```text
BatchCase
-> coarse/fine/adaptive runs
-> episode extraction
-> provisional comparison
-> selected convergence
-> primary attribution
-> grouped summary/report
```

Phase 2G8 只是 orchestration、aggregation 和 reporting，不自动修改 adaptive 参数，不改变 timestep policy，不修改 `solref/solimp`、`backend.step()` 或 MuJoCo contact solver。fixed-fine 只是 provisional baseline；只有经过 refinement 且 primary episode 收敛的结果才能称为 converged reference。改善率必须报告明确分母，分母只包含 reference checked and converged、primary matched、metric applicable、adaptive valid 的 case。

## Phase 2G9 当前能力

已支持 reference convergence diagnostics，用于解释 checked case 为什么未正式收敛：

- `ReferenceUnresolvedReason`：区分 `EPISODE_UNMATCHED`、`RESTITUTION_NOT_CONVERGED`、`PENETRATION_NOT_CONVERGED`、`DURATION_NOT_CONVERGED`、`START_TIME_NOT_CONVERGED`、`NON_MONOTONIC_REFINEMENT`、`METRIC_SAMPLING_SENSITIVITY`、`INVALID_LEVEL` 等原因。
- `ReferenceDiagnosticStatus`：报告层明确区分 `NOT_CHECKED`、`CONVERGED`、`NEAR_CONVERGED`、`NOT_CONVERGED` 和 `INVALID`；`NEAR_CONVERGED` 只用于诊断，不算正式 converged。
- `run_reference_convergence_diagnostics()`：对 selected checked cases 输出 fine、finer、ultra-fine 的 primary-impact level 指标，并为 unresolved case 可选追加 extra-fine / `h/8`。
- 每个 refinement level 记录 restitution、maximum penetration、contact duration、primary-impact start time、impact speed 和 separation speed。
- 每个指标记录 `D1 = |Q_fine - Q_finer|`、`D2 = |Q_finer - Q_ultra|`、`rho = D2 / D1`，用于识别非单调 refinement 和采样敏感性。
- `examples/22_mujoco_reference_convergence_diagnostics.py`：基于 smoke batch 导出 `artifacts/reference_diagnostics/metric_levels.csv`、`unresolved_cases.csv`、`diagnostics.json` 和 `report.md`。

Phase 2G9 只做诊断，不修改 `AdaptiveMuJoCoRunner`、substep policy、prediction horizon、`solref/solimp`、episode segmentation、reference tolerance 或 contact solver。未检查 reference 与未收敛 reference 继续分开统计。

## Phase 2H1 当前能力

已支持从已加载 MuJoCo scene 自动构造 adaptive pre-contact prediction candidates：

- `build_adaptive_prediction_candidates(scene=..., backend=...)`：从 `PhysicsSceneSpec`、compiled collider metadata 和 loaded backend 构造候选。
- `CompiledColliderMetadata`：记录 source collider id、runtime body id、MuJoCo geom name、geometry、world transform、dynamic/static 状态、collision group/mask 和 MuJoCo contact params。
- 自动支持 `SphereSphereAdaptiveCandidate`：两个 eligible sphere collider 生成一个稳定、去重、canonical ordered candidate。
- 自动支持 `SpherePlaneAdaptiveCandidate`：dynamic sphere + static box collider，在 box top 近似水平、尺寸足够大、sphere 初始投影位于顶面区域时，将 box 顶面近似为 `AnalyticPlane`。
- pair eligibility：不同 runtime body、至少一个 dynamic body、collision mask 允许、非 visual-only geom、当前 geometry 支持。
- unsupported geometry：box-box、capsule-box、mesh 等不会阻塞 MuJoCo 仿真，只会记录 diagnostic，当前不生成 adaptive prediction candidate。
- `create_adaptive_runner_from_scene()` 和 `AdaptiveMuJoCoRunner.from_scene()`：提供自动候选 + 手工候选的便捷构造；手工候选 API 继续可用。
- `examples/23_mujoco_automatic_adaptive_candidates.py`：展示 sphere-plane 和 sphere-sphere 自动候选，并直接运行 adaptive simulation。

必须区分两件事：

```text
MuJoCo runtime contact detection
-> 仍由 MuJoCo 根据 geom / contype / conaffinity 自动完成

Adaptive pre-contact prediction candidates
-> 只为 adaptive timestep 在碰撞前预测何时该加密 substeps
```

candidate builder 不创建或删除 MuJoCo contact pair，不修改 `contype/conaffinity`，不修改 `solref/solimp`，也不改变 `backend.step()` 或 adaptive timestep policy。

## 控制与外力接口

MuJoCo backend 已支持自由动态刚体的基础控制/扰动接口：

- `set_body_velocity()`：设置世界系线速度和角速度，可选择 `update_initial=True` 让 reset 后保留该初速度。
- `apply_force()`：施加世界系外力；如果提供 world-space `point`，会转换为等效力矩 `(point - COM) x force`。
- `apply_torque()`：施加世界系外力矩。
- `clear_applied_forces()`：清空 `xfrc_applied` 和 `qfrc_applied`。

这些接口只支持有 freejoint 的 dynamic body；static、kinematic 或 fixed-base body 会明确报错。

## Phase 2D2 当前能力

已验证：

- box drop：动态 box 能自由下落、产生 contact，并稳定停留在 ground 顶面。
- sphere drop：动态 sphere 能稳定落地，最终高度接近半径。
- resting contact：接近静止支撑位姿的 box 不会持续下沉或抖动。
- compound surface：由多个 collider 组成的静态桌面可以作为稳定支撑面。
- trajectory sampling：`simulate_body_trajectory()` 会 reset backend，并记录 `steps + 1` 个 `BodyStateSample`。
- maximum penetration：从目标 body 相关 contacts 中统计最大穿透深度。
- last-window stability：settled 判断使用最后一段时间窗口，而不是最后单步速度。
- deterministic replay：相同 scene 在 reset 后重复运行得到一致的末端状态、contact 序列和 metrics。

`settled` 阈值是评估配置，不是通用物理定律。默认 `SettlingCriteria` 关注最后窗口内的线速度、角速度、位置漂移、姿态漂移，并可要求最终仍存在目标 body contact。

当前仍未支持：

- MuJoCo internal per-contact impulse
- automatic restitution-to-solver-params mapping
- parameter optimization
- material parameter inversion
- initial velocity API
- general geometry collision prediction
- Hertz contact-time estimation
- event-time rollback
- automatic adaptive candidate generation
- automatic adaptive tuning
- automatic material-to-solver mapping
- quantitative friction validation
- joint
- actuator
- robot
- mesh
- GUI
- task framework

## 当前项目状态

项目仍处于早期仿真基础设施阶段，尚未实现完整仿真功能。当前代码已经具备可验证的 Physics IR、场景表示、MJCF 编译、MuJoCo 运行基础、接触映射和基础轨迹评估能力。

当前 `GeometrySpec` 已支持 box、sphere、cylinder、capsule、wedge/ramp、cone、frustum、ellipsoid、spherical cap 和 regular prism 等参数化语义。MuJoCo 直接编译覆盖 box、sphere、cylinder 和 capsule；wedge/ramp、cone、frustum 和 regular prism 已支持 deterministic convex mesh fallback。ellipsoid 和 spherical cap 仍待曲面 mesh fallback 或后端专用表达。

wedge/ramp、frustum 和 regular prism 已支持完整质量属性计算：wedge/ramp 与 regular prism 使用闭合多面体体积分解，frustum 使用连续圆台解析积分，结果包含完整 3x3 惯量张量、主惯量和主轴方向。

`MassProperties` 已扩展为可保存 `inertia_tensor` 和 `principal_axes`。MuJoCo 编译器会在主轴不与 body frame 对齐时输出 `<inertial quat="..." diaginertia="...">`，从而让后端消费 principal inertial frame。

## 开发原则

- visual mesh 与 collision mesh 分离。
- 统一 Physics IR，不让业务代码直接依赖具体后端。
- 先人工指定物理语义，再逐步加入自动推断。
- 所有实验必须支持固定随机种子和结果复现。
- `GeometrySpec` 表示最终物理尺寸，质量、体积和惯量计算不隐式读取 `Transform.scale`。
- MuJoCo numeric ID 只存在于 backend 内部，不进入 Physics IR 或业务层。
