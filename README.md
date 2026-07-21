# physical_simulation

`physical_simulation` 是一个独立 Python 子项目，用于把重建后的 3D 资产逐步转化为可验证、可复现、可接入物理后端的仿真资产与任务评估流程。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：在视觉几何重建完成之后，补充物理语义，构建后端无关的 Physics IR，并逐步接入碰撞体生成、刚体动力学、关节系统、机器人任务和动态评估。

当前已经支持参数化 Physics IR、场景表示、MJCF 编译、MuJoCo 模型加载、reset、单步 step、刚体世界状态读取、MuJoCo active contact 到 `ContactPoint` 的映射、单点 `ContactWrench` 读取，以及基础 drop/resting-contact 轨迹评估。冲量、关节、机器人和完整任务框架仍未实现。

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
- Phase 2D3B：Contact Force Aggregation and Impulse（计划中）。
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

`static_friction` 和 `restitution` 当前暂未映射到 MJCF explicit pair。`solref` 和 `solimp` 没有对应 Physics IR 参数，因此不显式设置，使用 MuJoCo 默认值。

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

- 跨接触点聚合 API。
- 关于刚体质心的 net contact torque。
- time-integrated impulse / collision impulse。
- 初始速度配置。
- `apply_force`。
- 定量摩擦验证。
- restitution mapping。
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

当前仍未新增公共聚合 API。关于刚体 COM 的 net force / net torque 只在测试和示例中局部计算，正式 `BodyContactWrench`、`BodyPairContactWrench`、impulse、initial velocity、`set_body_velocity` 和 `apply_force` 仍留到后续阶段。

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

- contact impulse
- restitution mapping
- quantitative friction validation
- `apply_force`
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
