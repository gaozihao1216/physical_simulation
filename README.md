# physical_simulation

`physical_simulation` 是一个独立 Python 子项目，用于把重建后的 3D 资产逐步转化为可验证、可复现、可接入物理后端的仿真资产与任务评估流程。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：在视觉几何重建完成之后，补充物理语义，构建后端无关的 Physics IR，并逐步接入碰撞体生成、刚体动力学、关节系统、机器人任务和动态评估。

当前已完成 Phase 1、Phase 1.5、Phase 2A、Phase 2B、Phase 2C1、Phase 2C2 和 Phase 2D1。项目已经可以通过 MuJoCo 加载场景、reset、推进单步仿真，读取刚体世界状态，并把 MuJoCo active contacts 映射为后端无关的 `ContactPoint`。接触力、力控制、关节系统、机器人任务和动态评估仍未实现。

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
|       |   |-- base.py
|       |   |-- errors.py
|       |   `-- mujoco_backend.py
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
- `backends`：MuJoCo、Isaac Sim 等物理后端适配器；当前支持 MuJoCo 模型加载、reset、单步推进和刚体状态读取。
- `runtime`：运行时状态对象，包括 `RigidBodyState`、`JointState`、`ContactPoint` 和 `SimulationStepResult`。
- `robots`：机器人模型加载、控制器和夹爪控制；当前仍是预留模块。
- `tasks`：下落、推动、稳定性、关节运动和抓取等测试任务；当前仍是预留模块。
- `evaluation`：仿真指标、失败分类和评估报告；当前仍是预留模块。
- `visualization`：场景显示、轨迹回放和调试可视化；当前仍是预留模块。
- `utils`：坐标变换、单位转换、XML 格式化和通用工具。
- `configs`：仿真参数、材料参数和任务配置。
- `docs`：系统架构和设计文档。
- `examples`：最小可运行示例。
- `scripts`：命令行运行入口。
- `tests/unit`：独立模块测试。
- `tests/integration`：完整流程或真实后端加载测试。

## 初步开发路线

- Phase 1：Physics IR / Parametric Physics Asset Representation。
- Phase 1.5：Physics IR Semantic Hardening and Scene Representation。
- Phase 2A：Transform Composition。
- Phase 2B：PhysicsSceneSpec -> MJCF Compiler。
- Phase 2C1：MuJoCo Model Loading and ID Mapping。
- Phase 2C2：Reset, Step and Rigid-Body State。
- Phase 2D1：MuJoCo Contact Mapping。
- Phase 2D2：Contact Force and Impulse。
- Phase 3：MuJoCo Backend。
- Phase 4：Rigid Body Simulation。
- Phase 5：Articulation。
- Phase 6：Robot Tasks。
- Phase 7：Dynamic Evaluation。

## Phase 2C2 当前能力

已支持：

- `MuJoCoBackend.reset()`：使用 MuJoCo 官方 reset API，并恢复加载时保存的 `qpos`、`qvel` 和 `act` 初始快照。
- `MuJoCoBackend.step(action=None)`：每次只推进一个 physics timestep，不做 decimation、不 sleep、不启动 viewer。
- simulation time：从真实 `MjData.time` 读取。
- step index：由 backend 维护，加载和 reset 后为 0，每次 step 增加 1。
- world body pose：从 MuJoCo `xpos` 和 `xquat` 读取，四元数顺序保持 `(w, x, y, z)`。
- world linear velocity 与 angular velocity：使用 `mj_objectVelocity` 读取世界方向六维速度，并按 MuJoCo `rot:lin` 顺序拆分。
- `SimulationStepResult`：返回稳定顺序的 `RigidBodyState` 快照，当前 `joint_states=()` 且 `contacts=()`。
- free-fall simulation：支持无接触自由落体趋势测试。
- reset determinism：相同步数运行在 reset 后可复现。

尚未支持：

- contact extraction
- contact force
- `apply_force`
- joint
- actuator
- robot
- mesh
- GUI
- task evaluation

## Phase 2D1 当前能力

已支持：

- MuJoCo active contact extraction：遍历 `data.contact[0:data.ncon]`。
- geom ID -> runtime body ID：公开结果只包含 `"{instance_id}/{body_id}"`，不泄漏 MuJoCo numeric ID 或 sanitized name。
- world-space contact position：`ContactPoint.position` 使用 MuJoCo 世界接触点。
- deterministic body ordering：`body_a/body_b` 按 runtime body ID 字典序稳定排序。
- body_a -> body_b contact normal：MuJoCo `contact.frame[:3]` 为 `geom1 -> geom2`，如果排序交换双方则同步翻转 normal。
- non-negative penetration depth：`penetration_depth = max(0.0, -contact.dist)`。
- multiple contact points：同一 body pair 的多个接触点全部保留，不合并、不平均。
- contact snapshots in `SimulationStepResult`：`reset()` 和 `step()` 返回当前 MuJoCo 状态对应的 contacts。
- fixed-base contact authoring：compiler 会为允许碰撞的 collision geom pair 生成显式 MuJoCo `<contact><pair>`，以便 fixed-base dynamic 与 static body 的接触也能进入 active contact。

尚未支持：

- normal force
- tangential force
- contact impulse
- resting stability evaluation
- friction validation
- restitution validation
- task success metrics
- robot interaction

## 当前项目状态

项目仍处于早期仿真基础设施阶段，尚未实现完整仿真功能。

当前代码已经具备可验证的 Physics IR、场景表示、MJCF 编译、MuJoCo 模型加载、reset、单步推进和刚体状态读取能力。接触、关节、机器人和评估仍是后续阶段。

## 开发原则

- visual mesh 与 collision mesh 分离。
- 统一 Physics IR，不让业务代码直接依赖具体后端。
- 先人工指定物理语义，再逐步加入自动推断。
- 所有实验必须支持固定随机种子和结果复现。
- `GeometrySpec` 表示最终物理尺寸，质量、体积和惯量计算不隐式读取 `Transform.scale`。
- MuJoCo numeric ID 只存在于 backend 内部，不进入 Physics IR 或业务层。
