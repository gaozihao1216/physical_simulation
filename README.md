# physical_simulation

`physical_simulation` 是一个独立 Python 子项目，用于把重建后的 3D 资产逐步转化为可验证、可复现、可接入物理后端的仿真资产与任务评估流程。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：在视觉几何重建完成之后，补充物理语义，构建后端无关的 Physics IR，并逐步接入碰撞体生成、刚体动力学、关节系统、机器人任务和动态评估。

当前已经支持参数化 Physics IR、场景表示、MJCF 编译、MuJoCo 模型加载、reset、单步 step、刚体世界状态读取，以及 MuJoCo active contact 到 `ContactPoint` 的映射。接触力、冲量、关节、机器人和评估系统仍未实现。

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
- `backends`：MuJoCo、Isaac Sim 等物理后端适配器；当前支持 MuJoCo 模型加载、reset、单步推进、刚体状态读取和 contact 映射。
- `runtime`：运行时状态对象，包括 `RigidBodyState`、`JointState`、`ContactPoint` 和 `SimulationStepResult`。
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
- Phase 2D2：Contact Force and Impulse。
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

尚未支持：

- normal force
- tangential force
- contact impulse
- resting stability evaluation
- friction validation
- restitution validation
- task success metrics
- robot interaction

`static_friction` 和 `restitution` 当前暂未映射到 MJCF explicit pair。`solref` 和 `solimp` 没有对应 Physics IR 参数，因此不显式设置，使用 MuJoCo 默认值。

## 当前项目状态

项目仍处于早期仿真基础设施阶段，尚未实现完整仿真功能。当前代码已经具备可验证的 Physics IR、场景表示、MJCF 编译、MuJoCo 运行基础和接触映射能力。

## 开发原则

- visual mesh 与 collision mesh 分离。
- 统一 Physics IR，不让业务代码直接依赖具体后端。
- 先人工指定物理语义，再逐步加入自动推断。
- 所有实验必须支持固定随机种子和结果复现。
- `GeometrySpec` 表示最终物理尺寸，质量、体积和惯量计算不隐式读取 `Transform.scale`。
- MuJoCo numeric ID 只存在于 backend 内部，不进入 Physics IR 或业务层。
