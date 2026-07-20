# physical_simulation

`physical_simulation` 是一个独立 Python 子项目，用于把重建后的 3D 资产逐步转化为可验证、可复现、可接入物理后端的仿真资产与任务评估流程。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：在视觉几何重建完成之后，补充物理语义，构建后端无关的 Physics IR，并逐步接入碰撞体生成、刚体动力学、关节系统、机器人任务和动态评估。

当前已完成 Phase 1、Phase 1.5、Phase 2A、Phase 2B，并完成 Phase 2C1 的 MuJoCo 模型加载与 ID 映射。项目仍未实现完整仿真循环、接触查询、关节系统、机器人控制或动态评估。

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
|       |   |-- errors.py
|       |   |-- mujoco_compiler.py
|       |   `-- mujoco_types.py
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

- `assets`：物理资产、刚体、碰撞体、材料、场景内基础几何等数据结构。
- `collision`：碰撞几何生成、primitive fitting、凸包和凸分解；当前仍是预留模块。
- `dynamics`：质量、重心、惯量、力、力矩和刚体动力学配置。
- `articulation`：关节、运动轴、自由度、限制和执行器；当前仍是预留模块。
- `compilers`：把后端无关的 `PhysicsSceneSpec` 编译为具体后端输入；当前支持 MJCF 生成。
- `backends`：MuJoCo、Isaac Sim 等物理后端适配器；当前支持 Phase 2C1 的 MuJoCo 模型加载与私有 ID 映射。
- `runtime`：仿真循环、状态管理、接触事件和传感器更新；当前只有运行时状态值对象。
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
- Phase 2C2：MuJoCo reset / step / body state extraction。
- Phase 3：MuJoCo Backend。
- Phase 4：Rigid Body Simulation。
- Phase 5：Articulation。
- Phase 6：Robot Tasks。
- Phase 7：Dynamic Evaluation。

## Phase 2C1 当前能力

已支持：

- `mujoco` 作为可选依赖，通过 `pip install -e ".[mujoco]"` 安装。
- `MuJoCoBackend.load_scene(scene)` 将 `PhysicsSceneSpec` 编译为 MJCF，并加载为真实 `mujoco.MjModel` 与 `mujoco.MjData`。
- 将 runtime body ID 映射为 MuJoCo 私有 numeric body ID。
- 将 collision geom numeric ID 映射回 runtime body ID。
- 支持重复加载、失败加载后保留旧状态、`close()` 清理内部状态。
- 真实 MuJoCo 集成测试覆盖 timestep、gravity、质量、惯量、body ID、geom ID、多实例和 compound collider。

尚未支持：

- `reset`
- `step`
- `get_body_state`
- `get_contacts`
- `apply_force`
- joint state
- robot task
- mesh geometry
- GUI / visualization

## 当前项目状态

项目骨架阶段，尚未实现完整仿真功能。

当前代码已经具备可验证的 Physics IR、场景表示、MJCF 编译和 MuJoCo 模型加载能力，但还没有可用的仿真循环与机器人任务执行。

## 开发原则

- visual mesh 与 collision mesh 分离。
- 统一 Physics IR，不让业务代码直接依赖具体后端。
- 先人工指定物理语义，再逐步加入自动推断。
- 所有实验必须支持固定随机种子和结果复现。
- `GeometrySpec` 表示最终物理尺寸，质量、体积和惯量计算不隐式读取 `Transform.scale`。
- MuJoCo numeric ID 只存在于 backend 内部，不进入 Physics IR 或业务层。
