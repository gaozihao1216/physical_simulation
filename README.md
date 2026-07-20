# physical_simulation

`physical_simulation` 是一个独立的 Python 子项目，用于物理场景构建、碰撞逻辑描述、刚体运动接口、关节系统、机器人控制和仿真评估。

## 项目目标

本项目负责 AIGC 流程中的物理仿真部分：将重建得到的 3D 资产转化为后端无关的物理资产，补充碰撞、动力学、关节和任务语义，并通过物理后端执行机器人任务与评估流程。

当前阶段已经进入 Phase 1，实现后端无关、可验证、可序列化的参数化 Physics IR。项目仍不接入 MuJoCo、Isaac Sim 等大型后端，也不实现 GLB 导入、真实碰撞检测或仿真循环。

## 与前半部分 3D Reconstruction 模块的边界

上游 3D reconstruction 模块负责几何重建、视觉网格生成、纹理生成和 GLB 等视觉资产导出。本项目从重建结果之后开始工作：

- 输入是已经重建完成的几何资产，例如 `Reconstructed GLB`。
- 不负责神经重建、网格生成、纹理生成或视觉资产训练。
- 负责物理语义，包括刚体、碰撞体、材料、质量属性、关节、机器人任务、运行状态和评估指标。

## 整体数据流

```text
Reconstructed GLB
-> Physical Asset
-> Collision and Dynamics Authoring
-> Physics Backend
-> Robot Task
-> Evaluation Report
```

## 统一规范

- 长度单位：米 `m`
- 质量单位：千克 `kg`
- 时间单位：秒 `s`
- 角度单位：弧度 `rad`
- 力单位：牛顿 `N`
- 力矩单位：`N*m`
- 坐标系：右手坐标系
- 上方向：`+Z`
- 内部旋转表示：四元数，顺序为 `(w, x, y, z)`

## 当前目录结构

```text
physical_simulation/
|-- README.md
|-- pyproject.toml
|-- .gitignore
|-- configs/
|   `-- README.md
|-- docs/
|   `-- architecture.md
|-- examples/
|   |-- README.md
|   |-- 01_static_ground.py
|   |-- 02_dynamic_box.py
|   |-- 03_dynamic_sphere.py
|   `-- 04_compound_table.py
|-- scripts/
|   `-- README.md
|-- src/
|   `-- physical_simulation/
|       |-- __init__.py
|       |-- assets/
|       |   |-- __init__.py
|       |   |-- builders.py
|       |   |-- collider.py
|       |   |-- geometry.py
|       |   |-- mass_properties.py
|       |   |-- material.py
|       |   |-- rigid_body.py
|       |   |-- transform.py
|       |   `-- visual.py
|       |-- dynamics/
|       |   |-- __init__.py
|       |   `-- inertia.py
|       |-- validation/
|       |   |-- __init__.py
|       |   |-- asset_validator.py
|       |   `-- errors.py
|       |-- serialization/
|       |   |-- __init__.py
|       |   `-- json_codec.py
|       |-- backends/
|       |   |-- __init__.py
|       |   `-- base.py
|       |-- collision/
|       |   `-- __init__.py
|       |-- articulation/
|       |   `-- __init__.py
|       |-- runtime/
|       |   `-- __init__.py
|       |-- robots/
|       |   `-- __init__.py
|       |-- tasks/
|       |   `-- __init__.py
|       |-- evaluation/
|       |   `-- __init__.py
|       |-- visualization/
|       |   `-- __init__.py
|       `-- utils/
|           `-- __init__.py
`-- tests/
    |-- __init__.py
    |-- unit/
    |   |-- __init__.py
    |   |-- test_builders.py
    |   |-- test_geometry.py
    |   |-- test_inertia.py
    |   |-- test_material.py
    |   |-- test_serialization.py
    |   |-- test_transform.py
    |   `-- test_validation.py
    `-- integration/
        `-- __init__.py
```

## 各模块职责

- `assets`：物理资产、Transform、基础几何、材质、质量属性、visual、collider、刚体规格和参数化构造器。
- `collision`：碰撞几何生成、primitive fitting、凸包和凸分解。当前阶段尚未实现。
- `dynamics`：质量、重心、惯量、力、力矩和刚体动力学配置；Phase 1 已实现基础几何的对角惯量计算。
- `articulation`：关节、运动轴、自由度、限制和执行器。当前阶段尚未实现。
- `backends`：MuJoCo、Isaac Sim 等物理后端适配器。当前阶段只保留抽象接口。
- `runtime`：仿真循环、状态管理、接触事件和传感器更新。当前阶段尚未实现。
- `robots`：机器人模型加载、控制器和夹爪控制。当前阶段尚未实现。
- `tasks`：下落、推动、稳定性、关节运动和抓取等测试任务。当前阶段尚未实现。
- `evaluation`：仿真指标、失败分类和评估报告。当前阶段尚未实现。
- `visualization`：场景显示、轨迹回放和调试可视化。当前阶段尚未实现。
- `utils`：坐标变换、单位转换、日志和通用工具。
- `validation`：统一参数验证和项目内异常类型。
- `serialization`：JSON 编解码、保存和加载。
- `configs`：仿真参数、材料参数和任务配置。
- `docs`：系统架构和设计文档。
- `examples`：最小可运行示例。
- `scripts`：命令行运行入口。
- `tests/unit`：独立模块测试。
- `tests/integration`：完整仿真流程测试。

## 初步开发路线

- Phase 1：Physics IR / Parametric Physics Asset Representation
- Phase 2：Collider Generation
- Phase 3：MuJoCo Backend
- Phase 4：Rigid Body Simulation
- Phase 5：Articulation
- Phase 6：Robot Tasks
- Phase 7：Dynamic Evaluation

## Phase 1: Parametric Physics Asset Representation

当前已经支持：

- `Transform`：位置、四元数旋转和缩放，包含有限值验证与四元数归一化。
- `BoxGeometry`、`SphereGeometry`、`CylinderGeometry`、`CapsuleGeometry`：基础参数化几何与体积计算。
- `PhysicsMaterialSpec`：摩擦、恢复系数和可选密度。
- `MassProperties`：质量、重心和对角惯量。
- Visual 与 Collider 分离：即使使用相同基础几何，也通过独立数据结构表达。
- 单刚体与复合 collider：一个 `RigidBodySpec` 可以包含多个 `VisualSpec` 和多个 `ColliderSpec`。
- 参数化构造器：`create_box`、`create_sphere`、`create_cylinder`、`create_capsule`、`create_ground`。
- JSON 序列化：稳定、可读，并支持 rigid body round-trip。
- 参数验证：对非法字段、实际值和预期范围给出明确异常。

当前尚未支持：

- GLB 导入
- `MeshGeometry`
- Convex Hull
- Joint
- Actuator
- MuJoCo
- Robot
- 真实碰撞检测
- 仿真循环

## 当前项目状态

项目骨架与 Phase 1 Physics IR 阶段，尚未实现完整仿真功能。

## 开发原则

- visual mesh 与 collision mesh 分离。
- 统一 Physics IR，不让业务代码直接依赖具体后端。
- 先人工指定物理语义，再逐步加入自动推断。
- 所有实验必须支持固定随机种子和结果复现。
