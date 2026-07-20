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

负责把 Physics IR 接入具体物理引擎。

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

`MjModel` 是加载后的 MuJoCo 模型，`MjData` 是 backend 内部可变状态。MuJoCo numeric body ID 和 geom ID 只保存在 `MuJoCoBackend` 内部，用于状态查询和后续接触映射，不进入 Physics IR 或业务层公共接口。

## Runtime Layer

负责 reset、step、状态查询、接触事件、传感器更新和确定性执行。

Phase 2C2 已支持 reset、单步 step、刚体世界位姿读取、世界线速度和角速度读取，并把结果封装为后端无关的 `SimulationStepResult` 快照。

当前 `SimulationStepResult.contacts` 始终为空，即使 MuJoCo 内部已经产生接触。接触提取和映射留到 Phase 2D。

## Robot Task Layer

负责机器人相关任务，例如下落、推动、稳定性、关节运动、抓取和夹爪控制。

当前仍未实现，后续会基于 Runtime Layer 提供的状态和控制接口构建。

## Evaluation Layer

负责从仿真轨迹和任务结果中计算指标、分类失败原因并生成评估报告。

当前仍未实现，后续会依赖可复现的仿真运行结果。
