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

`PhysicsSceneSpec` 是后端编译与运行时加载的输入。IR 中使用稳定的业务 ID 与 runtime body ID，不保存 MuJoCo numeric ID。

## Backend Layer

负责把 Physics IR 接入具体物理引擎。

Phase 2B 编译路径：

```text
PhysicsSceneSpec
-> MuJoCoCompiler
-> MuJoCoCompilationResult
-> MJCF
```

Phase 2C1 加载路径：

```text
MJCF
-> mujoco.MjModel
-> mujoco.MjData
-> backend-private numeric ID mappings
```

MuJoCo numeric body ID 和 geom ID 只保存在 `MuJoCoBackend` 内部，用于后续状态查询、接触映射和力施加。它们不进入 Physics IR，也不作为业务层公共标识。

## Runtime Layer

负责 reset、step、状态查询、接触事件、传感器更新和确定性执行。

当前只定义运行时状态值对象。Phase 2C1 不实现 reset、step、body state、contact 或 force。

## Robot Task Layer

负责机器人相关任务，例如下落、推动、稳定性、关节运动、抓取和夹爪控制。

当前仍未实现，后续会基于 Runtime Layer 提供的状态和控制接口构建。

## Evaluation Layer

负责从仿真轨迹和任务结果中计算指标、分类失败原因并生成评估报告。

当前仍未实现，后续会依赖可复现的仿真运行结果。
