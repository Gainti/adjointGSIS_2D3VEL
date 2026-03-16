# AdjointGSIS_2D3VEL

一个基于 **MPI 并行 + 非结构网格** 的二维离散速度法（DVM）求解器框架，当前主要包含：

- Fluent ASCII `.cas` 网格读取（2D）
- 基于 METIS 的网格划分与分区分发
- Halo 交换与并行迭代
- 原始（primal）DVM 求解流程
- 预留伴随（adjoint）与边界敏感度模块
- Tecplot (`.szplt`) 宏观量输出

---

## 1. 目录结构

```text
src/
  core/   配置读取、常量定义、工具函数、计时/性能统计
  mesh/   Fluent 网格解析、几何计算、分区、halo、网格形变
  dvm/    DVM 主求解器、伴随求解器、边界敏感度装配
  io/     残差与 Tecplot 输出
cases/
  demo/   示例算例（config.ini）
```

---

## 2. 主要功能与当前状态

### 已实现（主流程）

1. **读取配置与网格**
   - 通过 `--case <caseDir>` 读取 `<caseDir>/config.ini`
   - 读取 Fluent ASCII `.cas` 网格并构建单元/面拓扑

2. **并行划分与通信准备**
   - 根进程读取全局网格
   - 基于 METIS 对单元图划分
   - 分发局部网格并构建 halo 通信信息

3. **DVM 原始方程迭代**
   - 构建 3 维速度空间（`Nvx × Nvy × Nvz`）
   - 迭代推进并监控 `rho/ux/uy` 残差

4. **结果输出**
   - 输出 Tecplot 网格与单元中心变量（`macro`）

### 未完全开发

- 伴随 DVM 迭代流程（`initialAdj/stepAdj`）
- 边界灵敏度组装（`boundary_sensitivity`）
- 网格形变接口（`meshDeform`）

---

## 3. 依赖与环境

- C++17
- CMake ≥ 3.15
- MPI（`find_package(MPI REQUIRED)`）
- METIS + GKlib
- TecIO（当前通过 `./tecio/` 目录链接）

> 注意：`CMakeLists.txt` 中 METIS/GKlib 路径是硬编码的本地路径，使用前请按你的环境修改：
>
> - `METIS_INCLUDE_DIR`
> - `METIS_LIBRARY`
> - `GK_INCLUDE_DIR`
> - `GK_LIBRARY`

---

## 4. 编译

推荐使用仓库自带脚本：

```bash
./make.sh
```

等价命令：

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

默认可执行文件：

```text
build/solver
```

---

## 5. 运行

### 单进程

```bash
./build/solver --case cases/demo
```

### MPI 并行

```bash
mpirun -np 4 ./build/solver --case cases/demo
```

仓库中也提供脚本：

```bash
./run.sh
./mpirun.sh
```

---

## 6. 配置文件说明（`cases/demo/config.ini`）

### `[case]`

- `mesh_file`：网格文件路径（相对 `case` 目录拼接）
- `output_dir`：输出目录

### `[solver]` 关键参数

- `uwall`：壁面速度方向设置
- `tauw` / `delta` / `St`：物理参数
- `Nvx,Nvy,Nvz`：速度空间离散数
- `Lvx,Lvy,Lvz`：速度空间截断范围
- `maxIter`：最大迭代步
- `tol`：收敛阈值
- `printInterval`：打印间隔
- `checkInterval`：残差检查间隔

程序启动时会在 rank0 打印完整求解参数。

---

## 7. 输出结果

当前主流程会输出宏观量 Tecplot 文件：

- `<output_dir>/macro.szplt`（以及相关 Tecplot 文件）

代码中还包含残差 CSV 与伴随变量输出接口，可根据需要在 `main.cpp` 中启用。

---

## 8. 网格与模型限制

- 当前按 **2D Fluent ASCII** 网格流程设计
- 仅支持项目中已实现的边界处理与物理闭合
- 伴随与敏感度虽有代码基础，但默认流程未完全打开

---

## 9. 后续建议

- 将 METIS/GKlib/TecIO 改为可配置查找（避免硬编码路径）
- 在 `README` 中补充典型工况与物理模型说明（如控制方程、无量纲定义）
- 增加最小可复现实验（mesh + baseline log + 结果对比）
- 完整打通伴随求解与灵敏度输出流程
