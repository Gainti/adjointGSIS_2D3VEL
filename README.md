# AdjointGSIS_2D3VEL

基于非结构网格和有限体积方法的离散速度法并行求解线化玻尔兹曼方程的直接求解和伴随求解框架。

---

## 1. 依赖与环境

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

## 2. 编译

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

## 3. 运行

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

## 4. 配置文件说明（`cases/demo/config.ini`）

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

## 5. 输出结果

当前主流程会输出宏观量 Tecplot 文件：

- `<output_dir>/macro.szplt`（以及相关 Tecplot 文件）

代码中还包含残差 CSV 与伴随变量输出接口，可根据需要在 `main.cpp` 中启用。

---
