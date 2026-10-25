# TAWG-PSO UAV Path Planning

本项目包含论文使用的 TAWG-PSO 核心算法、路径规划任务配置、公共评价器、三类地形环境、交互式可视化和 `medium_1` 复现实验入口。

## 项目结构

```text
tawgpso/
├── algorithm/
│   └── tawg_pso.py             # Complete TAWG-PSO algorithm
├── configuration/
│   └── protocol.py             # Frozen parameters and task configuration
├── evaluation/
│   ├── objectives.py           # Objective, constraints and feasibility
│   ├── metrics.py              # Run metrics and statistical summaries
│   └── energy.py               # Energy, turn, pitch and climb metrics
├── visualization/
│   └── plotting.py             # Path and convergence visualization
├── environments/
│   ├── synthetic/              # sparse_1, medium_1 and dense_1
│   ├── scenarios/              # S1-S6 scenarios and GeoTIFF assets
│   └── external_dem/           # CI-EXT and MSH-EXT environments
├── main.py                     # medium_1 reproduction entry
├── requirements.txt
└── README.md
```

## 核心算法

`algorithm/tawg_pso.py` 包含粒子群位置与速度更新、自适应航点门控编码与解码、结构化初始化、约束状态反馈、动态参数调度、可行精英引导、停滞检测、局部扰动和弱粒子再注入。

冻结参数包括 `K_CAP=26`、`TAU=0.45`、`LAMBDA_K_MAX=0.02`、`LAMBDA_B=0.00`、热启动比例 `0.25`，以及 C7-W4 动态参数组。

## 环境与任务

### 自建地形

`environments/synthetic/` 包含统一的确定性地形生成器，以及 `sparse_1`、`medium_1`、`dense_1` 三组障碍物坐标。默认预览环境为 `medium_1`。

### S1-S6 场景

`environments/scenarios/` 包含六个场景的任务几何、起终点、威胁区定义和各场景 GeoTIFF。场景加载器将地形构造成 0–200 m 水平范围和 0–60 m 高程范围。

### 外部 DEM

`environments/external_dem/` 包含 CI-EXT 和 MSH-EXT 的原生米制地形加载、物理坐标解码、NoData 有效域、AGL 约束、圆柱禁飞区和路径采样审计。

- CI-EXT 使用单一集成 GeoTIFF `ci_ext.tif`，尺寸为 923×1001，水平分辨率为 5 m。地形源自公开 NMOPSO 地图场景中的 Christmas Island DEM。
- MSH-EXT 使用 `msh_ext_1m_window.tif` 作为 1 m 主评价地形，并包含 `msh_ext_5m_grid.tif` 诊断网格。地形源自 2017 Mount St. Helens bare-earth LiDAR DEM，论文数据标识为 USGS DOI `10.5066/P9H16RC7`。

两个外部场景包含相同的起终点、六个圆柱禁飞区、5 m 安全边界和 5–120 m AGL 范围。优化器使用归一化坐标，评价器使用物理米制坐标。

## 目标与评价器

`evaluation/objectives.py` 包含搜索目标、公共代价、碰撞与净距、高度与 AGL、平滑度、航点复杂度、可行性判定和评价预算。

`evaluation/metrics.py` 包含成功状态、fitness、三维路径长度、有效航点数、运行时间和统计汇总。

`evaluation/energy.py` 包含转角、俯仰变化、累计爬升、累计下降和能耗代理。

## 安装

论文服务器配置为 Ubuntu 24.04.4 LTS、CPython 3.12.3、Intel Xeon Gold 6330、112 个逻辑处理器和 128 GiB 内存。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行主实验

```bash
python main.py
```

默认实验使用 `medium_1`、随机种子 `2026001`、50 个粒子、400 次迭代和 20,100 次评价预算。运行结束后显示三维规划路径窗口和收敛曲线窗口，终端输出路径长度、fitness、能耗、有效航点数、最小净距和 AGL 指标。

使用显式保存目录可同时导出图片、`metrics.json` 和 `path.csv`：

```bash
python main.py --save-dir outputs/medium_1_seed2026001
```

## 显示地形

以下命令直接显示对应地形窗口：

```bash
python -m environments.synthetic.run_terrain --environment medium_1
python -m environments.scenarios.run_terrain --scenario s1
python -m environments.scenarios.run_terrain --scenario all
python -m environments.external_dem.run_terrain --scene ci_ext
python -m environments.external_dem.run_terrain --scene msh_ext
```

地形入口支持通过 `--save` 或 `--save-dir` 显式导出图片，并支持通过 `--no-show` 运行无窗口任务。

## 随机种子

单次复现入口的默认随机种子为 `2026001`。成组实验使用 `2026001` 至 `2026051` 的连续配对种子。
