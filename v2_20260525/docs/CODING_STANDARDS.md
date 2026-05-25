# 代码风格与接口规范文档

**版本**: v2.20260525
**适用范围**: 台湾输电线路智能路径规划系统 v2 及后续版本
**最后更新**: 2026-05-25

---

## 目录

1. [Python 代码风格规范](#1-python-代码风格规范)
2. [模块接口设计规范](#2-模块接口设计规范)
3. [配置管理规范](#3-配置管理规范)
4. [日志与调试规范](#4-日志与调试规范)
5. [文件命名与目录规范](#5-文件命名与目录规范)
6. [文档与注释规范](#6-文档与注释规范)
7. [版本迭代规范](#7-版本迭代规范)
8. [测试规范](#8-测试规范)

---

## 1. Python 代码风格规范

### 1.1 命名规范

| 元素       | 规范                          | 示例                              |
| ---------- | ----------------------------- | --------------------------------- |
| 模块       | 小写 + 下划线                 | `path_planning.py`, `cost_model.py` |
| 类         | 大驼峰 (CapWords)            | `CostSurfaceFuser`, `PathValidator` |
| 函数/方法  | 小写 + 下划线, 动词开头       | `compute_slope()`, `generate_hard_mask()` |
| 常量       | 全大写 + 下划线               | `MAX_SLOPE`, `BASE_RESOLUTION`    |
| 变量       | 小写 + 下划线                 | `dist_existing`, `hard_mask`      |
| 私有函数   | 单下划线前缀                  | `_reconstruct_path()`, `_landuse_tag_to_code()` |
| 私有变量   | 单下划线前缀                  | `_cache`, `_internal_state`       |

### 1.2 代码格式

- **缩进**: 4个空格 (禁止Tab)
- **行宽**: 最大100字符
- **空行**:
  - 模块级函数之间: 2个空行
  - 类方法之间: 1个空行
  - 逻辑段落之间: 1个空行
- **导入顺序** (分三组, 组间空行):
  1. 标准库 (`import sys`, `from pathlib import Path`)
  2. 第三方库 (`import numpy as np`, `from scipy.ndimage import gaussian_filter`)
  3. 本地模块 (`import config as cfg`, `from src.preprocessing import generate_hard_mask`)
- **字符串**: 默认使用双引号 `"`, 内部包含双引号时使用单引号 `'`
- **尾随逗号**: 多行列表/字典/元组最后一项加尾随逗号

### 1.3 类型注解

所有函数参数和返回值**必须**标注类型。示例:

```python
def compute_slope(dem: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Horn(1981)坡度计算 (度)"""
    ...

def geo_to_grid(lat: float, lon: float, transform: rasterio.Affine) -> tuple[int, int]:
    """WGS84坐标 -> 栅格行列号"""
    ...

def generate_hard_mask(
    data: dict[str, np.ndarray | gpd.GeoDataFrame | None],
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """生成硬约束掩膜"""
    ...
```

### 1.4 其他约定

- 使用 `np.array()` 而非 `np.ndarray()` 构造数组
- 避免可变默认参数
- 优先使用 f-string 格式化
- 异常捕获必须指定具体异常类型, 禁止裸 `except:`

---

## 2. 模块接口设计规范

### 2.1 统一输入输出契约

每个可独立运行的模块应实现:

```python
def process(
    input_data: dict,
    params: dict,
    verbose: bool = False,
) -> dict:
    """
    Args:
        input_data: 必须包含 'dem', 'crs', 'transform', 'bounds' 键
        params: 运行时参数覆盖
        verbose: 是否输出详细日志

    Returns:
        dict: 必须包含 'status', 'data', 'metadata', 'warnings', 'errors'
    """
```

### 2.2 输入数据字典规范

```python
input_data = {
    "dem": np.ndarray,              # 高程数组 (H, W)
    "crs": str,                      # 坐标参考系, 如 "EPSG:4326"
    "transform": rasterio.Affine,    # 仿射变换
    "bounds": tuple,                 # (lon_min, lat_min, lon_max, lat_max)
    # ... 可选附加数据
}
```

### 2.3 返回字典规范

```python
{
    "status": "success" | "failure",
    "data": {
        "result_1": np.ndarray,
        "result_2": gpd.GeoDataFrame,
        # ...
    },
    "metadata": {
        "processing_time_s": 12.5,
        "n_features": 18,
        "grid_shape": [2401, 4201],
    },
    "warnings": [
        "水域数据缺失, 跳过水域约束",
    ],
    "errors": [
        # 空列表表示无错误
    ],
}
```

### 2.4 自定义异常类

```python
class PathPlanningError(Exception):
    """路径规划模块基础异常"""
    pass

class HardConstraintViolationError(PathPlanningError):
    """硬约束违规异常 — 路径穿越禁止区域"""
    def __init__(self, point: tuple, constraint_type: str):
        self.point = point
        self.constraint_type = constraint_type
        super().__init__(f"硬约束违规: {constraint_type} at {point}")

class NoPathFoundError(PathPlanningError):
    """A*搜索失败 — 起点与终点间无可达路径"""
    pass

class InvalidEndpointError(PathPlanningError):
    """起点或终点在硬约束区内"""
    def __init__(self, endpoint: str, coord: tuple):
        self.endpoint = endpoint
        self.coord = coord
        super().__init__(f"{endpoint} ({coord}) 位于硬约束禁止区域")
```

命名规则: `{模块名}{错误类型}Error`
上下文信息要求: 异常必须包含触发位置、违规类型、相关数值

异常转标准返回字典机制:
```python
try:
    result = some_operation()
except PathPlanningError as e:
    return {
        "status": "failure",
        "data": {},
        "metadata": {},
        "warnings": [],
        "errors": [str(e)],
    }
```

---

## 3. 配置管理规范

### 3.1 配置分层

```
config.py (v2/config.py)
├── BASE_CONFIG (共享默认值, 不可直接修改)
│   ├── TAIWAN_BBOX
│   ├── WGS84 / PROJECTED_CRS
│   ├── BASE_RESOLUTION
│   ├── RF_N_ESTIMATORS, RF_MAX_DEPTH
│   └── FEATURE_BANDS
│
├── V2_STRICT_CONFIG (版本覆盖)
│   ├── MAX_SLOPE = 28
│   ├── WATER_BUFFER = 100
│   ├── MAX_ELEVATION = 3000
│   ├── PROTECTED_BUFFER = 200
│   ├── BUILDING_DENSITY_LIMIT = 500
│   ├── MAX_CURVATURE_RADIUS = 500
│   ├── ASTAR_HEURISTIC_WEIGHT = 1.2
│   ├── ASTAR_DIAGONAL_COST = 1.5
│   └── LABEL_WEIGHTS = {dist_existing: 0.65, ...}
│
└── TEST_CASES (测试线路定义)
    ├── case_01: way/365662519
    ├── case_02: way/203692582
    └── ...
```

### 3.2 覆盖规则

1. **基础配置禁止直接修改**: 所有默认值定义在模块顶部
2. **版本配置通过模块级变量覆盖**: v2/config.py 中的常量直接覆盖 v1 值
3. **运行时参数**: 通过 `params` 字典传入, 优先级最高

```python
# 运行时参数覆盖示例
params = {"MAX_SLOPE": 30}
effective_max_slope = params.get("MAX_SLOPE", cfg.MAX_SLOPE)
```

### 3.3 配置文件结构要求

- 每个配置段必须用分隔注释 `# ====...===` 标记
- 阈值修改必须添加 `# v2_strict: 从XX调整至YY` 注释
- 所有距离参数以米(m)为单位
- 所有百分比以0-1或0-100明确标注

---

## 4. 日志与调试规范

### 4.1 日志级别使用场景

| 级别       | 使用场景                                         | 示例 |
| ---------- | ------------------------------------------------ | ---- |
| `DEBUG`    | 开发调试信息, 中间变量值, 特征形状               | `每个batch的预测进度` |
| `INFO`     | 正常流程节点, 参数值, 统计摘要                   | `Phase2: 提取地形因子...`, `A*搜索完成 (0.5s)` |
| `WARNING`  | 可恢复异常, 数据缺失但已使用代理, 降级处理       | `水域数据缺失, 跳过水域约束` |
| `ERROR`    | 不可恢复错误, 程序将中止或跳过当前案例           | `起点在硬约束区 → 判定不通过` |
| `CRITICAL` | 系统级故障, 整个流程无法继续                     | `DEM文件不存在: /path/to/dem.tif` |

### 4.2 统一日志格式

```
[模块名] 消息内容
[Phase2] 栅格对齐至统一分辨率...
[Phase4] A*搜索完成 (0.5s), 路径长度: 558 像元
```

- 模块阶段使用 `[PhaseN]` 前缀
- 子模块使用 `[模块名]` 前缀
- 关键数值使用格式化输出: `f"坡度>{MAX_SLOPE}°禁止: {count} 像元"`

### 4.3 调试开关

- `verbose` 参数控制详细输出 (默认 False)
- 仅在 `verbose=True` 时输出中间值和进度条
- 进度报告间隔不超过 5 秒

---

## 5. 文件命名与目录规范

### 5.1 输出文件命名格式

```
{prefix}_{case_id}_{description}_{version}.{ext}
```

示例:
- `case_01_optimal_path_v2.shp`
- `case_01_quality_report_v2.json`
- `case_01_elevation_profile_v2.png`
- `validation_v2_report.json`

### 5.2 目录结构

```
v2_20260525/
├── config.py
├── validate_v2.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── cost_model.py
│   ├── path_planning.py
│   └── output.py
├── output/
│   ├── case_01/
│   │   ├── case_01_optimal_path.shp
│   │   ├── case_01_optimal_path.geojson
│   │   ├── case_01_statistics.json
│   │   ├── case_01_quality_report.json
│   │   ├── case_01_map_overview.png
│   │   └── case_01_elevation_profile.png
│   ├── case_02/
│   └── ...
├── docs/
│   └── CODING_STANDARDS.md
├── tests/
│   ├── test_preprocessing.py
│   ├── test_cost_model.py
│   ├── test_path_planning.py
│   └── test_pipeline.py
└── CHANGELOG.md
```

### 5.3 代码文件头部注释块

每个 `.py` 文件必须包含版本注释块:

```python
"""
v2_strict: {模块中文描述}
版本: v2.20260525
作者: path_planning_team
变更记录:
  - v2.20260525: {变更内容摘要}
  - v1.20260525: 初始版本
依赖: {列出依赖的模块和文件}
"""
```

---

## 6. 文档与注释规范

### 6.1 函数文档字符串 (Google Style)

```python
def compute_slope(dem: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Horn(1981)方法计算地形坡度。

    使用8邻域高程差分计算最大坡度角, 适用于SRTM DEM数据。

    Args:
        dem: 高程数组 (H, W), 单位米, NaN已填充
        transform: 地理仿射变换, 用于计算像元实际尺寸

    Returns:
        np.ndarray: 坡度数组 (H, W), 单位度, 范围[0, 90]

    Raises:
        ValueError: dem包含NaN值时抛出
    """
```

必须包含: `Args`, `Returns`; 可选: `Raises`, `Note`, `Example`

### 6.2 算法步骤注释

注释应解释**物理意义**而非复述代码:

```python
# 正确: 解释物理意义
# 走廊偏好: 输电线路共享基础设施可大幅降低成本
# 距现有线越近折扣越大, 1000m外无折扣

# 错误: 复述代码
# 将dist_existing除以1000, 用clip限制在0-1, 再乘以0.98加0.02
```

### 6.3 阈值调整注释

所有阈值修改必须注释依据:

```python
# v2_strict: MAX_SLOPE 从35°收紧至28°
# 依据: 台湾电力公司输电线路设计规范, 一般线路最大坡度25-30°
# 28°取保守上限, 考虑特殊塔型的适应性
MAX_SLOPE = 28
```

---

## 7. 版本迭代规范

### 7.1 版本号格式

```
v{主版本}.{日期}
v2.20260525
```

- 主版本: 重大架构变更时递增
- 日期: YYYYMMDD 格式, 表示该版本冻结日期

### 7.2 版本目录结构

```
project_root/
├── v1_20260525/          # 历史基线 (只读)
├── v2_20260525/          # 当前开发版本
├── v3_YYYYMMDD/          # 未来版本
└── shared/               # 跨版本共享文件 (内容完全一致)
```

### 7.3 CHANGELOG.md 要求

每个版本目录下必须包含 `CHANGELOG.md`, 记录:

```markdown
# CHANGELOG — v2.20260525

## 算法变更
- MAX_SLOPE: 35° → 28°
- WATER_BUFFER: 50m → 100m
- 新增 PROTECTED_BUFFER = 200m
- 新增 BUILDING_DENSITY_LIMIT = 500 栋/km²
- 新增 _strict_quality_gate() 7项质量审查

## 接口变更
- astar_search(): 起点/终点在硬约束区→返回None (v1自动搜索最近可行点)
- smooth_path(): B样条→线性插值

## 性能影响
- 硬约束区域从5.6%增加到预估15-20%
- A*搜索空间减小, 搜索速度提升
- 质量门控增加约0.1s/每条线路

## 兼容性说明
- v1输出格式不再直接兼容, 需通过validate_v2.py重新生成
- 共享模块(data_acquisition.py)接口保持一致
```

### 7.4 共享文件判定标准

文件放入 `shared/` 的条件:
1. 文件内容在 v1 和 v2 中**完全相同**
2. 不包含任何版本相关的硬编码阈值
3. 被两个版本的代码通过相对路径引用

---

## 8. 测试规范

### 8.1 测试文件组织

```
tests/
├── test_preprocessing.py   # 预处理模块单元测试
├── test_cost_model.py      # 成本建模模块单元测试
├── test_path_planning.py   # 路径规划模块单元测试
├── test_output.py          # 输出模块单元测试
└── test_pipeline.py        # 端到端集成测试
```

### 8.2 单元测试要求

每个测试文件必须包含:

1. **正常路径测试** (Happy Path):
```python
def test_compute_slope_flat_terrain():
    """平坦地形应返回全0坡度"""
    dem = np.ones((100, 100), dtype=np.float32) * 100.0
    slope = compute_slope(dem, mock_transform)
    assert np.allclose(slope, 0.0, atol=0.1)

def test_compute_slope_45degree():
    """45°斜坡应正确计算"""
    ...
```

2. **异常路径测试** (Error Handling):
```python
def test_hard_mask_missing_slope():
    """缺失坡度数据时应有合理回退"""
    ...

def test_astar_invalid_endpoint():
    """起点在硬约束区应返回None"""
    ...
```

3. **边界条件测试** (Edge Cases):
```python
def test_rdp_single_point():
    """单点输入应原样返回"""
    ...

def test_empty_geodataframe():
    """空GeoDataFrame应返回零数组"""
    ...
```

### 8.3 集成测试

`test_pipeline.py` 使用最小数据集验证完整流程:

```python
def test_full_pipeline_mini():
    """10×10像元微型DEM, 验证5阶段完整性"""
    mini_dem = np.random.rand(10, 10).astype(np.float32) * 100
    # ... 设置transform, 运行完整流程
    # 验证输出文件存在
    assert (output_dir / "case_mini_optimal_path.shp").exists()
    assert (output_dir / "case_mini_quality_report.json").exists()
```

### 8.4 测试数据

- 测试使用模拟数据 (NumPy随机数组), 不依赖外部GIS文件
- 所有测试应在 60 秒内完成
- 使用 `pytest` 框架: `pytest tests/ -v --timeout=60`

---

## 附录: 规范检查清单

- [ ] 所有函数有类型注解
- [ ] 所有公共函数有 Google Style docstring
- [ ] 导入分三组排列
- [ ] 阈值修改有 v2_strict 注释
- [ ] 代码文件有版本注释块
- [ ] 输出文件包含版本号
- [ ] 异常类型具体, 不裸 except
- [ ] 行宽不超过100字符
- [ ] 每个模块有对应测试文件
- [ ] CHANGELOG.md 已更新
