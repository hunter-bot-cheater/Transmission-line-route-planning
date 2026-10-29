# 塔位图邻接缓存有界化验证

日期：2026-10-29

## 改动

塔位图 A* 原先把每个已扩展节点的完整邻接边表永久存进字典。零转弯惩罚时，搜索状态就是节点，A* 不会因不同前驱重复扩展同一个节点；永久缓存会持续累积而没有重用收益。现在零转弯惩罚关闭邻接缓存；启用转弯惩罚时保留最多 64 个节点的 LRU 缓存，因为转弯状态可能重复查询同一节点。

变更文件：

- `transmission_path_planner/experimental_tower_graph.py`
- `tests/test_experimental_tower_graph.py`

## 服务器验证

为避免覆盖正在运行的基线，在 `/data1/B24041626/pp/tower_graph_bounded_cache_20261029_01` 创建独立代码副本，只在该副本中替换规划模块并加入测试文件。未改动现有运行目录或服务器全局环境。

服务器命令：

```bash
PYTHONPATH=/data1/B24041626/pp/tower_graph_bounded_cache_20261029_01/source \
  /data1/B21030328/anaconda3/bin/python3.11 -m pytest -q \
  /data1/B24041626/pp/tower_graph_bounded_cache_20261029_01/tests/test_experimental_tower_graph.py
```

结果：`37 passed in 6.16s`。项目虚拟环境中没有 pytest；使用服务器已有的 Python 3.11 / pytest 8.3.4 环境，没有安装依赖。

## 与全域路线任务的关系

正在运行的 `/data1/B24041626/pp/tower_graph_stride2_sensitivity_20261029_01` 进程 PID `2284077` 使用旧版源代码，作为性能基线保留。最近一次检查时进程仍在计算塔位图阶段，尚无 TowerGraph 路线或误差结果；该运行的 RSS 约 29.98 GB，服务器可用内存约 115.5 GB。不能将它写成有界缓存版本的运行结果。

待基线完成后，应使用本目录对应的新代码另开任务做同配置比较，并记录代码哈希、参数、耗时、峰值内存和线路几何差异。路线几何仍不得作为规划输入。
