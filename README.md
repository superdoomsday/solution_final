# 考试排程求解器（Z3）

基于 Z3 的考试排程求解器，支持三种建模方式、可选附加约束和分析工具。

## 文件说明

| 文件 | 用途 |
|---|---|
| `cw1_solver.py` | 主求解器（运行此文件） |
| `inst/` | 示例 sat 和 unsat 实例 |
| `tests/` | 自动化测试 |
| `benchmark.py` | 耗时和模型对比实验（会写入 `results/`） |
| `gen_instances.py` | 测试和基准实验使用的随机实例生成器 |
| `requirements.txt` | Python 依赖 |

## 安装

```bash
python3 -m pip install -r requirements.txt
```

`ortools` 是可选依赖，仅用于 `benchmark.py` 中的 CP-SAT 对比实验。

## 运行求解器

```bash
python3 cw1_solver.py inst/sat1.txt
python3 cw1_solver.py inst/*.txt --check
python3 cw1_solver.py inst/*.txt --model bool --check
python3 cw1_solver.py inst/*.txt --model cnf --check
python3 cw1_solver.py inst/unsat1.txt --explain
python3 cw1_solver.py inst/sat6.txt --count 100
python3 cw1_solver.py inst/sat10.txt --optimize gap
```

有解时会输出 `sat` 和考试安排；无解时会输出 `unsat`，随后显示求解耗时（毫秒）。

### 模型（`--model`）

| 名称 | 默认 | 思路 |
|---|---|---|
| `int` | 是 | 每场考试使用一个整数表示时段和教室 |
| `bool` | | 使用布尔表表示时段和教室分配 |
| `cnf` | | 使用相同的布尔表，并将每条规则表示为子句（文字的 OR） |

### 常用选项

- `--check` — 使用普通 Python（不调用 Z3）重新检查排程
- `--count N` — 最多统计 N 种排程
- `--explain` — 对无解实例，找出冲突的最小约束组
- `--optimize gap|makespan` — 寻找最优排程
- 附加约束（默认全部关闭）：`--max-per-day`、`--unavailable`、`--invigilators`、`--min-gap`、`--large-early`

## 测试

```bash
python3 -m pytest tests -q
```

## 基准实验

```bash
python3 benchmark.py --quick   # 快速冒烟测试
python3 benchmark.py           # 完整实验（数分钟）
```
