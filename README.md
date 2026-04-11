# open-match-lca

`open-match-lca` 是一个公开数据优先、脚本优先、可复现的 Python 研究仓库，用于实现论文级实验：`产品文本 -> NAICS-6 类别匹配 -> EPA 开放影响因子预测`，并提供可选扩展 `产品文本 -> USLCI process 候选检索`。

## 项目目标

- 主任务：将产品标题和描述映射到 NAICS-6 类别，并据此预测公开 EPA/USEEIO 影响因子。
- 扩展任务：在公开 USLCI 过程语料中做候选检索，生成过程级审阅包。
- 设计原则：公开数据优先、固定随机种子、脚本驱动、结果落盘、适合论文实验复现。

## 为什么默认主实验不使用商业数据库

- 默认主实验不得依赖 ecoinvent、GaBi 或任何商业数据库，以避免许可壁垒影响复现。
- 默认主实验不得依赖 openLCA GUI，以确保 CI、本地服务器和脚本环境都可运行。
- 主实验默认只使用公开数据源和开源模型，不调用任何闭源在线 API。
- `openLCA` 仅作为可选扩展：只有研究者已经合法拥有相应数据库许可时才可自行接入，本仓库默认路径不会引用商业库。

## 数据来源与许可说明

- Amazon CaML 风格产品数据：研究用途产品文本与 NAICS 标注。请研究者自行确认原始发布页面与许可条款。
- NAICS 语料：美国官方 NAICS 分类标题与层级信息。
- EPA 因子：EPA/USEEIO 公开排放或环境影响因子。
- USLCI：NREL/USLCI 公开过程数据或其发布包。

本仓库不内置任何商业数据库，不会静默回退到商业数据源。使用者必须自行核对各数据源的最新许可说明。

## 环境安装

必须使用项目目录下的虚拟环境，并使用 Python 3.11：

```bash
cd /Users/edward0112/Desktop/lca/open-match-lca
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
make install-dev
```

## 数据下载方法

当前仓库优先提供公开数据接口与解析器骨架。下载脚本会创建目标目录并记录目标清单；若某数据源需要研究者手动放置原始文件，脚本会给出明确提示，不会静默失败。

```bash
source .venv/bin/activate
python scripts/00_download_data.py --target all --out_dir data/raw --overwrite
```

## 一键运行 smoke test

```bash
source .venv/bin/activate
make smoke-test
```

当前 smoke test 覆盖：

- parser 基本读写与字段校验
- 数据切分可复现性
- 检索指标与回归指标
- RRF 融合
- top-k factor mixture 基线

## 一键运行完整实验

完整实验配置位于 `configs/exp/full.yaml`。当前仓库第一阶段先交付可运行骨架、schema、parser 和 smoke test；后续会逐步补齐 dense training、rerank、regression 和 process retrieval 主体。

```bash
source .venv/bin/activate
python scripts/01_prepare_main_data.py \
  --amazon_dir data/raw/amazon_caml \
  --epa_dir data/raw/epa_factors \
  --naics_dir data/raw/naics \
  --out_dir data/processed \
  --config configs/data/main.yaml

python scripts/02_make_splits.py \
  --input_products data/processed/products.parquet \
  --split_type random_stratified \
  --seed 13 \
  --out_dir data/splits
```

## 如何复现论文表格和图

- 运行评估脚本后，指标会落到 `reports/metrics/`
- 表格导出脚本会生成 CSV 和 LaTeX
- 图导出模块会将图保存到 `reports/figures/`

```bash
python scripts/12_export_paper_tables.py \
  --metrics_dir reports/metrics \
  --output_dir reports/tables \
  --format both
```

## 可选扩展

### USLCI process retrieval

- 属于 optional extension
- 若找不到 USLCI 原始数据，脚本会明确提示：该功能为 optional extension，主实验不受影响

### openLCA

- 仅当研究者已经合法拥有相关数据库许可时才可自行接入
- 不是主实验必要条件
- 默认配置和默认代码路径不依赖 openLCA GUI

## 限制与适用边界

- 该系统定位为 screening-level LCA 辅助检索与因子预测
- 不是完整审计级产品 LCA
- 预测结果不应替代专家审阅、供应链一手数据或正式核算流程

## 仓库状态

当前版本已完成：

- 仓库骨架
- `pyproject.toml` / `requirements.txt` / `Makefile`
- 数据 schema 与 parser 初版
- 可运行 smoke test
- README 初稿

后续将按以下顺序补齐：

1. 下载、解析、切分
2. BM25、dense zero-shot、top-1 factor baseline
3. dense fine-tune
4. hybrid 和 rerank
5. regression 和 uncertainty
6. process extension 与论文导出
