Download papers with `curl -sL https://arxiv.org/pdf/<id> -o "paper/<ShortName> - <Full Title> (<Venue Year>).pdf"` — filename must follow `ShortName - Full Title (Venue Year).pdf` (no colons, venue abbreviation + year in parentheses, e.g. `ICLR 2024`).
After downloading, file the paper into `paper/PAPER.md` via the strict-MECE hierarchy — 一级 research role (A 任务专用方法 / B 通用·预训练·多任务模型 / C 综述·实证·数据集), 二级 task(A)/范式(B)/类型(C), 三级 backbone for 预测 (Transformer / 线性·MLP / 状态空间·其他 / 模型无关增强) — and record orthogonal facets as columns (**关注属性**: 频域/通道/非平稳/不规则采样/… and **应用域**: 通用/金融), never as new 大类 (通用/金融/…, domain never becomes a 大类; e.g. finance+feature-engineering → 大类六 数据工程 · 金融); add a row with 文件名, 大小 (`ls -lh`), venue, 应用域, 方向, 关键机制, 发表日期 (arXiv first-submission), 收录日期 (顶会主会场首日 YYYY-MM-DD，arXiv 填 N/A), 来源 PDF link, then renumber sequentially and update the overview table, header count, and total size.

When writing PyTorch training code, device selection must follow the priority: **CUDA → MPS → CPU**. On macOS, `torch.backends.mps.is_available()` must be checked before falling back to CPU — never skip MPS. Use this pattern:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Also set `pin_memory=True` only when `device.type == "cuda"`; it is not supported on MPS.
