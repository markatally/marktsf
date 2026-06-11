# External Research Code

`external/` is git-ignored because it contains upstream research repositories and
local experiment outputs. Recreate it with:

```bash
scripts/sync_external.sh
```

By default the script checks out pinned revisions for reproducibility:

| Local path | Upstream | Branch | Pinned revision | Use |
|------------|----------|--------|-----------------|-----|
| `external/TSLib` | `https://github.com/thuml/Time-Series-Library.git` | `main` | `4e938a1` | Official baseline runners for DLinear, PatchTST, TiDE, TimeXer, and related models. |
| `external/ShiftingTime` | `https://github.com/srinathdama/ShiftingTime.git` | `main` | `3f9be2a` | ShiftingTime reference implementation. |

To intentionally refresh both repositories to their upstream `main` branches:

```bash
scripts/sync_external.sh --latest
```

Keep generated outputs, checkpoints, and local edits inside `external/`
untracked unless there is a deliberate reason to promote a small reproducibility
artifact into this repository.
