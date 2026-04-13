# Current Results Summary

This file is a lightweight running summary of the current recurrent-value experiments.
Replace bracketed placeholders with numbers or short notes as runs finish.

## Scope

- Date: `[YYYY-MM-DD]`
- Author: `[name]`
- Repo commit / branch: `[commit-or-branch]`
- Benchmark family: `[SAW / CGIVL / mixed]`
- Primary goal: `[e.g. determine whether recur_tied + FiLM beats MLP baseline]`

## Success Metrics

### SAW

| Task | Run |overall success| Best seed 0 | Best seed 1 | Best seed 2 | Best seed 3|
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `antmaze-large-navigate` | `[baseline]` | `[90+-3]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-large-navigate` | `[recurrent]` | `[ ]` | `[94.4]` | `[93.6]` | `[93.2]` | `[ ]` |

| `antmaze-giant-navigate` | `[baseline]` | `[73+-4]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-giant-navigate` | `[recurrent]` | `[ ]` | `[844]` | `[ ]` | `[ ]` | `[ ]` |

| `humanoidmaze-giant-navigate` | `[baseline]` | `[35+-4]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `humanoidmaze-giant-navigate` | `[recurrent]` | `[ ]` | `[42.8]` | `[ ]` | `[ ]` | `[ ]` |

### HIQL

| Task | Run |overall success| Best seed 0 | Best seed 1 | Best seed 2 | Best seed 3|
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `antmaze-large-stitch` | `[baseline]` | `[67+-5]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-large-stitch` | `[recurrent]` | `[ ]` | `[92]` | `[ ]` | `[ ]` | `[ ]` |

### CGIVL

| Task | Run |overall success| Best seed 0 | Best seed 1 | Best seed 2 | Best seed 3|
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `antmaze-large-stitch` | `[baseline]` | `[ ]` | `[66.4]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-large-stitch` | `[recurrent]` | `[ ]` | `[84.8]` | `[ ]` | `[ ]` | `[ ]` |

### CRL

| Task | Run |overall success| Best seed 0 | Best seed 1 | Best seed 2 | Best seed 3|
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `antmaze-large-stitch` | `[baseline]` | `[11+-2]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-large-stitch` | `[recurrent]` | `[ ]` | `[48]` | `[ ]` | `[ ]` | `[ ]` |

| `antmaze-giant-navigate` | `[baseline]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `antmaze-giant-navigate` | `[recurrent]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |





## Appendix: Commands

### SAW

```bash
[paste command]
```

### CGIVL / CGCIVL

```bash
[paste command]
```
