# Documentation

Reference docs for the SAE pipeline. Start here:

- **[architecture.md](architecture.md)** — System design, data flow, and the
  *why* behind the file layouts.
- **[pipeline.md](pipeline.md)** — Per-script walkthroughs (what each script
  does, its inputs and outputs, how to test it in isolation).
- **[data-formats.md](data-formats.md)** — Authoritative on-disk file specs
  (shapes, dtypes, JSON schemas) for every artefact the pipeline produces.
- **[configuration.md](configuration.md)** — Full reference for the `MONO_*`
  environment variables / `constants.py` tunables.
- **[performance.md](performance.md)** — Bottleneck analysis, per-stage time
  budgets, and tuning recipes for different hardware constraints.

For top-level usage, see [`../README.md`](../README.md). For agent-facing
guidance and project invariants, see [`../AGENTS.md`](../AGENTS.md).
