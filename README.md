# model-name-norm-bifrost-plugin

A [Bifrost](https://github.com/maximhq/bifrost) LLM plugin that rewrites a
request's model name to the exact name the target provider registered — so a
caller may use any spelling of a model (`GLM_5_2`, `glm 5 2`, `glm-5.2:free`)
and still reach the right provider entry.

## What it does

At startup it builds a canonical index from the models each provider's keys
declare. On every request (`PreRequestHook`, once per top-level request and
observed by every fallback attempt) it resolves the requested model — and each
declared fallback — against the target provider's own registered spellings and
rewrites the request in place. Unknown models are forwarded verbatim so they
fail at the provider with the provider's own error, not ours.

Resolution runs in three tiers, most precise first:

1. **Strict** — case/separator/prefix only. Preserves the caller's exact
   variant, so `claude-4-sonnet-thinking` never collapses to the base model.
2. **Canonical** — suffix-stripped. The PRD's headline behavior: `GLM_5_2`
   finds `ZAI/glm-5.2`.
3. **Fuzzy** (opt-in) — relaxed segment-prefix rule, scoped to the request's
   own provider.

## Wiring it into Bifrost

The plugin satisfies `schemas.LLMPlugin` and goes straight into
`schemas.BifrostConfig.LLMPlugins`:

```go
import (
    bifrost "github.com/maximhq/bifrost/core"
    "github.com/maximhq/bifrost/core/schemas"
    "github.com/YoungSx/model-name-norm-bifrost-plugin/plugin"
)

p, err := plugin.NewFromAccount(ctx, plugin.DefaultConfig(), account, logger)
if err != nil { return err }
bf, err := bifrost.Init(ctx, schemas.BifrostConfig{
    Account:    account,
    LLMPlugins: []schemas.LLMPlugin{p},
    Logger:     logger,
})
```

No `.so` build, no special loader: it is a normal Go dependency. (If you later
want to distribute it as a precompiled shared object, the
`-buildmode=plugin` artifact is a drop-in, but it is **not** required — the
in-process registration above is the supported path and what the tests
exercise.)

## Verified

- `go test ./...` — unit tests for both tiers, conflicts, fallback handling,
  wildcard/blacklist hygiene, fuzzy scoping, concurrency (`-race`).
- `go test -tags integration ./plugin` — drives a real `bifrost.Init` against
  a local echo server and asserts the downstream `model` field is rewritten to
  the provider's registered name (alias case) and preserved (variant case).
- `go run ./cmd/smoke` — end-to-end sanity over the production constructor.

## Packages

- `normalizer` — dependency-free canonicalization core + index.
- `plugin` — Bifrost adapter (`LLMPlugin`, account bridge).
- `cmd/smoke` — runnable end-to-end check.
