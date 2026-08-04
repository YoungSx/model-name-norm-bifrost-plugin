// Package plugin adapts the dependency-free normalizer core to the Bifrost
// plugin interface. It builds the model index once at startup and, on every
// request, rewrites the requested model name to the concrete name the target
// provider registered — so a caller may use any spelling of a model and still
// reach the right provider entry (PRD sections 3.2, 8, 9, 10).
//
// Wiring it into Bifrost is direct: the plugin satisfies schemas.LLMPlugin, so
// it goes in schemas.BifrostConfig.LLMPlugins. See NewFromAccount and the
// integration test in plugin/integration_test.go, which drives a real
// bifrost.Init and asserts on what the provider endpoint actually received.
package plugin

import (
	"strings"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	"github.com/maximhq/bifrost/core/schemas"
)

// PluginName is the stable identifier Bifrost uses to refer to this plugin in
// config and logs.
const PluginName = "model-name-normalization"

// Config is the plugin's full configuration. It embeds the core normalization
// config verbatim (so one YAML/JSON decode fills both) and adds the matching mode.
type Config struct {
	// Normalization is the `normalization` section of the plugin config.
	Normalization normalizer.Config `json:"normalization" yaml:"normalization"`
	// Matching controls lookup behavior (`matching` section).
	Matching MatchingConfig `json:"matching" yaml:"matching"`
}

// MatchingConfig mirrors the `matching` config block (PRD section 10).
type MatchingConfig struct {
	// Fuzzy enables the relaxed segment-prefix rule when the exact tiers miss.
	Fuzzy bool `json:"fuzzy" yaml:"fuzzy"`
}

// DefaultConfig returns the PRD-recommended plugin configuration: the default
// normalization pipeline with fuzzy matching disabled (exact-only, section 9).
func DefaultConfig() Config {
	return Config{
		Normalization: normalizer.DefaultConfig(),
		Matching:      MatchingConfig{Fuzzy: false},
	}
}

// Plugin is the Bifrost LLM plugin. After New returns it is immutable, so its
// hooks are safe for the concurrent request path without locking (the index and
// normalizer are both read-only). It satisfies schemas.LLMPlugin.
type Plugin struct {
	index  *normalizer.Index
	logger schemas.Logger
}

// compile-time assertion that Plugin implements the LLM plugin contract.
var _ schemas.LLMPlugin = (*Plugin)(nil)

// New builds the plugin: it constructs the normalizer, indexes every
// provider-registered model, and logs the collisions that affect resolution
// (PRD sections 11, 12). models is the full set of models every configured
// provider has registered; passing an empty slice yields a plugin that never
// rewrites (no index entries to match against). logger may be nil, in which case
// diagnostics are silently dropped.
//
// Names that carry no routing information are dropped here rather than indexed:
// blank entries, and the `*` wildcard that Bifrost's Key.Models whitelist uses
// to mean "any model is allowed" (schemas.WhiteList.IsUnrestricted). Indexing `*`
// as if it were a model would let a request be rewritten to the literal string
// `"*"`, which no provider can serve.
func New(cfg Config, models []normalizer.ProviderModel, logger schemas.Logger) *Plugin {
	n := normalizer.New(cfg.Normalization)
	idx := normalizer.BuildIndex(n, models, cfg.Matching.Fuzzy)

	p := &Plugin{index: idx, logger: logger}
	for _, c := range idx.Conflicts() {
		if len(c.Providers) > 1 {
			// PRD 12: a shared canonical is permitted (never overwritten) but must
			// be surfaced so operators know a request's provider decides the winner.
			p.logf(logWarn, "canonical model served by multiple providers",
				"canonical", c.Canonical, "providers", strings.Join(c.Providers, ","))
		}
		if len(c.Shadowed) > 0 {
			// Distinct models of one provider collapsed onto one canonical. They
			// remain reachable by their own spellings (strict tier); only a bare
			// canonical request is ambiguous, and resolves to the first registered.
			p.logf(logWarn, "canonical model shadows provider variants",
				"canonical", c.Canonical, "shadowed", strings.Join(c.Shadowed, ","))
		}
	}
	return p
}

// GetName returns the plugin's stable name.
func (p *Plugin) GetName() string { return PluginName }

// Cleanup releases resources. The plugin holds only immutable in-memory state,
// so there is nothing to free.
func (p *Plugin) Cleanup() error { return nil }

// IndexLen reports how many distinct canonical names the startup index holds. It
// is a read-only diagnostic — useful for startup logging and smoke checks — and
// does not expose the index itself, keeping the plugin's state encapsulated.
func (p *Plugin) IndexLen() int { return p.index.Len() }

// PreRequestHook is the canonical routing phase (PRD section 3.2): it runs once
// per request, before any fan-out, and its mutations are committed for every
// downstream plugin, the provider call, and all fallbacks. Here we resolve the
// requested model — and every declared fallback — to the exact name the target
// provider registered.
//
// Fallbacks are rewritten too, and must be: each fallback carries its own
// provider and model, and Bifrost's prepareFallbackRequest copies that model
// verbatim into the retry. PreLLMHook (which does run per attempt) cannot fix it
// either, because by then the loosely-spelled name is already the request's
// model with no record of which fallback it came from. Normalizing the whole
// routing plan here is the only place it holds for every attempt.
//
// The hook is deliberately conservative: a request is only rewritten on a hit
// for its own provider. A miss leaves it untouched so an unknown model still
// reaches the provider verbatim (and fails there with the provider's own error,
// not ours). Per the interface contract a returned error is non-blocking, so we
// never fail a request from normalization; we only log.
func (p *Plugin) PreRequestHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	if req == nil {
		return nil
	}
	provider, model, fallbacks := req.GetRequestFields()

	if model != "" {
		p.rewriteModel(req, string(provider), model)
	}
	p.rewriteFallbacks(req, fallbacks)
	return nil
}

// rewriteModel resolves the primary model and writes it back when it changed.
func (p *Plugin) rewriteModel(req *schemas.BifrostRequest, provider, model string) {
	res := p.index.ResolveForProvider(model, provider)
	p.logResolution(provider, model, res)
	if !res.OK() || res.Model.Original == model {
		return
	}

	req.SetModel(res.Model.Original)

	// SetModel is a type switch that does not cover every request type
	// GetRequestFields reports a model for (e.g. PassthroughRequest). Read back
	// rather than assume: a silent no-op here would otherwise look like a
	// successful rewrite in the logs.
	if _, after, _ := req.GetRequestFields(); after != res.Model.Original {
		p.logf(logDebug, "model rewrite not applied by request type",
			"provider", provider, "original", model, "target", res.Model.Original)
	}
}

// rewriteFallbacks resolves each fallback against its own provider. A new slice
// is built and installed only when something changed, so the caller's slice is
// never mutated in place (Bifrost shallow-copies the request per fallback
// attempt, so an aliased slice would be observed by every attempt).
func (p *Plugin) rewriteFallbacks(req *schemas.BifrostRequest, fallbacks []schemas.Fallback) {
	if len(fallbacks) == 0 {
		return
	}
	var updated []schemas.Fallback
	for i, fb := range fallbacks {
		if fb.Model == "" {
			continue
		}
		res := p.index.ResolveForProvider(fb.Model, string(fb.Provider))
		p.logResolution(string(fb.Provider), fb.Model, res)
		if !res.OK() || res.Model.Original == fb.Model {
			continue
		}
		if updated == nil {
			updated = make([]schemas.Fallback, len(fallbacks))
			copy(updated, fallbacks)
		}
		updated[i].Model = res.Model.Original
	}
	if updated != nil {
		req.SetFallbacks(updated)
	}
}

// logResolution emits the diagnostics for one resolution attempt: the PRD 6.7
// empty-fallback warning, a fuzzy-match notice, and a debug line on a miss
// (common and expected, so not a warning).
func (p *Plugin) logResolution(provider, model string, res normalizer.Resolution) {
	if res.Request.EmptyFallback {
		p.logf(logWarn, "normalization fallback",
			"original", res.Request.Original, "fallback", res.Request.Canonical)
	}
	switch res.Type {
	case normalizer.MatchNone:
		p.logf(logDebug, "no model match",
			"provider", provider, "original", model, "canonical", res.Request.Canonical)
	case normalizer.MatchFuzzy:
		p.logf(logInfo, "fuzzy model match",
			"provider", provider, "input", model,
			"canonical", res.Request.Canonical, "matched", res.MatchedKey,
			"target", res.Model.Original)
	}
}

// PreLLMHook is a no-op: all routing is committed in PreRequestHook, which the
// pipeline runs exactly once and observes everywhere — including on every
// fallback attempt, since PreRequestHook rewrites the fallback plan itself.
func (p *Plugin) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

// PostLLMHook is a no-op: normalization affects only the outbound request.
func (p *Plugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

// logLevel is a tiny internal enum so logf can route to the right Logger method
// without callers importing schemas log levels.
type logLevel int

const (
	logDebug logLevel = iota
	logInfo
	logWarn
)

// logf dispatches a structured log line to the configured logger, tolerating a
// nil logger (diagnostics are simply dropped).
func (p *Plugin) logf(level logLevel, msg string, args ...any) {
	if p.logger == nil {
		return
	}
	switch level {
	case logDebug:
		p.logger.Debug(msg, args...)
	case logInfo:
		p.logger.Info(msg, args...)
	case logWarn:
		p.logger.Warn(msg, args...)
	}
}
