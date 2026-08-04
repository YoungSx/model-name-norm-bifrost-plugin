// Package plugin adapts the dependency-free normalizer core to the Bifrost
// plugin interface. It builds the canonical index once at startup and, on every
// request, rewrites the requested model name to the concrete name the target
// provider registered — so a caller may use any spelling of a model and still
// reach the right provider entry (PRD sections 3.2, 8, 9, 10).
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
// config verbatim (so one YAML decode fills both) and adds the matching mode.
type Config struct {
	// Normalization is the `normalization` section of the plugin YAML.
	Normalization normalizer.Config `json:"normalization" yaml:"normalization"`
	// Matching controls lookup behavior (`matching` section).
	Matching MatchingConfig `json:"matching" yaml:"matching"`
}

// MatchingConfig mirrors the `matching` YAML block (PRD section 10).
type MatchingConfig struct {
	// Fuzzy enables the relaxed segment-prefix rule when an exact match misses.
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
	normalizer *normalizer.Normalizer
	index      *normalizer.Index
	logger     schemas.Logger
}

// compile-time assertion that Plugin implements the LLM plugin contract.
var _ schemas.LLMPlugin = (*Plugin)(nil)

// New builds the plugin: it constructs the normalizer, normalizes every
// provider-registered model to assemble the canonical index, and logs any
// cross-provider canonical conflicts (PRD sections 11, 12). models is the full
// set of models every configured provider has registered; passing an empty
// slice yields a plugin that normalizes requests but never rewrites them (no
// index entries to match against). logger may be nil, in which case diagnostics
// are silently dropped.
func New(cfg Config, models []normalizer.ProviderModel, logger schemas.Logger) *Plugin {
	n := normalizer.New(cfg.Normalization)
	idx, conflicts := normalizer.BuildIndex(n, models, cfg.Matching.Fuzzy)

	p := &Plugin{normalizer: n, index: idx, logger: logger}
	for _, c := range conflicts {
		// PRD 12: a shared canonical is permitted (never overwritten) but must be
		// surfaced so operators know a request's provider decides the winner.
		p.logf(logWarn, "duplicated canonical model",
			"canonical", c.Canonical,
			"providers", strings.Join(c.Providers, ","))
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
// downstream plugin, the provider call, and all fallbacks. Here we normalize the
// requested model and, when it resolves to a model the target provider actually
// registered, rewrite the request to that provider's exact name.
//
// The hook is deliberately conservative: a request is only rewritten on a hit
// for its own provider. A miss or an empty model leaves the request untouched so
// an unknown model still reaches the provider verbatim (and fails there with the
// provider's own error, not ours). A normalization fallback (PRD 6.7) is warned
// about but does not by itself suppress rewriting: its degenerate canonical (the
// original token, e.g. "free") is still matched, so a request that only differed
// by case or separators from a provider's literally-registered model is correctly
// resolved. Per the interface contract a returned error is non-blocking, so we
// never fail a request from normalization; we only log.
func (p *Plugin) PreRequestHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	if req == nil {
		return nil
	}
	provider, model, _ := req.GetRequestFields()
	if model == "" {
		return nil // model-less request type (e.g. list-models); nothing to do.
	}

	res := p.normalizer.Normalize(model)
	if res.EmptyFallback {
		// PRD 6.7 / 12: suffix stripping consumed the whole name; warn and leave
		// the request as-is rather than routing on a degenerate canonical.
		p.logf(logWarn, "normalization fallback",
			"original", res.Original, "fallback", res.Canonical)
	}

	pm, matched, mt, ok := p.index.MatchForProvider(res.Canonical, string(provider))
	if !ok {
		// No registered model for this provider under the canonical: forward the
		// request unchanged. Common and expected, so logged at debug only.
		p.logf(logDebug, "no canonical match",
			"provider", string(provider), "original", model, "canonical", res.Canonical)
		return nil
	}

	if mt == normalizer.MatchFuzzy {
		p.logf(logInfo, "fuzzy model match",
			"input", model, "canonical", res.Canonical, "matched", matched)
	}

	// Rewrite to the exact name the provider registered. Skip the write when it
	// already equals the request to avoid a pointless mutation.
	if pm.Original != model {
		req.SetModel(pm.Original)
	}
	return nil
}

// PreLLMHook is a no-op: all routing is committed in PreRequestHook, which the
// pipeline runs exactly once and observes everywhere. Doing the rewrite here
// instead would repeat on every fallback attempt for no benefit.
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
