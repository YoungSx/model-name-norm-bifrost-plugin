package plugin

import (
	"testing"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	"github.com/maximhq/bifrost/core/schemas"
)

// sampleModels mirrors the multi-provider registry used in the core index tests:
// the same logical model under three spellings, plus a couple of distinct ones.
func sampleModels() []normalizer.ProviderModel {
	return []normalizer.ProviderModel{
		{Provider: "zai", Original: "ZAI/glm-5.2"},
		{Provider: "zhipu", Original: "智谱/GLM_5_2"},
		{Provider: "openrouter", Original: "glm 5 2:free"},
		{Provider: "anthropic", Original: "Claude-4-Sonnet"},
		{Provider: "anthropic", Original: "claude-4-sonnet-thinking"},
		{Provider: "deepseek", Original: "deepseek-ai/DeepSeek-V4-Flash-fast"},
	}
}

// chatReq wraps a provider/model pair in the BifrostRequest shape the hook reads.
func chatReq(provider, model string) *schemas.BifrostRequest {
	return &schemas.BifrostRequest{
		ChatRequest: &schemas.BifrostChatRequest{
			Provider: schemas.ModelProvider(provider),
			Model:    model,
		},
	}
}

// modelOf reads back the (possibly rewritten) model from a request.
func modelOf(req *schemas.BifrostRequest) string {
	_, m, _ := req.GetRequestFields()
	return m
}

func newPlugin(t *testing.T, fuzzy bool) *Plugin {
	t.Helper()
	cfg := DefaultConfig()
	cfg.Matching.Fuzzy = fuzzy
	return New(cfg, sampleModels(), nil)
}

// TestPreRequestHook_RewritesToProviderName is the core PRD behavior: a caller's
// arbitrary spelling is rewritten to the exact name the target provider
// registered, so the downstream provider call uses a name it recognizes.
func TestPreRequestHook_RewritesToProviderName(t *testing.T) {
	p := newPlugin(t, false)

	cases := []struct {
		name     string
		provider string
		input    string
		want     string
	}{
		// All three spellings canonicalize to glm-5.2; each provider gets its own.
		{"zai underscore spelling", "zai", "GLM_5_2", "ZAI/glm-5.2"},
		{"zhipu space spelling", "zhipu", "glm 5 2", "智谱/GLM_5_2"},
		{"openrouter free suffix", "openrouter", "glm-5.2:free", "glm 5 2:free"},
		// Distinct model, single provider.
		{"deepseek stacked suffix", "deepseek", "deepseek-ai/deepseek-v4-flash-fast", "deepseek-ai/DeepSeek-V4-Flash-fast"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := chatReq(c.provider, c.input)
			if err := p.PreRequestHook(nil, req); err != nil {
				t.Fatalf("PreRequestHook error: %v", err)
			}
			if got := modelOf(req); got != c.want {
				t.Fatalf("model rewritten to %q, want %q", got, c.want)
			}
		})
	}
}

// TestPreRequestHook_WrongProviderUntouched verifies the hook only rewrites on a
// hit for the request's own provider: a canonical registered by other providers
// must not leak across, the request is forwarded verbatim.
func TestPreRequestHook_WrongProviderUntouched(t *testing.T) {
	p := newPlugin(t, false)
	// glm-5.2 exists for zai/zhipu/openrouter but not for anthropic.
	req := chatReq("anthropic", "glm-5.2")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "glm-5.2" {
		t.Fatalf("unmatched provider should be untouched, got %q", got)
	}
}

// TestPreRequestHook_UnknownModelUntouched confirms a model no provider
// registered is forwarded unchanged (fails downstream with the provider's own
// error, not ours).
func TestPreRequestHook_UnknownModelUntouched(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "totally-unknown-model")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "totally-unknown-model" {
		t.Fatalf("unknown model should be untouched, got %q", got)
	}
}

// TestPreRequestHook_FuzzyDisabledMiss checks that a suffixed request that isn't
// itself an index key misses when fuzzy is off, leaving the request untouched.
func TestPreRequestHook_FuzzyDisabledMiss(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("anthropic", "claude-4-sonnet-instruct")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "claude-4-sonnet-instruct" {
		t.Fatalf("fuzzy-off miss should be untouched, got %q", got)
	}
}

// TestPreRequestHook_FuzzyEnabledMatch verifies fuzzy segment-prefix routing
// resolves an unknown suffix to the registered base model for the same provider.
func TestPreRequestHook_FuzzyEnabledMatch(t *testing.T) {
	p := newPlugin(t, true)
	req := chatReq("anthropic", "claude-4-sonnet-instruct")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	// claude-4-sonnet-instruct → canonical claude-4-sonnet-instruct, fuzzy-matches
	// index key claude-4-sonnet whose anthropic entry is "Claude-4-Sonnet".
	if got := modelOf(req); got != "Claude-4-Sonnet" {
		t.Fatalf("fuzzy match rewrote to %q, want Claude-4-Sonnet", got)
	}
}

// TestPreRequestHook_EmptyModel guards the model-less request type: nothing to
// normalize, no panic, no mutation.
func TestPreRequestHook_EmptyModel(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "" {
		t.Fatalf("empty model should stay empty, got %q", got)
	}
}

// TestPreRequestHook_NilRequest ensures a nil request is tolerated.
func TestPreRequestHook_NilRequest(t *testing.T) {
	p := newPlugin(t, false)
	if err := p.PreRequestHook(nil, nil); err != nil {
		t.Fatalf("nil request should be a no-op, got %v", err)
	}
}

// TestPreRequestHook_EmptyFallbackUnregisteredForwarded covers the common
// empty-fallback case (PRD 6.7): a model that reduces to a degenerate canonical
// (here "free", where suffix stripping consumed the whole name) is forwarded
// verbatim — not because the fallback is special, but because no provider in the
// sample registry registered "free", so the lookup misses.
func TestPreRequestHook_EmptyFallbackUnregisteredForwarded(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "free")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "free" {
		t.Fatalf("unregistered empty-fallback model should be forwarded verbatim, got %q", got)
	}
}

// TestPreRequestHook_EmptyFallbackRegisteredStillRewrites pins the true contract:
// EmptyFallback is only a diagnostic flag, it does NOT suppress rewriting. When a
// provider actually registered the degenerate canonical, the request is rewritten
// to that provider's exact spelling like any other hit. Here "zai" registers the
// literal "free"; a request for "Free" normalizes (lowercase) to canonical "free",
// hits, and is rewritten — proving the hook does not special-case the fallback.
func TestPreRequestHook_EmptyFallbackRegisteredStillRewrites(t *testing.T) {
	models := []normalizer.ProviderModel{{Provider: "zai", Original: "free"}}
	p := New(DefaultConfig(), models, nil)
	req := chatReq("zai", "Free")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "free" {
		t.Fatalf("registered empty-fallback canonical should be rewritten to \"free\", got %q", got)
	}
}

// TestNew_NoModels builds a plugin over an empty registry: it must normalize but
// never rewrite, since there is nothing to match against.
func TestNew_NoModels(t *testing.T) {
	p := New(DefaultConfig(), nil, nil)
	req := chatReq("zai", "GLM_5_2")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "GLM_5_2" {
		t.Fatalf("empty index should leave request untouched, got %q", got)
	}
}

// TestGetNameAndCleanup pins the trivial BasePlugin surface.
func TestGetNameAndCleanup(t *testing.T) {
	p := newPlugin(t, false)
	if p.GetName() != PluginName {
		t.Fatalf("GetName = %q, want %q", p.GetName(), PluginName)
	}
	if err := p.Cleanup(); err != nil {
		t.Fatalf("Cleanup should be a no-op, got %v", err)
	}
}
