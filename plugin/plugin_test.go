package plugin

import (
	"testing"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	"github.com/maximhq/bifrost/core/schemas"
)

// sampleModels mirrors the multi-provider registry used in the core index tests.
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

func chatReq(provider, model string) *schemas.BifrostRequest {
	return &schemas.BifrostRequest{
		ChatRequest: &schemas.BifrostChatRequest{
			Provider: schemas.ModelProvider(provider),
			Model:    model,
		},
	}
}

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
// registered.
func TestPreRequestHook_RewritesToProviderName(t *testing.T) {
	p := newPlugin(t, false)
	cases := []struct {
		name, provider, input, want string
	}{
		{"zai underscore spelling", "zai", "GLM_5_2", "ZAI/glm-5.2"},
		{"zhipu space spelling", "zhipu", "glm 5 2", "智谱/GLM_5_2"},
		{"openrouter free suffix", "openrouter", "glm-5.2:free", "glm 5 2:free"},
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

// TestPreRequestHook_VariantNotCollapsed is the regression for the variant loss
// bug: a caller explicitly asking for the thinking variant must NOT be rewritten
// to the base model, even when the base was registered first and shares the
// loose canonical. The strict tier must keep the request's intent.
func TestPreRequestHook_VariantNotCollapsed(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("anthropic", "claude-4-sonnet-thinking")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "claude-4-sonnet-thinking" {
		t.Fatalf("explicit variant request collapsed to %q, want claude-4-sonnet-thinking", got)
	}
	// A request for the bare canonical spelling still gets the base model.
	req = chatReq("anthropic", "Claude-4-Sonnet")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "Claude-4-Sonnet" {
		t.Fatalf("base model request became %q, want Claude-4-Sonnet", got)
	}
}

// TestPreRequestHook_FallbacksNormalized is the regression for fallback handling:
// the routing plan's fallbacks must be normalized too, because Bifrost copies
// each fallback's model verbatim into the retry and PreLLMHook runs per-attempt
// with no record of which fallback fired.
func TestPreRequestHook_FallbacksNormalized(t *testing.T) {
	p := newPlugin(t, false)
	req := &schemas.BifrostRequest{
		ChatRequest: &schemas.BifrostChatRequest{
			Provider: "anthropic",
			Model:    "Claude-4-Sonnet",
			Fallbacks: []schemas.Fallback{
				{Provider: "zai", Model: "GLM_5_2"},
				{Provider: "deepseek", Model: "deepseek-ai/deepseek-v4-flash-fast"},
				{Provider: "zai", Model: "unregistered-model"},
			},
		},
	}
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	_, _, fbs := req.GetRequestFields()
	if fbs[0].Model != "ZAI/glm-5.2" {
		t.Fatalf("fallback[0] = %q, want ZAI/glm-5.2", fbs[0].Model)
	}
	if fbs[1].Model != "deepseek-ai/DeepSeek-V4-Flash-fast" {
		t.Fatalf("fallback[1] = %q, want deepseek-ai/DeepSeek-V4-Flash-fast", fbs[1].Model)
	}
	if fbs[2].Model != "unregistered-model" {
		t.Fatalf("unregistered fallback should be left untouched, got %q", fbs[2].Model)
	}
}

// TestPreRequestHook_FallbackAliasSafeSlice verifies the fallback slice is
// rebuilt rather than mutated in place: a second request sharing the original
// slice must still see the caller's original names.
func TestPreRequestHook_FallbackAliasSafeSlice(t *testing.T) {
	p := newPlugin(t, false)
	orig := []schemas.Fallback{{Provider: "zai", Model: "GLM_5_2"}}
	req := &schemas.BifrostRequest{ChatRequest: &schemas.BifrostChatRequest{
		Provider: "anthropic", Model: "Claude-4-Sonnet", Fallbacks: orig,
	}}
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	_, _, fbs := req.GetRequestFields()
	if fbs[0].Model != "ZAI/glm-5.2" {
		t.Fatalf("fallback not rewritten: %q", fbs[0].Model)
	}
	if orig[0].Model != "GLM_5_2" {
		t.Fatalf("caller's slice mutated in place: %q", orig[0].Model)
	}
}

// TestPreRequestHook_WrongProviderUntouched: a canonical registered by other
// providers must not leak across; the request is forwarded verbatim.
func TestPreRequestHook_WrongProviderUntouched(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("anthropic", "glm-5.2")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "glm-5.2" {
		t.Fatalf("unmatched provider should be untouched, got %q", got)
	}
}

// TestPreRequestHook_UnknownModelUntouched: no provider registered it → forward
// unchanged.
func TestPreRequestHook_UnknownModelUntouched(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "totally-unknown-model")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "totally-unknown-model" {
		t.Fatalf("unknown model should be untouched, got %q", got)
	}
}

// TestPreRequestHook_FuzzyEnabledMatch: fuzzy resolves an unknown suffix to the
// registered base model for the same provider.
func TestPreRequestHook_FuzzyEnabledMatch(t *testing.T) {
	p := newPlugin(t, true)
	req := chatReq("anthropic", "claude-4-sonnet-instruct")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "Claude-4-Sonnet" {
		t.Fatalf("fuzzy match rewrote to %q, want Claude-4-Sonnet", got)
	}
}

func TestPreRequestHook_FuzzyDisabledMiss(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("anthropic", "claude-4-sonnet-instruct")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "claude-4-sonnet-instruct" {
		t.Fatalf("fuzzy-off miss should be untouched, got %q", got)
	}
}

// TestPreRequestHook_FuzzyVersionMismatchRejected mirrors the core rule.
func TestPreRequestHook_FuzzyVersionMismatchRejected(t *testing.T) {
	p := newPlugin(t, true)
	req := chatReq("anthropic", "claude-5-sonnet")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "claude-5-sonnet" {
		t.Fatalf("version mismatch should be untouched, got %q", got)
	}
}

// TestPreRequestHook_EmptyModel and nil guards.
func TestPreRequestHook_EmptyModel(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "" {
		t.Fatalf("empty model should stay empty, got %q", got)
	}
}

func TestPreRequestHook_NilRequest(t *testing.T) {
	p := newPlugin(t, false)
	if err := p.PreRequestHook(nil, nil); err != nil {
		t.Fatalf("nil request should be a no-op, got %v", err)
	}
}

// TestPreRequestHook_EmptyFallbackForwarded: a model that empties under stripping
// (PRD 6.7) is forwarded verbatim because no provider registered the fallback
// canonical — handled in the resolve miss path, no special-casing.
func TestPreRequestHook_EmptyFallbackForwarded(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("zai", "free")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "free" {
		t.Fatalf("unregistered empty-fallback should be forwarded verbatim, got %q", got)
	}
}

// TestPreRequestHook_EmptyFallbackRegisteredStillRewrites pins the true contract:
// EmptyFallback is only diagnostic; when a provider actually registered the
// degenerate canonical, the request is still rewritten.
func TestPreRequestHook_EmptyFallbackRegisteredStillRewrites(t *testing.T) {
	p := New(DefaultConfig(), []normalizer.ProviderModel{{Provider: "zai", Original: "free"}}, nil)
	req := chatReq("zai", "Free")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "free" {
		t.Fatalf("registered empty-fallback canonical should be rewritten to \"free\", got %q", got)
	}
}

// TestNew_WildcardNotIndexed is the regression for the dangerous `*` indexing
// bug: Key.Models uses `*` to mean "allow all", so it must never become an index
// entry that could rewrite a request to the literal string "*".
func TestNew_WildcardNotIndexed(t *testing.T) {
	p := New(DefaultConfig(), []normalizer.ProviderModel{
		{Provider: "zai", Original: "*"},
		{Provider: "zai", Original: "glm-5.2"},
	}, nil)
	if p.IndexLen() != 1 {
		t.Fatalf("wildcard must be skipped; index has %d canonicals, want 1", p.IndexLen())
	}
	// A request for a model that's just the wildcard must not be rewritten.
	req := chatReq("zai", "*")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "*" {
		t.Fatalf("wildcard request should stay \"*\", got %q", got)
	}
}

// TestNew_BlankSkipped ensures empty model names don't pollute the index.
func TestNew_BlankSkipped(t *testing.T) {
	p := New(DefaultConfig(), []normalizer.ProviderModel{
		{Provider: "zai", Original: ""},
		{Provider: "zai", Original: "  "},
		{Provider: "zai", Original: "glm-5.2"},
	}, nil)
	if p.IndexLen() != 1 {
		t.Fatalf("blanks must be skipped; index has %d canonicals, want 1", p.IndexLen())
	}
}

// TestNew_NoModels builds over an empty registry: it never rewrites.
func TestNew_NoModels(t *testing.T) {
	p := New(DefaultConfig(), nil, nil)
	req := chatReq("zai", "GLM_5_2")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	if got := modelOf(req); got != "GLM_5_2" {
		t.Fatalf("empty index should leave request untouched, got %q", got)
	}
}

// TestGetNameAndCleanup pins the trivial surface.
func TestGetNameAndCleanup(t *testing.T) {
	p := newPlugin(t, false)
	if p.GetName() != PluginName {
		t.Fatalf("GetName = %q, want %q", p.GetName(), PluginName)
	}
	if err := p.Cleanup(); err != nil {
		t.Fatalf("Cleanup should be a no-op, got %v", err)
	}
}
