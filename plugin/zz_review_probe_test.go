package plugin

import (
	"testing"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	"github.com/maximhq/bifrost/core/schemas"
)

// Probe 1: a provider registering both a base model and a suffixed variant.
// Does an explicit request for the variant survive?
func TestProbe_VariantCollapse(t *testing.T) {
	p := newPlugin(t, false)
	req := chatReq("anthropic", "claude-4-sonnet-thinking")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	t.Logf("requested %q -> got %q", "claude-4-sonnet-thinking", modelOf(req))
	if got := modelOf(req); got != "claude-4-sonnet-thinking" {
		t.Errorf("VARIANT LOST: explicit request for the thinking variant became %q", got)
	}
}

// Probe 2: are routing fallbacks normalized?
func TestProbe_Fallbacks(t *testing.T) {
	p := newPlugin(t, false)
	req := &schemas.BifrostRequest{
		ChatRequest: &schemas.BifrostChatRequest{
			Provider: "anthropic",
			Model:    "Claude-4-Sonnet",
			Fallbacks: []schemas.Fallback{
				{Provider: "zai", Model: "GLM_5_2"},
			},
		},
	}
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	_, _, fbs := req.GetRequestFields()
	t.Logf("fallback after hook: %+v", fbs)
	if fbs[0].Model != "ZAI/glm-5.2" {
		t.Errorf("FALLBACK NOT NORMALIZED: got %q, want %q", fbs[0].Model, "ZAI/glm-5.2")
	}
}

// Probe 3: whitelist wildcard "*" from Key.Models pollutes the index?
func TestProbe_WildcardWhitelist(t *testing.T) {
	models := []normalizer.ProviderModel{
		{Provider: "zai", Original: "*"},
		{Provider: "zai", Original: "glm-5.2"},
	}
	p := New(DefaultConfig(), models, nil)
	t.Logf("index length with wildcard entry: %d", p.IndexLen())
	if p.IndexLen() != 1 {
		t.Errorf("WILDCARD INDEXED: index holds %d canonicals, wildcard should be skipped", p.IndexLen())
	}
}

// Probe 4: same-provider canonical collision — is it reported as a conflict?
func TestProbe_SameProviderCollisionWarning(t *testing.T) {
	n := normalizer.New(normalizer.DefaultConfig())
	_, conflicts := normalizer.BuildIndex(n, []normalizer.ProviderModel{
		{Provider: "anthropic", Original: "claude-4-sonnet"},
		{Provider: "anthropic", Original: "claude-4-sonnet-thinking"},
	}, false)
	t.Logf("conflicts reported: %+v", conflicts)
	if len(conflicts) == 0 {
		t.Error("SILENT MERGE: two distinct anthropic models collapsed to one canonical with no warning")
	}
}
