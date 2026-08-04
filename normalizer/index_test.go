package normalizer

import (
	"sort"
	"testing"
)

// sampleModels is the multi-provider registry used across index tests: the same
// logical model registered under three different naming conventions, plus a few
// distinct models — one of which (claude-4-sonnet / -thinking) exercises the
// variant-shadowing a single-loose-key index must handle.
func sampleModels() []ProviderModel {
	return []ProviderModel{
		{Provider: "zai", Original: "ZAI/glm-5.2"},
		{Provider: "zhipu", Original: "智谱/GLM_5_2"},
		{Provider: "openrouter", Original: "glm 5 2:free"},
		{Provider: "anthropic", Original: "Claude-4-Sonnet"},
		{Provider: "anthropic", Original: "claude-4-sonnet-thinking"},
		{Provider: "deepseek", Original: "deepseek-ai/DeepSeek-V4-Flash-fast"},
	}
}

func buildTestIndex(t *testing.T, fuzzy bool) *Index {
	t.Helper()
	n := New(DefaultConfig())
	return BuildIndex(n, sampleModels(), fuzzy)
}

// TestBuildIndex_Canonicalization pins the loose tier: glm-5.2 collapses three
// providers into one bucket, and the stacked-suffix deepseek keeps "flash".
func TestBuildIndex_Canonicalization(t *testing.T) {
	idx := buildTestIndex(t, false)

	bucket := idx.loose["glm-5.2"]
	if len(bucket) != 3 {
		t.Fatalf("glm-5.2: got %d provider models, want 3: %+v", len(bucket), bucket)
	}
	if len(idx.loose["deepseek-v4-flash"]) != 1 {
		t.Fatalf("deepseek-v4-flash should be one entry, got %+v", idx.loose["deepseek-v4-flash"])
	}
}

// TestBuildIndex_ConflictDetection verifies the cross-provider canonical is
// reported, and that anthropic's two models (base + thinking) are reported as a
// single-provider shadowing rather than a second cross-provider conflict.
func TestBuildIndex_ConflictDetection(t *testing.T) {
	idx := buildTestIndex(t, false)

	conflicts := idx.Conflicts()
	var multi []Conflict
	for _, c := range conflicts {
		if len(c.Providers) > 1 {
			multi = append(multi, c)
		}
	}
	if len(multi) != 1 {
		t.Fatalf("got %d cross-provider conflicts, want 1: %+v", len(multi), conflicts)
	}
	c := multi[0]
	if c.Canonical != "glm-5.2" {
		t.Fatalf("conflict canonical = %q, want glm-5.2", c.Canonical)
	}
	got := append([]string(nil), c.Providers...)
	sort.Strings(got)
	want := []string{"openrouter", "zai", "zhipu"}
	if !equalStrings(got, want) {
		t.Fatalf("conflict providers = %v, want %v", got, want)
	}

	// anthropic base + variant must surface as a shadowing warning, not a second
	// cross-provider conflict.
	var shadow []Conflict
	for _, cc := range conflicts {
		if len(cc.Shadowed) > 0 {
			shadow = append(shadow, cc)
		}
	}
	if len(shadow) != 1 || shadow[0].Canonical != "claude-4-sonnet" {
		t.Fatalf("expected one shadowing conflict on claude-4-sonnet, got %+v", shadow)
	}
	if shadow[0].Shadowed[0] != "claude-4-sonnet-thinking" {
		t.Fatalf("expected thinking to be shadowed, got %v", shadow[0].Shadowed)
	}
}

// TestBuildIndex_SameProviderDuplicateSpelling must NOT be a conflict: one
// provider registering two spellings of the same model (same strict key) is a
// legit alias load-balancing setup, flagged only as benign dedup.
func TestBuildIndex_SameProviderDuplicateSpelling(t *testing.T) {
	n := New(DefaultConfig())
	models := []ProviderModel{
		{Provider: "zai", Original: "GLM_5_2"},
		{Provider: "zai", Original: "glm-5-2"}, // same strict key after version norm
	}
	idx := BuildIndex(n, models, false)
	for _, c := range idx.Conflicts() {
		if c.Canonical == "glm-5.2" {
			t.Fatalf("duplicate spelling of one model should not conflict, got %+v", c)
		}
	}
}

// TestResolveForProvider_StrictWinsRotationBack is the regression for the variant
// collapse: an explicit request for the thinking variant must resolve to it,
// even though the loose key is shared with the base model that was registered
// first.
func TestResolveForProvider_StrictWinsVariant(t *testing.T) {
	idx := buildTestIndex(t, false)
	res := idx.ResolveForProvider("claude-4-sonnet-thinking", "anthropic")
	if !res.OK() || res.Type != MatchStrict {
		t.Fatalf("variant: ok=%v type=%s, want strict hit", res.OK(), res.Type)
	}
	if res.Model.Original != "claude-4-sonnet-thinking" {
		t.Fatalf("variant resolved to %q, want claude-4-sonnet-thinking", res.Model.Original)
	}

	// A bare canonical request still gets the base model (first registered).
	res = idx.ResolveForProvider("claude-4-sonnet", "anthropic")
	if res.Model.Original != "Claude-4-Sonnet" {
		t.Fatalf("base canonical resolved to %q, want Claude-4-Sonnet", res.Model.Original)
	}
}

// TestResolveForProvider_CrossSpellingAliasResolution is the PRD headline: an
// arbitrary spelling finds the target provider's registered name. Each of the
// three glm spellings resolves to its own provider's original.
func TestResolveForProvider_CrossSpellingAliasResolution(t *testing.T) {
	idx := buildTestIndex(t, false)
	cases := []struct {
		provider, in, want string
	}{
		{"zai", "GLM_5_2", "ZAI/glm-5.2"},
		{"zhipu", "glm 5 2", "智谱/GLM_5_2"},
		{"openrouter", "glm-5.2:free", "glm 5 2:free"},
		{"deepseek", "deepseek-ai/deepseek-v4-flash-fast", "deepseek-ai/DeepSeek-V4-Flash-fast"},
	}
	for _, c := range cases {
		res := idx.ResolveForProvider(c.in, c.provider)
		if !res.OK() || res.Model.Original != c.want {
			t.Fatalf("ResolveForProvider(%q,%q) = %+v, want %q", c.in, c.provider, res, c.want)
		}
	}
}

// TestResolveForProvider_NotInBucketMiss: the canonical exists but not for this
// provider — must miss so the request is left untouched.
func TestResolveForProvider_NotInBucketMiss(t *testing.T) {
	idx := buildTestIndex(t, false)
	// glm-5.2 has zai/zhipu/openrouter, not anthropic.
	if res := idx.ResolveForProvider("glm-5.2", "anthropic"); res.OK() {
		t.Fatal("glm-5.2 for anthropic should miss")
	}
}

// TestResolveForProvider_AbsentMiss and the empty model guard.
func TestResolveForProvider_AbsentMiss(t *testing.T) {
	idx := buildTestIndex(t, false)
	if res := idx.ResolveForProvider("does-not-exist", "zai"); res.OK() {
		t.Fatal("absent canonical should miss")
	}
	if res := idx.ResolveForProvider("", "zai"); res.OK() {
		t.Fatal("empty model should miss")
	}
	if res := idx.ResolveForProvider("   ", "zai"); res.OK() {
		t.Fatal("whitespace model should miss")
	}
}

// TestResolveForProvider_FuzzyScopedToProvider is the regression guard for the
// provider-scoped fuzzy rule. Two providers offer segment-prefix candidates for
// "gpt-5-mini": p1's "gpt-5" (gap 1) and p2's "gpt-5-mini-preview" (gap 1). The
// global-best key is "gpt-5", which belongs to p1. A naive "pick global best,
// then filter by provider" implementation would report a false miss for p2.
func TestResolveForProvider_FuzzyScopedToProvider(t *testing.T) {
	n := New(DefaultConfig())
	models := []ProviderModel{
		{Provider: "p1", Original: "gpt-5"},
		{Provider: "p2", Original: "gpt-5-mini-preview"},
	}
	idx := BuildIndex(n, models, true)

	res := idx.ResolveForProvider("gpt-5-mini", "p2")
	if !res.OK() || res.Type != MatchFuzzy || res.Model.Original != "gpt-5-mini-preview" {
		t.Fatalf("p2 gpt-5-mini: %+v, want fuzzy gpt-5-mini-preview", res)
	}
	res = idx.ResolveForProvider("gpt-5-mini", "p1")
	if !res.OK() || res.Model.Original != "gpt-5" {
		t.Fatalf("p1 gpt-5-mini: %+v, want gpt-5", res)
	}
	if res := idx.ResolveForProvider("gpt-5-mini", "p3"); res.OK() {
		t.Fatal("p3 has no candidate; must miss")
	}
}

// TestResolveForProvider_FuzzyRejectsVersionMismatch: claude-5-sonnet must never
// resolve to claude-4-sonnet.
func TestResolveForProvider_FuzzyRejectsVersionMismatch(t *testing.T) {
	idx := buildTestIndex(t, true)
	if res := idx.ResolveForProvider("claude-5-sonnet", "anthropic"); res.OK() {
		t.Fatalf("claude-5-sonnet should miss, got %+v", res)
	}
}

// TestResolveForProvider_FuzzyDisabled by default.
func TestResolveForProvider_FuzzyDisabled(t *testing.T) {
	idx := buildTestIndex(t, false)
	if res := idx.ResolveForProvider("claude-4-sonnet-instruct", "anthropic"); res.OK() {
		t.Fatalf("fuzzy-off miss should not resolve, got %+v", res)
	}
}

// TestBuildIndex_SkipsUnroutableNames documents the build-time hygiene that
// matches ModelsFromAccount: blank names and the whitelist wildcard `*` are not
// models and must never become index keys.
func TestBuildIndex_SkipsUnroutableNames(t *testing.T) {
	n := New(DefaultConfig())
	models := []ProviderModel{
		{Provider: "zai", Original: "*"},
		{Provider: "zai", Original: "  "},
		{Provider: "zai", Original: "glm-5.2"},
	}
	idx := BuildIndex(n, models, false)
	if idx.Len() != 1 {
		t.Fatalf("expected 1 canonical (glm-5.2), got %d", idx.Len())
	}
	if _, ok := idx.loose["*"]; ok {
		t.Fatal("wildcard '*' must never be indexed as a model")
	}
}

// TestIndex_Len pins the documented distinct-canonical count for the sample.
func TestIndex_Len(t *testing.T) {
	idx := buildTestIndex(t, false)
	// glm-5.2 (zai+zhipu+openrouter), claude-4-sonnet (base + thinking),
	// deepseek-v4-flash = 3.
	if idx.Len() != 3 {
		t.Fatalf("index length = %d, want 3", idx.Len())
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
