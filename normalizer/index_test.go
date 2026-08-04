package normalizer

import (
	"reflect"
	"sort"
	"testing"
)

// sampleModels is the multi-provider registry used across index tests: the same
// logical model registered under three different naming conventions, plus a few
// distinct models.
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

func buildTestIndex(t *testing.T, fuzzy bool) (*Index, []Conflict) {
	t.Helper()
	n := New(DefaultConfig())
	return BuildIndex(n, sampleModels(), fuzzy)
}

func TestBuildIndex_Canonicalization(t *testing.T) {
	idx, _ := buildTestIndex(t, false)

	// glm-5.2 collapses three providers into one canonical bucket.
	got, matched, mt := idx.Match("glm-5.2")
	if mt != MatchExact || matched != "glm-5.2" {
		t.Fatalf("glm-5.2: got match=%s type=%s, want exact glm-5.2", matched, mt)
	}
	if len(got) != 3 {
		t.Fatalf("glm-5.2: got %d provider models, want 3: %+v", len(got), got)
	}

	// deepseek-v4-flash keeps the non-suffix "flash" token but drops "fast".
	if _, _, mt := idx.Match("deepseek-v4-flash"); mt != MatchExact {
		t.Fatalf("deepseek-v4-flash: want exact match, got %s", mt)
	}
}

func TestBuildIndex_ConflictDetection(t *testing.T) {
	_, conflicts := buildTestIndex(t, false)

	if len(conflicts) != 1 {
		t.Fatalf("got %d conflicts, want 1: %+v", len(conflicts), conflicts)
	}
	c := conflicts[0]
	if c.Canonical != "glm-5.2" {
		t.Fatalf("conflict canonical = %q, want glm-5.2", c.Canonical)
	}
	got := append([]string(nil), c.Providers...)
	sort.Strings(got)
	want := []string{"openrouter", "zai", "zhipu"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("conflict providers = %v, want %v", got, want)
	}
}

func TestBuildIndex_SameProviderNoConflict(t *testing.T) {
	n := New(DefaultConfig())
	// One provider registering two spellings of the same model must not be
	// reported as a cross-provider conflict.
	models := []ProviderModel{
		{Provider: "zai", Original: "GLM_5_2"},
		{Provider: "zai", Original: "glm-5-2"},
	}
	_, conflicts := BuildIndex(n, models, false)
	if len(conflicts) != 0 {
		t.Fatalf("same-provider duplicates should not conflict, got %+v", conflicts)
	}
}

func TestMatch_ExactMissWithoutFuzzy(t *testing.T) {
	idx, _ := buildTestIndex(t, false)

	// A suffixed request that isn't itself an index key must miss when fuzzy is off.
	if _, _, mt := idx.Match("glm-5.2-turbo"); mt != MatchNone {
		t.Fatalf("glm-5.2-turbo without fuzzy: want none, got %s", mt)
	}
}

func TestMatch_FuzzySuffixDifference(t *testing.T) {
	idx, _ := buildTestIndex(t, true)

	// Request lacks the "-thinking" suffix that the registered model carries;
	// fuzzy segment-prefix should still resolve it. (Both normalize into the
	// index: claude-4-sonnet and claude-4-sonnet-thinking.)
	models, matched, mt := idx.Match("claude-4-sonnet-instruct")
	if mt != MatchFuzzy {
		t.Fatalf("claude-4-sonnet-instruct: want fuzzy, got %s", mt)
	}
	if matched != "claude-4-sonnet" {
		t.Fatalf("fuzzy matched %q, want claude-4-sonnet", matched)
	}
	if len(models) == 0 {
		t.Fatalf("fuzzy match returned no provider models")
	}
}

func TestMatch_FuzzyRejectsVersionMismatch(t *testing.T) {
	idx, _ := buildTestIndex(t, true)

	// The PRD explicitly forbids matching across a version/segment difference:
	// claude-5-sonnet must never resolve to claude-4-sonnet.
	if _, matched, mt := idx.Match("claude-5-sonnet"); mt != MatchNone {
		t.Fatalf("claude-5-sonnet: want none, got %s (matched %q)", mt, matched)
	}
}

func TestMatch_FuzzyDeterministicClosest(t *testing.T) {
	n := New(DefaultConfig())
	// Two candidates share the request's segment prefix; the one with the
	// smaller segment-count gap must win deterministically.
	models := []ProviderModel{
		{Provider: "p1", Original: "gpt-5"},
		{Provider: "p2", Original: "gpt-5-mini-preview"},
	}
	idx, _ := BuildIndex(n, models, true)

	_, matched, mt := idx.Match("gpt-5-mini")
	if mt != MatchFuzzy {
		t.Fatalf("gpt-5-mini: want fuzzy, got %s", mt)
	}
	// gpt-5-mini vs gpt-5 (gap 1) vs gpt-5-mini-preview (gap 1): tie broken
	// lexicographically → gpt-5 sorts before gpt-5-mini-preview... but gpt-5 is
	// a prefix (gap 1) and gpt-5-mini-preview is also gap 1. Lexical first wins.
	if matched != "gpt-5" {
		t.Fatalf("fuzzy tie-break matched %q, want gpt-5", matched)
	}
}

func TestIndex_Len(t *testing.T) {
	idx, _ := buildTestIndex(t, false)
	// Distinct canonicals: glm-5.2 (zai+zhipu+openrouter), claude-4-sonnet
	// (Claude-4-Sonnet and claude-4-sonnet-thinking both collapse here, since
	// "thinking" is a stripped dash-token), and deepseek-v4-flash = 3.
	if idx.Len() != 3 {
		t.Fatalf("index length = %d, want 3", idx.Len())
	}
}

func TestMatchForProvider_DisambiguatesConflict(t *testing.T) {
	idx, _ := buildTestIndex(t, false)

	// glm-5.2 is a shared canonical (zai, zhipu, openrouter). A request targeting
	// one provider must resolve to that provider's own registered spelling.
	pm, matched, mt, ok := idx.MatchForProvider("glm-5.2", "zhipu")
	if !ok || mt != MatchExact || matched != "glm-5.2" {
		t.Fatalf("zhipu glm-5.2: ok=%v type=%s matched=%q, want exact glm-5.2", ok, mt, matched)
	}
	if pm.Provider != "zhipu" || pm.Original != "智谱/GLM_5_2" {
		t.Fatalf("resolved %+v, want provider=zhipu original=智谱/GLM_5_2", pm)
	}
}

func TestMatchForProvider_ProviderNotInBucket(t *testing.T) {
	idx, _ := buildTestIndex(t, false)

	// The canonical exists, but not for this provider: must miss so the caller
	// leaves the request untouched.
	if _, _, _, ok := idx.MatchForProvider("glm-5.2", "anthropic"); ok {
		t.Fatal("glm-5.2 for anthropic should miss: no anthropic entry in that bucket")
	}
}

func TestMatchForProvider_Miss(t *testing.T) {
	idx, _ := buildTestIndex(t, false)

	if _, _, _, ok := idx.MatchForProvider("does-not-exist", "zai"); ok {
		t.Fatal("absent canonical should miss")
	}
}

// TestMatchForProvider_FuzzyScopedToProvider is the regression guard for the
// provider-scoped fuzzy rule. Two providers offer segment-prefix candidates for
// "gpt-5-mini": p1's "gpt-5" (gap 1) and p2's "gpt-5-mini-preview" (gap 1). The
// global-best key is "gpt-5" (lexicographically first), which belongs to p1. A
// naive "pick global best, then filter by provider" implementation would report
// a false miss for p2 — even though p2 has a perfectly good fuzzy candidate.
// Provider-scoped selection must resolve p2 to its own "gpt-5-mini-preview".
func TestMatchForProvider_FuzzyScopedToProvider(t *testing.T) {
	n := New(DefaultConfig())
	models := []ProviderModel{
		{Provider: "p1", Original: "gpt-5"},
		{Provider: "p2", Original: "gpt-5-mini-preview"},
	}
	idx, _ := BuildIndex(n, models, true)

	// p2 must resolve within its own candidates, not be shadowed by p1's closer key.
	pm, matched, mt, ok := idx.MatchForProvider("gpt-5-mini", "p2")
	if !ok || mt != MatchFuzzy {
		t.Fatalf("p2 gpt-5-mini: ok=%v type=%s, want fuzzy hit", ok, mt)
	}
	if matched != "gpt-5-mini-preview" || pm.Original != "gpt-5-mini-preview" {
		t.Fatalf("p2 resolved to matched=%q original=%q, want gpt-5-mini-preview", matched, pm.Original)
	}

	// p1 still resolves to its own "gpt-5".
	pm, matched, _, ok = idx.MatchForProvider("gpt-5-mini", "p1")
	if !ok || matched != "gpt-5" || pm.Original != "gpt-5" {
		t.Fatalf("p1 gpt-5-mini: ok=%v matched=%q original=%q, want gpt-5", ok, matched, pm.Original)
	}

	// A provider with no candidate at all still misses.
	if _, _, _, ok := idx.MatchForProvider("gpt-5-mini", "p3"); ok {
		t.Fatal("p3 has no candidate; must miss")
	}
}
