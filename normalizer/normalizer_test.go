package normalizer

import "testing"

// newDefault builds a Normalizer with the PRD-recommended defaults, matching the
// configuration every acceptance table in the spec assumes.
func newDefault() *Normalizer {
	return New(DefaultConfig())
}

// TestPRDTables walks every worked example and acceptance-table row from the PRD
// so the pipeline is pinned to the spec exactly.
func TestPRDTables(t *testing.T) {
	n := newDefault()
	cases := []struct {
		name string
		in   string
		want string
	}{
		// Section 4 canonical examples.
		{"prefix zai", "ZAI/glm-5.2", "glm-5.2"},
		{"prefix zhipu underscore", "智谱/GLM_5_2", "glm-5.2"},
		{"space and colon free", "glm 5 2:free", "glm-5.2"},
		{"stacked deepseek", "deepseek-ai/DeepSeek-V4-Flash-fast", "deepseek-v4-flash"},

		// Section 6.1 trim.
		{"trim spaces", " glm-5.2 ", "glm-5.2"},

		// Section 6.2 prefix.
		{"prefix deepseek-ai", "deepseek-ai/deepseek-v4-flash", "deepseek-v4-flash"},

		// Section 6.3 lowercase.
		{"lowercase claude", "Claude-4-Sonnet", "claude-4-sonnet"},

		// Section 6.4 separators.
		{"underscore", "GLM_5_2", "glm-5.2"},
		{"spaces", "glm 5 2", "glm-5.2"},
		{"compress dashes", "glm---5", "glm-5"},
		{"keep dot", "glm-5.2", "glm-5.2"},

		// Section 6.5 version.
		{"version gpt", "gpt-5-1", "gpt-5.1"},
		{"non-version llama", "llama-3-70b", "llama-3-70b"},

		// Section 6.6 suffixes.
		{"colon free", "glm-5.2:free", "glm-5.2"},
		{"colon date", "gpt-5:2025", "gpt-5"},
		{"dash thinking", "claude-4-sonnet-thinking", "claude-4-sonnet"},
		{"flash preserved", "deepseek-v4-flash", "deepseek-v4-flash"},
		{"flash-fast stripped", "deepseek-v4-flash-fast", "deepseek-v4-flash"},
		{"bracket 1M", "claude-4-sonnet[1M]", "claude-4-sonnet"},
		{"bracket beta", "model[beta]", "model"},

		// Section 13 acceptance tables.
		{"table prefix zhipu", "智谱/glm-5.2", "glm-5.2"},
		{"table prefix deepseek", "deepseek-ai/deepseek-v4", "deepseek-v4"},
		{"table sep glm---5.2", "glm---5.2", "glm-5.2"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := n.Normalize(c.in)
			if got.Canonical != c.want {
				t.Fatalf("Normalize(%q) = %q, want %q", c.in, got.Canonical, c.want)
			}
			if got.EmptyFallback {
				t.Fatalf("Normalize(%q) unexpectedly flagged EmptyFallback", c.in)
			}
		})
	}
}

// TestEmptyProtection covers section 6.7: a name consisting entirely of a
// stripped suffix must fall back to the pre-suffix value and flag it.
func TestEmptyProtection(t *testing.T) {
	n := newDefault()
	got := n.Normalize("free")
	if got.Canonical != "free" {
		t.Fatalf("Normalize(\"free\") = %q, want \"free\"", got.Canonical)
	}
	if !got.EmptyFallback {
		t.Fatal("Normalize(\"free\") should set EmptyFallback")
	}
}

// TestEmptyProtectionColonOnly ensures a bare ":suffix" input, which colon
// stripping would empty, also falls back rather than yielding "".
func TestEmptyProtectionColonOnly(t *testing.T) {
	n := newDefault()
	got := n.Normalize(":free")
	if got.Canonical == "" {
		t.Fatal("canonical must never be empty")
	}
	if !got.EmptyFallback {
		t.Fatal("expected EmptyFallback for input that empties under stripping")
	}
}

// TestWhitespaceOnly guards the degenerate all-whitespace input.
func TestWhitespaceOnly(t *testing.T) {
	n := newDefault()
	got := n.Normalize("   ")
	if !got.EmptyFallback {
		t.Fatal("expected EmptyFallback for whitespace-only input")
	}
}

// TestLastSlashWins verifies multi-segment prefixes keep only the final path
// component, per rule 6.2 ("take the content after the last '/'").
func TestLastSlashWins(t *testing.T) {
	n := newDefault()
	got := n.Normalize("org/team/glm-5-2")
	if got.Canonical != "glm-5.2" {
		t.Fatalf("got %q, want glm-5.2", got.Canonical)
	}
}

// TestVersionMultiSegment checks that a run of numeric segments joins with dots
// while a trailing non-numeric token stays separated by the canonical dash.
func TestVersionMultiSegment(t *testing.T) {
	n := newDefault()
	if got := n.Normalize("gpt-5-1-turbo").Canonical; got != "gpt-5.1-turbo" {
		t.Fatalf("got %q, want gpt-5.1-turbo", got)
	}
}

// TestStagesDisabled confirms every stage is opt-out: with a zero Config the
// pipeline only trims (all normalization flags default false).
func TestStagesDisabled(t *testing.T) {
	n := New(Config{})
	got := n.Normalize("  ZAI/GLM_5_2:free  ")
	if got.Canonical != "ZAI/GLM_5_2:free" {
		t.Fatalf("with all stages off, only trim should apply; got %q", got.Canonical)
	}
}

// TestConcurrentUse exercises the documented concurrency guarantee: a single
// Normalizer shared across goroutines must be race-free (run with -race).
func TestConcurrentUse(t *testing.T) {
	n := newDefault()
	done := make(chan string, 8)
	for i := 0; i < 8; i++ {
		go func() { done <- n.Normalize("ZAI/GLM_5_2").Canonical }()
	}
	for i := 0; i < 8; i++ {
		if got := <-done; got != "glm-5.2" {
			t.Fatalf("concurrent Normalize got %q, want glm-5.2", got)
		}
	}
}
