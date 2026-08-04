package normalizer

import "testing"

func TestProbeDollarSeparator(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Separator.Replacement = "$1"
	n := New(cfg)
	t.Logf("replacement=$1: %q -> %q", "glm 5 2", n.Normalize("glm 5 2").Canonical)

	cfg2 := DefaultConfig()
	cfg2.Separator.Replacement = "__"
	n2 := New(cfg2)
	t.Logf("replacement=__: %q -> %q", "glm 5 2 free", n2.Normalize("glm 5 2 free").Canonical)
	t.Logf("replacement=__ bracket: %q -> %q", "glm 5[1m]", n2.Normalize("glm 5[1m]").Canonical)
}

func TestProbeDotSeparatorIdempotence(t *testing.T) {
	n := New(DefaultConfig())
	once := n.Normalize("gpt-5-1-2").Canonical
	twice := n.Normalize(once).Canonical
	t.Logf("idempotence: %q -> %q -> %q", "gpt-5-1-2", once, twice)
	if once != twice {
		t.Errorf("NOT IDEMPOTENT: %q != %q", once, twice)
	}
}

func TestProbeUnicodeAndCase(t *testing.T) {
	n := New(DefaultConfig())
	for _, in := range []string{"GLM 5 2", "glm–5.2", "  ", "///", "a/", "model[unclosed", "gpt-5:", "GLM-5.2-LATEST-FREE"} {
		r := n.Normalize(in)
		t.Logf("%-24q -> %-20q fallback=%v", in, r.Canonical, r.EmptyFallback)
	}
}
