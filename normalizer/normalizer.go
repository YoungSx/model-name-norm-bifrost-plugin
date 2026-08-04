package normalizer

import (
	"regexp"
	"strings"
)

// separatorClass matches any run of characters treated as separators before
// unification: ASCII/Unicode whitespace, underscore and hyphen. A whole run
// collapses to a single canonical separator, satisfying both the "unify" and
// "compress consecutive" rules of the PRD in one pass.
var separatorClass = regexp.MustCompile(`[\s_\-]+`)

// bracketGroup matches a `[...]` group (non-greedy body, no nested brackets),
// removed anywhere in the string by the bracket-suffix rule.
var bracketGroup = regexp.MustCompile(`\[[^\]]*\]`)

// Result is the outcome of normalizing a single model name.
type Result struct {
	// Canonical is the normalized name used as the matching key.
	Canonical string
	// Original is the untouched input, preserved for logging and diagnostics.
	Original string
	// EmptyFallback is true when the pipeline would have emitted an empty
	// canonical and fell back to a non-empty earlier value instead. Callers
	// should emit a warning when this is set.
	EmptyFallback bool
}

// Normalizer applies the canonicalization pipeline. It is safe for concurrent
// use: all fields are read-only after construction, so a single instance built
// at startup serves every request without locking.
type Normalizer struct {
	cfg       Config
	separator string              // effective canonical separator (never empty)
	tokenSet  map[string]struct{} // dash-suffix tokens, lower-cased for O(1) lookup
}

// New builds a Normalizer from cfg, precomputing everything the hot path needs
// so per-request work is limited to string manipulation.
func New(cfg Config) *Normalizer {
	sep := cfg.Separator.Replacement
	if sep == "" {
		sep = "-"
	}
	tokens := make(map[string]struct{}, len(cfg.Suffix.Tokens))
	for _, t := range cfg.Suffix.Tokens {
		t = strings.ToLower(strings.TrimSpace(t))
		if t != "" {
			tokens[t] = struct{}{}
		}
	}
	return &Normalizer{cfg: cfg, separator: sep, tokenSet: tokens}
}

// Normalize runs the full pipeline, including suffix stripping, and returns the
// loose canonical name: the key under which every marketing/capability variant
// of one logical model collapses together (`claude-4-sonnet-thinking` and
// `Claude-4-Sonnet` both yield `claude-4-sonnet`). This is what lets a caller's
// arbitrary spelling find a provider's registered model.
//
// Note on ordering: the PRD lists version normalization (step 5) before suffix
// stripping (step 6), but its own canonical example `glm 5 2:free` → `glm-5.2`
// (section 4) is unachievable in that order — the `:free` colon suffix keeps the
// trailing `2` glued to `free` when the version stage runs, so `5-2` never
// becomes `5.2`. Running suffix stripping first, then version normalization on
// the cleaned name, satisfies every documented example and preserves the PRD's
// core invariant that one logical model yields one canonical form regardless of
// which marketing/capability suffixes were attached.
func (n *Normalizer) Normalize(original string) Result {
	return n.normalize(original, true)
}

// NormalizeStrict runs the pipeline with every suffix rule disabled, collapsing
// only cosmetic differences: surrounding whitespace, provider prefix, letter
// case, separator spelling and numeric-version punctuation.
//
// Strict canonicalization is what keeps distinct models distinct. Suffix
// stripping is deliberately lossy — it maps a family of names onto one key — so
// using it alone would make an explicit request for `claude-4-sonnet-thinking`
// indistinguishable from one for `claude-4-sonnet`, and a provider that
// registered both would always be handed the base model. The strict key
// preserves the caller's intent (`claude-4-sonnet-thinking` stays itself) while
// still absorbing spelling noise (`Claude 4 Sonnet Thinking` maps to the same
// strict key). Index lookups try strict first and only fall back to the loose
// key, so variant fidelity wins whenever the provider actually offers the
// variant. See Index.ResolveForProvider.
func (n *Normalizer) NormalizeStrict(original string) Result {
	return n.normalize(original, false)
}

// normalize is the shared pipeline. stripSuffixes selects the loose (true) or
// strict (false) variant; every other stage is identical, so the two keys can
// never disagree about casing, prefixes or separators.
func (n *Normalizer) normalize(original string, stripSuffixes bool) Result {
	s := original

	// 1. Trim surrounding whitespace and invisible characters.
	s = trim(s)

	// 2. Prefix stripping: drop everything up to and including the last '/'.
	if n.cfg.Prefix.StripAfterLastSlash {
		if i := strings.LastIndex(s, "/"); i >= 0 {
			s = s[i+1:]
		}
	}

	// 3. Lowercase.
	if n.cfg.Lowercase {
		s = strings.ToLower(s)
	}

	// 4. Separator normalization: unify `_`/whitespace/`-` into the canonical
	//    separator and collapse consecutive separators.
	if n.cfg.Separator.Normalize {
		s = separatorClass.ReplaceAllString(s, n.separator)
		s = strings.Trim(s, n.separator)
	}

	res := Result{Original: original}

	// 5. Suffix stripping (colon → bracket → dash-token), loose mode only. The
	//    pre-suffix value is retained for empty protection.
	preSuffix := s
	if stripSuffixes {
		s = n.stripSuffixes(s)
	}

	// 6. Empty protection: never emit an empty canonical when a non-empty
	//    earlier value exists. Checked before version normalization since
	//    version on an empty value is meaningless. Prefix stripping alone can
	//    empty the name (input `a/`), so this applies in strict mode too.
	if s == "" {
		fallback := preSuffix
		if fallback == "" {
			fallback = trim(original)
		}
		res.Canonical = fallback
		res.EmptyFallback = true
		return res
	}

	// 7. Version normalization on the cleaned name: conservative
	//    digit-digit → digit.digit.
	if n.cfg.Version.NormalizeNumericVersion {
		s = n.normalizeVersion(s)
	}

	// 8. Output canonical name.
	res.Canonical = s
	return res
}

// trim removes leading/trailing Unicode whitespace. strings.TrimSpace already
// covers the invisible-character cases called out in the PRD (spaces, tabs,
// non-breaking runs are handled by unicode.IsSpace).
func trim(s string) string {
	return strings.TrimSpace(s)
}

// normalizeVersion joins adjacent purely-numeric segments with '.' instead of
// the canonical separator. Splitting on the separator (rather than using a
// regex) sidesteps Go's lack of look-around and makes the "both sides must be
// complete numeric tokens" rule exact: `glm-5-2` → `glm-5.2`, while
// `llama-3-70b` is left untouched because `70b` is not purely numeric.
func (n *Normalizer) normalizeVersion(s string) string {
	if !strings.Contains(s, n.separator) {
		return s
	}
	parts := strings.Split(s, n.separator)
	var b strings.Builder
	b.WriteString(parts[0])
	for i := 1; i < len(parts); i++ {
		if isAllDigits(parts[i-1]) && isAllDigits(parts[i]) {
			b.WriteByte('.')
		} else {
			b.WriteString(n.separator)
		}
		b.WriteString(parts[i])
	}
	return b.String()
}

// stripSuffixes applies the three suffix families in an order that keeps each
// one meaningful: colon truncation first (it removes a trailing free-form
// segment), then bracket removal (which can expose a new trailing token), then
// repeated dash-token removal.
func (n *Normalizer) stripSuffixes(s string) string {
	// A. Colon suffix: everything from the first ':' onward is dropped.
	if n.cfg.Suffix.Colon.Enabled {
		if i := strings.Index(s, ":"); i >= 0 {
			s = s[:i]
		}
	}

	// C. Bracket suffix: remove every `[...]` group, then trim separators the
	//    removal may have left dangling (e.g. `claude-4-sonnet-[1m]`).
	if n.cfg.Suffix.Bracket.Enabled {
		s = bracketGroup.ReplaceAllString(s, "")
		s = strings.Trim(s, n.separator)
	}

	// B. Dash-token suffix: strip complete trailing tokens repeatedly so stacked
	//    suffixes (`...-flash-fast`) collapse fully, but only when the token is
	//    configured — non-listed trailing words like `flash` are preserved. When
	//    the name is reduced to a single configured token (e.g. input `free`) it
	//    is consumed too, deliberately producing an empty string so the caller's
	//    empty-protection stage can fall back to the original and warn (PRD 6.7).
	if len(n.tokenSet) > 0 && n.separator != "" {
		for s != "" {
			idx := strings.LastIndex(s, n.separator)
			last := s[idx+len(n.separator):] // whole string when idx == -1
			if _, ok := n.tokenSet[last]; !ok {
				break
			}
			if idx < 0 {
				s = ""
				break
			}
			s = s[:idx]
		}
	}

	return s
}

// isAllDigits reports whether s is non-empty and consists solely of ASCII
// digits — the definition of a version-number segment for rule 6.5.
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}
