package normalizer

import (
	"sort"
	"strings"
)

// ProviderModel is one registered model as seen from a provider, plus the
// canonical form it normalized to. It is the value stored in the canonical
// index and returned on a successful lookup.
type ProviderModel struct {
	// Provider is the Bifrost provider key (e.g. "zai", "zhipu").
	Provider string
	// Original is the model name exactly as the provider registered it.
	Original string
	// Canonical is the normalized key this model was indexed under.
	Canonical string
}

// MatchType describes how a lookup resolved, for logging and diagnostics.
type MatchType int

const (
	// MatchNone means no provider model matched the request.
	MatchNone MatchType = iota
	// MatchExact means the request's canonical form equaled an index key.
	MatchExact
	// MatchFuzzy means the request matched under the relaxed segment-prefix rule.
	MatchFuzzy
)

// String renders a MatchType for log lines.
func (m MatchType) String() string {
	switch m {
	case MatchExact:
		return "exact"
	case MatchFuzzy:
		return "fuzzy"
	default:
		return "none"
	}
}

// Conflict records a canonical name that several providers registered under —
// allowed by the PRD (section 11) but surfaced as a startup warning (section 12).
type Conflict struct {
	Canonical string
	Providers []string
}

// Index is the startup-computed canonical→providers map. It is immutable after
// Build returns, so concurrent request-time lookups need no locking.
type Index struct {
	exact      map[string][]ProviderModel
	sortedKeys []string            // canonical keys, sorted, for deterministic fuzzy matching
	segCache   map[string][]string // canonical → its separator-split segments
	separator  string
	fuzzy      bool
}

// BuildIndex normalizes every provider model and assembles the canonical index.
// Multiple providers mapping to one canonical are kept side by side (never
// overwritten); those collisions are returned as conflicts for the caller to
// log. Input order is preserved within each canonical bucket so logs and lookup
// results are stable.
func BuildIndex(n *Normalizer, models []ProviderModel, fuzzy bool) (*Index, []Conflict) {
	exact := make(map[string][]ProviderModel, len(models))
	for _, m := range models {
		canonical := n.Normalize(m.Original).Canonical
		m.Canonical = canonical
		exact[canonical] = append(exact[canonical], m)
	}

	keys := make([]string, 0, len(exact))
	segCache := make(map[string][]string, len(exact))
	for k := range exact {
		keys = append(keys, k)
		segCache[k] = strings.Split(k, n.separator)
	}
	sort.Strings(keys)

	var conflicts []Conflict
	for _, k := range keys {
		bucket := exact[k]
		if len(bucket) < 2 {
			continue
		}
		// Report the distinct providers behind a shared canonical. Duplicates of
		// the same provider (e.g. two keys) don't count as a conflict.
		seen := make(map[string]struct{}, len(bucket))
		providers := make([]string, 0, len(bucket))
		for _, m := range bucket {
			if _, ok := seen[m.Provider]; ok {
				continue
			}
			seen[m.Provider] = struct{}{}
			providers = append(providers, m.Provider)
		}
		if len(providers) > 1 {
			conflicts = append(conflicts, Conflict{Canonical: k, Providers: providers})
		}
	}

	return &Index{
		exact:      exact,
		sortedKeys: keys,
		segCache:   segCache,
		separator:  n.separator,
		fuzzy:      fuzzy,
	}, conflicts
}

// Match resolves a request's canonical name to provider models. It tries an
// exact hit first (the default mode, PRD section 9), then—only when fuzzy is
// enabled—the relaxed segment-prefix rule (section 10). The matched index key
// is returned for logging; on a miss the models slice is nil and type MatchNone.
func (idx *Index) Match(canonical string) (models []ProviderModel, matched string, mt MatchType) {
	if m, ok := idx.exact[canonical]; ok {
		return m, canonical, MatchExact
	}
	if !idx.fuzzy {
		return nil, "", MatchNone
	}
	if key, ok := idx.bestFuzzyKey(canonical, acceptAny); ok {
		return idx.exact[key], key, MatchFuzzy
	}
	return nil, "", MatchNone
}

// acceptAny is the no-op acceptability predicate used by the provider-agnostic
// Match path: every segment-prefix candidate is eligible.
func acceptAny(string) bool { return true }

// MatchForProvider resolves canonical to a single provider model belonging to
// the requested provider. A request always targets one provider, so a canonical
// that several providers share (a conflict) is disambiguated here: only the
// entry whose Provider equals provider is a hit. When multiple entries under one
// canonical belong to the same provider (different spellings of one model), the
// first in registration order wins, keeping the choice deterministic. The bool
// is false when the canonical is absent or no entry matches the provider, in
// which case the caller should leave the request untouched.
//
// Fuzzy resolution is provider-scoped: rather than picking the single global-best
// key and then checking whether it happens to serve the provider (which would
// report a false miss whenever the global best belongs to a different provider),
// it selects the best key *among those that actually serve the provider*. So a
// request for provider p2 still resolves to p2's `gpt-5-mini-preview` even when
// p1's `gpt-5` is the closer key overall.
func (idx *Index) MatchForProvider(canonical, provider string) (pm ProviderModel, matched string, mt MatchType, ok bool) {
	// Exact path: a direct canonical hit that includes an entry for the provider.
	if pm, ok := firstForProvider(idx.exact[canonical], provider); ok {
		return pm, canonical, MatchExact, true
	}
	if !idx.fuzzy {
		return ProviderModel{}, "", MatchNone, false
	}

	// Fuzzy path: among keys that both satisfy the segment-prefix relation and
	// carry an entry for this provider, pick the closest (smallest segment-count
	// gap, lexicographically-first key on a tie).
	key, found := idx.bestFuzzyKey(canonical, func(k string) bool {
		_, ok := firstForProvider(idx.exact[k], provider)
		return ok
	})
	if !found {
		return ProviderModel{}, "", MatchNone, false
	}
	pm, _ = firstForProvider(idx.exact[key], provider)
	return pm, key, MatchFuzzy, true
}

// firstForProvider returns the first model in bucket registered by provider,
// preserving registration order so the choice is deterministic.
func firstForProvider(bucket []ProviderModel, provider string) (ProviderModel, bool) {
	for _, m := range bucket {
		if m.Provider == provider {
			return m, true
		}
	}
	return ProviderModel{}, false
}

// Len reports the number of distinct canonical names in the index.
func (idx *Index) Len() int { return len(idx.exact) }

// bestFuzzyKey finds the best index key related to canonical under the
// segment-prefix rule: one name's dash-separated segments must be a prefix of the
// other's. This permits suffix, separator and version-alias differences (which
// already collapse during normalization) while forbidding any mismatch inside a
// shared segment — so `claude-4-sonnet` matches `claude-4-sonnet-thinking` but
// never `claude-5-sonnet`. Among candidates the smallest segment-count gap wins,
// then the lexicographically-first key, keeping results deterministic.
//
// accept filters the candidate set before selection: a key is only eligible when
// accept(key) is true. This is what makes provider-scoped fuzzy matching correct
// — the closest key is chosen from those that actually serve the provider, rather
// than choosing the global best and rejecting it after the fact. The
// provider-agnostic path passes acceptAny.
func (idx *Index) bestFuzzyKey(canonical string, accept func(string) bool) (string, bool) {
	reqSegs := strings.Split(canonical, idx.separator)
	best := ""
	bestGap := -1
	for _, key := range idx.sortedKeys { // sorted → deterministic tie-break
		if !accept(key) {
			continue
		}
		keySegs := idx.segCache[key]
		if !segmentPrefix(reqSegs, keySegs) {
			continue
		}
		gap := len(keySegs) - len(reqSegs)
		if gap < 0 {
			gap = -gap
		}
		if bestGap == -1 || gap < bestGap {
			best, bestGap = key, gap
		}
	}
	return best, bestGap != -1
}

// segmentPrefix reports whether the shorter of a and b is an exact element-wise
// prefix of the longer. Equal-length slices only match when identical (which the
// exact path already handles, but this keeps the relation total and symmetric).
func segmentPrefix(a, b []string) bool {
	shorter, longer := a, b
	if len(shorter) > len(longer) {
		shorter, longer = longer, shorter
	}
	for i := range shorter {
		if shorter[i] != longer[i] {
			return false
		}
	}
	return true
}
