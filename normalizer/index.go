package normalizer

import (
	"sort"
	"strings"
)

// ProviderModel is one registered model as seen from a provider, plus the two
// keys it was indexed under. It is the value stored in the index and returned on
// a successful lookup.
type ProviderModel struct {
	// Provider is the Bifrost provider key (e.g. "zai", "zhipu").
	Provider string
	// Original is the model name exactly as the provider registered it.
	Original string
	// Canonical is the loose (suffix-stripped) key this model was indexed under.
	Canonical string
	// Strict is the variant-preserving key this model was indexed under.
	Strict string
}

// MatchType describes how a lookup resolved, for logging and diagnostics.
type MatchType int

const (
	// MatchNone means no registered model matched the request.
	MatchNone MatchType = iota
	// MatchStrict means the request and the registered model agree once only
	// cosmetic differences (case, separators, prefix) are removed — the request's
	// exact variant was found.
	MatchStrict
	// MatchCanonical means they agree only after suffix stripping: the request
	// named a different variant (or none) of the same logical model.
	MatchCanonical
	// MatchFuzzy means the request matched under the relaxed segment-prefix rule.
	MatchFuzzy
)

// String renders a MatchType for log lines.
func (m MatchType) String() string {
	switch m {
	case MatchStrict:
		return "strict"
	case MatchCanonical:
		return "canonical"
	case MatchFuzzy:
		return "fuzzy"
	default:
		return "none"
	}
}

// Conflict records a loose canonical name that more than one registered model
// collapsed onto. Both shapes are permitted by the PRD (section 11) but are
// surfaced as startup warnings (section 12) because each changes what a bare
// canonical request resolves to:
//
//   - Providers with len > 1: several providers offer this canonical, so the
//     request's own provider decides the winner.
//   - Shadowed non-empty: one provider registered several genuinely different
//     models here (e.g. a base model and its `-thinking` variant). They stay
//     individually reachable by their own spellings via the strict tier, but a
//     request for the bare canonical resolves to the first registered.
type Conflict struct {
	// Canonical is the shared loose key.
	Canonical string
	// Providers lists the distinct providers in this bucket, registration order.
	Providers []string
	// Shadowed lists originals that lose a bare-canonical lookup to an earlier
	// registration by the same provider.
	Shadowed []string
}

// Index is the startup-computed model index. It holds two tiers — strict
// (variant-preserving) and loose (suffix-stripped) — and owns the Normalizer
// that produced them, so a lookup can never be performed with different
// normalization rules than the build used. It is immutable after BuildIndex
// returns, so concurrent request-time lookups need no locking.
type Index struct {
	n          *Normalizer
	strict     map[string][]ProviderModel
	loose      map[string][]ProviderModel
	sortedKeys []string            // loose keys, sorted, for deterministic fuzzy matching
	segCache   map[string][]string // loose key → its separator-split segments
	conflicts  []Conflict
	fuzzy      bool
}

// BuildIndex normalizes every provider model into both tiers and assembles the
// index. Models sharing a key are kept side by side (never overwritten) in
// registration order, so lookups and logs are stable; the collisions that change
// resolution are recorded for Conflicts. Entries with a blank name, or whose
// normalization yields a blank key, are skipped as unroutable.
func BuildIndex(n *Normalizer, models []ProviderModel, fuzzy bool) *Index {
	strict := make(map[string][]ProviderModel, len(models))
	loose := make(map[string][]ProviderModel, len(models))
	for _, m := range models {
		name := strings.TrimSpace(m.Original)
		if name == "" || name == "*" {
			// Blank names and the `*` whitelist wildcard carry no routing
			// information: `*` means "any model is allowed", not a model
			// literally named `*`. Indexing it would let a request be rewritten
			// to the literal "*", which no provider can serve.
			continue
		}
		m.Canonical = n.Normalize(m.Original).Canonical
		m.Strict = n.NormalizeStrict(m.Original).Canonical
		if m.Canonical == "" || m.Strict == "" {
			continue
		}
		strict[m.Strict] = append(strict[m.Strict], m)
		loose[m.Canonical] = append(loose[m.Canonical], m)
	}

	keys := make([]string, 0, len(loose))
	segCache := make(map[string][]string, len(loose))
	for k := range loose {
		keys = append(keys, k)
		segCache[k] = strings.Split(k, n.separator)
	}
	sort.Strings(keys)

	return &Index{
		n:          n,
		strict:     strict,
		loose:      loose,
		sortedKeys: keys,
		segCache:   segCache,
		conflicts:  buildConflicts(loose, keys),
		fuzzy:      fuzzy,
	}
}

// buildConflicts walks the loose buckets in sorted key order and reports the
// collisions an operator needs to know about: a canonical served by more than
// one provider, and a canonical where one provider's later registrations are
// shadowed by an earlier one.
func buildConflicts(loose map[string][]ProviderModel, keys []string) []Conflict {
	var conflicts []Conflict
	for _, k := range keys {
		bucket := loose[k]
		if len(bucket) < 2 {
			continue
		}

		var providers []string
		seenProvider := make(map[string]struct{}, len(bucket))
		// Per provider, the strict keys already claimed — a repeat strict key is
		// merely another spelling of a model we've seen, not a distinct model.
		claimed := make(map[string]map[string]struct{}, len(bucket))
		var shadowed []string

		for _, m := range bucket {
			if _, ok := seenProvider[m.Provider]; !ok {
				seenProvider[m.Provider] = struct{}{}
				providers = append(providers, m.Provider)
			}
			seen, ok := claimed[m.Provider]
			if !ok {
				claimed[m.Provider] = map[string]struct{}{m.Strict: {}}
				continue
			}
			if _, dup := seen[m.Strict]; dup {
				continue
			}
			seen[m.Strict] = struct{}{}
			shadowed = append(shadowed, m.Original)
		}

		if len(providers) > 1 || len(shadowed) > 0 {
			conflicts = append(conflicts, Conflict{Canonical: k, Providers: providers, Shadowed: shadowed})
		}
	}
	return conflicts
}

// Conflicts returns the startup diagnostics computed by BuildIndex, in sorted
// canonical order. The slice is owned by the Index; callers must not mutate it.
func (idx *Index) Conflicts() []Conflict { return idx.conflicts }

// Len reports the number of distinct loose canonical names in the index.
func (idx *Index) Len() int { return len(idx.loose) }

// Resolution is the outcome of resolving one requested model name.
type Resolution struct {
	// Model is the provider's registered model. Only meaningful when OK.
	Model ProviderModel
	// MatchedKey is the index key that matched, for logging.
	MatchedKey string
	// Type says how it resolved (or MatchNone).
	Type MatchType
	// Request is the loose normalization of the requested name, carried so
	// callers can log diagnostics (notably EmptyFallback) without renormalizing.
	Request Result
}

// OK reports whether the request resolved to a registered model.
func (r Resolution) OK() bool { return r.Type != MatchNone }

// ResolveForProvider maps a requested model name to a model the given provider
// actually registered. A request always targets one provider, so a canonical
// shared by several providers is disambiguated here: only entries whose Provider
// equals provider are eligible.
//
// Tiers are tried in order of fidelity, which is what keeps the plugin from
// silently changing which model a caller asked for:
//
//  1. Strict — the request names this variant (up to case/separator/prefix
//     noise). `claude-4-sonnet-thinking` resolves to the provider's
//     `claude-4-sonnet-thinking`, never to its `Claude-4-Sonnet`.
//  2. Canonical — the request names some variant of the same logical model, or
//     the bare model. This is the PRD's headline behavior: `GLM_5_2` finds
//     `ZAI/glm-5.2`. Within one provider the first registration wins, keeping
//     the choice deterministic (and reported via Conflicts).
//  3. Fuzzy — only when enabled: the relaxed segment-prefix rule.
//
// Fuzzy resolution is provider-scoped: rather than picking the single
// global-best key and then checking whether it happens to serve the provider
// (which would report a false miss whenever the global best belongs to another
// provider), it selects the best key among those that actually serve the
// provider. So a request for p2 still resolves to p2's `gpt-5-mini-preview`
// even when p1's `gpt-5` is the closer key overall.
//
// On a miss the returned Resolution has Type MatchNone and callers should leave
// the request untouched.
func (idx *Index) ResolveForProvider(model, provider string) Resolution {
	res := Resolution{Request: idx.n.Normalize(model)}
	if strings.TrimSpace(model) == "" {
		return res
	}

	// Tier 1: strict — the caller's own variant.
	strictKey := idx.n.NormalizeStrict(model).Canonical
	if pm, ok := firstForProvider(idx.strict[strictKey], provider); ok {
		res.Model, res.MatchedKey, res.Type = pm, strictKey, MatchStrict
		return res
	}

	// Tier 2: loose canonical — some variant of the same logical model.
	if pm, ok := firstForProvider(idx.loose[res.Request.Canonical], provider); ok {
		res.Model, res.MatchedKey, res.Type = pm, res.Request.Canonical, MatchCanonical
		return res
	}

	if !idx.fuzzy {
		return res
	}

	// Tier 3: fuzzy — closest segment-prefix key that serves this provider.
	key, found := idx.bestFuzzyKey(res.Request.Canonical, func(k string) bool {
		_, ok := firstForProvider(idx.loose[k], provider)
		return ok
	})
	if !found {
		return res
	}
	pm, _ := firstForProvider(idx.loose[key], provider)
	res.Model, res.MatchedKey, res.Type = pm, key, MatchFuzzy
	return res
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

// bestFuzzyKey finds the best loose key related to canonical under the
// segment-prefix rule: one name's separator-split segments must be a prefix of
// the other's. This permits suffix, separator and version-alias differences
// (which already collapse during normalization) while forbidding any mismatch
// inside a shared segment — so `claude-4-sonnet` matches
// `claude-4-sonnet-thinking` but never `claude-5-sonnet`. Among candidates the
// smallest segment-count gap wins, then the lexicographically-first key, keeping
// results deterministic.
//
// accept filters the candidate set before selection: a key is only eligible when
// accept(key) is true. This is what makes provider-scoped fuzzy matching correct
// — the closest key is chosen from those that actually serve the provider,
// rather than choosing the global best and rejecting it after the fact.
func (idx *Index) bestFuzzyKey(canonical string, accept func(string) bool) (string, bool) {
	reqSegs := strings.Split(canonical, idx.n.separator)
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
// exact tiers already handle, but this keeps the relation total and symmetric).
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
