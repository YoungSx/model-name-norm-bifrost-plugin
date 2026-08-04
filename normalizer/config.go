// Package normalizer implements the model-name normalization pipeline described
// in the PRD. It is intentionally free of any Bifrost dependency so the core
// canonicalization logic can be unit-tested and reused in isolation.
package normalizer

// Config mirrors the `normalization` section of the plugin YAML one-to-one so a
// single decode step (JSON/YAML) populates it directly. Zero values are safe:
// every stage is opt-in, and DefaultConfig provides the PRD-recommended setup.
type Config struct {
	Lowercase bool            `json:"lowercase" yaml:"lowercase"`
	Prefix    PrefixConfig    `json:"prefix" yaml:"prefix"`
	Separator SeparatorConfig `json:"separator" yaml:"separator"`
	Version   VersionConfig   `json:"version" yaml:"version"`
	Suffix    SuffixConfig    `json:"suffix" yaml:"suffix"`
}

// PrefixConfig controls provider-prefix stripping (everything up to and
// including the last '/').
type PrefixConfig struct {
	StripAfterLastSlash bool `json:"strip_after_last_slash" yaml:"strip_after_last_slash"`
}

// SeparatorConfig controls separator unification. Replacement is the canonical
// separator all of `_`, whitespace and `-` collapse into (default "-").
type SeparatorConfig struct {
	Normalize   bool   `json:"normalize" yaml:"normalize"`
	Replacement string `json:"replacement" yaml:"replacement"`
}

// VersionConfig controls conservative numeric-version normalization
// (digit-digit → digit.digit).
type VersionConfig struct {
	NormalizeNumericVersion bool `json:"normalize_numeric_version" yaml:"normalize_numeric_version"`
}

// SuffixConfig controls the three suffix-stripping families: colon, bracket and
// trailing dash-token.
type SuffixConfig struct {
	Colon   ToggleConfig `json:"colon" yaml:"colon"`
	Bracket ToggleConfig `json:"bracket" yaml:"bracket"`
	Tokens  []string     `json:"tokens" yaml:"tokens"`
}

// ToggleConfig is a simple enabled flag wrapper, matching the nested YAML shape
// (`colon: { enabled: true }`).
type ToggleConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
}

// DefaultConfig returns the recommended configuration from the PRD: every stage
// enabled, "-" as the canonical separator, and the common marketing/capability
// suffix tokens.
func DefaultConfig() Config {
	return Config{
		Lowercase: true,
		Prefix:    PrefixConfig{StripAfterLastSlash: true},
		Separator: SeparatorConfig{Normalize: true, Replacement: "-"},
		Version:   VersionConfig{NormalizeNumericVersion: true},
		Suffix: SuffixConfig{
			Colon:   ToggleConfig{Enabled: true},
			Bracket: ToggleConfig{Enabled: true},
			Tokens:  []string{"free", "fast", "thinking", "latest"},
		},
	}
}
