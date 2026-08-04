package plugin

import (
	"context"
	"fmt"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	"github.com/maximhq/bifrost/core/schemas"
)

// ModelSource is the minimal slice of the Bifrost schemas.Account contract this
// package needs to enumerate registered models. Depending on the narrow
// interface rather than the full Account keeps the adapter testable with a tiny
// fake and documents exactly which Account methods the plugin relies on.
type ModelSource interface {
	// GetConfiguredProviders returns every provider configured on the account.
	GetConfiguredProviders() ([]schemas.ModelProvider, error)
	// GetKeysForProvider returns the keys for a provider; each key carries the
	// list of models it can serve (schemas.Key.Models).
	GetKeysForProvider(ctx context.Context, provider schemas.ModelProvider) ([]schemas.Key, error)
}

// compile-time proof that a real *schemas.Account satisfies ModelSource, so the
// adapter stays in lock-step with the upstream contract.
var _ ModelSource = (schemas.Account)(nil)

// ModelsFromAccount enumerates every (provider, model) pair the account exposes
// and returns them as normalizer.ProviderModel values ready for New. It is the
// bridge between Bifrost's account configuration and the dependency-free core:
// the plugin's startup index (PRD section 8) is built from whatever models the
// providers' keys declare.
//
// Only names that could legitimately be a rewrite target are collected, mirroring
// the eligibility rules Bifrost itself applies when selecting a key
// (`key.Models.IsAllowed(model) && !key.BlacklistedModels.IsBlocked(model)`):
//
//   - A disabled key contributes nothing.
//   - The `*` wildcard is skipped. In Key.Models it means "any model is allowed"
//     (schemas.WhiteList.IsUnrestricted), not a model literally named `*`; a
//     provider configured that way accepts the caller's name as-is and needs no
//     rewriting. Indexing `*` would let a request be rewritten to `"*"`.
//   - A model blacklisted on the same key is skipped: rewriting a request to a
//     name that key refuses to serve would turn a working request into a routing
//     failure.
//
// Model names are de-duplicated per provider (the same model may appear across
// several keys for load-balancing) so a provider contributes each spelling once,
// preserving first-seen order for deterministic index buckets and logs.
//
// A provider whose keys enumerate no eligible models contributes nothing rather
// than failing: many providers accept any model name, and such a provider simply
// isn't matchable until it declares its catalogue. Errors from the account are
// surfaced (wrapped) so a genuine configuration failure at startup is not
// silently swallowed.
func ModelsFromAccount(ctx context.Context, src ModelSource) ([]normalizer.ProviderModel, error) {
	if src == nil {
		return nil, nil
	}
	providers, err := src.GetConfiguredProviders()
	if err != nil {
		return nil, fmt.Errorf("model-normalization: list providers: %w", err)
	}

	var models []normalizer.ProviderModel
	for _, provider := range providers {
		keys, err := src.GetKeysForProvider(ctx, provider)
		if err != nil {
			return nil, fmt.Errorf("model-normalization: keys for provider %q: %w", provider, err)
		}
		seen := make(map[string]struct{})
		for _, k := range keys {
			if k.Enabled != nil && !*k.Enabled {
				continue
			}
			for _, name := range k.Models {
				if name == "" || name == "*" {
					continue
				}
				if k.BlacklistedModels.IsBlocked(name) {
					continue
				}
				if _, dup := seen[name]; dup {
					continue
				}
				seen[name] = struct{}{}
				models = append(models, normalizer.ProviderModel{
					Provider: string(provider),
					Original: name,
				})
			}
		}
	}
	return models, nil
}

// NewFromAccount is the production constructor: it enumerates the account's
// models (ModelsFromAccount) and builds the plugin over them. Use it when wiring
// the plugin into a live Bifrost instance; use New directly in tests or when the
// model set is already known.
//
// The resulting plugin goes straight into schemas.BifrostConfig.LLMPlugins:
//
//	p, err := plugin.NewFromAccount(ctx, plugin.DefaultConfig(), account, logger)
//	if err != nil { return err }
//	bf, err := bifrost.Init(ctx, schemas.BifrostConfig{
//	    Account:    account,
//	    LLMPlugins: []schemas.LLMPlugin{p},
//	    Logger:     logger,
//	})
//
// The index is a snapshot of the account at construction time. Bifrost can gain
// providers, keys and models at runtime; those are not reflected until the plugin
// is rebuilt, and a request for a model added later simply misses and is
// forwarded verbatim (never rewritten to something stale).
func NewFromAccount(ctx context.Context, cfg Config, src ModelSource, logger schemas.Logger) (*Plugin, error) {
	models, err := ModelsFromAccount(ctx, src)
	if err != nil {
		return nil, err
	}
	return New(cfg, models, logger), nil
}
