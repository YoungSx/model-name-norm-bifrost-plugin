package plugin

import (
	"context"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// fakeSource is a tiny in-memory ModelSource for exercising the account adapter
// without a live Bifrost account. providers is returned verbatim; keys maps a
// provider to the keys GetKeysForProvider yields. Either error field, when set,
// is returned from the corresponding method.
type fakeSource struct {
	providers    []schemas.ModelProvider
	keys         map[schemas.ModelProvider][]schemas.Key
	providersErr error
	keysErr      error
}

func (f *fakeSource) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return f.providers, f.providersErr
}

func (f *fakeSource) GetKeysForProvider(_ context.Context, p schemas.ModelProvider) ([]schemas.Key, error) {
	if f.keysErr != nil {
		return nil, f.keysErr
	}
	return f.keys[p], nil
}

func key(models ...string) schemas.Key {
	return schemas.Key{Models: models}
}

// TestModelsFromAccount_EnumeratesAndDedups covers the happy path: every
// (provider, model) pair is surfaced, and a model repeated across keys of the
// same provider (load-balancing) is contributed once, in first-seen order.
func TestModelsFromAccount_EnumeratesAndDedups(t *testing.T) {
	src := &fakeSource{
		providers: []schemas.ModelProvider{"zai", "anthropic"},
		keys: map[schemas.ModelProvider][]schemas.Key{
			"zai": {
				key("ZAI/glm-5.2", "glm-4"),
				key("ZAI/glm-5.2"), // duplicate across a second key
			},
			"anthropic": {
				key("Claude-4-Sonnet"),
			},
		},
	}

	models, err := ModelsFromAccount(context.Background(), src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// zai: ZAI/glm-5.2 (once), glm-4; anthropic: Claude-4-Sonnet = 3 pairs.
	if len(models) != 3 {
		t.Fatalf("got %d models, want 3: %+v", len(models), models)
	}
	// First-seen order within a provider must be preserved.
	if models[0].Provider != "zai" || models[0].Original != "ZAI/glm-5.2" {
		t.Fatalf("models[0] = %+v, want zai/ZAI/glm-5.2", models[0])
	}
	if models[1].Original != "glm-4" {
		t.Fatalf("models[1].Original = %q, want glm-4", models[1].Original)
	}
	if models[2].Provider != "anthropic" || models[2].Original != "Claude-4-Sonnet" {
		t.Fatalf("models[2] = %+v, want anthropic/Claude-4-Sonnet", models[2])
	}
}

// TestModelsFromAccount_EmptyModelsSkipped verifies a provider whose keys declare
// no models (or blank entries) contributes nothing rather than erroring.
func TestModelsFromAccount_EmptyModelsSkipped(t *testing.T) {
	src := &fakeSource{
		providers: []schemas.ModelProvider{"openai"},
		keys: map[schemas.ModelProvider][]schemas.Key{
			"openai": {key(), key("")},
		},
	}
	models, err := ModelsFromAccount(context.Background(), src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("empty/blank models should contribute nothing, got %+v", models)
	}
}

// TestModelsFromAccount_ProvidersError surfaces a provider-listing failure.
func TestModelsFromAccount_ProvidersError(t *testing.T) {
	sentinel := errors.New("boom")
	src := &fakeSource{providersErr: sentinel}
	if _, err := ModelsFromAccount(context.Background(), src); !errors.Is(err, sentinel) {
		t.Fatalf("want wrapped sentinel error, got %v", err)
	}
}

// TestModelsFromAccount_KeysError surfaces a per-provider key failure.
func TestModelsFromAccount_KeysError(t *testing.T) {
	sentinel := errors.New("keys down")
	src := &fakeSource{
		providers: []schemas.ModelProvider{"zai"},
		keysErr:   sentinel,
	}
	if _, err := ModelsFromAccount(context.Background(), src); !errors.Is(err, sentinel) {
		t.Fatalf("want wrapped sentinel error, got %v", err)
	}
}

// TestModelsFromAccount_NilSource tolerates a nil source (no models, no error).
func TestModelsFromAccount_NilSource(t *testing.T) {
	models, err := ModelsFromAccount(context.Background(), nil)
	if err != nil || models != nil {
		t.Fatalf("nil source should yield (nil, nil), got (%+v, %v)", models, err)
	}
}

// TestNewFromAccount_BuildsMatchablePlugin is the end-to-end wiring check: build
// the plugin straight from an account, then confirm a request is rewritten to the
// provider's registered spelling — proving the adapter feeds the index correctly.
func TestNewFromAccount_BuildsMatchablePlugin(t *testing.T) {
	src := &fakeSource{
		providers: []schemas.ModelProvider{"zai"},
		keys: map[schemas.ModelProvider][]schemas.Key{
			"zai": {key("ZAI/glm-5.2")},
		},
	}
	p, err := NewFromAccount(context.Background(), DefaultConfig(), src, nil)
	if err != nil {
		t.Fatalf("NewFromAccount error: %v", err)
	}
	req := chatReq("zai", "glm 5 2")
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatalf("PreRequestHook error: %v", err)
	}
	if got := modelOf(req); got != "ZAI/glm-5.2" {
		t.Fatalf("model rewritten to %q, want ZAI/glm-5.2", got)
	}
}
