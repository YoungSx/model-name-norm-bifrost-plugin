//go:build integration

package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/normalizer"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
)

// TestIntegration_RewritesModelOnLiveBifrost is the end-to-end proof the plugin
// works inside a real Bifrost: bring up bifrost.Init with the plugin in
// LLMPlugins, point the OpenAI provider at a local echo server, send a loosely
// spelled chat request, and assert the server actually received the provider's
// registered model name — i.e. that PreRequestHook rewrote it before dispatch.
//
// Run with: go test -tags integration ./plugin -run Integration -v
// (gated behind a build tag so `go test ./...` stays hermetic and offline-free.)
func TestIntegration_RewritesModelOnLiveBifrost(t *testing.T) {
	var (
		mu       sync.Mutex
		gotModel string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("echo server: bad JSON: %v (body=%q)", err, string(body))
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		mu.Lock()
		gotModel = payload.Model
		mu.Unlock()

		// Minimal OpenAI-shaped non-streaming chat response.
		resp := map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"model":   payload.Model,
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Account that declares the OpenAI provider with the registered model names,
	// exactly as ModelsFromAccount consumes them. The "ZAI/glm-5.2" prefix models
	// are loaded across two providers (OpenAI-keyed here) so the plugin can show a
	// cross-spelling rewrite from the loose spelling "GLM_5_2".
	acct := &integrationAccount{
		providers: []schemas.ModelProvider{schemas.OpenAI},
		configs: map[schemas.ModelProvider]*schemas.ProviderConfig{
			schemas.OpenAI: {
				NetworkConfig: schemas.NetworkConfig{
					BaseURL:                        server.URL,
					DefaultRequestTimeoutInSeconds: 30,
				},
			},
		},
		keys: map[schemas.ModelProvider][]schemas.Key{
			schemas.OpenAI: {{
				// OpenAI rejects keyless requests (CanProviderKeyValueBeEmpty=false),
				// so the key must carry a non-empty Value or Bifrost's key pool
				// filter rejects it before the echo server is ever reached.
				Value:  schemas.SecretVar{Val: "sk-test"},
				Models: []string{"ZAI/glm-5.2", "gpt-5.2", "Claude-4-Sonnet", "claude-4-sonnet-thinking"},
			}},
		},
	}

	// Build the plugin from the same account, register it with Bifrost, init.
	p, err := NewFromAccount(context.Background(), DefaultConfig(), acct, nil)
	if err != nil {
		t.Fatalf("NewFromAccount: %v", err)
	}

	bf, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account:    acct,
		LLMPlugins: []schemas.LLMPlugin{p},
		Logger:     &silentLogger{},
	})
	if err != nil {
		t.Fatalf("bifrost.Init: %v", err)
	}
	defer bf.Shutdown()

	bctx, cancel := schemas.NewBifrostContextWithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Loose spelling of a registered model. The provider registered "ZAI/glm-5.2";
	// "GLM_5_2" canonicalizes to the same loose key, so the dispatched model name
	// must be the registered one, not the caller's.
	req := &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "GLM_5_2",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: ptrContent("hi")}},
	}
	resp, bifErr := bf.ChatCompletionRequest(bctx, req)
	if bifErr != nil {
		t.Fatalf("ChatCompletionRequest error: %+v", bifErr)
	}
	if resp == nil {
		t.Fatal("nil response with no error")
	}

	mu.Lock()
	gm := gotModel
	mu.Unlock()
	if gm != "ZAI/glm-5.2" {
		t.Fatalf("downstream model = %q, want ZAI/glm-5.2 (plugin did not rewrite before dispatch)", gm)
	}
}

// TestIntegration_VariantPreservedOnLiveBifrost verifies the strict tier inside
// a real Bifrost: an explicit request for the thinking variant reaches the
// provider as the variant, not the base model.
func TestIntegration_VariantPreservedOnLiveBifrost(t *testing.T) {
	var (
		mu       sync.Mutex
		gotModel string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &payload)
		mu.Lock()
		gotModel = payload.Model
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}},
		})
	}))
	defer server.Close()

	acct := &integrationAccount{
		providers: []schemas.ModelProvider{schemas.OpenAI},
		configs: map[schemas.ModelProvider]*schemas.ProviderConfig{
			schemas.OpenAI: {NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, DefaultRequestTimeoutInSeconds: 30}},
		},
		keys: map[schemas.ModelProvider][]schemas.Key{
			schemas.OpenAI: {{
				Value:  schemas.SecretVar{Val: "sk-test"},
				Models: []string{"Claude-4-Sonnet", "claude-4-sonnet-thinking"},
			}},
		},
	}
	p, err := NewFromAccount(context.Background(), DefaultConfig(), acct, nil)
	if err != nil {
		t.Fatalf("NewFromAccount: %v", err)
	}
	bf, err := bifrost.Init(context.Background(), schemas.BifrostConfig{Account: acct, LLMPlugins: []schemas.LLMPlugin{p}, Logger: &silentLogger{}})
	if err != nil {
		t.Fatalf("bifrost.Init: %v", err)
	}
	defer bf.Shutdown()

	bctx, cancel := schemas.NewBifrostContextWithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "claude-4-sonnet-thinking",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: ptrContent("hi")}},
	}
	if _, bifErr := bf.ChatCompletionRequest(bctx, req); bifErr != nil {
		t.Fatalf("ChatCompletionRequest error: %+v", bifErr)
	}
	mu.Lock()
	gm := gotModel
	mu.Unlock()
	if gm != "claude-4-sonnet-thinking" {
		t.Fatalf("downstream model = %q, want claude-4-sonnet-thinking (variant collapsed in Bifrost)", gm)
	}
}

// integrationAccount is a full in-memory schemas.Account for driving bifrost.Init
// without a config file or real network.
type integrationAccount struct {
	providers []schemas.ModelProvider
	configs   map[schemas.ModelProvider]*schemas.ProviderConfig
	keys      map[schemas.ModelProvider][]schemas.Key
}

func (a *integrationAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return a.providers, nil
}
func (a *integrationAccount) GetKeysForProvider(_ context.Context, p schemas.ModelProvider) ([]schemas.Key, error) {
	return a.keys[p], nil
}
func (a *integrationAccount) GetConfigForProvider(p schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	cfg, ok := a.configs[p]
	if !ok {
		return nil, fmt.Errorf("provider not configured: %s", p)
	}
	return cfg, nil
}

var _ schemas.Account = (*integrationAccount)(nil)

type silentLogger struct{}

func (*silentLogger) Debug(string, ...any)              {}
func (*silentLogger) Info(string, ...any)               {}
func (*silentLogger) Warn(string, ...any)               {}
func (*silentLogger) Error(string, ...any)              {}
func (*silentLogger) Fatal(string, ...any)             {}
func (*silentLogger) SetLevel(schemas.LogLevel)         {}
func (*silentLogger) SetOutputType(schemas.LoggerOutputType) {}
func (*silentLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

var _ schemas.Logger = (*silentLogger)(nil)

// pathContent wraps a plain string in the pointer-shaped Content field Bifrost
// requires, so a minimal "hi" user message renders as JSON {"role":"user","content":"hi"}.
func ptrContent(s string) *schemas.ChatMessageContent {
	cs := s
	return &schemas.ChatMessageContent{ContentStr: &cs}
}

// keep normalizer import even if unused in some build paths.
var _ = normalizer.MatchNone
