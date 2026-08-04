// Command smoke is an end-to-end sanity check for the model-name normalization
// plugin. It does not use mocks of the plugin's own types: it implements the full
// Bifrost schemas.Account contract with an in-memory account, builds the plugin
// through the production constructor (plugin.NewFromAccount), and drives real
// schemas.BifrostRequest values through PreRequestHook — exactly the path a live
// Bifrost instance would take. Each case asserts the model was rewritten (or
// left alone) as the PRD requires; any deviation exits non-zero so this can gate
// a build.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/YoungSx/model-name-norm-bifrost-plugin/plugin"
	"github.com/maximhq/bifrost/core/schemas"
)

// smokeAccount is a complete, in-memory schemas.Account. It models three
// providers registering the same logical model under different spellings, plus a
// couple of distinct models — the multi-provider scenario from the PRD.
type smokeAccount struct {
	providers []schemas.ModelProvider
	models    map[schemas.ModelProvider][]string
}

func (a *smokeAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return a.providers, nil
}

func (a *smokeAccount) GetKeysForProvider(_ context.Context, p schemas.ModelProvider) ([]schemas.Key, error) {
	return []schemas.Key{{Models: a.models[p]}}, nil
}

func (a *smokeAccount) GetConfigForProvider(_ schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	return &schemas.ProviderConfig{}, nil
}

// compile-time proof the smoke account satisfies the real Bifrost contract.
var _ schemas.Account = (*smokeAccount)(nil)

func main() {
	acct := &smokeAccount{
		providers: []schemas.ModelProvider{"zai", "zhipu", "openrouter", "anthropic", "deepseek"},
		models: map[schemas.ModelProvider][]string{
			"zai":        {"ZAI/glm-5.2"},
			"zhipu":      {"智谱/GLM_5_2"},
			"openrouter": {"glm 5 2:free"},
			"anthropic":  {"Claude-4-Sonnet", "claude-4-sonnet-thinking"},
			"deepseek":   {"deepseek-ai/DeepSeek-V4-Flash-fast"},
		},
	}

	cfg := plugin.DefaultConfig()
	cfg.Matching.Fuzzy = true // exercise both exact and fuzzy paths

	p, err := plugin.NewFromAccount(context.Background(), cfg, acct, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "smoke: NewFromAccount failed: %v\n", err)
		os.Exit(1)
	}

	type tc struct {
		desc     string
		provider string
		input    string
		want     string // expected model after the hook
	}
	cases := []tc{
		{"zai underscore spelling → provider name", "zai", "GLM_5_2", "ZAI/glm-5.2"},
		{"zhipu space spelling → provider name", "zhipu", "glm 5 2", "智谱/GLM_5_2"},
		{"openrouter free-suffix → provider name", "openrouter", "glm-5.2:free", "glm 5 2:free"},
		{"deepseek stacked suffix → provider name", "deepseek", "deepseek-ai/deepseek-v4-flash-fast", "deepseek-ai/DeepSeek-V4-Flash-fast"},
		{"anthropic exact spelling preserved", "anthropic", "Claude-4-Sonnet", "Claude-4-Sonnet"},
		{"anthropic thinking variant not collapsed", "anthropic", "claude-4-sonnet-thinking", "claude-4-sonnet-thinking"},
		{"anthropic fuzzy suffix → base model", "anthropic", "claude-4-sonnet-instruct", "Claude-4-Sonnet"},
		{"wildcard whitelist not leaked to model", "zai", "*", "*"},
		{"cross-provider canonical not leaked", "anthropic", "glm-5.2", "glm-5.2"},
		{"unknown model forwarded verbatim", "zai", "totally-unknown", "totally-unknown"},
		{"version mismatch rejected under fuzzy", "anthropic", "claude-5-sonnet", "claude-5-sonnet"},
	}

	failures := 0
	for _, c := range cases {
		req := &schemas.BifrostRequest{
			ChatRequest: &schemas.BifrostChatRequest{
				Provider: schemas.ModelProvider(c.provider),
				Model:    c.input,
			},
		}
		if err := p.PreRequestHook(nil, req); err != nil {
			fmt.Fprintf(os.Stderr, "FAIL  %s: hook error: %v\n", c.desc, err)
			failures++
			continue
		}
		_, got, _ := req.GetRequestFields()
		status := "ok  "
		if got != c.want {
			status = "FAIL"
			failures++
		}
		fmt.Printf("%s  %-42s [%s] %q -> %q (want %q)\n", status, c.desc, c.provider, c.input, got, c.want)
	}

	fmt.Printf("\nindex canonicals: %d, cases: %d, failures: %d\n", p.IndexLen(), len(cases), failures)
	if failures > 0 {
		os.Exit(1)
	}
	fmt.Println("smoke: PASS")
}
