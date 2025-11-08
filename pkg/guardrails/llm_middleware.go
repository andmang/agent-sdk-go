package guardrails

import (
	"context"

	"github.com/andmang/agent-sdk-go/pkg/interfaces"
)

// LLMMiddleware implements middleware for LLM calls
type LLMMiddleware struct {
	llm      interfaces.LLM
	pipeline *Pipeline
}

// NewLLMMiddleware creates a new LLM middleware
func NewLLMMiddleware(llm interfaces.LLM, pipeline *Pipeline) *LLMMiddleware {
	return &LLMMiddleware{
		llm:      llm,
		pipeline: pipeline,
	}
}

// Generate generates text from a prompt
func (m *LLMMiddleware) Generate(ctx context.Context, prompt string, options map[string]interface{}) (string, error) {
	// Process request through guardrails
	processedPrompt, err := m.pipeline.ProcessRequest(ctx, prompt)
	if err != nil {
		return "", err
	}

	// Call the underlying LLM
	// Pass an empty slice of options instead of nil to avoid nil pointer dereference
	response, err := m.llm.Generate(ctx, processedPrompt, []interfaces.GenerateOption{}...)
	if err != nil {
		return "", err
	}

	// Process response through guardrails
	processedResponse, err := m.pipeline.ProcessResponse(ctx, response)
	if err != nil {
		return "", err
	}

	return processedResponse, nil
}
