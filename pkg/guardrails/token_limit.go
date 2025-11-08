package guardrails

import (
	"context"
	"fmt"
	"strings"
)

// TokenCounter is an interface for counting tokens in text
type TokenCounter interface {
	CountTokens(text string) (int, error)
}

// SimpleTokenCounter implements a simple token counter
type SimpleTokenCounter struct{}

// CountTokens counts tokens in text (simple approximation)
func (s *SimpleTokenCounter) CountTokens(text string) (int, error) {
	// Simple approximation: count words and punctuation
	return len(strings.Fields(text)), nil
}

// TokenLimit implements a guardrail that limits the number of tokens
type TokenLimit struct {
	maxTokens    int
	counter      TokenCounter
	action       Action
	truncateMode string // "start", "end", or "middle"
}

// NewTokenLimit creates a new token limit guardrail
func NewTokenLimit(maxTokens int, counter TokenCounter, action Action, truncateMode string) *TokenLimit {
	if counter == nil {
		counter = &SimpleTokenCounter{}
	}

	if truncateMode == "" {
		truncateMode = "end"
	}

	return &TokenLimit{
		maxTokens:    maxTokens,
		counter:      counter,
		action:       action,
		truncateMode: truncateMode,
	}
}

// Type returns the type of guardrail
func (t *TokenLimit) Type() GuardrailType {
	return TokenLimitGuardrail
}

// CheckRequest checks if a request violates the guardrail
func (t *TokenLimit) CheckRequest(ctx context.Context, request string) (bool, string, error) {
	tokens, err := t.counter.CountTokens(request)
	if err != nil {
		return false, request, fmt.Errorf("failed to count tokens: %w", err)
	}

	if tokens > t.maxTokens {
		modified, err := t.truncate(request)
		if err != nil {
			return false, request, err
		}
		return true, modified, nil
	}

	return false, request, nil
}

// CheckResponse checks if a response violates the guardrail
func (t *TokenLimit) CheckResponse(ctx context.Context, response string) (bool, string, error) {
	tokens, err := t.counter.CountTokens(response)
	if err != nil {
		return false, response, fmt.Errorf("failed to count tokens: %w", err)
	}

	if tokens > t.maxTokens {
		modified, err := t.truncate(response)
		if err != nil {
			return false, response, err
		}
		return true, modified, nil
	}

	return false, response, nil
}

// Action returns the action to take when the guardrail is triggered
func (t *TokenLimit) Action() Action {
	return t.action
}

// truncate truncates text to the maximum token limit
func (t *TokenLimit) truncate(text string) (string, error) {
	words := strings.Fields(text)

	if len(words) <= t.maxTokens {
		return text, nil
	}

	switch t.truncateMode {
	case "start":
		return strings.Join(words[len(words)-t.maxTokens:], " "), nil
	case "middle":
		half := t.maxTokens / 2
		return strings.Join(words[:half], " ") + " ... " + strings.Join(words[len(words)-half:], " "), nil
	case "end":
		fallthrough
	default:
		return strings.Join(words[:t.maxTokens], " ") + " ...", nil
	}
}
