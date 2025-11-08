package guardrails

import (
	"context"
	"regexp"
)

// PiiFilter implements a guardrail that filters personally identifiable information
type PiiFilter struct {
	patterns map[string]*regexp.Regexp
	action   Action
}

// NewPiiFilter creates a new PII filter guardrail
func NewPiiFilter(action Action) *PiiFilter {
	patterns := map[string]*regexp.Regexp{
		"email":       regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`),
		"phone":       regexp.MustCompile(`\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b`),
		"ssn":         regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
		"credit_card": regexp.MustCompile(`\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b`),
		"ip_address":  regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
	}

	return &PiiFilter{
		patterns: patterns,
		action:   action,
	}
}

// Type returns the type of guardrail
func (p *PiiFilter) Type() GuardrailType {
	return PiiFilterGuardrail
}

// CheckRequest checks if a request violates the guardrail
func (p *PiiFilter) CheckRequest(ctx context.Context, request string) (bool, string, error) {
	modified := request
	triggered := false

	for name, pattern := range p.patterns {
		if pattern.MatchString(modified) {
			triggered = true
			modified = pattern.ReplaceAllString(modified, "[REDACTED "+name+"]")
		}
	}

	return triggered, modified, nil
}

// CheckResponse checks if a response violates the guardrail
func (p *PiiFilter) CheckResponse(ctx context.Context, response string) (bool, string, error) {
	modified := response
	triggered := false

	for name, pattern := range p.patterns {
		if pattern.MatchString(modified) {
			triggered = true
			modified = pattern.ReplaceAllString(modified, "[REDACTED "+name+"]")
		}
	}

	return triggered, modified, nil
}

// Action returns the action to take when the guardrail is triggered
func (p *PiiFilter) Action() Action {
	return p.action
}
