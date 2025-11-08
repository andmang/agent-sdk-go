package guardrails

import (
	"context"
	"regexp"
	"strings"
)

// ContentFilter implements a guardrail that filters inappropriate content
type ContentFilter struct {
	blockedWords []string
	action       Action
	regex        *regexp.Regexp
}

// NewContentFilter creates a new content filter guardrail
func NewContentFilter(blockedWords []string, action Action) *ContentFilter {
	// Escape special characters and join with OR
	pattern := strings.Join(blockedWords, "|")
	regex := regexp.MustCompile(`(?i)\b(` + pattern + `)\b`)

	return &ContentFilter{
		blockedWords: blockedWords,
		action:       action,
		regex:        regex,
	}
}

// Type returns the type of guardrail
func (c *ContentFilter) Type() GuardrailType {
	return ContentFilterGuardrail
}

// CheckRequest checks if a request violates the guardrail
func (c *ContentFilter) CheckRequest(ctx context.Context, request string) (bool, string, error) {
	if c.regex.MatchString(request) {
		modified := c.regex.ReplaceAllString(request, "****")
		return true, modified, nil
	}
	return false, request, nil
}

// CheckResponse checks if a response violates the guardrail
func (c *ContentFilter) CheckResponse(ctx context.Context, response string) (bool, string, error) {
	if c.regex.MatchString(response) {
		modified := c.regex.ReplaceAllString(response, "****")
		return true, modified, nil
	}
	return false, response, nil
}

// Action returns the action to take when the guardrail is triggered
func (c *ContentFilter) Action() Action {
	return c.action
}
