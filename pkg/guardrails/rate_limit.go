package guardrails

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/andmang/agent-sdk-go/pkg/multitenancy"
)

// RateLimit implements a guardrail that limits the rate of requests
type RateLimit struct {
	requestsPerMinute int
	requestCounts     map[string][]time.Time
	mu                sync.Mutex
	action            Action
}

// NewRateLimit creates a new rate limit guardrail
func NewRateLimit(requestsPerMinute int, action Action) *RateLimit {
	return &RateLimit{
		requestsPerMinute: requestsPerMinute,
		requestCounts:     make(map[string][]time.Time),
		action:            action,
	}
}

// Type returns the type of guardrail
func (r *RateLimit) Type() GuardrailType {
	return RateLimitGuardrail
}

// CheckRequest checks if a request violates the guardrail
func (r *RateLimit) CheckRequest(ctx context.Context, request string) (bool, string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Get organization ID from context
	orgID, err := multitenancy.GetOrgID(ctx)
	if err != nil {
		// If no organization ID is found, use a default key
		orgID = "default"
	}

	// Get current time
	now := time.Now()

	// Clean up old requests (older than 1 minute)
	var recentRequests []time.Time
	for _, t := range r.requestCounts[orgID] {
		if now.Sub(t) < time.Minute {
			recentRequests = append(recentRequests, t)
		}
	}
	r.requestCounts[orgID] = recentRequests

	// Check if rate limit is exceeded
	if len(recentRequests) >= r.requestsPerMinute {
		return true, fmt.Sprintf("Rate limit exceeded: %d requests per minute", r.requestsPerMinute), nil
	}

	// Add current request to count
	r.requestCounts[orgID] = append(r.requestCounts[orgID], now)

	return false, request, nil
}

// CheckResponse checks if a response violates the guardrail
func (r *RateLimit) CheckResponse(ctx context.Context, response string) (bool, string, error) {
	// Rate limits typically apply to requests, not responses
	return false, response, nil
}

// Action returns the action to take when the guardrail is triggered
func (r *RateLimit) Action() Action {
	return r.action
}
