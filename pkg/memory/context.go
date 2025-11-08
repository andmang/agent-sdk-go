package memory

import (
	"context"
)

// Key type for context values
type contextKey string

// ConversationIDKey is the key used to store conversation ID in context
const ConversationIDKey contextKey = "conversation_id"

// WithConversationID adds a conversation ID to the context
func WithConversationID(ctx context.Context, conversationID string) context.Context {
	return context.WithValue(ctx, ConversationIDKey, conversationID)
}

// GetConversationID retrieves the conversation ID from the context
func GetConversationID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(ConversationIDKey).(string)
	return id, ok
}
