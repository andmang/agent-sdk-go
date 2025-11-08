package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/andmang/agent-sdk-go/pkg/interfaces"
	"github.com/andmang/agent-sdk-go/pkg/llm"
	"github.com/andmang/agent-sdk-go/pkg/logging"
	"github.com/andmang/agent-sdk-go/pkg/multitenancy"
	"github.com/andmang/agent-sdk-go/pkg/retry"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
)

// Define a custom type for context keys to avoid collisions
type contextKey string

// Define constants for context keys
const organizationKey contextKey = "organization"

// OpenAIClient implements the LLM interface for OpenAI
type OpenAIClient struct {
	Client          openai.Client
	ChatService     openai.ChatService
	ResponseService openai.Client
	Model           string
	apiKey          string
	baseURL         string
	logger          logging.Logger
	retryExecutor   *retry.Executor
}

// Option represents an option for configuring the OpenAI client
type Option func(*OpenAIClient)

// WithModel sets the model for the OpenAI client
func WithModel(model string) Option {
	return func(c *OpenAIClient) {
		c.Model = model
	}
}

// isReasoningModel returns true if the model is a reasoning model that requires temperature = 1
func isReasoningModel(model string) bool {
	reasoningModels := []string{
		"o1-", "o1-mini", "o1-preview",
		"o3-", "o3-mini",
		"o4-", "o4-mini",
		"gpt-5", "gpt-5-mini", "gpt-5-nano",
	}

	for _, prefix := range reasoningModels {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// getTemperatureForModel returns the appropriate temperature for a model
func (c *OpenAIClient) getTemperatureForModel(requestedTemp float64) float64 {
	if isReasoningModel(c.Model) {
		if requestedTemp != 1.0 {
			c.logger.Debug(context.Background(), "Overriding temperature for reasoning model", map[string]interface{}{
				"model":                 c.Model,
				"requested_temperature": requestedTemp,
				"forced_temperature":    1.0,
				"reason":                "reasoning models only support temperature = 1",
			})
		}
		return 1.0
	}
	return requestedTemp
}

// WithLogger sets the logger for the OpenAI client
func WithLogger(logger logging.Logger) Option {
	return func(c *OpenAIClient) {
		c.logger = logger
	}
}

// WithRetry configures retry policy for the client
func WithRetry(opts ...retry.Option) Option {
	return func(c *OpenAIClient) {
		c.retryExecutor = retry.NewExecutor(retry.NewPolicy(opts...))
	}
}

// WithBaseURL sets the base URL for the OpenAI client
func WithBaseURL(baseURL string) Option {
	return func(c *OpenAIClient) {
		c.baseURL = baseURL
		// Recreate the client and services with the new base URL
		c.Client = openai.NewClient(option.WithAPIKey(c.apiKey), option.WithBaseURL(baseURL))
		c.ChatService = openai.NewChatService(option.WithAPIKey(c.apiKey), option.WithBaseURL(baseURL))
		c.ResponseService = openai.NewClient(option.WithAPIKey(c.apiKey), option.WithBaseURL(baseURL))
	}
}

// NewClient creates a new OpenAI client
func NewClient(apiKey string, options ...Option) *OpenAIClient {
	// Create client with default options
	client := &OpenAIClient{
		Client:          openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL("https://api.openai.com/v1")),
		ChatService:     openai.NewChatService(option.WithAPIKey(apiKey), option.WithBaseURL("https://api.openai.com/v1")),
		ResponseService: openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL("https://api.openai.com/v1")),
		Model:           "gpt-4o-mini",
		apiKey:          apiKey,
		baseURL:         "https://api.openai.com/v1",
		logger:          logging.New(),
	}

	// Apply options
	for _, option := range options {
		option(client)
	}

	return client
}

// Generate generates text from a prompt
func (c *OpenAIClient) Generate(ctx context.Context, prompt string, options ...interfaces.GenerateOption) (string, error) {
	// Apply options
	params := &interfaces.GenerateOptions{
		LLMConfig: &interfaces.LLMConfig{
			Temperature: 0.7,
		},
	}

	for _, option := range options {
		option(params)
	}

	// Get organization ID from context if available
	orgID, _ := multitenancy.GetOrgID(ctx)
	if orgID != "" {
		ctx = context.WithValue(ctx, organizationKey, orgID)
	}

	// Build messages starting with memory context
	messages := []openai.ChatCompletionMessageParamUnion{}

	// Add system message if available
	if params.SystemMessage != "" {
		messages = append(messages, openai.SystemMessage(params.SystemMessage))
		c.logger.Debug(ctx, "Using system message", map[string]interface{}{"system_message": params.SystemMessage})
	}

	// Build messages using unified builder
	builder := newMessageHistoryBuilder(c.logger)
	messages = append(messages, builder.buildMessages(ctx, prompt, params.Memory)...)

	// Create request
	req := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(c.Model),
		Messages: messages,
	}

	if params.LLMConfig != nil {
		req.Temperature = openai.Float(c.getTemperatureForModel(params.LLMConfig.Temperature))
		// Reasoning models don't support top_p parameter
		if !isReasoningModel(c.Model) {
			req.TopP = openai.Float(params.LLMConfig.TopP)
		}
		req.FrequencyPenalty = openai.Float(params.LLMConfig.FrequencyPenalty)
		req.PresencePenalty = openai.Float(params.LLMConfig.PresencePenalty)
		if len(params.LLMConfig.StopSequences) > 0 {
			req.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: params.LLMConfig.StopSequences}
		}
		// Set reasoning effort for reasoning models
		if isReasoningModel(c.Model) && params.LLMConfig.Reasoning != "" {
			req.ReasoningEffort = shared.ReasoningEffort(params.LLMConfig.Reasoning)
			c.logger.Debug(ctx, "Setting reasoning effort", map[string]interface{}{"reasoning_effort": params.LLMConfig.Reasoning})
		}
	}

	// Set response format if provided
	if params.ResponseFormat != nil {
		// Convert to the new API's response format structure
		jsonSchema := shared.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   params.ResponseFormat.Name,
			Schema: params.ResponseFormat.Schema,
		}

		req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				Type:       "json_schema",
				JSONSchema: jsonSchema,
			},
		}
		c.logger.Debug(ctx, "Using response format", map[string]interface{}{"format": *params.ResponseFormat})
	}

	// Set organization ID if available
	if orgID, ok := ctx.Value(organizationKey).(string); ok && orgID != "" {
		req.User = openai.String(orgID)
	}

	var resp *openai.ChatCompletion
	var err error

	operation := func() error {
		var reasoningEffort string
		if params.LLMConfig != nil && params.LLMConfig.Reasoning != "" {
			reasoningEffort = params.LLMConfig.Reasoning
		} else {
			reasoningEffort = "none"
		}

		c.logger.Debug(ctx, "Executing OpenAI API request", map[string]interface{}{
			"model":             c.Model,
			"temperature":       req.Temperature,
			"top_p":             req.TopP,
			"frequency_penalty": req.FrequencyPenalty,
			"presence_penalty":  req.PresencePenalty,
			"stop_sequences":    req.Stop,
			"messages":          len(req.Messages),
			"response_format":   params.ResponseFormat != nil,
			"reasoning_effort":  reasoningEffort,
		})

		resp, err = c.ChatService.Completions.New(ctx, req)
		if err != nil {
			c.logger.Error(ctx, "Error from OpenAI API", map[string]interface{}{
				"error": err.Error(),
				"model": c.Model,
			})
			return fmt.Errorf("failed to generate text: %w", err)
		}
		return nil
	}

	if c.retryExecutor != nil {
		c.logger.Debug(ctx, "Using retry mechanism for OpenAI request", map[string]interface{}{
			"model": c.Model,
		})
		err = c.retryExecutor.Execute(ctx, operation)
	} else {
		err = operation()
	}

	if err != nil {
		return "", err
	}

	// Return response
	if len(resp.Choices) > 0 {
		c.logger.Debug(ctx, "Successfully received response from OpenAI", map[string]interface{}{
			"model": c.Model,
		})
		return resp.Choices[0].Message.Content, nil
	}

	return "", fmt.Errorf("no response from OpenAI API")
}

// Chat uses the ChatCompletion API to have a conversation (messages) with a model
func (c *OpenAIClient) Chat(ctx context.Context, messages []llm.Message, params *llm.GenerateParams) (string, error) {
	if params == nil {
		params = llm.DefaultGenerateParams()
	}

	// Convert messages to the OpenAI Chat format
	chatMessages := make([]openai.ChatCompletionMessageParamUnion, len(messages))
	for i, msg := range messages {
		switch msg.Role {
		case "system":
			chatMessages[i] = openai.SystemMessage(msg.Content)
		case "user":
			chatMessages[i] = openai.UserMessage(msg.Content)
		case "assistant":
			chatMessages[i] = openai.AssistantMessage(msg.Content)
		case "tool":
			// For tool messages, we need to handle tool call ID
			// Use the ToolCallID from the Message struct
			chatMessages[i] = openai.ToolMessage(msg.Content, msg.ToolCallID)
		default:
			// Default to user message for unknown roles
			chatMessages[i] = openai.UserMessage(msg.Content)
		}
	}

	// Create chat request
	req := openai.ChatCompletionNewParams{
		Model:            openai.ChatModel(c.Model),
		Messages:         chatMessages,
		Temperature:      openai.Float(c.getTemperatureForModel(params.Temperature)),
		FrequencyPenalty: openai.Float(params.FrequencyPenalty),
		PresencePenalty:  openai.Float(params.PresencePenalty),
	}

	// Reasoning models don't support top_p parameter
	if !isReasoningModel(c.Model) {
		req.TopP = openai.Float(params.TopP)
	}

	if len(params.StopSequences) > 0 {
		req.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: params.StopSequences}
	}

	// Set reasoning effort for reasoning models
	if isReasoningModel(c.Model) && params.Reasoning != "" {
		req.ReasoningEffort = shared.ReasoningEffort(params.Reasoning)
		c.logger.Debug(ctx, "Setting reasoning effort", map[string]interface{}{"reasoning_effort": params.Reasoning})
	}

	var resp *openai.ChatCompletion
	var err error

	operation := func() error {
		c.logger.Debug(ctx, "Executing OpenAI Chat API request", map[string]interface{}{
			"model":             c.Model,
			"temperature":       req.Temperature,
			"top_p":             req.TopP,
			"frequency_penalty": req.FrequencyPenalty,
			"presence_penalty":  req.PresencePenalty,
			"stop_sequences":    req.Stop,
			"messages":          len(req.Messages),
			"reasoning_effort":  params.Reasoning,
		})

		resp, err = c.ChatService.Completions.New(ctx, req)
		if err != nil {
			c.logger.Error(ctx, "Error from OpenAI Chat API", map[string]interface{}{
				"error": err.Error(),
				"model": c.Model,
			})
			return fmt.Errorf("failed to create chat completion: %w", err)
		}
		return nil
	}

	if c.retryExecutor != nil {
		c.logger.Debug(ctx, "Using retry mechanism for OpenAI Chat request", map[string]interface{}{
			"model": c.Model,
		})
		err = c.retryExecutor.Execute(ctx, operation)
	} else {
		err = operation()
	}

	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no completions returned")
	}

	c.logger.Debug(ctx, "Successfully received chat response from OpenAI", map[string]interface{}{
		"model": c.Model,
	})

	return resp.Choices[0].Message.Content, nil
}

func (c *OpenAIClient) GenerateWithTools(ctx context.Context, prompt string, tools []interfaces.Tool, options ...interfaces.GenerateOption) (string, error) {
	// Convert options to params
	params := &interfaces.GenerateOptions{}
	for _, opt := range options {
		if opt != nil {
			opt(params)
		}
	}

	if params.LLMConfig == nil {
		params.LLMConfig = &interfaces.LLMConfig{
			Temperature: 0.7,
			TopP:        1.0,
		}
	}
	maxIterations := params.MaxIterations
	if maxIterations == 0 {
		maxIterations = 10 // A reasonable default
	}

	// === START OF THE FIX ===
	// Build the list of tool definitions ONCE.
	openaiTools := make([]openai.ChatCompletionToolUnionParam, len(tools))
	for i, tool := range tools {
		properties := make(map[string]interface{})
		required := []string{}

		for name, param := range tool.Parameters() {
			paramSchema := map[string]interface{}{
				"description": param.Description,
			}

			paramType := param.Type
			if paramType == "integer" {
				paramType = "number"
			}
			paramSchema["type"] = paramType

			if paramType == "array" {
				itemsSchema := map[string]interface{}{}
				if param.Items != nil && param.Items.Type != "" {
					itemsSchema["type"] = param.Items.Type
				} else {
					itemsSchema["type"] = "string" // Default for arrays missing item spec
				}
				paramSchema["items"] = itemsSchema
			}

			if param.Enum != nil {
				paramSchema["enum"] = param.Enum
			}

			properties[name] = paramSchema
			if param.Required {
				required = append(required, name)
			}
		}

		// This correctly creates a single tool definition.
		toolDef := shared.FunctionDefinitionParam{
			Name:        tool.Name(),
			Description: openai.String(tool.Description()),
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": properties,
				"required":   required,
			},
		}
		openaiTools[i] = openai.ChatCompletionFunctionTool(toolDef)
	}
	// === END OF THE FIX ===

	// Build message history
	builder := newMessageHistoryBuilder(c.logger)
	messages := builder.buildMessages(ctx, prompt, params.Memory)

	if params.SystemMessage != "" {
		// Ensure system message is at the start
		messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(params.SystemMessage)}, messages...)
	}

	// Loop for tool calls
	for iteration := 0; iteration < maxIterations; iteration++ {
		req := openai.ChatCompletionNewParams{
			Model:            openai.ChatModel(c.Model),
			Messages:         messages,
			Tools:            openaiTools,
			Temperature:      openai.Float(c.getTemperatureForModel(params.LLMConfig.Temperature)),
			FrequencyPenalty: openai.Float(params.LLMConfig.FrequencyPenalty),
			PresencePenalty:  openai.Float(params.LLMConfig.PresencePenalty),
		}
		if !isReasoningModel(c.Model) {
			req.TopP = openai.Float(params.LLMConfig.TopP)
			req.ParallelToolCalls = openai.Bool(true)
		}
		if len(params.LLMConfig.StopSequences) > 0 {
			req.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: params.LLMConfig.StopSequences}
		}

		resp, err := c.ChatService.Completions.New(ctx, req)
		if err != nil {
			c.logger.Error(ctx, "Error from OpenAI API", map[string]interface{}{"error": err.Error()})
			return "", fmt.Errorf("failed to create chat completion: %w", err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no completions returned")
		}

		responseMessage := resp.Choices[0].Message
		messages = append(messages, responseMessage.ToParam())

		if len(responseMessage.ToolCalls) == 0 {
			return strings.TrimSpace(responseMessage.Content), nil
		}

		// Process tool calls and append results to messages for the next loop iteration
		for _, toolCall := range responseMessage.ToolCalls {
			var selectedTool interfaces.Tool
			for _, t := range tools {
				if t.Name() == toolCall.Function.Name {
					selectedTool = t
					break
				}
			}

			var toolResultContent string
			if selectedTool == nil {
				toolResultContent = fmt.Sprintf("Error: tool not found: %s", toolCall.Function.Name)
			} else {
				result, err := selectedTool.Execute(ctx, toolCall.Function.Arguments)
				if err != nil {
					toolResultContent = fmt.Sprintf("Error: %v", err)
				} else {
					toolResultContent = result
				}
			}
			messages = append(messages, openai.ToolMessage(toolResultContent, toolCall.ID))
		}
	}

	return "", fmt.Errorf("max iterations reached without a final answer")
}

// Name implements interfaces.LLM.Name
func (c *OpenAIClient) Name() string {
	return "openai"
}

// SupportsStreaming implements interfaces.LLM.SupportsStreaming
func (c *OpenAIClient) SupportsStreaming() bool {
	return true
}

// GetModel returns the model name being used
func (c *OpenAIClient) GetModel() string {
	return c.Model
}

// WithTemperature creates a GenerateOption to set the temperature
func WithTemperature(temperature float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.Temperature = temperature
	}
}

// WithTopP creates a GenerateOption to set the top_p
func WithTopP(topP float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.TopP = topP
	}
}

// WithFrequencyPenalty creates a GenerateOption to set the frequency penalty
func WithFrequencyPenalty(frequencyPenalty float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.FrequencyPenalty = frequencyPenalty
	}
}

// WithPresencePenalty creates a GenerateOption to set the presence penalty
func WithPresencePenalty(presencePenalty float64) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.PresencePenalty = presencePenalty
	}
}

// WithStopSequences creates a GenerateOption to set the stop sequences
func WithStopSequences(stopSequences []string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.LLMConfig.StopSequences = stopSequences
	}
}

// WithSystemMessage creates a GenerateOption to set the system message
func WithSystemMessage(systemMessage string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.SystemMessage = systemMessage
	}
}

// WithResponseFormat creates a GenerateOption to set the response format
func WithResponseFormat(format interfaces.ResponseFormat) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		options.ResponseFormat = &format
	}
}

// WithReasoning creates a GenerateOption to set the reasoning effort for reasoning models
// For OpenAI reasoning models (o1, o3, o4, gpt-5 series), valid values are:
// "minimal", "low", "medium", "high"
// This parameter is only used with reasoning models and is ignored for standard models.
func WithReasoning(reasoning string) interfaces.GenerateOption {
	return func(options *interfaces.GenerateOptions) {
		if options.LLMConfig == nil {
			options.LLMConfig = &interfaces.LLMConfig{}
		}
		options.LLMConfig.Reasoning = reasoning
	}
}
