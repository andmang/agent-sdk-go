# LLM Providers

This document explains how to use the LLM (Large Language Model) providers in the Agent SDK.

## Overview

The Agent SDK supports multiple LLM providers, including OpenAI and Anthropic. Each provider has its own implementation but shares a common interface.

## Supported Providers

### OpenAI

```go
import (
    "github.com/andmang/agent-sdk-go/pkg/llm/openai"
    "github.com/andmang/agent-sdk-go/pkg/config"
)

// Get configuration
cfg := config.Get()

// Create OpenAI client
client := openai.NewClient(cfg.LLM.OpenAI.APIKey)

// Optional: Configure the client
client = openai.NewClient(
    cfg.LLM.OpenAI.APIKey,
    openai.WithModel("gpt-4o-mini"),
    openai.WithTemperature(0.7),
)
```

### Anthropic

```go
import (
    "github.com/andmang/agent-sdk-go/pkg/llm/anthropic"
    "github.com/andmang/agent-sdk-go/pkg/config"
)

// Get configuration
cfg := config.Get()

// Create Anthropic client
client := anthropic.NewClient(cfg.LLM.Anthropic.APIKey)

// Optional: Configure the client
client = anthropic.NewClient(
    cfg.LLM.Anthropic.APIKey,
    anthropic.WithModel("claude-3-opus-20240229"),
    anthropic.WithTemperature(0.7),
)
```

## Using LLM Providers

### Text Generation

Generate text based on a prompt:

```go
import "context"

// Generate text
response, err := client.Generate(context.Background(), "What is the capital of France?")
if err != nil {
    log.Fatalf("Failed to generate text: %v", err)
}
fmt.Println(response)
```

### Chat Completion

Generate a response to a conversation:

```go
import (
    "context"
    "github.com/andmang/agent-sdk-go/pkg/llm"
)

// Create messages
messages := []llm.Message{
    {
        Role:    "system",
        Content: "You are a helpful AI assistant.",
    },
    {
        Role:    "user",
        Content: "What is the capital of France?",
    },
}

// Generate chat completion
response, err := client.Chat(context.Background(), messages)
if err != nil {
    log.Fatalf("Failed to generate chat completion: %v", err)
}
fmt.Println(response)
```

### Generation with Tools

Generate a response that can use tools:

```go
import (
    "context"
    "github.com/andmang/agent-sdk-go/pkg/interfaces"
    "github.com/andmang/agent-sdk-go/pkg/tools/websearch"
)

// Create tools
searchTool := websearch.New(googleAPIKey, googleSearchEngineID)

// Generate with tools
response, err := client.GenerateWithTools(
    context.Background(),
    "What's the weather in San Francisco?",
    []interfaces.Tool{searchTool},
)
if err != nil {
    log.Fatalf("Failed to generate with tools: %v", err)
}
fmt.Println(response)
```

## Configuration Options

### Common Options

These options are available for all LLM providers:

```go
// Temperature controls randomness (0.0 to 1.0)
WithTemperature(0.7)

// TopP controls diversity via nucleus sampling
WithTopP(0.9)

// FrequencyPenalty reduces repetition of token sequences
WithFrequencyPenalty(0.0)

// PresencePenalty reduces repetition of topics
WithPresencePenalty(0.0)

// StopSequences specifies sequences that stop generation
WithStopSequences([]string{"###"})

// Reasoning controls how the model explains its thinking
// Options: "none", "minimal", "comprehensive"
WithReasoning("minimal")
```
