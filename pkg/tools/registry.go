package tools

import (
	"sync"

	"github.com/andmang/agent-sdk-go/pkg/interfaces"
)

// Registry implements the ToolRegistry interface
type Registry struct {
	tools map[string]interfaces.Tool
	mu    sync.RWMutex
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]interfaces.Tool),
	}
}

// Register registers a tool with the registry
func (r *Registry) Register(tool interfaces.Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[tool.Name()] = tool
}

// Get returns a tool by name
func (r *Registry) Get(name string) (interfaces.Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

// List returns all registered tools
func (r *Registry) List() []interfaces.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var tools []interfaces.Tool
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	return tools
}
