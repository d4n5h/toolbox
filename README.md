# Toolbox

A TypeScript/JavaScript vector store for LangChain tools with semantic search. Manage hundreds or thousands of tools and dynamically select only the relevant ones for each agent. Similar to LangGraph's BigTool, but for the JS/TS ecosystem.

## Features

- **Semantic Search** - Retrieve tools using natural language queries with vector embeddings
- **Any Vector Store** - Works with Pinecone, Chroma, Weaviate, FAISS, Qdrant, Milvus, MongoDB Atlas, Azure AI Search, or in-memory
- **Any Embeddings** - Supports Ollama, OpenAI, HuggingFace, Cohere, and any LangChain-compatible embeddings
- **Tool Management** - Namespaces (flat/hierarchical), metadata filtering, versioning, relationships, batch operations
- **Analytics & Optimization** - Usage tracking, query caching, tool validation
- **Schema Introspection** - Enhanced Zod v4 schema extraction with descriptions and examples
- **LangGraph Integration** - Dynamic tool loading with state-based discovery

## Installation

```bash
bun install toolbox
# or
npm install toolbox
```

## Quick Start

```typescript
import { Toolbox } from "toolbox";
import { OllamaEmbeddings } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Create an embeddings model
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
});

// Initialize the toolbox
const toolbox = new Toolbox({
  embeddings,
  defaultRetrievalLimit: 5,
});

// Create a tool
const calculatorTool = tool(
  async (input: { a: number; b: number; operation: string }) => {
    const { a, b, operation } = input;
    switch (operation) {
      case "add":
        return (a + b).toString();
      case "subtract":
        return (a - b).toString();
      case "multiply":
        return (a * b).toString();
      case "divide":
        return b !== 0 ? (a / b).toString() : "Error: Division by zero";
      default:
        return "Error: Unknown operation";
    }
  },
  {
    name: "calculator",
    description: "Performs basic arithmetic operations on two numbers",
    schema: z.object({
      a: z.number().describe("The first number"),
      b: z.number().describe("The second number"),
      operation: z.enum(["add", "subtract", "multiply", "divide"]).describe("The operation to perform"),
    }),
  }
);

// Add tools to the toolbox
await toolbox.addTool(calculatorTool);

// Retrieve relevant tools based on a query
const results = await toolbox.retrieveTools("math calculation");
for (const result of results) {
  console.log(`${result.tool.name} (score: ${result.score.toFixed(4)})`);
}
```

## API Reference

### Constructor

```typescript
new Toolbox(options: ToolboxOptions)
```

**Options (choose one):**

#### Option 1: Custom vector store

```typescript
{
  vectorStore: VectorStore,        // Any LangChain vector store
  defaultRetrievalLimit?: number,  // Default: 5
  enableValidation?: boolean,       // Default: false
  enableCache?: boolean,           // Default: false
  cacheSize?: number              // Default: 100
}
```

#### Option 2: Embeddings (creates MemoryVectorStore)

```typescript
{
  embeddings: Embeddings,         // Any LangChain embeddings
  defaultRetrievalLimit?: number, // Default: 5
  enableValidation?: boolean,     // Default: false
  enableCache?: boolean,         // Default: false
  cacheSize?: number            // Default: 100
}
```

### Core Methods

```typescript
// Add tools
addTool(tool, metadata?, toolId?): Promise<string>
addTools(tools[], onProgress?): Promise<string[]>

// Retrieve tools
retrieveTools(query, options?): Promise<ToolRetrievalResult[]>
// options: { limit?, namespace?, tags?, category?, minVersion? }

// Get tools
getTool(toolId): StructuredTool | undefined
getAllTools(): StructuredTool[]
getAllToolIds(): string[]
getToolsByIds(toolIds[]): StructuredTool[]
getToolCount(): number

// Remove tools
removeTool(toolId): Promise<boolean>

// Agent integration
createSearchAndLoadTool(stateRef, options?): StructuredTool  // Recommended
createSearchTool(options?): StructuredTool

// Advanced features
updateTool(toolId, tool, version?): Promise<string>
getToolVersions(toolId): string[]
validateTool(tool): ValidationResult
getToolsByNamespace(namespace, exact?): StructuredTool[]
getNamespaces(topLevelOnly?): string[]
getNamespaceHierarchy(): Map<string, string[]>
setCustomRetrievalFunction(fn?): void
getUsageStats(toolId?): ToolUsageStats | Map<string, ToolUsageStats>
getMostUsedTools(limit): ToolUsageStats[]
trackToolUsage(toolId, success): void
getToolDependencies(toolId): StructuredTool[]
getToolsThatDependOn(toolId): StructuredTool[]
addToolRelationship(toolId, dependencies[]): void
bulkUpdateMetadata(toolIds[], metadata): Promise<void>
clearCache(): void
extractZodSchema(schema): any  // Enhanced Zod v4 schema extraction
```

## LangGraph Integration

### Dynamic Tool Loading (Recommended)

Agents start with minimal tools and discover/load more as needed:

```typescript
import { Toolbox, createLangGraphAgentWithToolbox } from "toolbox";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { HumanMessage } from "@langchain/core/messages";

const toolbox = new Toolbox({ 
  embeddings: new OllamaEmbeddings({ model: "nomic-embed-text" }) 
});
await toolbox.addTools([calculatorTool, weatherTool]);

const agentWrapper = await createLangGraphAgentWithToolbox({
  toolbox,
  model: new ChatOllama({ model: "glm-4.6:cloud" }),
  initialTools: [], // Agent discovers tools dynamically
  systemPrompt: "Use search_and_load_tools to discover relevant tools.",
});

const response = await agentWrapper.agent.invoke({
  messages: [new HumanMessage("Calculate 15 * 8")],
});
```

**How it works:** Agent calls `search_and_load_tools` → tools are automatically added to state → agent can use them immediately. Returns tool info including schemas extracted via Zod v4's `toJSONSchema()`.

## Comparison with LangGraph BigTool

| Feature | BigTool (Python) | Toolbox (TS/JS) |
|---------|-----------------|-----------------|
| Language | Python only | TypeScript/JavaScript |
| Vector Stores | LangGraph persistence only | Any LangChain vector store |
| Schema Introspection | Basic | ✅ Enhanced (Zod v4) |
| Persistence | Built-in Postgres | Via vector store |
| Cross-Platform | Python only | Node.js, Bun, Browser |
| Official Support | ✅ LangChain team | Community |

**Toolbox advantages:** Better schema support, flexible vector stores, cross-platform, composable retrieval  
**BigTool advantages:** Official support, built-in Postgres, deep LangGraph integration

## Vector Stores & Embeddings

Toolbox supports **any LangChain-compatible vector store** (Pinecone, Chroma, Weaviate, FAISS, Qdrant, Milvus, MongoDB Atlas, Azure AI Search) or **any embeddings** (Ollama, OpenAI, HuggingFace, Cohere, etc.).

**Custom vector store:**

```typescript
const toolbox = new Toolbox({ vectorStore: myVectorStore });
```

**Embeddings (creates MemoryVectorStore):**

```typescript
const toolbox = new Toolbox({ 
  embeddings: new OllamaEmbeddings({ model: "nomic-embed-text" }) 
});
```

## Examples

**Basic example:** `examples/agent-example.ts` - Dynamic tool loading with LangGraph  
**Comprehensive example:** `examples/comprehensive-example.ts` - Tests all features

```bash
bun run examples/comprehensive-example.ts
```

## How It Works

1. Add tools → Toolbox creates documents with name, description, and schema
2. Embed documents → Stored in your vector store
3. Query tools → Semantic search using cosine similarity
4. Retrieve results → Tools sorted by relevance score

## Requirements

- Node.js 18+ or Bun
- TypeScript 5+
- LangChain dependencies

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
