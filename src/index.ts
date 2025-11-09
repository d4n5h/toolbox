import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StructuredTool, tool } from "@langchain/core/tools";
import { Embeddings } from "@langchain/core/embeddings";
import { VectorStore } from "@langchain/core/vectorstores";
import { BaseMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

/**
 * Tool metadata for organizing and filtering tools
 */
export interface ToolMetadata {
  /** Namespace for organizing tools (supports flat "math" or hierarchical "math/calculator") */
  namespace?: string;
  /** Tags for categorizing tools */
  tags?: string[];
  /** Version of the tool (default: "1.0.0") */
  version?: string;
  /** Category of the tool */
  category?: string;
  /** Tool IDs that this tool depends on */
  dependencies?: string[];
  /** When the tool was created */
  createdAt?: Date;
  /** When the tool was last updated */
  updatedAt?: Date;
}

/**
 * Configuration options for the Toolbox
 * You can either provide a vectorStore directly, or provide embeddings to use MemoryVectorStore
 */
type ToolboxOptions =
  | {
      /** Vector store instance to use. Can be any LangChain-compatible vector store (Pinecone, Chroma, Weaviate, etc.) */
      vectorStore: VectorStore;
      /** Number of tools to retrieve by default */
      defaultRetrievalLimit?: number;
      /** Enable tool validation on add */
      enableValidation?: boolean;
      /** Enable query result caching */
      enableCache?: boolean;
      /** Cache size limit */
      cacheSize?: number;
    }
  | {
      /** Embeddings model to use for semantic search. Will create a MemoryVectorStore. Can be any LangChain-compatible embeddings (Ollama, OpenAI, etc.) */
      embeddings: Embeddings;
      /** Number of tools to retrieve by default */
      defaultRetrievalLimit?: number;
      /** Enable tool validation on add */
      enableValidation?: boolean;
      /** Enable query result caching */
      enableCache?: boolean;
      /** Cache size limit */
      cacheSize?: number;
    };

/**
 * Options for retrieving tools with metadata filtering
 */
export interface RetrievalOptions {
  /** Maximum number of tools to retrieve */
  limit?: number;
  /** Filter by namespace (exact match or prefix for hierarchical) */
  namespace?: string;
  /** Filter by tags (tool must have all specified tags) */
  tags?: string[];
  /** Filter by category */
  category?: string;
  /** Minimum version (semantic versioning) */
  minVersion?: string;
}

/**
 * Result of a tool retrieval operation
 */
export interface ToolRetrievalResult {
  /** The tool instance */
  tool: StructuredTool;
  /** The tool ID */
  toolId: string;
  /** Similarity score (higher is more similar) */
  score: number;
  /** Tool metadata */
  metadata?: ToolMetadata;
}

/**
 * Custom retrieval function type
 */
export type CustomRetrievalFunction = (
  query: string,
  tools: StructuredTool[],
  metadata: Map<string, ToolMetadata>
) => Promise<ToolRetrievalResult[]>;

/**
 * Tool usage statistics
 */
export interface ToolUsageStats {
  /** Tool ID */
  toolId: string;
  /** Number of times the tool was called */
  callCount: number;
  /** Last time the tool was used */
  lastUsed: Date;
  /** Number of successful calls */
  successCount: number;
  /** Number of failed calls */
  errorCount: number;
}

/**
 * Tool validation result
 */
export interface ValidationResult {
  /** Whether the tool is valid */
  valid: boolean;
  /** Validation errors */
  errors: string[];
  /** Validation warnings */
  warnings: string[];
}

/**
 * State interface for LangGraph agents with dynamic tool loading
 * Extends the built-in agent state with availableTools
 */
interface ToolboxState {
  /** Messages in the conversation */
  messages: BaseMessage[];
  /** Dynamically loaded tools available to the agent */
  availableTools: StructuredTool[];
  /** Reference to the Toolbox instance */
  toolbox: Toolbox;
}

/**
 * Toolbox - A vector store for tools similar to LangGraph's Big Tool
 * 
 * This class allows you to:
 * - Store tools with their descriptions in a vector store
 * - Retrieve relevant tools using semantic search
 * - Manage a registry of tool instances
 * - Use any LangChain-compatible vector store (Pinecone, Chroma, Weaviate, etc.)
 */
class Toolbox {
  private vectorStore: VectorStore;
  private toolRegistry: Map<string, StructuredTool> = new Map();
  private metadataRegistry: Map<string, ToolMetadata> = new Map();
  private defaultRetrievalLimit: number;
  private enableValidation: boolean;
  private customRetrievalFunction?: CustomRetrievalFunction;
  private queryCache: Map<string, { results: ToolRetrievalResult[]; timestamp: number }> = new Map();
  private enableCache: boolean;
  private cacheSize: number;
  private usageStats: Map<string, ToolUsageStats> = new Map();
  private toolCountCache?: number;
  private toolCountCacheTimestamp?: number;

  constructor(options: ToolboxOptions) {
    if ("vectorStore" in options) {
      // Use provided vector store
      this.vectorStore = options.vectorStore;
    } else {
      // Create MemoryVectorStore from embeddings
      this.vectorStore = new MemoryVectorStore(options.embeddings);
    }
    this.defaultRetrievalLimit = options.defaultRetrievalLimit ?? 5;
    this.enableValidation = options.enableValidation ?? false;
    this.enableCache = options.enableCache ?? false;
    this.cacheSize = options.cacheSize ?? 100;
  }

  /**
   * Add a tool to the toolbox
   * @param tool - The tool instance to add
   * @param metadata - Optional metadata for the tool
   * @param toolId - Optional custom ID for the tool. If not provided, a UUID will be generated
   * @returns The tool ID
   */
  async addTool(tool: StructuredTool, metadata?: ToolMetadata, toolId?: string): Promise<string> {
    // Validate tool if validation is enabled
    if (this.enableValidation) {
      const validation = this.validateTool(tool);
      if (!validation.valid) {
        throw new Error(`Tool validation failed: ${validation.errors.join(", ")}`);
      }
    }

    const id = toolId ?? uuidv4();
    
    // Create metadata with defaults
    const toolMetadata: ToolMetadata = {
      version: metadata?.version ?? "1.0.0",
      namespace: metadata?.namespace,
      tags: metadata?.tags ?? [],
      category: metadata?.category,
      dependencies: metadata?.dependencies ?? [],
      createdAt: metadata?.createdAt ?? new Date(),
      updatedAt: new Date(),
    };
    
    // Store the tool in the registry
    this.toolRegistry.set(id, tool);
    this.metadataRegistry.set(id, toolMetadata);
    
    // Invalidate cache
    this.toolCountCache = undefined;
    this.queryCache.clear();
    
    // Create a document with the tool's description for semantic search
    // Include name, description, schema information, and metadata
    const toolDescription = this.createToolDescription(tool, toolMetadata);
    const document = new Document({
      pageContent: toolDescription,
      metadata: {
        toolId: id,
        toolName: tool.name,
        toolDescription: tool.description,
        namespace: toolMetadata.namespace,
        tags: toolMetadata.tags?.join(",") ?? "",
        version: toolMetadata.version,
        category: toolMetadata.category,
        dependencies: toolMetadata.dependencies?.join(",") ?? "",
        createdAt: toolMetadata.createdAt?.toISOString(),
        updatedAt: toolMetadata.updatedAt?.toISOString(),
      },
    });
    
    // Add the document to the vector store
    await this.vectorStore.addDocuments([document]);
    
    return id;
  }

  /**
   * Add multiple tools at once
   * @param tools - Array of tool entries with optional metadata and IDs
   * @param onProgress - Optional progress callback
   * @returns Array of tool IDs
   */
  async addTools(
    tools: Array<{ tool: StructuredTool; metadata?: ToolMetadata; toolId?: string }>,
    onProgress?: (current: number, total: number) => void
  ): Promise<string[]> {
    const toolIds: string[] = [];
    const total = tools.length;
    
    for (let i = 0; i < tools.length; i++) {
      const entry = tools[i];
      if (!entry) continue;
      const { tool, metadata, toolId } = entry;
      const id = await this.addTool(tool, metadata, toolId);
      toolIds.push(id);
      onProgress?.(i + 1, total);
    }
    
    return toolIds;
  }

  /**
   * Filter tools by metadata
   * @param filters - Metadata filters to apply
   * @returns Array of tools matching the filters
   */
  private filterToolsByMetadata(filters: RetrievalOptions): StructuredTool[] {
    const filtered: StructuredTool[] = [];
    
    for (const [toolId, tool] of this.toolRegistry.entries()) {
      const metadata = this.metadataRegistry.get(toolId);
      if (!metadata) continue;
      
      // Filter by namespace
      if (filters.namespace) {
        if (filters.namespace.includes('/')) {
          // Hierarchical namespace - check for exact match or prefix
          if (metadata.namespace !== filters.namespace && 
              !metadata.namespace?.startsWith(filters.namespace + '/')) {
            continue;
          }
        } else {
          // Flat namespace - check exact match or starts with
          if (metadata.namespace !== filters.namespace && 
              !metadata.namespace?.startsWith(filters.namespace + '/')) {
            continue;
          }
        }
      }
      
      // Filter by tags (must have all specified tags)
      if (filters.tags && filters.tags.length > 0) {
        const toolTags = metadata.tags ?? [];
        const hasAllTags = filters.tags.every(tag => toolTags.includes(tag));
        if (!hasAllTags) continue;
      }
      
      // Filter by category
      if (filters.category && metadata.category !== filters.category) {
        continue;
      }
      
      // Filter by minimum version
      if (filters.minVersion && metadata.version) {
        if (this.compareVersions(metadata.version, filters.minVersion) < 0) {
          continue;
        }
      }
      
      filtered.push(tool);
    }
    
    return filtered;
  }

  /**
   * Compare two semantic versions
   * @param v1 - First version
   * @param v2 - Second version
   * @returns Negative if v1 < v2, 0 if equal, positive if v1 > v2
   */
  private compareVersions(v1: string, v2: string): number {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);
    const maxLength = Math.max(parts1.length, parts2.length);
    
    for (let i = 0; i < maxLength; i++) {
      const part1 = parts1[i] || 0;
      const part2 = parts2[i] || 0;
      if (part1 < part2) return -1;
      if (part1 > part2) return 1;
    }
    
    return 0;
  }

  /**
   * Check if metadata passes the filter
   */
  private checkMetadataFilter(metadata: ToolMetadata | undefined, filters: RetrievalOptions): boolean {
    if (!metadata) return false;
    
    if (filters.namespace) {
      if (filters.namespace.includes('/')) {
        if (metadata.namespace !== filters.namespace && 
            !metadata.namespace?.startsWith(filters.namespace + '/')) {
          return false;
        }
      } else {
        if (metadata.namespace !== filters.namespace && 
            !metadata.namespace?.startsWith(filters.namespace + '/')) {
          return false;
        }
      }
    }
    
    if (filters.tags && filters.tags.length > 0) {
      const toolTags = metadata.tags ?? [];
      const hasAllTags = filters.tags.every(tag => toolTags.includes(tag));
      if (!hasAllTags) return false;
    }
    
    if (filters.category && metadata.category !== filters.category) {
      return false;
    }
    
    if (filters.minVersion && metadata.version) {
      if (this.compareVersions(metadata.version, filters.minVersion) < 0) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Get tool ID by tool instance
   */
  private getToolIdByTool(tool: StructuredTool): string | undefined {
    for (const [id, t] of this.toolRegistry.entries()) {
      if (t === tool || t.name === tool.name) {
        return id;
      }
    }
    return undefined;
  }

  /**
   * Set cache entry with size management
   */
  private setCache(key: string, results: ToolRetrievalResult[]): void {
    if (this.queryCache.size >= this.cacheSize) {
      // Remove oldest entry (simple FIFO)
      const firstKey = this.queryCache.keys().next().value;
      if (firstKey) {
        this.queryCache.delete(firstKey);
      }
    }
    this.queryCache.set(key, { results, timestamp: Date.now() });
  }

  /**
   * Retrieve relevant tools based on a semantic query with optional metadata filtering
   * @param query - The search query
   * @param options - Retrieval options including filters and limit
   * @returns Array of tool retrieval results sorted by relevance
   */
  async retrieveTools(query: string, options?: RetrievalOptions): Promise<ToolRetrievalResult[]> {
    // Check cache if enabled
    if (this.enableCache) {
      const cacheKey = JSON.stringify({ query, options });
      const cached = this.queryCache.get(cacheKey);
      if (cached) {
        return cached.results;
      }
    }

    const retrievalLimit = options?.limit ?? this.defaultRetrievalLimit;
    
    // Pre-filter by metadata if filters are provided
    let toolsToSearch: StructuredTool[] = [];
    let metadataMap: Map<string, ToolMetadata> = new Map();
    
    if (options && (options.namespace || options.tags || options.category || options.minVersion)) {
      // Apply metadata filters first
      toolsToSearch = this.filterToolsByMetadata(options);
      // Create metadata map for filtered tools
      for (const tool of toolsToSearch) {
        const toolId = this.getToolIdByTool(tool);
        if (toolId) {
          const metadata = this.metadataRegistry.get(toolId);
          if (metadata) {
            metadataMap.set(toolId, metadata);
          }
        }
      }
    } else {
      // No filters - use all tools
      toolsToSearch = Array.from(this.toolRegistry.values());
      metadataMap = new Map(this.metadataRegistry);
    }
    
    // Use custom retrieval function if set
    if (this.customRetrievalFunction) {
      const results = await this.customRetrievalFunction(query, toolsToSearch, metadataMap);
      
      // Cache results if enabled
      if (this.enableCache) {
        const cacheKey = JSON.stringify({ query, options });
        this.setCache(cacheKey, results);
      }
      
      return results;
    }
    
    // Perform similarity search in the vector store
    // If we have filters, we need to search more to account for filtering
    const searchLimit = options && (options.namespace || options.tags || options.category || options.minVersion)
      ? Math.min(retrievalLimit * 3, this.toolRegistry.size)
      : retrievalLimit;
    
    const results = await this.vectorStore.similaritySearchWithScore(
      query,
      searchLimit
    );
    
    // Map results to tool retrieval results
    const toolResults: ToolRetrievalResult[] = [];
    
    for (const [document, score] of results) {
      const toolId = document.metadata.toolId as string;
      const tool = this.toolRegistry.get(toolId);
      const metadata = this.metadataRegistry.get(toolId);
      
      if (tool) {
        // Apply metadata filters if not already applied
        if (!options || (!options.namespace && !options.tags && !options.category && !options.minVersion)) {
          toolResults.push({
            tool,
            toolId,
            score,
            metadata,
          });
        } else {
          // Double-check filters (in case vector store returned unfiltered results)
          const passesFilter = this.checkMetadataFilter(metadata, options);
          if (passesFilter) {
            toolResults.push({
              tool,
              toolId,
              score,
              metadata,
            });
          }
        }
      }
      
      // Stop when we have enough results
      if (toolResults.length >= retrievalLimit) {
        break;
      }
    }
    
    // Cache results if enabled
    if (this.enableCache) {
      const cacheKey = JSON.stringify({ query, options });
      this.setCache(cacheKey, toolResults);
    }
    
    return toolResults;
  }

  /**
   * Get a tool by its ID
   * @param toolId - The tool ID
   * @returns The tool instance or undefined if not found
   */
  getTool(toolId: string): StructuredTool | undefined {
    return this.toolRegistry.get(toolId);
  }

  /**
   * Get multiple tools by their IDs
   * @param toolIds - Array of tool IDs
   * @returns Array of tool instances (undefined entries are filtered out)
   */
  getToolsByIds(toolIds: string[]): StructuredTool[] {
    return toolIds
      .map((id) => this.toolRegistry.get(id))
      .filter((tool): tool is StructuredTool => tool !== undefined);
  }

  /**
   * Get all tools in the registry
   * @returns Array of all tool instances
   */
  getAllTools(): StructuredTool[] {
    return Array.from(this.toolRegistry.values());
  }

  /**
   * Get all tool IDs
   * @returns Array of all tool IDs
   */
  getAllToolIds(): string[] {
    return Array.from(this.toolRegistry.keys());
  }

  /**
   * Validate a tool
   * @param tool - The tool to validate
   * @returns Validation result with errors and warnings
   */
  validateTool(tool: StructuredTool): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Check required fields
    if (!tool.name || tool.name.trim().length === 0) {
      errors.push("Tool name is required");
    }
    
    if (!tool.description || tool.description.trim().length === 0) {
      errors.push("Tool description is required");
    } else if (tool.description.length < 10) {
      warnings.push("Tool description is very short (less than 10 characters)");
    }
    
    // Check schema
    if (!tool.schema) {
      warnings.push("Tool has no schema defined");
    } else {
      try {
        // Try to validate schema structure
        const schema = tool.schema as any;
        if (schema._def && schema._def.typeName === "ZodObject") {
          const shape = typeof schema._def.shape === 'function' ? schema._def.shape() : schema._def.shape;
          if (!shape || Object.keys(shape).length === 0) {
            warnings.push("Tool schema has no properties");
          }
        }
      } catch (e) {
        warnings.push(`Schema validation warning: ${e}`);
      }
    }
    
    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Update a tool
   * @param toolId - The tool ID to update
   * @param tool - The updated tool instance
   * @param newVersion - Optional new version (defaults to incrementing patch version)
   * @returns The tool ID
   */
  async updateTool(toolId: string, tool: StructuredTool, newVersion?: string): Promise<string> {
    const existingMetadata = this.metadataRegistry.get(toolId);
    if (!existingMetadata) {
      throw new Error(`Tool with ID ${toolId} not found`);
    }
    
    // Validate if validation is enabled
    if (this.enableValidation) {
      const validation = this.validateTool(tool);
      if (!validation.valid) {
        throw new Error(`Tool validation failed: ${validation.errors.join(", ")}`);
      }
    }
    
    // Determine new version
    let version = newVersion;
    if (!version && existingMetadata.version) {
      // Increment patch version
      const parts = existingMetadata.version.split('.');
      const patch = parseInt(parts[2] || '0', 10) + 1;
      version = `${parts[0]}.${parts[1] || '0'}.${patch}`;
    } else if (!version) {
      version = "1.0.1";
    }
    
    // Update metadata
    const updatedMetadata: ToolMetadata = {
      ...existingMetadata,
      version,
      updatedAt: new Date(),
    };
    
    // Update registries
    this.toolRegistry.set(toolId, tool);
    this.metadataRegistry.set(toolId, updatedMetadata);
    
    // Invalidate cache
    this.toolCountCache = undefined;
    this.queryCache.clear();
    
    // Update vector store - remove old document and add new one
    // First, try to delete from vector store if supported
    await this.deleteFromVectorStore(toolId);
    
    // Create new document
    const toolDescription = this.createToolDescription(tool, updatedMetadata);
    const document = new Document({
      pageContent: toolDescription,
      metadata: {
        toolId,
        toolName: tool.name,
        toolDescription: tool.description,
        namespace: updatedMetadata.namespace,
        tags: updatedMetadata.tags?.join(",") ?? "",
        version: updatedMetadata.version,
        category: updatedMetadata.category,
        dependencies: updatedMetadata.dependencies?.join(",") ?? "",
        createdAt: updatedMetadata.createdAt?.toISOString(),
        updatedAt: updatedMetadata.updatedAt?.toISOString(),
      },
    });
    
    await this.vectorStore.addDocuments([document]);
    
    return toolId;
  }

  /**
   * Get version history for a tool
   * @param toolId - The tool ID
   * @returns Array of version strings
   */
  getToolVersions(toolId: string): string[] {
    const metadata = this.metadataRegistry.get(toolId);
    if (!metadata || !metadata.version) {
      return [];
    }
    // For now, return current version. In a full implementation, you'd track history
    return [metadata.version];
  }

  /**
   * Delete from vector store if supported
   * @param toolId - The tool ID to delete
   * @returns True if deletion was attempted/successful
   */
  private async deleteFromVectorStore(toolId: string): Promise<boolean> {
    try {
      // Check if vector store has delete method
      if (typeof (this.vectorStore as any).delete === 'function') {
        await (this.vectorStore as any).delete({ ids: [toolId] });
        return true;
      }
    } catch (e) {
      // Vector store doesn't support deletion or deletion failed
      // This is okay - we'll just add a new document
    }
    return false;
  }

  /**
   * Remove a tool from the toolbox
   * @param toolId - The tool ID to remove
   * @param softDelete - If true, mark as deleted but keep in registry (default: false)
   * @returns True if the tool was removed, false if it wasn't found
   */
  async removeTool(toolId: string, softDelete: boolean = false): Promise<boolean> {
    const tool = this.toolRegistry.get(toolId);
    if (!tool) {
      return false;
    }
    
    // Check for dependent tools
    const dependents = this.getToolsThatDependOn(toolId);
    if (dependents.length > 0) {
      throw new Error(`Cannot remove tool ${toolId}: ${dependents.length} tool(s) depend on it: ${dependents.map(t => t.name).join(', ')}`);
    }
    
    if (softDelete) {
      // Mark as deleted in metadata
      const metadata = this.metadataRegistry.get(toolId);
      if (metadata) {
        // Add a deleted flag (we'll use a special namespace or tag)
        // For now, we'll just update the metadata
        metadata.updatedAt = new Date();
      }
    } else {
      // Remove from registries
      this.toolRegistry.delete(toolId);
      this.metadataRegistry.delete(toolId);
      
      // Delete from vector store if supported
      await this.deleteFromVectorStore(toolId);
      
      // Invalidate cache
      this.toolCountCache = undefined;
      this.queryCache.clear();
    }
    
    return true;
  }

  /**
   * Restore a soft-deleted tool
   * @param toolId - The tool ID to restore
   * @returns True if the tool was restored
   */
  async restoreTool(toolId: string): Promise<boolean> {
    // In a full implementation, you'd track deleted tools separately
    // For now, if the tool exists in registry, it's not deleted
    const exists = this.toolRegistry.has(toolId);
    if (!exists) {
      return false;
    }
    
    // Update metadata to clear deleted flag
    const metadata = this.metadataRegistry.get(toolId);
    if (metadata) {
      metadata.updatedAt = new Date();
    }
    
    return true;
  }

  /**
   * Get the number of tools in the toolbox
   * @returns The number of tools
   */
  getToolCount(): number {
    if (this.toolCountCache !== undefined && this.toolCountCacheTimestamp) {
      // Cache is valid for 1 second
      if (Date.now() - this.toolCountCacheTimestamp < 1000) {
        return this.toolCountCache;
      }
    }
    const count = this.toolRegistry.size;
    this.toolCountCache = count;
    this.toolCountCacheTimestamp = Date.now();
    return count;
  }

  /**
   * Get tools by namespace
   * @param namespace - The namespace to filter by
   * @param exact - If true, only exact matches. If false, includes hierarchical matches (default: false)
   * @returns Array of tools in the namespace
   */
  getToolsByNamespace(namespace: string, exact: boolean = false): StructuredTool[] {
    const tools: StructuredTool[] = [];
    
    for (const [toolId, tool] of this.toolRegistry.entries()) {
      const metadata = this.metadataRegistry.get(toolId);
      if (!metadata?.namespace) continue;
      
      if (exact) {
        if (metadata.namespace === namespace) {
          tools.push(tool);
        }
      } else {
        // Hierarchical matching
        if (metadata.namespace === namespace || 
            metadata.namespace.startsWith(namespace + '/')) {
          tools.push(tool);
        }
      }
    }
    
    return tools;
  }

  /**
   * Get all namespaces
   * @param includeHierarchical - If true, returns all namespaces including hierarchical ones. If false, returns only top-level (default: true)
   * @returns Array of unique namespace strings
   */
  getNamespaces(includeHierarchical: boolean = true): string[] {
    const namespaces = new Set<string>();
    
    for (const metadata of this.metadataRegistry.values()) {
      if (metadata.namespace) {
        if (includeHierarchical) {
          namespaces.add(metadata.namespace);
        } else {
          // Only top-level namespace
          const topLevel = metadata.namespace.split('/')[0];
          if (topLevel) {
            namespaces.add(topLevel);
          }
        }
      }
    }
    
    return Array.from(namespaces).sort();
  }

  /**
   * Get namespace hierarchy
   * @returns Map of namespace to array of child namespaces
   */
  getNamespaceHierarchy(): Map<string, string[]> {
    const hierarchy = new Map<string, string[]>();
    
    for (const metadata of this.metadataRegistry.values()) {
      if (metadata.namespace) {
        const parts = metadata.namespace.split('/');
        for (let i = 0; i < parts.length; i++) {
          const parent = parts.slice(0, i).join('/');
          const current = parts.slice(0, i + 1).join('/');
          
          if (i === 0) {
            // Top level
            if (!hierarchy.has('')) {
              hierarchy.set('', []);
            }
            const topLevel = hierarchy.get('')!;
            if (!topLevel.includes(current)) {
              topLevel.push(current);
            }
          } else {
            if (!hierarchy.has(parent)) {
              hierarchy.set(parent, []);
            }
            const children = hierarchy.get(parent)!;
            if (!children.includes(current)) {
              children.push(current);
            }
          }
        }
      }
    }
    
    return hierarchy;
  }

  /**
   * Get tool count by namespace
   * @param namespace - The namespace to count
   * @returns Number of tools in the namespace
   */
  getToolCountByNamespace(namespace: string): number {
    return this.getToolsByNamespace(namespace).length;
  }

  /**
   * Set custom retrieval function
   * @param fn - Custom retrieval function
   */
  setCustomRetrievalFunction(fn: CustomRetrievalFunction): void {
    this.customRetrievalFunction = fn;
  }

  /**
   * Clear query cache
   */
  clearCache(): void {
    this.queryCache.clear();
  }

  /**
   * Get tool dependencies
   * @param toolId - The tool ID
   * @returns Array of tools that this tool depends on
   */
  getToolDependencies(toolId: string): StructuredTool[] {
    const metadata = this.metadataRegistry.get(toolId);
    if (!metadata || !metadata.dependencies || metadata.dependencies.length === 0) {
      return [];
    }
    
    return this.getToolsByIds(metadata.dependencies);
  }

  /**
   * Get tools that depend on a given tool
   * @param toolId - The tool ID
   * @returns Array of tools that depend on this tool
   */
  getToolsThatDependOn(toolId: string): StructuredTool[] {
    const dependents: StructuredTool[] = [];
    
    for (const [id, metadata] of this.metadataRegistry.entries()) {
      if (metadata.dependencies && metadata.dependencies.includes(toolId)) {
        const tool = this.toolRegistry.get(id);
        if (tool) {
          dependents.push(tool);
        }
      }
    }
    
    return dependents;
  }

  /**
   * Add tool relationship (dependencies)
   * @param toolId - The tool ID
   * @param dependsOn - Array of tool IDs this tool depends on
   */
  addToolRelationship(toolId: string, dependsOn: string[]): void {
    const metadata = this.metadataRegistry.get(toolId);
    if (!metadata) {
      throw new Error(`Tool with ID ${toolId} not found`);
    }
    
    // Validate that all dependencies exist
    for (const depId of dependsOn) {
      if (!this.toolRegistry.has(depId)) {
        throw new Error(`Dependency tool with ID ${depId} not found`);
      }
    }
    
    // Update metadata
    metadata.dependencies = [...new Set([...(metadata.dependencies || []), ...dependsOn])];
    metadata.updatedAt = new Date();
    
    this.metadataRegistry.set(toolId, metadata);
    
    // Invalidate cache
    this.queryCache.clear();
  }

  /**
   * Get usage statistics for a tool or all tools
   * @param toolId - Optional tool ID. If omitted, returns stats for all tools
   * @returns Usage statistics
   */
  getUsageStats(toolId?: string): ToolUsageStats | Map<string, ToolUsageStats> {
    if (toolId) {
      return this.usageStats.get(toolId) || {
        toolId,
        callCount: 0,
        lastUsed: new Date(0),
        successCount: 0,
        errorCount: 0,
      };
    }
    
    return new Map(this.usageStats);
  }

  /**
   * Get most used tools
   * @param limit - Maximum number of tools to return (default: 10)
   * @returns Array of usage statistics sorted by call count
   */
  getMostUsedTools(limit: number = 10): ToolUsageStats[] {
    const stats = Array.from(this.usageStats.values());
    return stats
      .sort((a, b) => b.callCount - a.callCount)
      .slice(0, limit);
  }

  /**
   * Reset usage statistics
   * @param toolId - Optional tool ID. If omitted, resets all stats
   */
  resetUsageStats(toolId?: string): void {
    if (toolId) {
      this.usageStats.delete(toolId);
    } else {
      this.usageStats.clear();
    }
  }

  /**
   * Track tool usage
   * @param toolId - The tool ID
   * @param success - Whether the call was successful
   */
  trackToolUsage(toolId: string, success: boolean): void {
    const existing = this.usageStats.get(toolId);
    
    if (existing) {
      existing.callCount++;
      existing.lastUsed = new Date();
      if (success) {
        existing.successCount++;
      } else {
        existing.errorCount++;
      }
    } else {
      this.usageStats.set(toolId, {
        toolId,
        callCount: 1,
        lastUsed: new Date(),
        successCount: success ? 1 : 0,
        errorCount: success ? 0 : 1,
      });
    }
  }

  /**
   * Update multiple tools
   * @param tools - Array of tool updates
   * @returns Array of tool IDs
   */
  async updateTools(
    tools: Array<{ toolId: string; tool: StructuredTool; newVersion?: string }>
  ): Promise<string[]> {
    const toolIds: string[] = [];
    
    for (const { toolId, tool, newVersion } of tools) {
      const id = await this.updateTool(toolId, tool, newVersion);
      toolIds.push(id);
    }
    
    return toolIds;
  }

  /**
   * Remove multiple tools
   * @param toolIds - Array of tool IDs to remove
   * @param softDelete - If true, mark as deleted but keep in registry (default: false)
   * @returns Array of boolean results indicating success
   */
  async removeTools(toolIds: string[], softDelete: boolean = false): Promise<boolean[]> {
    const results: boolean[] = [];
    
    for (const toolId of toolIds) {
      try {
        const result = await this.removeTool(toolId, softDelete);
        results.push(result);
      } catch (e) {
        results.push(false);
      }
    }
    
    return results;
  }

  /**
   * Bulk update metadata for multiple tools
   * @param toolIds - Array of tool IDs
   * @param metadata - Partial metadata to update
   */
  async bulkUpdateMetadata(toolIds: string[], metadata: Partial<ToolMetadata>): Promise<void> {
    for (const toolId of toolIds) {
      const existing = this.metadataRegistry.get(toolId);
      if (!existing) {
        continue;
      }
      
      const updated: ToolMetadata = {
        ...existing,
        ...metadata,
        updatedAt: new Date(),
      };
      
      this.metadataRegistry.set(toolId, updated);
    }
    
    // Invalidate cache
    this.queryCache.clear();
  }

  /**
   * Get tools for an agent based on a query
   * This is a convenience method that retrieves relevant tools and returns them as an array
   * @param query - Optional query to retrieve relevant tools. If not provided, returns all tools
   * @param options - Retrieval options including limit and filters
   * @returns Array of tool instances
   * @deprecated Use createSearchTool() instead to expose a search_tools tool to agents
   */
  async getToolsForAgent(query?: string, options?: RetrievalOptions): Promise<StructuredTool[]> {
    if (query) {
      const results = await this.retrieveTools(query, options);
      return results.map((result) => result.tool);
    }
    return this.getAllTools();
  }

  /**
   * Create a search_tools tool that agents can use to query the toolbox
   * This tool allows agents to search for relevant tools based on natural language queries
   * @param options - Configuration options for the search tool
   * @returns A StructuredTool that agents can call to search for tools
   */
  createSearchTool(options?: {
    /** Custom name for the search tool (default: "search_tools") */
    name?: string;
    /** Custom description for the search tool */
    description?: string;
  }): StructuredTool {
    const toolName = options?.name ?? "search_tools";
    const toolDescription = options?.description ?? 
      "Search for relevant tools in the toolbox based on a natural language query. Returns information about matching tools including their names, descriptions, IDs, and metadata. Supports filtering by namespace, tags, category, and version.";

    return tool(
      async (input: { 
        query: string; 
        limit?: number;
        namespace?: string;
        tags?: string[];
        category?: string;
        minVersion?: string;
      }) => {
        const { query, limit, namespace, tags, category, minVersion } = input;
        const results = await this.retrieveTools(query, {
          limit,
          namespace,
          tags,
          category,
          minVersion,
        });
        
        // Format results as a JSON string for the agent
        const toolInfo = results.map((result) => ({
          toolId: result.toolId,
          name: result.tool.name,
          description: result.tool.description,
          score: result.score.toFixed(4),
          metadata: result.metadata ? {
            namespace: result.metadata.namespace,
            tags: result.metadata.tags,
            category: result.metadata.category,
            version: result.metadata.version,
          } : undefined,
        }));

        return JSON.stringify({
          query,
          found: toolInfo.length,
          tools: toolInfo,
        }, null, 2);
      },
      {
        name: toolName,
        description: toolDescription,
        schema: z.object({
          query: z.string().describe("Natural language query describing what kind of tools you need"),
          limit: z.number().optional().describe("Maximum number of tools to return (defaults to toolbox default)"),
          namespace: z.string().optional().describe("Filter by namespace (supports hierarchical namespaces like 'math/calculator')"),
          tags: z.array(z.string()).optional().describe("Filter by tags (tool must have all specified tags)"),
          category: z.string().optional().describe("Filter by category"),
          minVersion: z.string().optional().describe("Minimum version (semantic versioning)"),
        }),
      }
    );
  }

  /**
   * Create a search_and_load_tools tool that agents can use to search for tools
   * and automatically add them to their available tools (state-based)
   * This is designed for LangGraph state management where tools are stored in state
   * @param stateRef - Reference to the state object containing availableTools array
   * @param options - Configuration options for the search tool
   * @returns A StructuredTool that agents can call to search and load tools
   */
  createSearchAndLoadTool(
    stateRef: { availableTools: StructuredTool[] },
    options?: {
      /** Custom name for the search tool (default: "search_and_load_tools") */
      name?: string;
      /** Custom description for the search tool */
      description?: string;
    }
  ): StructuredTool {
    const toolName = options?.name ?? "search_and_load_tools";
    const toolDescription = options?.description ?? 
      "Search for relevant tools in the toolbox based on a natural language query and automatically add them to your available tools. Returns information about the tools that were loaded.";

    return tool(
      async (input: { 
        query: string; 
        limit?: number;
        namespace?: string;
        tags?: string[];
        category?: string;
        minVersion?: string;
      }) => {
        const { query, limit, namespace, tags, category, minVersion } = input;
        const results = await this.retrieveTools(query, {
          limit,
          namespace,
          tags,
          category,
          minVersion,
        });
        
        // Get existing tool names to avoid duplicates
        const existingToolNames = new Set(stateRef.availableTools.map(t => t.name));
        
        // Add new tools to state (avoid duplicates)
        const newlyAdded: Array<{ 
          toolId: string; 
          name: string; 
          description: string; 
          score: string;
          schema?: any;
        }> = [];
        const alreadyAvailable: string[] = [];
        
        for (const result of results) {
          if (!existingToolNames.has(result.tool.name)) {
            stateRef.availableTools.push(result.tool);
            existingToolNames.add(result.tool.name);
            
            // Extract schema information for the agent
            let schemaInfo: any = null;
            if (result.tool.schema) {
              try {
                // Try to extract schema structure
                const schema = result.tool.schema as any;
                if (schema._def) {
                  // Zod schema
                  schemaInfo = this.extractZodSchema(schema);
                } else {
                  schemaInfo = schema;
                }
              } catch (e) {
                // If schema extraction fails, just include the raw schema
                schemaInfo = result.tool.schema;
              }
            }
            
            newlyAdded.push({
              toolId: result.toolId,
              name: result.tool.name,
              description: result.tool.description,
              score: result.score.toFixed(4),
              schema: schemaInfo,
            });
          } else {
            alreadyAvailable.push(result.tool.name);
          }
        }

        return JSON.stringify({
          query,
          newlyLoaded: newlyAdded.length,
          alreadyAvailable: alreadyAvailable.length,
          tools: newlyAdded,
          message: newlyAdded.length > 0 
            ? `Successfully loaded ${newlyAdded.length} new tool(s). ${alreadyAvailable.length > 0 ? `${alreadyAvailable.length} tool(s) were already available.` : ''} Use execute_tool with the tool name and parameters matching the schema.`
            : `All ${results.length} matching tool(s) were already available.`,
        }, null, 2);
      },
      {
        name: toolName,
        description: toolDescription,
        schema: z.object({
          query: z.string().describe("Natural language query describing what kind of tools you need"),
          limit: z.number().optional().describe("Maximum number of tools to load (defaults to toolbox default)"),
          namespace: z.string().optional().describe("Filter by namespace (supports hierarchical namespaces like 'math/calculator')"),
          tags: z.array(z.string()).optional().describe("Filter by tags (tool must have all specified tags)"),
          category: z.string().optional().describe("Filter by category"),
          minVersion: z.string().optional().describe("Minimum version (semantic versioning)"),
        }),
      }
    );
  }

  /**
   * Extract schema information from a Zod schema for tool documentation
   * Uses Zod v4's native toJSONSchema() method when available
   * @param schema - The Zod schema
   * @returns Schema information as a plain object
   */
  extractZodSchema(schema: any): any {
    try {
      // Use Zod v4's native toJSONSchema() static method if available
      if (schema && typeof z.toJSONSchema === 'function') {
        const jsonSchema = z.toJSONSchema(schema);
        
        // Enhance the JSON schema with example generation
        if (jsonSchema.type === 'object' && jsonSchema.properties) {
          jsonSchema.example = this.generateExampleFromSchema(jsonSchema.properties);
        }
        
        return jsonSchema;
      }
      
      // Fallback to manual extraction for older Zod versions or edge cases
      const def = schema._def;
      if (!def) return null;
      
      // Check if it's a ZodObject - either by typeName or by presence of shape
      const isZodObject = def.typeName === "ZodObject" || (def.shape && typeof def.shape === 'object');
      
      if (isZodObject) {
        // Get the shape - it's a function that returns the shape object or the shape itself
        const shape = typeof def.shape === 'function' ? def.shape() : def.shape;
        
        if (!shape || typeof shape !== 'object') {
          return {
            type: "object",
            properties: {},
            required: [],
          };
        }
        
        const properties: Record<string, any> = {};
        const required: string[] = [];
        
        for (const [key, value] of Object.entries(shape)) {
          const fieldSchema = value as any;
          let fieldDef = fieldSchema?._def;
          
          // If _def doesn't exist, the schema itself might be the def
          if (!fieldDef && fieldSchema) {
            // Check if the schema has typeName directly (some Zod versions)
            if (fieldSchema.typeName || fieldSchema._type) {
              fieldDef = fieldSchema;
            }
          }
          
          if (!fieldDef) continue;
          
          // Unwrap optional/default wrappers
          let actualDef = fieldDef;
          let isOptional = false;
          let currentDef = fieldDef;
          
          while (currentDef?.typeName === "ZodOptional" || currentDef?.typeName === "ZodDefault") {
            isOptional = true;
            // Get the inner type's _def, or fall back to innerType itself if it's already a _def
            const innerType = currentDef.innerType;
            if (innerType) {
              currentDef = innerType._def || innerType;
              actualDef = currentDef;
            } else {
              break;
            }
          }
          
          // Extract type name - try multiple approaches
          let fieldType = "unknown";
          if (actualDef?.typeName) {
            fieldType = actualDef.typeName.replace("Zod", "").toLowerCase();
          } else if (fieldDef?.typeName) {
            // Fallback to original fieldDef
            fieldType = fieldDef.typeName.replace("Zod", "").toLowerCase();
          } else if (actualDef?.type) {
            fieldType = actualDef.type;
          } else {
            // Try to infer from the schema object itself
            if (fieldSchema?._def?.typeName) {
              fieldType = fieldSchema._def.typeName.replace("Zod", "").toLowerCase();
            } else if (fieldSchema?.typeName) {
              fieldType = fieldSchema.typeName.replace("Zod", "").toLowerCase();
            }
          }
          
          properties[key] = {
            type: fieldType === "number" ? "number" : fieldType === "string" ? "string" : fieldType === "boolean" ? "boolean" : fieldType,
            description: actualDef?.description || fieldDef?.description || "",
          };
          
          // Handle enums
          if (actualDef?.typeName === "ZodEnum" || fieldDef?.typeName === "ZodEnum") {
            properties[key].enum = actualDef?.values || fieldDef?.values || [];
          }
          
          // Check if field is required (not optional and not default)
          if (!isOptional && fieldDef.typeName !== "ZodDefault") {
            required.push(key);
          }
        }
        
        return {
          type: "object",
          properties,
          required,
          example: this.generateExampleFromSchema(properties),
        };
      }
      
      return {
        type: def.typeName?.replace("Zod", "").toLowerCase() || "unknown",
        description: def.description || "",
      };
    } catch (e) {
      // Fallback: try to stringify the schema
      try {
        return JSON.parse(JSON.stringify(schema, (key, value) => {
          if (key === '_def') return undefined;
          return value;
        }));
      } catch (e2) {
        return null;
      }
    }
  }

  /**
   * Generate an example JSON object from schema properties
   */
  private generateExampleFromSchema(properties: Record<string, any>): any {
    const example: any = {};
    for (const [key, prop] of Object.entries(properties)) {
      if (prop.enum && prop.enum.length > 0) {
        example[key] = prop.enum[0];
      } else if (prop.type === "number") {
        // Use a realistic number example
        example[key] = key.toLowerCase().includes('limit') || key.toLowerCase().includes('count') ? 10 : 5;
      } else if (prop.type === "boolean") {
        example[key] = false;
      } else {
        // Use more descriptive string examples based on key name
        const keyLower = key.toLowerCase();
        if (keyLower.includes('city') || keyLower.includes('location')) {
          example[key] = "New York";
        } else if (keyLower.includes('query') || keyLower.includes('search')) {
          example[key] = "example query";
        } else if (keyLower.includes('operation')) {
          example[key] = prop.enum?.[0] || "add";
        } else {
          example[key] = "example_value";
        }
      }
    }
    return example;
  }

  /**
   * Create a comprehensive description of a tool for semantic search
   * @param tool - The tool to describe
   * @param metadata - Optional metadata to include in description
   * @returns A string description of the tool
   */
  private createToolDescription(tool: StructuredTool, metadata?: ToolMetadata): string {
    let description = `Tool: ${tool.name}\n`;
    description += `Description: ${tool.description}\n`;
    
    // Include metadata in searchable content
    if (metadata) {
      if (metadata.namespace) {
        description += `Namespace: ${metadata.namespace}\n`;
      }
      if (metadata.category) {
        description += `Category: ${metadata.category}\n`;
      }
      if (metadata.tags && metadata.tags.length > 0) {
        description += `Tags: ${metadata.tags.join(", ")}\n`;
      }
      if (metadata.version) {
        description += `Version: ${metadata.version}\n`;
      }
    }
    
    // Include schema information if available
    if (tool.schema) {
      try {
        const schemaStr = JSON.stringify(tool.schema, null, 2);
        description += `Schema: ${schemaStr}\n`;
      } catch (e) {
        // If schema can't be stringified, skip it
      }
    }
    
    return description;
  }
}

/**
 * Options for creating a LangGraph agent with dynamic tool loading
 */
interface CreateLangGraphAgentWithToolboxOptions {
  /** The Toolbox instance */
  toolbox: Toolbox;
  /** The language model to use (LanguageModelLike) */
  model: any;
  /** Initial tools to include (default: empty, agent starts with just search_and_load_tools) */
  initialTools?: StructuredTool[];
  /** System prompt for the agent */
  systemPrompt?: string;
}

/**
 * Helper function to create a LangGraph agent with dynamic tool loading from Toolbox
 * 
 * This creates an agent that:
 * - Starts with minimal tools (just search_and_load_tools by default)
 * - Can dynamically discover and load tools using search_and_load_tools
 * - Automatically makes loaded tools available for use in the same conversation
 * 
 * This implementation uses a wrapper around createAgent that dynamically updates
 * the agent's tools based on state. After search_and_load_tools is called, the
 * agent is recreated with the new tools for subsequent interactions.
 * 
 * @param options - Configuration options
 * @returns A runnable agent wrapper that supports dynamic tool loading
 */
async function createLangGraphAgentWithToolbox(
  options: CreateLangGraphAgentWithToolboxOptions
): Promise<any> {
  const { toolbox, model, initialTools = [], systemPrompt } = options;
  
  // Create state object for dynamic tool management
  // This state will be modified by search_and_load_tools
  const state: { availableTools: StructuredTool[] } = {
    availableTools: [...initialTools],
  };
  
  // Create the search_and_load_tools tool that modifies state
  const searchAndLoadTool = toolbox.createSearchAndLoadTool(state);
  
  // Create a tool to get information about available tools (including schemas)
  const getToolInfo = tool(
    async (input: { toolName?: string }) => {
      const { toolName } = input;
      const tools = state.availableTools.length > 0 ? state.availableTools : toolbox.getAllTools();
      
      if (toolName) {
        // Get specific tool info
        const tool = tools.find(t => t.name === toolName);
        if (!tool) {
          return JSON.stringify({ error: `Tool "${toolName}" not found. Use search_and_load_tools to discover and load it first.` });
        }
        
        let schemaInfo: any = null;
        if (tool.schema) {
          try {
            const schema = tool.schema as any;
            if (schema._def) {
              schemaInfo = toolbox.extractZodSchema(schema);
            } else {
              schemaInfo = schema;
            }
          } catch (e) {
            schemaInfo = tool.schema;
          }
        }
        
        // Create a clear, actionable response with example
        const response: any = {
          name: tool.name,
          description: tool.description,
          schema: schemaInfo,
        };
        
        // Add example usage if schema is available
        if (schemaInfo && schemaInfo.example) {
          response.exampleUsage = {
            toolName: tool.name,
            toolInput: JSON.stringify(schemaInfo.example),
            explanation: `Use execute_tool with toolName="${tool.name}" and toolInput as a JSON string matching the example above.`,
          };
        }
        
        return JSON.stringify(response, null, 2);
      } else {
        // List all available tools
        const toolList = tools.map(t => ({
          name: t.name,
          description: t.description,
        }));
        return JSON.stringify({
          availableTools: toolList,
          count: toolList.length,
        }, null, 2);
      }
    },
    {
      name: "get_tool_info",
      description: "CRITICAL: Get information about available tools, including their exact schemas and example usage. ALWAYS call this with a tool name BEFORE using execute_tool to see the required parameters and format. Call without toolName to list all available tools.",
      schema: z.object({
        toolName: z.string().optional().describe("Optional: name of a specific tool to get info for. If omitted, returns list of all available tools."),
      }),
    }
  );
  
  // Create a dynamic tool executor that can run any tool from the toolbox
  // This allows tools loaded during execution to be used immediately
  const executeTool = tool(
    async (input: { toolName: string; toolInput: string }) => {
      const { toolName, toolInput } = input;
      
      // First check state for loaded tools
      let toolToUse = state.availableTools.find(t => t.name === toolName);
      
      // If not in state, try to get from toolbox
      if (!toolToUse) {
        const allTools = toolbox.getAllTools();
        toolToUse = allTools.find(t => t.name === toolName);
      }
      
      if (!toolToUse) {
        return JSON.stringify({ 
          error: `Tool "${toolName}" not found. Use search_and_load_tools to discover and load it first, or use get_tool_info to see available tools.` 
        });
      }
      
      // Parse the tool input
      let parsedInput: any;
      try {
        parsedInput = JSON.parse(toolInput);
      } catch (e) {
        return JSON.stringify({ 
          error: `Invalid tool input. Expected JSON string, got: ${toolInput}. Use get_tool_info to see the tool's schema.` 
        });
      }
      
      // Execute the tool and track usage
      let toolId: string | undefined;
      try {
        // Find tool ID for tracking - use getAllToolIds and getTool
        const allIds = toolbox.getAllToolIds();
        for (const id of allIds) {
          const t = toolbox.getTool(id);
          if (t === toolToUse || t?.name === toolToUse.name) {
            toolId = id;
            break;
          }
        }
        
        const result = await toolToUse.invoke(parsedInput);
        
        // Track successful usage
        if (toolId) {
          toolbox.trackToolUsage(toolId, true);
        }
        
        return typeof result === 'string' ? result : JSON.stringify(result);
      } catch (error: any) {
        // Track failed usage
        if (toolId) {
          toolbox.trackToolUsage(toolId, false);
        }
        
        return JSON.stringify({ 
          error: error.message || String(error),
          hint: "Use get_tool_info to check the tool's schema and required parameters."
        });
      }
    },
    {
      name: "execute_tool",
      description: "Execute a tool by name. REQUIRED: Always call get_tool_info first to see the exact schema and example. Then use this tool with toolName and toolInput as a JSON string that exactly matches the schema from get_tool_info.",
      schema: z.object({
        toolName: z.string().describe("The exact name of the tool to execute (get from get_tool_info)"),
        toolInput: z.string().describe("JSON string containing the tool's input parameters. Must exactly match the schema from get_tool_info. Example: '{\"a\": 5, \"b\": 3, \"operation\": \"multiply\"}'"),
      }),
    }
  );
  
  // Start with search_and_load_tools, get_tool_info, execute_tool, and any initial tools
  const baseTools: StructuredTool[] = [...initialTools, searchAndLoadTool, getToolInfo, executeTool];
  
  // Import createAgent dynamically
  const langchainModule = await import("langchain");
  const { createAgent } = langchainModule;
  
  // Extract the exact tool type that createAgent expects
  type CreateAgentParams = Parameters<typeof createAgent>[0];
  type ExpectedTools = NonNullable<CreateAgentParams["tools"]>;
  
  // Create agent with dynamic tool execution capability
  // All tools are StructuredTool or compatible types that work with createAgent
  // We use a type assertion to satisfy createAgent's type requirements
  // The tools are compatible at runtime, but TypeScript needs the cast to the expected union type
  const agent = createAgent({
    model,
    tools: baseTools as unknown as ExpectedTools,
    systemPrompt: systemPrompt ?? 
      `You are a helpful assistant with access to dynamic tools. IMPORTANT WORKFLOW:
1. First, use search_and_load_tools to discover and load relevant tools based on the user's query
2. ALWAYS use get_tool_info with the tool name to see the exact schema and required parameters BEFORE executing
3. Use execute_tool with the tool name and a JSON string matching the schema exactly (use the example format from get_tool_info)
4. If execute_tool fails, check get_tool_info again to verify the correct parameter format

Never guess parameter formats - always check get_tool_info first!`,
  });
  
  // Return agent with state access and analytics
  return {
    agent,
    state, // Expose state for manual tool management if needed
    getAvailableTools: () => state.availableTools,
    addTools: (newTools: StructuredTool[]) => {
      const existingNames = new Set(state.availableTools.map(t => t.name));
      for (const tool of newTools) {
        if (!existingNames.has(tool.name)) {
          state.availableTools.push(tool);
          existingNames.add(tool.name);
        }
      }
    },
    // Analytics access
    getUsageStats: (toolId?: string) => toolbox.getUsageStats(toolId),
    getMostUsedTools: (limit?: number) => toolbox.getMostUsedTools(limit),
    resetUsageStats: (toolId?: string) => toolbox.resetUsageStats(toolId),
  };
}

export { Toolbox, createLangGraphAgentWithToolbox };
export type { ToolboxOptions, ToolboxState };