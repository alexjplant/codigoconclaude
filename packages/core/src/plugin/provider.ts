import { AnthropicPlugin } from "./provider/anthropic"
import { DynamicProviderPlugin } from "./provider/dynamic"
import { GithubCopilotPlugin } from "./provider/github-copilot"
import { OpenAIPlugin } from "./provider/openai"
import { OpenAICompatiblePlugin } from "./provider/openai-compatible"
import { OpencodePlugin } from "./provider/opencode"
import type { PluginInternal } from "./internal"
import type { Scope } from "effect"

export const ProviderPlugins: PluginInternal.Plugin<PluginInternal.Requirements | Scope.Scope>[] = [
  AnthropicPlugin,
  GithubCopilotPlugin,
  OpencodePlugin,
  OpenAICompatiblePlugin,
  OpenAIPlugin,
  DynamicProviderPlugin,
]