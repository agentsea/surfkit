# AI Provider Configuration

Surfkit supports multiple AI providers that are compatible with the OpenAI API format. You can specify which provider to use with the `--provider` flag.

## Supported Providers

- `openai` (default) - OpenAI API
- `openrouter` - OpenRouter API
- `azure` - Azure OpenAI API
- `gemini` - Google Gemini API
- `ollama` - Ollama API
- `mistral` - Mistral AI API
- `deepseek` - DeepSeek API
- `xai` - xAI API
- `groq` - Groq API
- `arceeai` - ArceeAI API
- Any custom provider that is compatible with the OpenAI API

## Using a Provider

You can specify a provider when creating an agent or solving a task:

```bash
# Create an agent using OpenRouter
surfkit create agent --provider openrouter

# Solve a task using Azure OpenAI
surfkit solve "Create a simple web app" --provider azure --provider-base-url "https://your-resource.openai.azure.com/openai"
```

## Environment Variables

For each provider, you need to set the corresponding API key as an environment variable:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-api-key-here"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key-here"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-api-key-here"
export AZURE_OPENAI_API_VERSION="2023-05-15" # Optional
```

## Custom Providers

For custom providers not in the predefined list, you need to specify both the API key and base URL:

```bash
# Set environment variables for a custom provider
export CUSTOM_API_KEY="your-custom-api-key"
export CUSTOM_BASE_URL="https://your-custom-api-endpoint.com/v1"

# Use the custom provider
surfkit solve "Create a simple web app" --provider custom
```

## Configuration File

You can also configure providers in a JSON configuration file at `~/.surfkit/config.json`:

```json
{
  "model": "gpt-4o",
  "provider": "openrouter",
  "providers": {
    "openai": {
      "name": "OpenAI",
      "baseURL": "https://api.openai.com/v1",
      "envKey": "OPENAI_API_KEY"
    },
    "openrouter": {
      "name": "OpenRouter",
      "baseURL": "https://openrouter.ai/api/v1",
      "envKey": "OPENROUTER_API_KEY"
    },
    "custom": {
      "name": "Custom Provider",
      "baseURL": "https://your-custom-api-endpoint.com/v1",
      "envKey": "CUSTOM_API_KEY"
    }
  }
}
```

## Listing Available Providers

To see all available providers and their configuration:

```bash
surfkit providers
```
