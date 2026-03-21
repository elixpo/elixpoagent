"""Global configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    model_config = {"env_prefix": "PANDA_LLM_"}

    api_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    max_context_tokens: int = 128_000
    temperature: float = 0.0


class GitHubSettings(BaseSettings):
    model_config = {"env_prefix": "PANDA_GITHUB_"}

    app_id: str = ""
    private_key_path: str = ""
    webhook_secret: str = ""
    bot_username: str = "elixpoo"


class AgentSettings(BaseSettings):
    model_config = {"env_prefix": "PANDA_"}

    max_agent_steps: int = 50
    max_tokens_per_session: int = 500_000
    max_concurrent_sessions: int = 5
    session_storage_path: str = "/data/sessions"
    workspace_path: str = "/data/workspaces"


class SandboxSettings(BaseSettings):
    model_config = {"env_prefix": "PANDA_SANDBOX_"}

    mode: str = "none"  # "none", "nsjail", "docker"
    timeout: int = 120
    memory_limit: str = "512M"


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    github: GitHubSettings = GitHubSettings()
    agent: AgentSettings = AgentSettings()
    sandbox: SandboxSettings = SandboxSettings()

    api_secret_key: str = "dev-secret-change-me"
    cors_origins: list[str] = ["http://localhost:3000"]
    debug: bool = False

    model_config = {"env_prefix": "PANDA_"}


settings = Settings()
