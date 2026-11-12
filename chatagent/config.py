"""Central configuration, loaded from environment / .env.

Everything the agent needs to run is read here so the rest of the code never
touches os.environ directly. Copy .env.example to .env and fill in your key.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional; env vars still work without it
    pass

# --- Model -----------------------------------------------------------------
# GroqCloud. Groq's docs tell you to use the official `openai` SDK and just
# override base_url, so that is what agent.py does.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Text models with tool use, as served to this account (check your own with
# `client.models.list()` -- Groq's docs list models your key may not have):
#   openai/gpt-oss-120b   131k ctx,  500 T/s  - best judgement, the default
#   openai/gpt-oss-20b    131k ctx, 1000 T/s  - twice as fast, weaker reasoning
#   qwen/qwen3.8-27b      131k ctx            - PREVIEW: can vanish without notice
# The agent's hardest job is judging *when* to search, so the 120B is the right
# default; switch to the 20b if you want raw speed over that judgement.
# Neither gpt-oss model does parallel tool calls, which costs us nothing here:
# this agent issues one search at a time.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq's OpenAI-compatible endpoint takes `max_completion_tokens`, not the
# older `max_tokens`.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
# Safety stop for the tool loop, so a model that keeps calling web_search
# cannot spin forever at your expense.
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "4"))

# --- Search ----------------------------------------------------------------
# If TAVILY_API_KEY is set we use Tavily (higher quality); otherwise we fall
# back to DuckDuckGo, which needs no API key so the agent runs out of the box.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # seconds

# --- Memory ----------------------------------------------------------------
DB_PATH = os.getenv("CHATAGENT_DB", "chat_memory.db")
RECENT_TURNS = int(os.getenv("RECENT_TURNS", "12"))  # live context window (turns)
RECALL_K = int(os.getenv("RECALL_K", "4"))           # how many old turns to recall
