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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Sonnet is a good speed/quality default for a chat agent; set ANTHROPIC_MODEL
# to a Haiku model for even lower latency, or an Opus model for max quality.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))

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
