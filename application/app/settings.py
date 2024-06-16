import os
from dotenv import load_dotenv

load_dotenv()

# dev settings
DEBUG_MODE = os.getenv("DEBUG_MODE")
LOG_LEVEL = os.getenv("LOG_LEVEL")

# marqo
MARQO_URL = os.getenv("MARQO_URL")
MARQO_INDEX_NAME = os.getenv("MARQO_INDEX_NAME")
MARQO_LIMIT = os.getenv("MARQO_LIMIT")