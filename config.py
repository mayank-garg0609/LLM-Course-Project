import os
from dotenv import load_dotenv
load_dotenv()

# ---- tunable settings (change here, nowhere else) ----
T1 = 0.2            # R below T1  -> VERIFIED
T2 = 0.5            # R at/above T2 -> BLOCKED
TOP_K_SEARCH = 5    # search results fetched per claim
TOP_K_EVIDENCE = 3  # passages kept after embedding rerank
W_CON = 1.0         # weight of a contradicted claim in R
W_NEI = 0.5         # weight of a "not enough info" claim in R
MAX_CLAIMS = 8      # cap claims per response (cost + latency)
MAX_WORKERS = 4     # parallel verifier calls

# claims matching any of these regexes are never sent to web search (separate with ;;)
# e.g. PRIVATE_REGEX=ProjectX;;\\b\\d{10}\\b
PRIVATE_PATTERNS = [p for p in os.environ.get("PRIVATE_REGEX", "").split(";;") if p]
