"""
Pre-flight check: verify API keys work, print usage + cost.
Run once before writing any real code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Check env vars loaded ---
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

print("=" * 60)
print("PRE-FLIGHT CHECK")
print("=" * 60)

if not ANTHROPIC_KEY or not ANTHROPIC_KEY.startswith("sk-ant-"):
    print("❌ ANTHROPIC_API_KEY missing or invalid format")
    exit(1)
print(f"✅ Anthropic key loaded (ends with ...{ANTHROPIC_KEY[-6:]})")

if not TAVILY_KEY or not TAVILY_KEY.startswith("tvly-"):
    print("❌ TAVILY_API_KEY missing or invalid format")
    exit(1)
print(f"✅ Tavily key loaded (ends with ...{TAVILY_KEY[-6:]})")

# --- 2. Test Anthropic API ---
print("\n--- Testing Anthropic API ---")
from anthropic import Anthropic

client = Anthropic()

try:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'API works' in exactly 3 words."}
        ],
    )
    text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Haiku 4.5 pricing (approx): $1/M input, $5/M output
    cost = (input_tokens / 1_000_000) * 1.0 + (output_tokens / 1_000_000) * 5.0

    print(f"✅ Anthropic response: '{text}'")
    print(f"   Input tokens:  {input_tokens}")
    print(f"   Output tokens: {output_tokens}")
    print(f"   Cost:          ${cost:.6f}")
except Exception as e:
    print(f"❌ Anthropic API failed: {type(e).__name__}: {e}")
    exit(1)

# --- 3. Test Tavily API ---
print("\n--- Testing Tavily API ---")
from tavily import TavilyClient

try:
    tavily = TavilyClient(api_key=TAVILY_KEY)
    results = tavily.search(query="latest Python version", max_results=2)
    print(f"✅ Tavily returned {len(results.get('results', []))} results")
    print(f"   First result title: {results['results'][0]['title'][:80]}")
except Exception as e:
    print(f"❌ Tavily API failed: {type(e).__name__}: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED — ready to build")
print("=" * 60)