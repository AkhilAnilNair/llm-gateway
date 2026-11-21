import asyncio
import time
import aiohttp
import numpy as np

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"

SIMPLE_PROMPTS = [
    "What is the capital of Japan?",
    "Explain what an IP address is in one sentence.",
    "Summarize the difference between RAM and ROM briefly.",
    "Translate 'Thank you very much' to French.",
    "What does DNS stand for in networking?"
]

HARD_PROMPTS = [
    "Prove by induction that 2^n > n for all positive integers n.",
    "Design a fault-tolerant distributed lock service using Raft and explain lease expiration.",
    "Write a lock-free stack in Java using Compare-And-Swap and explain how to prevent the ABA problem.",
    "Explain how to solve the 0/1 Knapsack problem with dynamic programming and derive its space-optimized recurrence.",
    "Compare MVCC in PostgreSQL vs. InnoDB and explain how phantom reads are prevented under Serializable isolation."
]

async def send_req(session, prompt, tier="auto", allow_write=True, sem=None):
    headers = {
        "Content-Type": "application/json",
        "X-Route-Tier": tier,
        "X-Cache-Threshold": "0.90",
        "X-Cache-Write": "true" if allow_write else "false"
    }
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 150
    }
    async with sem:
        start = time.perf_counter()
        first_token = None
        async with session.post(GATEWAY_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                print(f"[HTTP {resp.status}] {await resp.text()}")
                return 0, 0
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:") and "[DONE]" not in decoded:
                    if first_token is None:
                        first_token = time.perf_counter()
        end = time.perf_counter()
        return (first_token - start) * 1000 if first_token else 0, (end - start) * 1000

async def main():
    timeout = aiohttp.ClientTimeout(total=120, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Exp 1: Concurrency=4 Mixed Workload (Tests Cache + Complexity Router + Queue Overflow)
        print("=== EXP 1: Concurrency=4 Full Intelligent Gateway (Cache + Router + Overflow) ===")
        workload = (SIMPLE_PROMPTS + HARD_PROMPTS) * 2  # 20 requests
        sem = asyncio.Semaphore(4)
        results = await asyncio.gather(*[send_req(session, p, "auto", True, sem) for p in workload])

        ttfts = [r[0] for r in results if r[0] > 0]
        e2es  = [r[1] for r in results if r[1] > 0]
        print(f"TTFT (ms) -> p50: {np.percentile(ttfts, 50):.1f} | p95: {np.percentile(ttfts, 95):.1f}")
        print(f"E2E  (ms) -> p50: {np.percentile(e2es, 50):.1f} | p95: {np.percentile(e2es, 95):.1f}")

if __name__ == "__main__":
    asyncio.run(main())