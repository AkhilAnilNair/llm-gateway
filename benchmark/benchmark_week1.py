import asyncio
import time
import aiohttp
import numpy as np

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"

PROMPTS = [
    "Explain how a TCP 3-way handshake works in two sentences.",
    "Write a Python function to check if a string is a palindrome.",
    "Summarize the difference between a process and a thread.",
    "What is the time complexity of binary search and why?",
    "Explain consistent hashing in distributed systems briefly."
]

async def send_request(session, prompt, tier, semaphore):
    headers = {
        "Content-Type": "application/json",
        "X-Route-Tier": tier
    }
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.2
    }

    async with semaphore:
        start_time = time.perf_counter()
        first_token_time = None
        chunk_count = 0

        async with session.post(GATEWAY_URL, json=payload, headers=headers) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:") and "[DONE]" not in decoded:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunk_count += 1

        end_time = time.perf_counter()
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_ms = (end_time - start_time) * 1000
        tpot_ms = ((end_time - first_token_time) * 1000 / max(chunk_count - 1, 1)) if first_token_time else 0
        return ttft_ms, tpot_ms, total_ms, chunk_count

async def run_suite(tier="weak", total_requests=20, concurrency=2):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, PROMPTS[i % len(PROMPTS)], tier, semaphore)
            for i in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)

    ttfts = [r[0] for r in results]
    tpots = [r[1] for r in results]
    e2es  = [r[2] for r in results]

    print(f"\n=== Tier: {tier.upper()} | Requests: {total_requests} | Concurrency: {concurrency} ===")
    print(f"TTFT (ms) -> p50: {np.percentile(ttfts, 50):.1f} | p95: {np.percentile(ttfts, 95):.1f}")
    print(f"TPOT (ms) -> p50: {np.percentile(tpots, 50):.1f} | p95: {np.percentile(tpots, 95):.1f}")
    print(f"E2E  (ms) -> p50: {np.percentile(e2es, 50):.1f} | p95: {np.percentile(e2es, 95):.1f}")

async def main():
    # Warm-up request
    await run_suite(tier="weak", total_requests=2, concurrency=1)
    # Actual Week 1 Baseline Tests
    await run_suite(tier="weak", total_requests=20, concurrency=1)
    await run_suite(tier="weak", total_requests=20, concurrency=4)
    await run_suite(tier="strong", total_requests=20, concurrency=2)

if __name__ == "__main__":
    asyncio.run(main())
