import asyncio
import json
import time
import aiohttp
import numpy as np

GATEWAY_URL = "http://localhost:8080/v1/chat/completions"

# 5 Canonical Prompts to seed the cache
SEED_PROMPTS = [
    "Explain how a TCP 3-way handshake works in two sentences.",
    "Write a Python function to check if a string is a palindrome.",
    "Summarize the difference between a process and a thread.",
    "What is the time complexity of binary search and why?",
    "Explain consistent hashing in distributed systems briefly."
]

# Mixed Workload: 50% Paraphrased Hits (semantic match) + 50% Novel Misses
MIXED_WORKLOAD = [
    "Can you briefly explain how the TCP three-way handshake works?",          # Paraphrase of #1
    "What is the role of the page table in operating system virtual memory?",  # Novel Miss
    "Write a short Python function that checks whether a string is a palindrome.", # Paraphrase of #2
    "Explain how Raft consensus elects a leader in a distributed cluster.",     # Novel Miss
    "What is the main difference between an OS process and a thread?",         # Paraphrase of #3
    "How does a B-Tree index speed up database range queries?",                # Novel Miss
    "Why is the time complexity of binary search O(log n)?",                   # Paraphrase of #4
    "Explain the difference between optimistic and pessimistic locking.",      # Novel Miss
    "Briefly explain how consistent hashing works in distributed systems.",     # Paraphrase of #5
    "How does garbage collection work in the Java Virtual Machine?"            # Novel Miss
]

# Adversarial Near-Miss Pairs (Similar words, opposite/different meaning!)
ADVERSARIAL_PROBES = [
    ("Write a Python script to delete all files in /tmp.", "Write a Python script to list all files in /tmp."),
    ("Explain how the TCP protocol works in networking.", "Explain how the UDP protocol works in networking.")
]

async def send_request(session, prompt, tier="weak", threshold=0.90, semaphore=None):
    headers = {
        "Content-Type": "application/json",
        "X-Route-Tier": tier,
        "X-Cache-Threshold": str(threshold)
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
        is_cache_hit = False

        async with session.post(GATEWAY_URL, json=payload, headers=headers) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:") and "[DONE]" not in decoded:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if "redis-semantic-cache" in decoded:
                        is_cache_hit = True

        end_time = time.perf_counter()
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_ms = (end_time - start_time) * 1000
        return ttft_ms, total_ms, is_cache_hit

async def run_experiment_1():
    print("\n--- Step 1: Seeding Redis Vector Index with 5 Canonical Prompts ---")
    sem = asyncio.Semaphore(2)
    async with aiohttp.ClientSession() as session:
        for p in SEED_PROMPTS:
            await send_request(session, p, tier="weak", threshold=0.90, semaphore=sem)

        print("--- Step 2: Running Mixed Workload (20 Reqs, Concurrency=4, Threshold=0.85) ---")
        sem_c4 = asyncio.Semaphore(4)
        tasks = [
            send_request(session, MIXED_WORKLOAD[i % len(MIXED_WORKLOAD)], "weak", 0.85, sem_c4)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

    all_ttfts = [r[0] for r in results]
    hit_ttfts = [r[0] for r in results if r[2]]
    miss_ttfts = [r[0] for r in results if not r[2]]
    hit_rate = (len(hit_ttfts) / len(results)) * 100

    print(f"\n=== EXP 1: Concurrency=4 with Semantic Cache (Hit Rate: {hit_rate:.1f}%) ===")
    print(f"Overall TTFT (ms) -> p50: {np.percentile(all_ttfts, 50):.1f} | p95: {np.percentile(all_ttfts, 95):.1f}")
    if hit_ttfts:
        print(f"Cache HIT TTFT    -> p50: {np.percentile(hit_ttfts, 50):.1f} ms (Bypassed GPU Queue!)")
    if miss_ttfts:
        print(f"Cache MISS TTFT   -> p50: {np.percentile(miss_ttfts, 50):.1f} | p95: {np.percentile(miss_ttfts, 95):.1f}")

async def run_experiment_2():
    print("\n=== EXP 2: Similarity Threshold (θ) Sensitivity & False-Positive Analysis ===")
    sem = asyncio.Semaphore(1)
    async with aiohttp.ClientSession() as session:
        # Seed adversarial base prompts
        for seed_p, _ in ADVERSARIAL_PROBES:
            await send_request(session, seed_p, "weak", 0.99, sem)

        paraphrases = [MIXED_WORKLOAD[0], MIXED_WORKLOAD[2], MIXED_WORKLOAD[4], MIXED_WORKLOAD[6], MIXED_WORKLOAD[8]]
        adversarials = [probe for _, probe in ADVERSARIAL_PROBES]

        print(f"{'Threshold (θ)':<15} | {'True Paraphrase Hit %':<24} | {'False Positive (Poison) %':<25}")
        print("-" * 70)
        for theta in [0.75, 0.85, 0.90, 0.96]:
            tp_hits = 0
            for p in paraphrases:
                _, _, hit = await send_request(session, p, "weak", theta, sem)
                if hit: tp_hits += 1

            fp_hits = 0
            for adv in adversarials:
                _, _, hit = await send_request(session, adv, "weak", theta, sem)
                if hit: fp_hits += 1

            print(f"{theta:<15.2f} | {(tp_hits/len(paraphrases))*100:<23.1f}% | {(fp_hits/len(adversarials))*100:<24.1f}%")

async def main():
    await run_experiment_1()
    await run_experiment_2()

if __name__ == "__main__":
    asyncio.run(main())