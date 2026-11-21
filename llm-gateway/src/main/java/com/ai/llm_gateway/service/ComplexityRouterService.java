package com.ai.llm_gateway.service;

import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ComplexityRouterService {

    private final EmbeddingService embeddingService;
    private float[] easyCentroid;
    private float[] hardCentroid;

    private static final List<String> EASY_ANCHORS = List.of(
            "Summarize this short paragraph in two sentences.",
            "What is the capital of France?",
            "Explain what HTTP stands for in simple terms.",
            "Translate 'Good morning' into Spanish.",
            "Define what a variable is in programming.",
            "List three benefits of drinking water.",
            "Rewrite this email to sound more polite.",
            "What is the difference between a list and a tuple in Python?"
    );

    private static final List<String> HARD_ANCHORS = List.of(
            "Prove by mathematical induction that the sum of the first n odd integers is n^2.",
            "Design a distributed rate limiter using Redis and token buckets that handles 100k RPS across multiple regions.",
            "Write a lock-free concurrent queue in Java using AtomicReference and explain the ABA problem.",
            "Solve this dynamic programming problem: find the longest increasing subsequence in O(n log n) time and prove its complexity.",
            "Compare Raft and Paxos consensus algorithms under network partitions and explain split-brain mitigation.",
            "Debug a race condition in a multi-threaded C++ producer-consumer ring buffer using memory barriers.",
            "Derive the backpropagation gradient for a multi-head self-attention layer in a Transformer.",
            "Optimize a PostgreSQL query with 5 joins and 100M rows using B-Tree vs. BRIN indexes and explain the query plan."
    );

    public ComplexityRouterService(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
    }

    @PostConstruct
    public void initCentroids() {
        this.easyCentroid = computeNormalizedCentroid(EASY_ANCHORS);
        this.hardCentroid = computeNormalizedCentroid(HARD_ANCHORS);
    }

    /**
     * Returns a complexity score C in [0.0, 1.0].
     * C >= 0.50 means the prompt is closer to Hard reasoning anchors.
     * C <  0.50 means the prompt is closer to Easy factual/summary anchors.
     */
    public double scoreComplexity(float[] promptVector) {
        double simEasy = cosineSimilarity(promptVector, easyCentroid);
        double simHard = cosineSimilarity(promptVector, hardCentroid);

        // Softmax-style scaling (temperature = 0.1) to map similarity delta cleanly into [0, 1]
        double diff = (simHard - simEasy) / 0.10;
        return 1.0 / (1.0 + Math.exp(-diff));
    }

    private float[] computeNormalizedCentroid(List<String> prompts) {
        float[] sum = new float[384];
        for (String p : prompts) {
            float[] vec = embeddingService.embed(p);
            for (int i = 0; i < 384; i++) {
                sum[i] += vec[i];
            }
        }
        return normalize(sum);
    }

    private float[] normalize(float[] vec) {
        double norm = 0.0;
        for (float v : vec) norm += v * v;
        norm = Math.sqrt(norm);
        float[] out = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            out[i] = (float) (vec[i] / (norm + 1e-10));
        }
        return out;
    }

    private double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
    }

}
