package com.ai.llm_gateway.service;

import io.micrometer.core.instrument.MeterRegistry;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.search.*;
import redis.clients.jedis.search.schemafields.TextField;
import redis.clients.jedis.search.schemafields.VectorField;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Service
public class SemanticCacheService {
    private static final String INDEX_NAME = "idx:llm_cache";
    private final JedisPooled jedis;
    private final EmbeddingService embeddingService;
    private final MeterRegistry meterRegistry;
    private final double defaultThreshold;

    public SemanticCacheService(
            @Value("${gateway.redis.host:localhost}") String redisHost,
            @Value("${gateway.redis.port:6379}") int redisPort,
            @Value("${gateway.cache.similarity-threshold:0.92}") double defaultThreshold,
            EmbeddingService embeddingService,
            MeterRegistry meterRegistry) {
        this.jedis = new JedisPooled(redisHost, redisPort);
        this.embeddingService = embeddingService;
        this.meterRegistry = meterRegistry;
        this.defaultThreshold = defaultThreshold;
    }

    @PostConstruct
    public void createVectorIndex() {
        try {
            jedis.ftInfo(INDEX_NAME);
        } catch (Exception e) {
            Map<String, Object> attr = new HashMap<>();
            attr.put("TYPE", "FLOAT32");
            attr.put("DIM", 384);
            attr.put("DISTANCE_METRIC", "COSINE");

            jedis.ftCreate(INDEX_NAME,
                    FTCreateParams.createParams().on(IndexDataType.HASH).prefix("cache:prompt:"),
                    TextField.of("prompt"),
                    TextField.of("response"),
                    VectorField.builder()
                            .fieldName("embedding")
                            .algorithm(VectorField.VectorAlgorithm.HNSW)
                            .attributes(attr)
                            .build()
            );
        }
    }

    public record CacheLookupResult(boolean hit, String response, float[] embedding, double similarity) {}

    public CacheLookupResult lookup(String prompt, Double customThreshold) {
        double threshold = (customThreshold != null) ? customThreshold : defaultThreshold;
        float[] vector = embeddingService.embed(prompt);

        // If threshold > 1.0 (e.g., 1.1), force a cache bypass for baseline testing
        if (threshold > 1.0) {
            meterRegistry.counter("llm.gateway.cache.requests", "result", "bypass").increment();
            return new CacheLookupResult(false, null, vector, 0.0);
        }

        byte[] vectorBytes = floatArrayToBytes(vector);

        Query q = new Query("*=>[KNN 1 @embedding $vec AS vector_distance]")
                .addParam("vec", vectorBytes)
                .returnFields("prompt", "response", "vector_distance")
                .setSortBy("vector_distance", true)
                .limit(0, 1)
                .dialect(2);

        SearchResult result = jedis.ftSearch(INDEX_NAME, q);
        if (!result.getDocuments().isEmpty()) {
            Document doc = result.getDocuments().get(0);
            double distance = Double.parseDouble(doc.getString("vector_distance"));
            double similarity = 1.0 - distance;

            if (similarity >= threshold) {
                meterRegistry.counter("llm.gateway.cache.requests", "result", "hit").increment();
                return new CacheLookupResult(true, doc.getString("response"), vector, similarity);
            }
        }
        meterRegistry.counter("llm.gateway.cache.requests", "result", "miss").increment();
        return new CacheLookupResult(false, null, vector, 0.0);
    }

    public void store(String prompt, float[] embedding, String fullResponse) {
        if (fullResponse == null || fullResponse.isBlank()) return;
        String key = "cache:prompt:" + UUID.randomUUID();
        Map<byte[], byte[]> hash = new HashMap<>();
        hash.put("prompt".getBytes(StandardCharsets.UTF_8), prompt.getBytes(StandardCharsets.UTF_8));
        hash.put("response".getBytes(StandardCharsets.UTF_8), fullResponse.getBytes(StandardCharsets.UTF_8));
        hash.put("embedding".getBytes(StandardCharsets.UTF_8), floatArrayToBytes(embedding));
        jedis.hset(key.getBytes(StandardCharsets.UTF_8), hash);
        jedis.expire(key.getBytes(StandardCharsets.UTF_8), 3600); // 1-hour TTL
    }

    private byte[] floatArrayToBytes(float[] input) {
        ByteBuffer buffer = ByteBuffer.allocate(input.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : input) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }
}
