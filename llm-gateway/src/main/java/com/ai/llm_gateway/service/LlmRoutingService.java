package com.ai.llm_gateway.service;

import com.ai.llm_gateway.dto.ChatCompletionRequest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class LlmRoutingService {
    private final WebClient localMetalClient;
    private final WebClient cloudGroqClient;
    private final SemanticCacheService semanticCacheService;
    private final MeterRegistry meterRegistry;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public LlmRoutingService(
            @Qualifier("localMetalClient") WebClient localMetalClient,
            @Qualifier("cloudGroqClient") WebClient cloudGroqClient,
            SemanticCacheService semanticCacheService,
            MeterRegistry meterRegistry) {
        this.localMetalClient = localMetalClient;
        this.cloudGroqClient = cloudGroqClient;
        this.semanticCacheService = semanticCacheService;
        this.meterRegistry = meterRegistry;
    }

    public Flux<ServerSentEvent<String>> routeAndStream(ChatCompletionRequest request, String tier, Double customThreshold) {
        long requestStartNs = System.nanoTime();

        String userPrompt = extractLatestUserPrompt(request);

        return Mono.fromCallable(() -> semanticCacheService.lookup(userPrompt, customThreshold))
                .subscribeOn(Schedulers.boundedElastic())
                .flatMapMany(cacheResult -> {
                    // 1. TIER-1 SEMANTIC CACHE HIT: Short-circuit the LLM completely!
                    if (cacheResult.hit()) {
                        long hitLatencyNs = System.nanoTime() - requestStartNs;
                        meterRegistry.timer("llm.gateway.ttft", "node", "redis-semantic-cache")
                                .record(Duration.ofNanos(hitLatencyNs));
                        meterRegistry.timer("llm.gateway.e2e.latency", "node", "redis-semantic-cache")
                                .record(Duration.ofNanos(hitLatencyNs));

                        String sseJson = formatAsOpenAiChunk(cacheResult.response(), cacheResult.similarity());
                        return Flux.just(
                                ServerSentEvent.<String>builder().data(sseJson).build(),
                                ServerSentEvent.<String>builder().data("[DONE]").build()
                        );
                    }

                    // 2. CACHE MISS: Route to Downstream LLM & Asynchronously Populate Cache
                    return streamFromDownstreamLlm(request, tier, requestStartNs, userPrompt, cacheResult.embedding());
                });
    }

    private Flux<ServerSentEvent<String>> streamFromDownstreamLlm(
            ChatCompletionRequest request,
            String tier,
            long requestStartNs,
            String userPrompt,
            float[] promptEmbedding) {

        AtomicLong firstTokenTimeNs = new AtomicLong(0);
        AtomicBoolean firstTokenSeen = new AtomicBoolean(false);
        AtomicInteger tokenChunks = new AtomicInteger(0);
        StringBuilder fullResponseBuffer = new StringBuilder();

        boolean isStrongTier = "strong".equalsIgnoreCase(tier);
        WebClient client = isStrongTier ? cloudGroqClient : localMetalClient;
        String targetModel = isStrongTier ? "openai/gpt-oss-20b" : "qwen2.5:1.5b";
        String nodeLabel = isStrongTier ? "cloud-groq-70b" : "local-m2-1.5b";

        return client.post()
                .uri("/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .bodyValue(request.withModel(targetModel))
                .retrieve()
                .bodyToFlux(String.class)
                .filter(chunk -> !chunk.isBlank())
                .doOnNext(chunk -> {
                    if (firstTokenSeen.compareAndSet(false, true)) {
                        long now = System.nanoTime();
                        firstTokenTimeNs.set(now);
                        meterRegistry.timer("llm.gateway.ttft", "node", nodeLabel)
                                .record(Duration.ofNanos(now - requestStartNs));
                    }
                    if (!"[DONE]".equals(chunk.trim())) {
                        tokenChunks.incrementAndGet();
                        extractDeltaContent(chunk, fullResponseBuffer);
                    }
                })
                .map(chunk -> ServerSentEvent.<String>builder().data(chunk).build())
                .doOnComplete(() -> {
                    long endTimeNs = System.nanoTime();
                    long totalNs = endTimeNs - requestStartNs;
                    meterRegistry.timer("llm.gateway.e2e.latency", "node", nodeLabel)
                            .record(Duration.ofNanos(totalNs));

                    int chunks = tokenChunks.get();
                    if (chunks > 1 && firstTokenTimeNs.get() > 0) {
                        long decodeDurationNs = endTimeNs - firstTokenTimeNs.get();
                        long tpotNs = decodeDurationNs / (chunks - 1);
                        meterRegistry.timer("llm.gateway.tpot", "node", nodeLabel)
                                .record(Duration.ofNanos(tpotNs));
                    }
                    meterRegistry.counter("llm.gateway.tokens.total", "node", nodeLabel)
                            .increment(chunks);

                    // Save assembled response + vector to Redis in background thread
                    String assembledText = fullResponseBuffer.toString();
                    if (!assembledText.isBlank()) {
                        Mono.fromRunnable(() -> semanticCacheService.store(userPrompt, promptEmbedding, assembledText))
                                .subscribeOn(Schedulers.boundedElastic())
                                .subscribe();
                    }
                });
    }
    private String extractLatestUserPrompt(ChatCompletionRequest request) {
        if (request.messages() == null || request.messages().isEmpty()) return "";
        return request.messages().get(request.messages().size() - 1).content();
    }

    private void extractDeltaContent(String rawJsonChunk, StringBuilder buffer) {
        try {
            String cleanJson = rawJsonChunk.startsWith("data:")
                    ? rawJsonChunk.substring(5).trim() : rawJsonChunk.trim();
            if ("[DONE]".equals(cleanJson)) return;
            JsonNode root = objectMapper.readTree(cleanJson);
            JsonNode contentNode = root.path("choices").path(0).path("delta").path("content");
            if (!contentNode.isMissingNode() && !contentNode.isNull()) {
                buffer.append(contentNode.asText());
            }
        } catch (Exception ignored) {}
    }
    private String formatAsOpenAiChunk(String cachedContent, double similarity) {
        try {
            Map<String, Object> payload = Map.of(
                    "id", "chatcmpl-cache-hit",
                    "object", "chat.completion.chunk",
                    "model", "redis-semantic-cache",
                    "similarity", similarity,
                    "choices", List.of(Map.of(
                            "index", 0,
                            "delta", Map.of("content", cachedContent),
                            "finish_reason", "stop"
                    ))
            );
            return objectMapper.writeValueAsString(payload);
        } catch (Exception e) {
            return "{\"choices\":[{\"delta\":{\"content\":\"Cached response\"}}]}";
        }
    }
}

