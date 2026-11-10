package com.ai.llm_gateway.service;

import com.ai.llm_gateway.dto.ChatCompletionRequest;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class LlmRoutingService {
    private final WebClient localMetalClient;
    private final WebClient cloudGroqClient;
    private final MeterRegistry meterRegistry;

    public LlmRoutingService(
            @Qualifier("localMetalClient") WebClient localMetalClient,
            @Qualifier("cloudGroqClient") WebClient cloudGroqClient,
            MeterRegistry meterRegistry) {
        this.localMetalClient = localMetalClient;
        this.cloudGroqClient = cloudGroqClient;
        this.meterRegistry = meterRegistry;
    }

    public Flux<ServerSentEvent<String>> routeAndStream(ChatCompletionRequest request, String tier) {
        long requestStartNs = System.nanoTime();
        AtomicLong firstTokenTimeNs = new AtomicLong(0);
        AtomicBoolean firstTokenSeen = new AtomicBoolean(false);
        AtomicInteger tokenChunks = new AtomicInteger(0);

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
                });
    }
}
