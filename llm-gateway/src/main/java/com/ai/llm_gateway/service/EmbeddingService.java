package com.ai.llm_gateway.service;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import io.micrometer.core.instrument.MeterRegistry;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class EmbeddingService {
    private ZooModel<String, float[]> model;
    private Predictor<String, float[]> predictor;
    private final MeterRegistry meterRegistry;

    public EmbeddingService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    @PostConstruct
    public void init() throws Exception {
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .optEngine("OnnxRuntime")
                .optModelUrls("djl://ai.djl.huggingface.onnxruntime/sentence-transformers/all-MiniLM-L6-v2")
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    public synchronized float[] embed(String prompt) {
        long startNs = System.nanoTime();
        try {
            return predictor.predict(prompt);
        } catch (TranslateException e) {
            throw new RuntimeException("ONNX Embedding calculation failed", e);
        } finally {
            meterRegistry.timer("llm.gateway.embedding.latency")
                    .record(Duration.ofNanos(System.nanoTime() - startNs));
        }
    }

    @PreDestroy
    public void close() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }
}
