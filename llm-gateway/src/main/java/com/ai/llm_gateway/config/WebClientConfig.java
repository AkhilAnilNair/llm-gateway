package com.ai.llm_gateway.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient localMetalCLient(WebClient.Builder builder,
                                      @Value("${gateway.nodes.local-url") String localUrl
                                      ) {
        return builder.baseUrl(localUrl).build();
    }

    @Bean
    public WebClient cloudGroqClient(WebClient.Builder builder,
                                     @Value("${gateway.nodes.cloud-url}") String cloudUrl,
                                     @Value("${gateway.nodes.cloud-api-key}") String apiKey) {
        return builder.baseUrl(cloudUrl)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .build();
    }
}
