package com.ai.llm_gateway.config;

import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import javax.net.ssl.SSLException;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient.Builder webClientBuilder(){
        return WebClient.builder();
    }

    @Bean
    public WebClient localMetalClient(WebClient.Builder builder,
                                      @Value("${gateway.nodes.local-url}") String localUrl
                                      ) {
        return builder.clone().baseUrl(localUrl.trim()).build();
    }

    @Bean
    public WebClient cloudGroqClient(WebClient.Builder builder,
                                     @Value("${gateway.nodes.cloud-url}") String cloudUrl,
                                     @Value("${gateway.nodes.cloud-api-key}") String apiKey) throws SSLException {
        SslContext sslContext = SslContextBuilder.forClient()
                .trustManager(InsecureTrustManagerFactory.INSTANCE)
                .build();

        HttpClient httpClient = HttpClient.create()
                .secure(sslSpec -> sslSpec.sslContext(sslContext));


        return builder.clone().baseUrl(cloudUrl.trim())
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .build();
    }
}
