package com.ai.llm_gateway.controller;

import com.ai.llm_gateway.dto.ChatCompletionRequest;
import com.ai.llm_gateway.service.LlmRoutingService;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/v1/chat")
public class GatewayController {

    private final LlmRoutingService routingService;

    public GatewayController(LlmRoutingService routingService) {
        this.routingService = routingService;
    }

    @PostMapping(value = "/completions", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> completions(
            @RequestBody ChatCompletionRequest request,
            @RequestHeader(value = "X-Route-Tier", defaultValue = "weak") String routeTier) {
        return routingService.routeAndStream(request, routeTier);
    }

}
