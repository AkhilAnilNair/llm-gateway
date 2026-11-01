package com.ai.llm_gateway.dto;

import java.util.List;

public record ChatCompletionRequest (
        String model,
        List<Message> messages,
        Boolean stream,
        Double temperature
){

    public record Message(String role, String content) {}

    public ChatCompletionRequest withModel(String targetModel) {
        return new ChatCompletionRequest(targetModel,
                this.messages,
                true,
                this.temperature != null? this.temperature : 0.2
        );
    }
}
