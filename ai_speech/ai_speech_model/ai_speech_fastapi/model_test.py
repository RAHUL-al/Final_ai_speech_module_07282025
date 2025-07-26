from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain.prompts import PromptTemplate
import torch
import logging

async def process_user_utterance(text, selecting_topic, selecting_accent, selecting_mood, selecting_class):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    try:
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "selecting_topic", "selecting_class", "selecting_mood", "selecting_accent"],
            template="""
                ROLE: You are a friendly, knowledgeable teaching assistant. Your purpose is to:
                1. Answer questions conversationally
                2. Never reveal internal project details
                3. Keep responses under 50 words
                4. Use scraping when needed

                RULES:
                - NEVER mention:
                * Model architecture/type
                * Team members/credentials
                * Code implementation
                * Technical specifications
                - If unsure, say "Let me check that for you" and use scraping()
                - Responses must be 1-2 sentences max
                - Always redirect technical questions to general knowledge

                CONTEXT: {context}
                TOPIC: {selecting_topic}
                CLASS: {selecting_class}
                MOOD: {selecting_mood}
                ACCENT: {selecting_accent}

                USER QUESTION: {question}

                RESPONSE FORMAT:
                [Concise 1-2 sentence response] 
                [Optional follow-up to keep the conversation going]
            """
        )

        prompt = prompt_template.format(
            context="This is a test context.",
            question=text,
            selecting_topic=selecting_topic,
            selecting_class=selecting_class,
            selecting_mood=selecting_mood,
            selecting_accent=selecting_accent
        )

        result = llm.invoke(prompt)
        logging.info(f"AI Response: {result}")

    except Exception as e:
        logging.error(f"QA Error: {str(e)}")


import asyncio
asyncio.run(process_user_utterance("Hello ? how are you?", "Math", "American", "Happy", "5th Grade"))
