from dotenv import load_dotenv
import soundfile as sf
import requests
import os
import time
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datetime import datetime
import asyncio
import threading
from gector.modeling import GECToR
from gector.predict import predict, load_verb_dict
from kokoro import KPipeline
from IPython.display import display, Audio
from pydub import AudioSegment
import re
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from firebase_admin import credentials, firestore, initialize_app
from firebase import db
import asyncio
import json
import logging
from contextlib import asynccontextmanager
import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration
import os
import logging
from transformers.utils import logging as hf_logging
from difflib import SequenceMatcher
import tempfile
from datetime import datetime
import logging
import os
import soundfile as sf
from pydub import AudioSegment
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# def _compare_sentences(reference, corrected):
#     reference_tokens = reference.lower().split()
#     corrected_tokens = corrected.lower().split()
#     smoothie = SmoothingFunction().method4
#     score = sentence_bleu([reference_tokens], corrected_tokens, smoothing_function=smoothie)
#     return round(score * 10, 2)

# def _grammar_check_sync(spoken_text, original_text):
#     model_id = "gotutiyan/gector-roberta-base-5k"
#     model = GECToR.from_pretrained(model_id)

#     if torch.cuda.is_available():
#         model = model.cuda()

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     encode, decode = load_verb_dict('data/verb-form-vocab.txt')

#     srcs = [spoken_text.strip()]

#     corrected = predict(
#         model, tokenizer, srcs, encode, decode,
#         keep_confidence=0.0,
#         min_error_prob=0.0,
#         n_iteration=5,
#         batch_size=1,
#     )

#     corrected_sent = corrected[0] if corrected else spoken_text
#     grammar_score = _compare_sentences(original_text, corrected_sent)
#     percentage_score = round(grammar_score)

#     return percentage_score


# score = _grammar_check_sync("I has a apple.", "I have an apple.")
# print(f"Grammar Score: {score}%")


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher
import spacy
import numpy as np

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def _compare_sentences(reference, corrected):
    reference = reference.lower().strip()
    corrected = corrected.lower().strip()
    bleu_score = _get_bleu_score(reference, corrected)
    seq_ratio = SequenceMatcher(None, reference, corrected).ratio()
    grammar_errors = _check_grammar_rules(reference, corrected)
    semantic_sim = _get_semantic_similarity(reference, corrected)
    final_score = (
        0.3 * bleu_score + 
        0.2 * seq_ratio * 10 + 
        0.3 * (1 - grammar_errors) * 10 +
        0.2 * semantic_sim * 10
    )
    
    return min(10, max(0, round(final_score, 2)))

def _get_bleu_score(ref, cor):
    ref_tokens = ref.split()
    cor_tokens = cor.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cor_tokens, smoothing_function=smoothie)

def _check_grammar_rules(ref, cor):
    """Check specific grammar rules and count errors"""
    error_count = 0

    ref_doc = nlp(ref)
    cor_doc = nlp(cor)

    for ref_token, cor_token in zip(ref_doc, cor_doc):
        if ref_token.text in ['a', 'an'] and cor_token.text in ['a', 'an']:
            if ref_token.text != cor_token.text:
                error_count += 1

    ref_verbs = [token.text for token in ref_doc if token.pos_ == "VERB"]
    cor_verbs = [token.text for token in cor_doc if token.pos_ == "VERB"]
    if ref_verbs != cor_verbs:
        error_count += 1

    max_errors = max(len(ref.split()), len(cor.split()))
    return min(1, error_count / max(1, max_errors))

def _get_semantic_similarity(ref, cor):
    """Get semantic similarity using spaCy word vectors"""
    ref_doc = nlp(ref)
    cor_doc = nlp(cor)
    return ref_doc.similarity(cor_doc)

def _grammar_check_sync(spoken_text, original_text):
    model_id = "gotutiyan/gector-roberta-base-5k"
    model = GECToR.from_pretrained(model_id)

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encode, decode = load_verb_dict('data/verb-form-vocab.txt')

    srcs = [spoken_text.strip()]

    corrected = predict(
        model, tokenizer, srcs, encode, decode,
        keep_confidence=0.0,
        min_error_prob=0.0,
        n_iteration=5,
        batch_size=1,
    )

    corrected_sent = corrected[0] if corrected else spoken_text
    grammar_score = _compare_sentences(original_text, corrected_sent)
    
    return grammar_score

