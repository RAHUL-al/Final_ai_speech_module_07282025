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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("essay")
hf_logging.set_verbosity_error()




async def async_with_timeout(coro, timeout: int, name=""):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] Task '{name}' timed out after {timeout} seconds")
        return f"[TIMEOUT] Task '{name}' took too long"
    except Exception as e:
        logger.exception(f"[ERROR] Task '{name}' failed: {e}")
        return f"[ERROR] Task '{name}' failed"


load_dotenv()

class Topic:
    def __init__(self):
        self.total_fluency = 0.0
        self.total_pronunciation = 0.0
        self.total_emotion = {}
        self.chunk_count = 0
        self.model_name = OllamaLLM(model="mistral")
        self.model_path = "silero_vad/silero-vad/src/silero_vad/data/silero_vad.jit"
        self.model = torch.jit.load(self.model_path)
        self.model.eval()
        self.sample_rate = 16000
        self.window_size = 512
        self.hop_size = 160
        self.threshold = 0.3

    def reset_realtime_stats(self):
        self.total_fluency = 0.0
        self.total_pronunciation = 0.0
        self.total_emotion = {}
        self.chunk_count = 0

    def update_realtime_stats(self, fluency, pronunciation, emotion):
        try:
            self.total_fluency += float(fluency)
        except:
            pass

        try:
            self.total_pronunciation += float(pronunciation)
        except:
            pass
        self.total_emotion[emotion] = self.total_emotion.get(emotion, 0) + 1
        self.chunk_count += 1

    def get_average_realtime_scores(self):
        if self.chunk_count == 0:
            return {"fluency": 0, "pronunciation": 0, "emotion": "unknown"}
        avg_fluency = round(self.total_fluency / self.chunk_count, 2)
        avg_pronunciation = round(self.total_pronunciation / self.chunk_count, 2)
        dominant_emotion = max(self.total_emotion.items(), key=lambda x: x[1])[0] if self.total_emotion else "unknown"
        return {
            "fluency": avg_fluency,
            "pronunciation": avg_pronunciation,
            "emotion": dominant_emotion,
        }

    async def topic_data_model_for_Qwen(self, username: str, prompt: str) -> str:
        try:
            model_name = OllamaLLM(model="mistral")
            response = model_name.invoke(prompt)

            asyncio.create_task(self.text_to_speech(response, username))

            return response
        except Exception as e:
            print(f"[mistral API Error] {e}")
            return "mistral model failed to generate a response."



    async def silvero_vad(self, audio_path: str):
        return await asyncio.to_thread(self._silvero_vad_sync, audio_path)

    def _silvero_vad_sync(self, audio_path: str) -> dict:
        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)

        max_abs = wav.abs().max()
        if max_abs > 0:
            wav = wav / max_abs
        else:
            return {"duration": 0.0}

        wav = wav.squeeze()
        if len(wav) < self.window_size:
            return {"duration": float(len(wav) / self.sample_rate)}

        speech_times = []

        for i in range(0, len(wav) - self.window_size + 1, self.hop_size):
            chunk = wav[i:i + self.window_size].unsqueeze(0)

            try:
                with torch.no_grad():
                    prob = self.model(chunk, self.sample_rate).item()
            except Exception as e:
                print(f"Skipping chunk due to error: {e}")
                continue

            if prob > self.threshold:
                time = i / self.sample_rate
                speech_times.append(time)

        if not speech_times:
            return {"duration": float(len(wav) / self.sample_rate)}

        max_silence = max(
            (speech_times[i] - speech_times[i - 1]) for i in range(1, len(speech_times))
        ) if len(speech_times) > 1 else 0.0

        return {"duration": max_silence}


    

    async def grammar_checking(self, spoken_text):
        return await asyncio.to_thread(self._grammar_check_sync, spoken_text)

    def _grammar_check_sync(self, spoken_text):
        model_id = "textattack/roberta-base-CoLA"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        use_safetensors=True
    ).to(device)
        inputs = tokenizer(
            spoken_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prob = probs[0][1].item()
            prob = round(prob * 10, 2)
        return prob
    

    async def overall_scoring_by_id(self, essay_id: str):
        # try:
        essay_doc = db.collection("essays").document(essay_id).get()
        if not essay_doc.exists:
            return {"error": f"No essay found with id {essay_id}"}

        essay_data = essay_doc.to_dict()
        return essay_data
        #     original_text = essay_data.get("content", "")
        #     username = essay_data.get("username")
        #     if not username:
        #         return {"error": "Username missing in essay data"}

        #     chunks = essay_data.get("chunks", [])
        #     if not chunks:
        #         return {"error": "No audio chunks available for this essay. Please ensure audio recording was completed."}

        #     spoken_text = " ".join(chunk.get("text", "") for chunk in chunks)

        #     avg_scores = essay_data.get("average_scores", {})
        #     pronunciation = avg_scores.get("pronunciation")
        #     fluency = avg_scores.get("fluency")
        #     emotion = avg_scores.get("emotion")

        #     if pronunciation is None or fluency is None or not emotion:
        #         fluency_scores = [float(c.get("fluency", 0)) for c in chunks if c.get("fluency") is not None]
        #         pronunciation_scores = [float(c.get("pronunciation", 0)) for c in chunks if c.get("pronunciation") is not None]
        #         emotion_counts = {}
        #         for c in chunks:
        #             em = c.get("emotion")
        #             if em:
        #                 emotion_counts[em] = emotion_counts.get(em, 0) + 1

        #         pronunciation = round(sum(pronunciation_scores) / len(pronunciation_scores), 2) if pronunciation_scores else "[ERROR]"
        #         fluency = round(sum(fluency_scores) / len(fluency_scores), 2) if fluency_scores else "[ERROR]"
        #         emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "[ERROR]"

        #     grammar = await self.grammar_checking(spoken_text)

        #     prompt = f"""
        #         You are an AI English teacher evaluating a student's spoken response based on their performance. You will be provided with the student's spoken text and a reference essay.

        #         Here are the available scores (only use the ones that are valid and available, ignore or skip any missing or erroneous ones like '[ERROR]' or None):
        #         - Pronunciation: {pronunciation}
        #         - Grammar: {grammar}
        #         - Fluency: {fluency}
        #         - Emotion: {emotion}

        #         Reference Essay:
        #         \"\"\"{original_text}\"\"\"

        #         Spoken Text:
        #         \"\"\"{spoken_text}\"\"\"

        #         Based on the above, return an honest but encouraging evaluation in **JSON format** with the following keys:
        #         - before giving final understanding, topic_grip and suggestions plese look on all the score of pronunciation, grammar, fluency, emotion. Here are you get the score is out of 10.
        #         - "understanding": Describe how well the spoken text reflects understanding of the reference essay.
        #         - "topic_grip": Comment on how well the speaker stayed on topic and conveyed key points.
        #         - "suggestions": A list of 3 teacher-style suggestions to improve the student's speaking and comprehension.
        #         - And also suggest some suggestion with example where is the problem in the speech text and what are you need to speak give 1-5 examples in suggestions.

        #         'blocklist': [
        #             'you', 'thank you', 'tchau', 'thanks', 'ok', 'Obrigado.', 'E aí', '',
        #             'me', 'hello', 'hi', 'hey', 'okay', 'thanks', 'thank', 'obrigado',
        #             'tchau.', 'bye', 'goodbye', 'me.', 'you.', 'thank you.'
        #         ],

        #         Important:
        #         - You got some data of list blocklist, then you need to concentrate on some word found from blocklist then try to understand it is worked as a raw sentence or make a little meaning because here this content is coming from database which is speak by user then used it else did not take word means filter that from existing content.
        #         - Do not include invalid, missing, or placeholder values in the response.
        #         - Do not talk about improving the AI itself; focus on guiding a human student.
        #         - Keep the tone supportive and constructive, like a real teacher giving oral feedback.
        #     """

        #     summary_response = await self.topic_data_model_for_Qwen(username, prompt)

        #     try:
        #         result = json.loads(str(summary_response))
        #     except Exception as e:
        #         logger.warning(f"Could not parse JSON: {e}")
        #         result = {"raw_response": str(summary_response)}

        #     result.update({
        #         "pronunciation": pronunciation,
        #         "grammar": grammar,
        #         "fluency": fluency,
        #         "emotion": emotion
        #     })

        #     return result

        # except Exception as e:
        #     logger.exception(f"[ERROR] overall_scoring_by_id failed: {e}")
        #     return {"error": "Internal Server Error"}

	
    async def speech_to_text(self, audio_path: str, username: str, device=None) -> str:
        try:
            # Run the synchronous _speech_to_text in a thread
            return await asyncio.to_thread(
                self._speech_to_text, 
                audio_path, 
                device
            )
        except Exception as e:
            print(f"[ERROR] Failed in speech_to_text: {e}")
            return ""

    def _speech_to_text(self, audio_path: str, device=None) -> str:
        token = "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"
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "audio/wav"
        }

        if not os.path.exists(audio_path):
            print(f"[Error] Audio file not found: {audio_path}")
            return ""

        try:
            with open(audio_path, "rb") as f:
                data = f.read()

            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()

            result = response.json()
            text = result.get("text", "").strip()
            print(f"Transcribed [{os.path.basename(audio_path)}]: {text}")
            return text

        except Exception as e:
            print(f"[Error] Failed to transcribe {audio_path}: {e}")
            return ""


    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for transcription (placeholder implementation)"""
        if not text:
            return 0.0
            
        # Simple heuristic - longer texts get higher confidence
        word_count = len(text.split())
        base_conf = min(0.3 + (word_count * 0.05), 0.9)  # 0.3-0.9 range
        
        # Penalize common error patterns
        if any(patt in text.lower() for patt in ['...', '??', 'unintelligible']):
            base_conf *= 0.7
            
        return round(base_conf, 2)
    async def text_to_speech(self, text_data, username):
        return await asyncio.to_thread(self._text_to_speech_sync, text_data, username)

    def _text_to_speech_sync(self, text_data, username):
        output_dir = os.path.join("text_to_speech_audio_folder", username)
        os.makedirs(output_dir, exist_ok=True)

        pipeline = KPipeline(lang_code='a')
        text = text_data
        max_chars = 400
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        generated_files = []
        for i, chunk in enumerate(chunks):
            print(f"\n chunk {i+1}: {chunk}\n")
            generator = pipeline(chunk, voice='af_heart')
            for j, (gs, ps, audio) in enumerate(generator):
                filename = os.path.join(output_dir, f'chunk_{i+1}_part_{j+1}.wav')
                sf.write(filename, audio, 24000)
                generated_files.append(filename)

        combined = AudioSegment.empty()
        for file in generated_files:
            audio = AudioSegment.from_wav(file)
            combined += audio

        output_path = os.path.join(output_dir, f"{username}_output.wav")
        combined.export(output_path, format="wav")
        print(f"Exported final audio to {output_path}")

        for file in generated_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting file: {file} - {e}")

        return output_path
    
    
    async def text_to_speech_assistant(self, text_data, username, session_temp_dir=None):
        return await asyncio.to_thread(
            self._text_to_speech_sync_assistant, 
            text_data, 
            username,
            session_temp_dir
        )

    def _text_to_speech_sync_assistant(self, text_data, username, session_temp_dir=None):
        # Use provided temp dir or fallback to default location
        output_dir = session_temp_dir if session_temp_dir else os.path.join("text_to_speech_audio_folder", username)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique output filename
        timestamp = int(datetime.now().timestamp())
        output_path = os.path.join(output_dir, f"tts_{username}_{timestamp}.wav")
        
        try:
            pipeline = KPipeline(lang_code='a')
            combined = AudioSegment.empty()
            
            # Process text in memory without intermediate files
            for _, _, audio in pipeline(text_data, voice='af_heart'):
                with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                    sf.write(tmp.name, audio, 24000)
                    combined += AudioSegment.from_wav(tmp.name)
            
            combined.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            # Create silent fallback in same directory
            silent_path = os.path.join(output_dir, "silent_fallback.wav")
            AudioSegment.silent(duration=1000).export(silent_path, format="wav")
            return silent_path
