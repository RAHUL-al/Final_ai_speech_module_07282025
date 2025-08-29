from scipy import signal
import noisereduce as nr
import numpy as np
from dotenv import load_dotenv
import soundfile as sf
import librosa
import os
import torchaudio
from transformers import AutoTokenizer
import torch.nn.functional as F
from datetime import datetime
import asyncio
from kokoro import KPipeline
from pydub import AudioSegment
import re
import torch
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from firebase import db
import asyncio
import torch
import torchaudio
import os
import logging
from transformers.utils import logging as hf_logging
from datetime import datetime
import soundfile as sf
from pydub import AudioSegment
import warnings
import torch
import io
from huggingface_hub import InferenceClient
from langchain.tools import Tool
import time
from fastapi.responses import JSONResponse
from starlette import status

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

            asyncio.create_task(self.text_to_speech_assistant(response, username))

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
        
            


    async def overall_scoring_by_speech_module(self, essay_id: str):
        try:
            essay_doc = db.collection("essays").document(essay_id).get()
            if not essay_doc.exists:
                return JSONResponse(
                    content={"error": f"No essay found with id {essay_id}"},
                    status_code=status.HTTP_404_NOT_FOUND
                )

            essay_data = essay_doc.to_dict()
            
            username = essay_data.get("username")
            if not username:
                return JSONResponse(
                    content={"error": "Username missing in essay data"},
                    status_code=status.HTTP_400_BAD_REQUEST
                )

            chunks = essay_data.get("chunks", [])
            if not chunks:
                return JSONResponse(
                    content={"error": "No audio chunks available for this essay. Please ensure audio recording was completed."},
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            
            data = essay_data.get("overall_scoring")
            if not data:
                return JSONResponse(
                    content={"error": "Overall scoring not found in essay data"},
                    status_code=status.HTTP_404_NOT_FOUND
                )

            return JSONResponse(
                content={"overall_scoring": data},
                status_code=status.HTTP_200_OK
            )
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

	
    async def speech_to_text(self, audio_path: str, device=None) -> str:
        try:
            return await asyncio.to_thread(
                self._speech_to_text, 
                audio_path, 
                device
            )
        except Exception as e:
            print(f"[ERROR] Failed in speech_to_text: {e}")
            return ""

    # def _speech_to_text(self, audio_path: str, device=None) -> str:
    #     token = "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"
    #     API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    #     headers = {
    #         "Authorization": f"Bearer {token}",
    #         "Content-Type": "audio/wav"
    #     }

    #     if not os.path.exists(audio_path):
    #         print(f"[Error] Audio file not found: {audio_path}")
    #         return ""

    #     try:
    #         with open(audio_path, "rb") as f:
    #             data = f.read()

    #         response = requests.post(API_URL, headers=headers, data=data)
    #         response.raise_for_status()

    #         result = response.json()
    #         text = result.get("text", "").strip()
    #         print(f"Transcribed [{os.path.basename(audio_path)}]: {text}")
    #         return text

    #     except Exception as e:
    #         print(f"[Error] Failed to transcribe {audio_path}: {e}")
    #         return ""

    def _speech_to_text(self, audio_path: str, device=None) -> str:
        client = InferenceClient(api_key="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")
        model_id = "openai/whisper-large-v3"

        def transcribe_audio(audio_path: str) -> str:
            resp = client.automatic_speech_recognition(audio_path, model=model_id)
            return getattr(resp, "text", resp.get("text") if isinstance(resp, dict) else str(resp))

        asr_tool = Tool(
            name="SpeechRecognizer",
            func=transcribe_audio,
            description="Transcribes an audio file to text",
            language="en"
        )
        return asr_tool.run(audio_path)


    def _calculate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
            
        word_count = len(text.split())
        base_conf = min(0.3 + (word_count * 0.05), 0.9)
        
        if any(patt in text.lower() for patt in ['...', '??', 'unintelligible']):
            base_conf *= 0.7
            
        return round(base_conf, 2)
    

    async def text_to_speech(self, text_data, username):
        return await asyncio.to_thread(self._text_to_speech_sync, text_data, username)
    
    def _text_to_speech_sync(self, text_data, username, session_temp_dir=None):
        output_dir = session_temp_dir if session_temp_dir else os.path.join("text_to_speech_audio_folder", username)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(datetime.now().timestamp())
        output_path = os.path.join(output_dir, f"tts_{username}_{timestamp}.wav")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline = KPipeline(lang_code='a')
            
            if hasattr(pipeline, 'model'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pipeline.model.to(device)
                pipeline.model.eval()
            
            combined = AudioSegment.empty()
            
            for _, _, audio in pipeline(text_data, voice='af_heart'):
                with io.BytesIO() as buffer:
                    sf.write(buffer, audio, 24000, format='WAV')
                    buffer.seek(0)
                    combined += AudioSegment.from_wav(buffer)
            
            combined.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            silent_path = os.path.join(output_dir, "silent_fallback.wav")
            AudioSegment.silent(duration=1000).export(silent_path, format="wav")
            return silent_path
    
    
    # async def text_to_speech_assistant(self, text_data, username, session_temp_dir=None):
    #     return await asyncio.to_thread(
    #         self._text_to_speech_sync_assistant, 
    #         text_data, 
    #         username,
    #         session_temp_dir
    #     )
    
    # def _text_to_speech_sync_assistant(self, text_data, username, session_temp_dir=None):
    #     output_dir = session_temp_dir if session_temp_dir else os.path.join("text_to_speech_audio_folder", username)
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     timestamp = int(datetime.now().timestamp())
    #     output_path = os.path.join(output_dir, f"tts_{username}_{timestamp}.wav")
        
    #     try:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             pipeline = KPipeline(lang_code='a')
            
    #         if hasattr(pipeline, 'model'):
    #             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #             pipeline.model.to(device)
    #             pipeline.model.eval()
            
    #         combined = AudioSegment.empty()
            
    #         for _, _, audio in pipeline(text_data, voice='af_heart'):
    #             with io.BytesIO() as buffer:
    #                 sf.write(buffer, audio, 24000, format='WAV')
    #                 buffer.seek(0)
    #                 combined += AudioSegment.from_wav(buffer)
            
    #         combined.export(output_path, format="wav")
    #         return output_path
            
    #     except Exception as e:
    #         logging.error(f"TTS generation failed: {e}")
    #         silent_path = os.path.join(output_dir, "silent_fallback.wav")
    #         AudioSegment.silent(duration=1000).export(silent_path, format="wav")
    #         return silent_path

    async def text_to_speech_assistant(self, text_data, username, session_temp_dir=None):
        return await asyncio.to_thread(self._text_to_speech_sync, text_data, username, session_temp_dir)

    def _text_to_speech_sync(self, text_data, username, session_temp_dir=None):
        output_dir = session_temp_dir if session_temp_dir else os.path.join("text_to_speech_audio_folder", username)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(datetime.now().timestamp())
        output_path = os.path.join(output_dir, f"tts_{username}_{timestamp}.wav")
        
        # Validate input text
        if not text_data or not isinstance(text_data, str) or text_data.strip() == "":
            logging.warning(f"Empty or invalid text data for TTS: {text_data}")
            text_data = "No text provided"
        
        # Truncate very long text to prevent memory issues
        if len(text_data) > 1000:
            logging.warning(f"Truncating long TTS text from {len(text_data)} to 1000 characters")
            text_data = text_data[:1000] + "..."  # Add ellipsis to indicate truncation
        
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Initialize pipeline with robust device handling
                    pipeline = self._initialize_tts_pipeline()
                    
                    if not pipeline:
                        raise Exception("Failed to initialize TTS pipeline")
                    
                    combined = AudioSegment.empty()
                    audio_generated = False
                    
                    # Process text in chunks if it's too long
                    text_chunks = self._split_text_for_tts(text_data)
                    
                    for chunk in text_chunks:
                        if not chunk.strip():
                            continue
                            
                        for _, _, audio in pipeline(chunk, voice='af_heart'):
                            if audio is not None and len(audio) > 0:
                                with io.BytesIO() as buffer:
                                    sf.write(buffer, audio, 24000, format='WAV')
                                    buffer.seek(0)
                                    chunk_audio = AudioSegment.from_wav(buffer)
                                    combined += chunk_audio
                                    audio_generated = True
                    
                    if not audio_generated:
                        raise Exception("No audio generated from TTS pipeline")
                    
                    # Ensure minimum audio duration
                    if len(combined) < 100:  # Less than 100ms
                        combined = AudioSegment.silent(duration=500)  # Create minimal silent audio
                    
                    combined.export(output_path, format="wav")
                    
                    # Verify the file was created successfully
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logging.info(f"TTS successfully generated: {output_path}")
                        return output_path
                    else:
                        raise Exception("Generated TTS file is empty or missing")
                    
            except Exception as e:
                logging.warning(f"TTS attempt {attempt + 1} failed: {e}")
                
                # Specific CUDA error handling
                if "CUDA" in str(e).upper() or "cuda" in str(e).lower():
                    logging.info("CUDA error detected, forcing CPU mode for next attempt")
                    self._clear_cuda_cache()
                    # Force CPU mode for next attempt
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
        
        # If all retries failed, return fallback audio
        return self._create_fallback_audio(output_dir, output_path)

    def _initialize_tts_pipeline(self):
        """Initialize TTS pipeline with robust error handling"""
        try:
            pipeline = KPipeline(lang_code='a')
            
            # Device handling with multiple fallback strategies
            device = None
            device_strategies = [
                self._try_cuda_device,
                self._try_cpu_device,
                self._try_meta_tensor_fix
            ]
            
            for strategy in device_strategies:
                try:
                    device = strategy(pipeline)
                    if device:
                        break
                except Exception as e:
                    logging.debug(f"Device strategy failed: {e}")
                    continue
            
            if not device:
                logging.warning("Could not determine device, proceeding without device assignment")
                return pipeline
            
            return pipeline
            
        except Exception as e:
            logging.error(f"Failed to initialize TTS pipeline: {e}")
            return None

    def _try_cuda_device(self, pipeline):
        """Try to use CUDA if available"""
        if not torch.cuda.is_available():
            return None
        
        try:
            device = torch.device('cuda')
            if hasattr(pipeline, 'model'):
                pipeline.model.to(device)
                pipeline.model.eval()
            logging.info("TTS using CUDA device")
            return device
        except Exception as e:
            logging.warning(f"CUDA device failed: {e}")
            return None

    def _try_cpu_device(self, pipeline):
        """Fallback to CPU"""
        try:
            device = torch.device('cpu')
            if hasattr(pipeline, 'model'):
                pipeline.model.to(device)
                pipeline.model.eval()
            logging.info("TTS using CPU device")
            return device
        except Exception as e:
            logging.warning(f"CPU device failed: {e}")
            return None

    def _try_meta_tensor_fix(self, pipeline):
        """Handle meta tensor issue specifically"""
        if not hasattr(pipeline, 'model'):
            return None
        
        try:
            # Alternative approach for meta tensor issue
            device = torch.device('cpu')
            pipeline.model = pipeline.model.to_empty(device=device)
            pipeline.model.eval()
            logging.info("TTS using meta tensor fix")
            return device
        except Exception as e:
            logging.warning(f"Meta tensor fix failed: {e}")
            return None

    def _split_text_for_tts(self, text, max_chunk_length=200):
        """Split text into manageable chunks for TTS"""
        if len(text) <= max_chunk_length:
            return [text]
        
        # Split by sentences if possible
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
        except Exception as e:
            logging.warning(f"Failed to clear CUDA cache: {e}")

    def _create_fallback_audio(self, output_dir, original_output_path):
        """Create fallback audio when TTS fails completely"""
        try:
            # Try to create a simple beep sound as fallback
            sample_rate = 24000
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            with io.BytesIO() as buffer:
                sf.write(buffer, audio_data, sample_rate, format='WAV')
                buffer.seek(0)
                audio = AudioSegment.from_wav(buffer)
                audio.export(original_output_path, format="wav")
            
            logging.info(f"Created fallback audio: {original_output_path}")
            return original_output_path
            
        except Exception as fallback_error:
            logging.error(f"Fallback audio creation also failed: {fallback_error}")
            
            # Ultimate fallback - silent audio
            silent_path = os.path.join(output_dir, "silent_fallback.wav")
            try:
                AudioSegment.silent(duration=1000).export(silent_path, format="wav")
                return silent_path
            except Exception:
                # If everything fails, return the path anyway
                return original_output_path

class AdvancedAudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.dynamic_noise_threshold = -45
        self.last_noise_level = None
        self.n_fft = 1024
        self.win_length = 512
        self.hop_length = 256

    def _update_noise_threshold(self, current_dBFS: float):
        if self.last_noise_level is None:
            self.last_noise_level = current_dBFS
        else:
            alpha = 0.2
            self.last_noise_level = alpha * current_dBFS + (1 - alpha) * self.last_noise_level
            
        if self.last_noise_level < -50:
            self.dynamic_noise_threshold = -45
        elif self.last_noise_level < -40:
            self.dynamic_noise_threshold = -40
        else:
            self.dynamic_noise_threshold = -35

    async def process_audio_chunk(self, audio_bytes: bytes) -> bytes:
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            current_dBFS = 10 * np.log10(np.mean(audio_float**2)) if len(audio_float) > 0 else -90
            self._update_noise_threshold(current_dBFS)
            
            processed_audio = await self._apply_full_processing(audio_float)
            processed_int16 = (processed_audio * 32767).astype(np.int16)
            return processed_int16.tobytes()
        except Exception as e:
            logging.error(f"Audio processing error: {str(e)}")
            return audio_bytes

    async def _apply_full_processing(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
            
        try:
            if len(audio) < self.n_fft:
                audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')
            
            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=0.8,
                n_fft=self.n_fft,
                win_length=self.n_fft // 2,
                hop_length=self.hop_length,
                thresh_n_mult_nonstationary=1.5
            )
            
            wiener_filtered = self._wiener_filter(reduced_noise)
            spectral_clean = self._spectral_subtraction(wiener_filtered)
            
            b, a = signal.butter(5, 100, btype='highpass', fs=self.sample_rate)
            filtered = signal.lfilter(b, a, spectral_clean)
            compressed = self._dynamic_range_compression(filtered)
            
            return compressed
        except Exception as e:
            logging.error(f"Audio processing failed: {str(e)}")
            return audio

    

    def _wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_profile is None and len(audio) > int(0.05 * self.sample_rate):
            noise_est = audio[:int(0.05 * self.sample_rate)]
            self.noise_profile = np.abs(np.fft.rfft(noise_est))
            
        if self.noise_profile is not None:
            audio_fft = np.fft.rfft(audio)
            audio_mag = np.abs(audio_fft)
            wiener_mag = (audio_mag**2 - self.noise_profile**2) / audio_mag
            wiener_mag = np.maximum(wiener_mag, 0)
            processed_fft = wiener_mag * np.exp(1j * np.angle(audio_fft))
            return np.fft.irfft(processed_fft)
        return audio
        
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        try:
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            if self.noise_profile is None:
                self.noise_profile = np.mean(np.abs(librosa.stft(
                    audio[:self.n_fft * 5],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )), axis=1)
            
            if self.noise_profile.shape[0] != magnitude.shape[0]:
                min_len = min(self.noise_profile.shape[0], magnitude.shape[0])
                magnitude = magnitude[:min_len]
                self.noise_profile = self.noise_profile[:min_len]
            
            magnitude_clean = magnitude - self.noise_profile[:, np.newaxis]
            magnitude_clean = np.maximum(magnitude_clean, 0.1 * self.noise_profile[:, np.newaxis])
            
            phase = np.angle(stft)
            clean_stft = magnitude_clean * np.exp(1j * phase)
            return librosa.istft(clean_stft, hop_length=self.hop_length)
        except Exception as e:
            logging.error(f"Spectral subtraction failed: {str(e)}")
            return audio
        
    def _dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        threshold = 0.1
        ratio = 2.0
        abs_audio = np.abs(audio)
        above_threshold = abs_audio - threshold
        above_threshold[above_threshold < 0] = 0
        reduction = above_threshold / ratio
        output = np.sign(audio) * (threshold + reduction)
        return output * 0.9
