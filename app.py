import os
import wave
import re
import io
import hashlib
import argparse
import subprocess
import numpy as np
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Default ElevenLabs Premade Voice IDs (acoustically matched to Gemini's Kore & Charon):
# Speaker 1 (Fah): Sarah (EXAVITQu4vr4xnSDxMaL) - F0 ~187.5Hz, Centroid ~1360Hz (closest match to Kore: F0 ~181.8Hz, Centroid ~1326Hz)
# Speaker 2 (Uncle Quin): Roger (CwhRBWXzGAHq8TQ4Fs17) - F0 ~100.8Hz, Centroid ~1183Hz (closest match to Charon: F0 ~94.1Hz, Centroid ~1109Hz)
DEFAULT_ELEVENLABS_VOICES = {
    "Speaker 1": os.getenv("ELEVENLABS_VOICE_SPEAKER_1", "EXAVITQu4vr4xnSDxMaL"),
    "Speaker 2": os.getenv("ELEVENLABS_VOICE_SPEAKER_2", "CwhRBWXzGAHq8TQ4Fs17"),
}

ELEVENLABS_MODEL_ALIASES = {
    "v4": "eleven_v4",
    "v4_turbo": "eleven_v4_turbo",
    "v3": "eleven_v3",
    "v2": "eleven_multilingual_v2",
    "flash": "eleven_flash_v2_5",
    "turbo": "eleven_turbo_v2_5",
}

def parse_transcript(filename="input.txt"):
    """
    Parses style context and speaker dialogue from the transcript file.
    Supports multi-line speaker paragraphs.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Error: '{filename}' not found.")
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    style_match = re.search(r"Style:\s*(.*)", content)
    context_match = re.search(r"Sample Context:\s*([\s\S]*?)(?=##|\Z)", content)
    
    style_context = ""
    if style_match:
        style_context += f"Style: {style_match.group(1).strip()}. "
    if context_match:
        style_context += f"Context: {context_match.group(1).strip()}"

    speaker_lines = []
    current_speaker = None
    current_text = []

    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match Speaker lines like "Speaker 1:"
        match = re.match(r"^(Speaker \d+):\s*(.*)", line)
        if match:
            if current_speaker:
                speaker_lines.append((current_speaker, " ".join(current_text)))
            current_speaker = match.group(1)
            current_text = [match.group(2).strip()]
        else:
            # Multi-line text continuation (skip headers/metadata)
            if (current_speaker and 
                not line.startswith("#") and 
                not line.startswith("Style:") and 
                not line.startswith("Sample Context:")):
                current_text.append(line)
                
    if current_speaker:
        speaker_lines.append((current_speaker, " ".join(current_text)))
        
    return style_context, speaker_lines

def translate_emotional_cues(text: str) -> str:
    """
    Translates common Thai emotional/conversational expressions in the dialogue
    into standard English TTS tag cues (e.g. [laughs], [sighs]).
    """
    processed = text
    
    # 1. Bracketed or parenthesized cues
    cues_map = {
        "หัวเราะ": "laughs",
        "ถอนหายใจ": "sighs",
        "หืม": "puzzled",
        "อ้าว": "surprised",
        "อึ้ก": "gasp",
        "โห": "wowed",
        "ยิ้ม": "smiling",
        "ทำหน้าประหลาดใจ": "surprised",
        "ประหลาดใจ": "surprised"
    }
    for th, en in cues_map.items():
        processed = re.sub(rf"[(\[][\s]*{th}[\s]*[)\]]", f"[{en}]", processed)
        
    # 2. Spoken leading/trailing emotional expressions
    processed = re.sub(r"(^|[\s,.:;?])อึ้ก([.\s]*)", r"\1[gasp]\2", processed)
    processed = re.sub(r"(^|[\s,.:;?])โห([.\s]*)", r"\1[wowed]\2", processed)
    processed = re.sub(r"(^|[\s,.:;?])อ้าว([.\s]*)", r"\1[surprised] \2", processed)
    processed = re.sub(r"(^|[\s,.:;?])หืม(\?|[.\s]*)", r"\1[puzzled] หืม\2", processed)
    
    return processed

def generate_silence(duration_seconds=0.6, rate=24000, sample_width=2, channels=1):
    """
    Generates a byte string of silent PCM audio.
    """
    return b'\x00' * int(rate * sample_width * channels * duration_seconds)

def extract_clean_pcm(raw_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """
    Extracts pure 16-bit mono PCM audio from Gemini TTS output.
    Strips RIFF WAVE headers and trailing metadata chunks (such as C2PA/SynthID
    in gemini-3.8-flash-tts) that otherwise play as a burst of static/hiss at
    the end of each speaker turn. Also applies a 15ms fade-in/fade-out.
    """
    if raw_bytes.startswith(b"RIFF"):
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            pcm_bytes = wf.readframes(wf.getnframes())
    else:
        pcm_bytes = raw_bytes

    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes

    fade_len = min(int(0.015 * sample_rate), len(samples) // 4)
    if fade_len > 0:
        samples[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        samples[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

def decode_mp3_to_pcm(mp3_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """
    Decodes MP3 bytes to 16-bit mono linear PCM at `sample_rate` Hz using bundled ffmpeg.
    Used as an automatic fallback if an ElevenLabs account tier restricts raw pcm_24000 output.
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=True,
    )
    return proc.stdout

def list_elevenlabs_voices(api_key: str):
    """Fetches and prints available voices from the ElevenLabs account (or public catalog if key is scoped)."""
    resp = requests.get(
        f"{ELEVENLABS_BASE_URL}/voices",
        headers={"xi-api-key": api_key},
        timeout=30,
    )
    if resp.status_code == 401:
        resp = requests.get(f"{ELEVENLABS_BASE_URL}/voices", timeout=30)
    resp.raise_for_status()
    voices = resp.json().get("voices", [])
    print(f"🎤 Found {len(voices)} ElevenLabs voices:")
    for v in voices:
        labels = v.get("labels", {})
        label_str = ", ".join(f"{k}={val}" for k, val in labels.items())
        print(f"  - {v.get('name'):<22} | ID: {v.get('voice_id')} | Category: {v.get('category', 'N/A'):<10} | {label_str}")

def list_elevenlabs_models(api_key: str):
    """Fetches and prints available TTS models from the ElevenLabs account."""
    resp = requests.get(
        f"{ELEVENLABS_BASE_URL}/models",
        headers={"xi-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    models = resp.json()
    print(f"🤖 Found {len(models)} ElevenLabs models:")
    for m in models:
        if m.get("can_do_text_to_speech"):
            print(f"  - {m.get('model_id'):<28} | {m.get('name')} ({m.get('description', '')[:70]}...)")

def generate_elevenlabs_audio(
    api_key: str,
    voice_id: str,
    text: str,
    model_id: str = "eleven_v4",
    previous_text: str | None = None,
    next_text: str | None = None,
    sample_rate: int = 24000,
) -> tuple[bytes, str]:
    """
    Generates 24kHz 16-bit mono PCM audio from ElevenLabs TTS API.
    Returns (clean_pcm_bytes, effective_model_id).
    Automatically falls back from pcm_24000 to mp3_44100_128 if raw PCM is restricted on the account tier,
    and falls back from eleven_v4 to eleven_v3 if eleven_v4 is not yet enabled on the account.
    """
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.25,
            "use_speaker_boost": True,
        },
    }
    # Models like eleven_v3 do not support previous_text / next_text in some versions; include only when supported
    if previous_text and not model_id.startswith("eleven_v3"):
        payload["previous_text"] = previous_text[-500:]
    if next_text and not model_id.startswith("eleven_v3"):
        payload["next_text"] = next_text[:500]

    output_format = f"pcm_{sample_rate}"
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}?output_format={output_format}"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # Handle model fallback if eleven_v4 is not available on this account yet
    if resp.status_code in (400, 404, 422) and model_id == "eleven_v4":
        err_text = resp.text.lower()
        if "model" in err_text:
            print("   ℹ️ 'eleven_v4' is not available on this account/endpoint yet; falling back to 'eleven_v3'...")
            model_id = "eleven_v3"
            payload["model_id"] = model_id
            payload.pop("previous_text", None)
            payload.pop("next_text", None)
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # Handle previous_text/next_text rejection if a model rejects context params
    if resp.status_code in (400, 422) and ("previous_text" in resp.text or "next_text" in resp.text):
        payload.pop("previous_text", None)
        payload.pop("next_text", None)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # Handle output_format tier restriction (e.g. Free/Starter tier disallowing pcm_24000)
    if resp.status_code == 403 and ("output_format" in resp.text.lower() or "pcm" in resp.text.lower()):
        print("   ℹ️ Raw PCM output restricted by account tier; using MP3 and converting to 24kHz PCM...")
        fallback_url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}?output_format=mp3_44100_128"
        resp = requests.post(fallback_url, headers=headers, json=payload, timeout=120)
        if not resp.ok:
            raise RuntimeError(f"ElevenLabs API Error ({resp.status_code}): {resp.text}")
        raw_pcm = decode_mp3_to_pcm(resp.content, sample_rate=sample_rate)
        return extract_clean_pcm(raw_pcm, sample_rate=sample_rate), model_id

    if not resp.ok:
        raise RuntimeError(f"ElevenLabs API Error ({resp.status_code}): {resp.text}")

    return extract_clean_pcm(resp.content, sample_rate=sample_rate), model_id

def is_likely_repeated(pcm_bytes: bytes, text: str, sample_rate: int = 24000, min_cps: float = 11.2) -> tuple[bool, float, float]:
    """
    Detects whether a generated TTS turn likely repeated a sentence by checking
    the speaking rate (characters per second). Normal Thai podcast speech on
    Gemini TTS is ~12.0-15.5 chars/s; repeated sentences drop below ~11.2 chars/s.
    """
    duration = (len(pcm_bytes) / 2) / sample_rate
    if duration <= 0 or len(text) < 80:
        return False, duration, (len(text) / duration if duration > 0 else 0.0)
    cps = len(text) / duration
    return cps < min_cps, duration, cps

def main():
    parser = argparse.ArgumentParser(description="🎙️ Thai 2-Person Podcast Audio Generator (ElevenLabs & Gemini TTS)")
    parser.add_argument("--input", "-i", default="input.txt", help="Path to input transcript file (default: input.txt)")
    parser.add_argument("--output", "-o", default="output.wav", help="Path to output WAV file (default: output.wav)")
    parser.add_argument(
        "--provider", "-p",
        choices=["elevenlabs", "gemini"],
        default=os.getenv("TTS_PROVIDER", "elevenlabs"),
        help="TTS provider to use: 'elevenlabs' or 'gemini' (default: elevenlabs)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model ID (default: 'eleven_v4' for ElevenLabs, 'gemini-3.8-flash-tts' for Gemini). Shorthands: v4, v4_turbo, v3, v2, flash",
    )
    parser.add_argument("--voice1", default=None, help="Override voice ID/name for Speaker 1 (Fah)")
    parser.add_argument("--voice2", default=None, help="Override voice ID/name for Speaker 2 (Uncle Quin)")
    parser.add_argument("--list-voices", action="store_true", help="List available ElevenLabs voices and exit")
    parser.add_argument("--list-models", action="store_true", help="List available ElevenLabs TTS models and exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache and force regeneration")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache directory and exit")
    parser.add_argument("--file-style", action="store_true", help="Use style from input.txt instead of conversational style (Gemini)")
    args = parser.parse_args()

    # Clear cache action
    if args.clear_cache:
        import shutil
        if os.path.exists(".cache"):
            shutil.rmtree(".cache")
            print("🧹 Cache cleared successfully.")
        else:
            print("🧹 Cache is already empty.")
        return

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    if args.list_voices:
        if not elevenlabs_api_key:
            print("❌ Error: ELEVENLABS_API_KEY is not set in .env or environment.")
            return
        list_elevenlabs_voices(elevenlabs_api_key)
        return

    if args.list_models:
        if not elevenlabs_api_key:
            print("❌ Error: ELEVENLABS_API_KEY is not set in .env or environment.")
            return
        list_elevenlabs_models(elevenlabs_api_key)
        return

    provider = args.provider.lower()
    if provider == "elevenlabs":
        raw_model = args.model or os.getenv("ELEVENLABS_MODEL", "eleven_v4")
        model_name = ELEVENLABS_MODEL_ALIASES.get(raw_model.lower(), raw_model)
        if not elevenlabs_api_key:
            print("❌ Error: ELEVENLABS_API_KEY is not set in .env or environment.")
            return
    else:
        model_name = args.model or os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")

    try:
        style_context, dialogue = parse_transcript(args.input)
        if not dialogue:
            print("❌ No dialogue found.")
            return

        print(f"📖 Loaded '{args.input}' with {len(dialogue)} dialogue turns.")
        print(f"🏷️  Provider: {provider} | 🤖 Model: {model_name}")

        dialogue_segments = []

        if provider == "elevenlabs":
            speaker_voices = {
                "Speaker 1": args.voice1 or DEFAULT_ELEVENLABS_VOICES["Speaker 1"],
                "Speaker 2": args.voice2 or DEFAULT_ELEVENLABS_VOICES["Speaker 2"],
            }
            print(f"ℹ️ ElevenLabs Voices -> Speaker 1 (Fah): {speaker_voices['Speaker 1']} | Speaker 2 (Uncle Quin): {speaker_voices['Speaker 2']}")

            cleaned_turns = [translate_emotional_cues(t) for _, t in dialogue]

            for idx, (speaker, text) in enumerate(dialogue):
                clean_text = cleaned_turns[idx]
                voice_id = speaker_voices.get(
                    speaker,
                    speaker_voices["Speaker 1"] if "1" in speaker else speaker_voices["Speaker 2"],
                )

                print(f"🎙️ [{idx + 1}/{len(dialogue)}] Speaker: {speaker} | Voice ID: {voice_id}")
                print(f"   Original: {text[:60]}...")
                if text != clean_text:
                    print(f"   Enhanced: {clean_text[:60]}...")

                cache_payload = f"elevenlabs:{model_name}:{voice_id}:{clean_text}".encode("utf-8")
                cache_hash = hashlib.sha256(cache_payload).hexdigest()
                cache_file = os.path.join(".cache", f"{cache_hash}.raw")

                pcm_bytes = None
                if not args.no_cache and os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        cached_raw = f.read()
                    pcm_bytes = extract_clean_pcm(cached_raw)
                    dur = (len(pcm_bytes) / 2) / 24000
                    cps = len(clean_text) / dur if dur > 0 else 0.0
                    print(f"   ⚡ [Cache Hit] ({dur:.1f}s, {cps:.1f} chars/s)")

                if pcm_bytes is None:
                    prev_text = cleaned_turns[idx - 1] if idx > 0 else None
                    next_text = cleaned_turns[idx + 1] if idx < len(cleaned_turns) - 1 else None
                    print(f"   🌐 [Cache Miss] Generating audio via ElevenLabs API ({model_name})...")
                    pcm_bytes, effective_model = generate_elevenlabs_audio(
                        api_key=elevenlabs_api_key,
                        voice_id=voice_id,
                        text=clean_text,
                        model_id=model_name,
                        previous_text=prev_text,
                        next_text=next_text,
                    )
                    if effective_model != model_name:
                        model_name = effective_model
                        # Update cache file path to match effective model
                        cache_payload = f"elevenlabs:{model_name}:{voice_id}:{clean_text}".encode("utf-8")
                        cache_hash = hashlib.sha256(cache_payload).hexdigest()
                        cache_file = os.path.join(".cache", f"{cache_hash}.raw")

                    dur = (len(pcm_bytes) / 2) / 24000
                    cps = len(clean_text) / dur if dur > 0 else 0.0
                    print(f"   ✅ Generated clean audio ({dur:.1f}s, {cps:.1f} chars/s)")

                    if not args.no_cache and pcm_bytes is not None:
                        os.makedirs(".cache", exist_ok=True)
                        with open(cache_file, "wb") as f:
                            f.write(pcm_bytes)

                dialogue_segments.append(pcm_bytes)

        else:
            client = genai.Client()

            # Define speaker personas/voices for conversational podcast
            speaker_configs = {
                "Speaker 1": {
                    "voice": args.voice1 or "Kore",
                    "style": "Style: Casual podcast conversation, warm, energetic, friendly young female co-host, natural Thai speech, natural conversational pacing, clear articulation, realistic pauses."
                },
                "Speaker 2": {
                    "voice": args.voice2 or "Charon",
                    "style": "Style: Casual podcast conversation, seasoned 40+ year old male host (mid-40s uncle figure), deep and resonant baritone voice, calm, wise, warm, composed demeanor, steady and measured pacing, natural Thai speech, clear articulation, realistic pauses."
                }
            }

            if args.file_style:
                print("ℹ️ Using overall style context from transcript file.")
            else:
                print(f"ℹ️ Using optimized conversational podcast styles (Fah -> {speaker_configs['Speaker 1']['voice']}, Uncle Quin -> {speaker_configs['Speaker 2']['voice']}).")

            for idx, (speaker, text) in enumerate(dialogue):
                # 1. Translate emotional expressions
                clean_text = translate_emotional_cues(text)

                # 2. Get speaker config
                config = speaker_configs.get(speaker, {
                    "voice": speaker_configs["Speaker 1"]["voice"] if "1" in speaker else speaker_configs["Speaker 2"]["voice"],
                    "style": speaker_configs["Speaker 1"]["style"] if "1" in speaker else speaker_configs["Speaker 2"]["style"]
                })

                voice_name = config["voice"]
                style_brief = style_context if (args.file_style and style_context) else config["style"]

                # Combine brief and text with explicit anti-repetition directive
                full_prompt = (
                    f"[{style_brief}]. Read the following text naturally and verbatim "
                    f"exactly once from start to finish, without repeating any words or phrases: {clean_text}"
                )

                print(f"🎙️ [{idx + 1}/{len(dialogue)}] Speaker: {speaker} | Voice: {voice_name}")
                print(f"   Original: {text[:60]}...")
                if text != clean_text:
                    print(f"   Enhanced: {clean_text[:60]}...")

                # 3. Check / fetch cache (cache key depends on model, voice, style, and cleaned text)
                cache_payload = f"{model_name}:{voice_name}:{style_brief}:{clean_text}".encode("utf-8")
                cache_hash = hashlib.sha256(cache_payload).hexdigest()
                cache_file = os.path.join(".cache", f"{cache_hash}.raw")

                pcm_bytes = None
                if not args.no_cache and os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        cached_raw = f.read()
                    candidate_pcm = extract_clean_pcm(cached_raw)
                    repeated, dur, cps = is_likely_repeated(candidate_pcm, clean_text)
                    if repeated:
                        print(f"   ⚠️ [Cache Invalidated] Detected likely sentence repetition in cache ({dur:.1f}s, {cps:.1f} chars/s). Regenerating...")
                    else:
                        print(f"   ⚡ [Cache Hit] ({dur:.1f}s, {cps:.1f} chars/s)")
                        pcm_bytes = candidate_pcm

                if pcm_bytes is None:
                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        print(f"   🌐 [Cache Miss] Generating audio via Gemini API (attempt {attempt}/{max_attempts})...")
                        response = client.models.generate_content(
                            model=model_name,
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                                    )
                                ),
                            )
                        )

                        if not response.candidates or not response.candidates[0].content.parts:
                            raise ValueError(f"Failed to generate audio for line {idx + 1}")

                        part = response.candidates[0].content.parts[0]
                        pcm_bytes = extract_clean_pcm(part.inline_data.data)
                        repeated, dur, cps = is_likely_repeated(pcm_bytes, clean_text)
                        if repeated and attempt < max_attempts:
                            print(f"   ⚠️ Detected likely repeated phrase ({dur:.1f}s, {cps:.1f} chars/s). Retrying...")
                            continue
                        print(f"   ✅ Generated clean audio ({dur:.1f}s, {cps:.1f} chars/s)")
                        break

                    # Save clean PCM to cache
                    if not args.no_cache and pcm_bytes is not None:
                        os.makedirs(".cache", exist_ok=True)
                        with open(cache_file, "wb") as f:
                            f.write(pcm_bytes)

                dialogue_segments.append(pcm_bytes)

        if not dialogue_segments:
            print("❌ No audio segments were successfully generated.")
            return

        # Fixed parameters for Gemini TTS audio (audio/l16; rate=24000; channels=1)
        sample_rate = 24000
        sample_width = 2  # 16-bit linear PCM is 2 bytes per sample
        channels = 1      # Mono

        print(f"\n🎧 Stitching audio segments (Rate: {sample_rate}Hz, Channels: {channels})...")
        
        final_pcm_stream = bytearray()
        for idx, pcm_data in enumerate(dialogue_segments):
            final_pcm_stream.extend(pcm_data)
            
            # Insert conversational silence between speakers (0.6 seconds)
            if idx < len(dialogue_segments) - 1:
                silence = generate_silence(
                    duration_seconds=0.6,
                    rate=sample_rate,
                    sample_width=sample_width,
                    channels=channels
                )
                final_pcm_stream.extend(silence)

        # Write final WAV file
        with wave.open(args.output, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(final_pcm_stream))

        print(f"🎉 Success! Podcast output saved to '{args.output}'")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()