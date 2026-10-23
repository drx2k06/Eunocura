"""
Multilingual patient chatbot.
- Speech-to-Text with translation: local faster-whisper (no cloud)
- Multilingual reply generation: local open-weight SLM
- Text-to-Speech: local pyttsx3 system voices
"""

import httpx
import base64
import asyncio
import json
import ssl
import certifi
import websockets
import io
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from pydub import AudioSegment

from src.llm.gemma import chat_model

# `slm` profile: multilingual patient-facing replies.
llm = chat_model("slm")

# Supported languages mapping
SUPPORTED_LANGUAGES = {
    "hindi": {"code": "hi-IN", "name": "Hindi"},
    "tamil": {"code": "ta-IN", "name": "Tamil"},
    "telugu": {"code": "te-IN", "name": "Telugu"},
    "bengali": {"code": "bn-IN", "name": "Bengali"},
    "gujarati": {"code": "gu-IN", "name": "Gujarati"},
    "kannada": {"code": "kn-IN", "name": "Kannada"},
    "malayalam": {"code": "ml-IN", "name": "Malayalam"},
    "marathi": {"code": "mr-IN", "name": "Marathi"},
    "odia": {"code": "or-IN", "name": "Odia"},
    "punjabi": {"code": "pa-IN", "name": "Punjabi"},
    "english": {"code": "en-IN", "name": "English"}
}



async def convert_audio_to_wav(audio_data: bytes) -> bytes:
    """
    Convert audio to WAV format with 16000 sample rate.

    Args:
        audio_data: Raw audio bytes (any format)

    Returns:
        WAV audio bytes with 16000 sample rate
    """
    try:
        # Load audio from bytes (pydub auto-detects format)
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Convert to mono, 16000 Hz, 16-bit PCM WAV
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz sample rate
        audio = audio.set_sample_width(2)  # 16-bit

        # Export to WAV format
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()

        print(f"✅ Converted audio: {len(audio_data)} bytes → {len(wav_bytes)} bytes (16kHz WAV)")
        return wav_bytes

    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        # If conversion fails, return original data
        return audio_data


async def speech_to_text_translate(audio_data: bytes, source_language: str = "hi-IN") -> Dict[str, Any]:
    """
    Transcribe speech and translate it to English - locally.

    faster-whisper runs on this machine; no audio leaves the device and no key is
    needed. Same return shape the previous cloud implementation used.
    """
    from src.services.local_speech import atranscribe_bytes

    return await atranscribe_bytes(audio_data, language=source_language)


async def generate_multilingual_response(
    user_message: str,
    target_language: str = "hi-IN",
    conversation_history: Optional[list] = None
) -> str:
    """
    Generate a response in the target language using Gemini.

    Args:
        user_message: User's message in English (already translated)
        target_language: Target language code (e.g., "hi-IN")
        conversation_history: Previous conversation messages

    Returns:
        Response text in target language
    """
    try:
        # Validate user message is not empty
        if not user_message or user_message.strip() == "":
            raise ValueError("User message cannot be empty")

        # Sanitize target language
        target_language = target_language.strip()
        print(f"🔍 DEBUG generate_multilingual_response called with target_language: '{target_language}'")

        # Get language name using direct lookup
        lang_name = None
        for lang_info in SUPPORTED_LANGUAGES.values():
            if lang_info["code"] == target_language:
                lang_name = lang_info["name"]
                break

        if not lang_name:
            lang_name = "Hindi"  # Default fallback
            print(f"⚠️ WARNING: Language code '{target_language}' not found in supported languages, defaulting to Hindi")
            print(f"ℹ️ Supported codes: {[l['code'] for l in SUPPORTED_LANGUAGES.values()]}")
        else:
            print(f"✅ DEBUG: Found language name: {lang_name} for code: {target_language}")

        # Build system prompt with language instruction
        system_prompt = f"""You are a helpful health assistant chatbot for patients.

🚨 CRITICAL LANGUAGE INSTRUCTION 🚨
You MUST respond COMPLETELY and ONLY in {lang_name} language.
- Do NOT use English at all, except for medical terms without common translations
- Do NOT write any part of your response in English
- Every single word must be in {lang_name} language, NOT in any other Indian language
- If {lang_name} uses Devanagari script (like Hindi, Marathi), use it correctly for that specific language
- {lang_name} is NOT the same as Hindi - ensure vocabulary and grammar are correct for {lang_name}
- If you don't understand the question, respond in {lang_name} asking for clarification
- NEVER mix languages in your response

Guidelines:
1. Be empathetic, friendly, and professional
2. Provide helpful information about medications, diet, and general health queries
3. Keep responses concise and easy to understand (2-4 sentences maximum)
4. If asked about serious medical conditions, advise consulting a doctor (in {lang_name})
5. For emergencies, advise calling emergency services immediately (in {lang_name})
6. ALWAYS provide a complete, meaningful answer - never give generic greetings unless appropriate

Language Target: {lang_name} ONLY (NOT Hindi, NOT English, NOT any other language)
Output Format: Pure {lang_name} text only

Remember:
- Your ENTIRE response MUST be in {lang_name} language
- Do not respond in English, Hindi, or any other language
- The user specifically requested {lang_name}, so use the correct vocabulary, grammar, and expressions for {lang_name}"""

        # Build messages
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history if provided (last 5 messages)
        if conversation_history:
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for msg in recent_history:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=content))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        # Generate response using Gemini
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Validate response is not empty
        if not response_text or response_text.strip() == "":
            raise ValueError("Gemini returned empty response")

        # Log for debugging
        print(f"🤖 Gemini response (first 100 chars): {response_text[:100]}")

        return response_text

    except Exception as e:
        print(f"❌ Error in generate_multilingual_response: {e}")
        raise


async def text_to_speech(
    text: str,
    target_language: str = "en-IN",
    speaker: Optional[str] = None,
    model: Optional[str] = None
) -> bytes:
    """
    Synthesize speech with the local system voices (pyttsx3).

    `speaker` and `model` are accepted and ignored - they named cloud voices. Signature
    kept so existing request payloads keep working.
    """
    from src.services.local_speech import asynthesize

    return await asynthesize(text, language=target_language)


async def process_multilingual_message(
    audio_data: Optional[bytes] = None,
    text_input: Optional[str] = None,
    source_language: str = "hi-IN",
    target_language: str = "hi-IN",
    conversation_history: Optional[list] = None,
    generate_audio: bool = True
) -> Dict[str, Any]:
    """
    Process a multilingual chatbot message (audio or text input).

    Args:
        audio_data: Audio bytes if using voice input
        text_input: Text string if using text input
        source_language: Source language for STT
        target_language: Target language for response
        conversation_history: Previous conversation messages
        generate_audio: Whether to generate audio response

    Returns:
        Dict with response_text, response_audio (base64), and translated_input
    """
    try:
        # Sanitize language inputs
        target_language = target_language.strip()
        source_language = source_language.strip()

        # Step 1: Get English text from input
        if audio_data:
            # Speech to English translation
            stt_result = await speech_to_text_translate(audio_data, source_language)
            english_text = stt_result["translated_text"]
            print(f"🎤 Speech translated to English: {english_text}")
        elif text_input:
            # If text input is not in English, we assume it's already in target language
            # For now, we'll use it directly (you could add text translation here)
            english_text = text_input
            print(f"📝 Text input: {english_text}")
        else:
            raise ValueError("Either audio_data or text_input must be provided")

        # Validate we have text to process
        english_text = english_text.strip()
        if not english_text:
            raise ValueError("Could not extract text from input. Please try again with a clearer message.")

        print(f"✅ Sending to Gemini: '{english_text}' (target language: {target_language})")

        # Step 2: Generate response in target language using Gemini
        response_text = await generate_multilingual_response(
            user_message=english_text,
            target_language=target_language,
            conversation_history=conversation_history
        )
        print(f"💬 Generated response in {target_language}: {response_text[:100]}...")

        # Step 3: Convert response to speech (if requested)
        response_audio_base64 = None
        if generate_audio:
            response_audio = await text_to_speech(
                text=response_text,
                target_language=target_language,
            )
            response_audio_base64 = base64.b64encode(response_audio).decode('utf-8')
            print(f"🔊 Generated audio response")

        return {
            "response_text": response_text,
            "response_audio": response_audio_base64,
            "translated_input": english_text if audio_data else None
        }

    except Exception as e:
        print(f"❌ Error in process_multilingual_message: {e}")
        raise
