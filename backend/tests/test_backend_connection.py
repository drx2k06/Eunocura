"""Regression test suite for frontend-backend connection, error resilience, and degraded mode."""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def test_root_endpoint():
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("service") == "EUNOCURA"
        assert data.get("version") == "2.0.0"


def test_healthz_degraded_contract():
    req = urllib.request.Request(f"{BASE_URL}/healthz")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "checks" in data
        assert "model" in data
        assert "mongo" in data
        assert "storage" in data
        assert "rag" in data
        # When mongo is offline, status is degraded, not ok
        if not data["checks"]["mongo"]:
            assert data["status"] == "degraded"


def test_auth_login_when_mongo_unreachable():
    payload = json.dumps({"email": "doctor@eunocura.local", "password": "Password123!"}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 503 or 401, not 200"
    except urllib.error.HTTPError as e:
        # Must return 503 (database unavailable) rather than 500 internal server error
        assert e.code == 503
        data = json.loads(e.read().decode("utf-8"))
        assert "detail" in data
        assert "Database unavailable" in data["detail"] or "MongoDB" in data["detail"]


def test_chatbot_text_when_model_unreachable():
    payload = json.dumps({
        "message": "What should I eat?",
        "target_language": "en-IN",
        "generate_audio": False
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chatbot/text",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 503 when model is offline"
    except urllib.error.HTTPError as e:
        assert e.code == 503
        data = json.loads(e.read().decode("utf-8"))
        assert "detail" in data
        assert "Model server unreachable" in data["detail"] or "ollama" in data["detail"].lower()


def test_chatbot_voice_when_speech_unreachable():
    boundary = "----TestBoundary"
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"audio_file\"; filename=\"test.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "dummy_wav_data\r\n"
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"source_language\"\r\n\r\n"
        "en-IN\r\n"
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"target_language\"\r\n\r\n"
        "en-IN\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/chatbot/voice",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 503 when speech is offline"
    except urllib.error.HTTPError as e:
        assert e.code == 503
        data = json.loads(e.read().decode("utf-8"))
        detail = data.get("detail", "")
        assert "faster-whisper" in detail or "transcription" in detail.lower() or "speech" in detail.lower()


def test_protected_upload_pdf_requires_auth():
    boundary = "----TestBoundary2"
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"test.pdf\"\r\n"
        "Content-Type: application/pdf\r\n\r\n"
        "%PDF-1.4...\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/doctors/upload-pdf",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 401 without token"
    except urllib.error.HTTPError as e:
        assert e.code == 401


if __name__ == "__main__":
    print("\nRunning frontend-backend connection regression checks...\n")
    test_root_endpoint()
    print("  ok    test_root_endpoint (HTTP 200)")
    test_healthz_degraded_contract()
    print("  ok    test_healthz_degraded_contract (HTTP 200 with honest status)")
    test_auth_login_when_mongo_unreachable()
    print("  ok    test_auth_login_when_mongo_unreachable (HTTP 503 instead of 500)")
    test_chatbot_text_when_model_unreachable()
    print("  ok    test_chatbot_text_when_model_unreachable (HTTP 503 instead of 500)")
    test_chatbot_voice_when_speech_unreachable()
    print("  ok    test_chatbot_voice_when_speech_unreachable (HTTP 503 instead of 500)")
    test_protected_upload_pdf_requires_auth()
    print("  ok    test_protected_upload_pdf_requires_auth (HTTP 401)")
    print("\n  6/6 connection regression checks passed!\n")
