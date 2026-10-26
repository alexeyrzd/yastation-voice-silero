"""Локальный TTS-сервер твоим клонированным голосом (Silero) с OpenAI-совместимым
эндпоинтом POST /audio/speech. Сделан специально, чтобы его дёргал quasard —
там для TTS уже есть готовый "openai"-движок (alice_server/tts.py, метод _openai):
он шлёт POST {base_url}/audio/speech с json {model, voice, input, response_format}
и ждёт в ответ аудио-файл. Этот сервер отвечает тем же самым, но голосом, полученным
через https://github.com/alexeyrzd/silero-clone-lite (clone_voice.py/extended_voice.py).

Требуется рядом клон silero-clone-lite — путь к нему передаётся через --repo
(по умолчанию ../silero-clone-lite).

Запуск:
    python silero_tts_server.py --repo ../silero-clone-lite --voice voices/myvoice_extended_dec.pt --port 8501

В config.toml quasard:
    [tts]
    engine = "openai"
    base_url = "http://127.0.0.1:8501"
    voice = "myvoice"      # сервер это поле игнорирует, голос фиксирован через --voice
    api_key = "any"         # сервер не проверяет ключ
"""
import argparse
import io
import os
import sys
import threading

import soundfile as sf
import torch
from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

app = FastAPI()
_lock = threading.Lock()
_state: dict = {}


cv = None  # заполняется в main() после sys.path.insert(--repo)


def load_voice(voice_path: str, model_path: str, device: str):
    torch.set_grad_enabled(False)
    S = cv.load_silero(model_path, device)
    params = dict(S.m.named_parameters())
    ck = torch.load(voice_path, map_location="cpu")
    E = ck["voice"] if isinstance(ck, dict) else ck
    if isinstance(ck, dict) and "offsets" in ck:
        for n, off in ck["offsets"].items():
            params[n].add_(off.to(device))
    _state["S"] = S
    _state["E"] = E.to(device)
    _state["device"] = device


@app.post("/audio/speech")
async def audio_speech(req: Request):
    body = await req.json()
    text = body.get("input", "")
    with _lock:
        audio = cv.render(_state["S"], cv.normalize_text(text), _state["E"])
    buf = io.BytesIO()
    peak = max(1e-6, float(abs(audio).max()))
    sf.write(buf, audio / peak * 0.9, 48000, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main():
    global cv
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../silero-clone-lite",
                     help="путь к клону https://github.com/alexeyrzd/silero-clone-lite")
    ap.add_argument("--voice", required=True, help="voices/<имя>.pt — из clone_voice.py или extended_voice.py")
    ap.add_argument("--model", default=None, help="по умолчанию <repo>/v5_ru.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

    repo_path = os.path.abspath(args.repo)
    sys.path.insert(0, repo_path)
    import clone_voice as cv_module
    cv = cv_module

    model_path = args.model or os.path.join(repo_path, "v5_ru.pt")
    print(f"Загружаю Silero ({model_path}) и голос {args.voice} на {args.device}...")
    load_voice(args.voice, model_path, args.device)
    print(f"Готово. Слушаю http://{args.host}:{args.port}/audio/speech")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
