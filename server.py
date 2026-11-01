"""
Guitar AI Studio Web Server
Lightweight, robust HTTP server with partial content streaming, REST API for chord harmony,
Demucs stem separation (karaoke / play-along), and on-demand guitar tablature extraction.
"""

import os
import sys
import json
import time
import shutil
import mimetypes
from pathlib import Path
from typing import Optional
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

# Add root directory to sys.path so our modules are importable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from chord_detector import ChordDetector
from stem_extractor import slice_audio, check_stems_exist

WEB_DIR = ROOT_DIR / "web"
UPLOADS_DIR = ROOT_DIR / "uploads"
CACHE_DIR = ROOT_DIR / ".cache"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy instances
chord_detector = ChordDetector()
_tab_extractor = None


def get_tab_extractor():
    """Lazy loader so server starts up instantly without waiting for ONNX/TF."""
    global _tab_extractor
    if _tab_extractor is None:
        from tab_extractor import GuitarTabExtractor
        _tab_extractor = GuitarTabExtractor()
    return _tab_extractor


def find_audio_file(filename: str) -> Optional[Path]:
    """Finds an audio file in project root, uploads, or cache directory."""
    if not filename:
        return None
    decoded = unquote(filename).strip()

    # Check if referencing a cache subpath (e.g. cache/song_hash/guitar.wav)
    if decoded.startswith("cache/"):
        rel_path = decoded[len("cache/") :]
        target = CACHE_DIR / rel_path
        if target.exists() and target.is_file():
            return target

    clean_name = os.path.basename(decoded)
    candidates = [
        UPLOADS_DIR / clean_name,
        ROOT_DIR / clean_name,
        CACHE_DIR / clean_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


class StudioRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")
        # Prevent browser caching of CSS, JS and API responses
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. API: List available tracks
        if path == "/api/tracks":
            self._handle_list_tracks()
            return

        # 2. API: Get chords for a track
        if path == "/api/chords":
            self._handle_get_chords(query)
            return

        # 3. API: Check stems status
        if path == "/api/stems-status":
            self._handle_stems_status(query)
            return

        # 4. Audio streaming with HTTP Range support
        if path.startswith("/audio/"):
            raw_filename = path[len("/audio/") :]
            filename = unquote(raw_filename)
            self._handle_stream_audio(filename)
            return

        # Default: Serve static files from web/
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. API: Upload audio
        if path == "/api/upload":
            self._handle_upload()
            return

        # 2. API: Separate stems (Demucs)
        if path == "/api/separate-stems":
            self._handle_separate_stems()
            return

        # 3. API: Generate tabs for snippet
        if path == "/api/tabs":
            self._handle_generate_tabs()
            return

        self._send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, data: dict, status: int = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_list_tracks(self):
        tracks = []
        # Check standard demo tracks
        for name, label in [
            ("sample_song.wav", "Demo: Acoustic Chords (C - G - Am - F)"),
            ("sample_riff.wav", "Demo: Electric Guitar Riff (E - G - A - D)"),
        ]:
            p = ROOT_DIR / name
            if p.exists():
                tracks.append({"filename": name, "label": label, "size": p.stat().st_size})

        # Check uploads
        for f in UPLOADS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
                tracks.append({"filename": f.name, "label": f.name, "size": f.stat().st_size})

        self._send_json({"tracks": tracks})

    def _handle_get_chords(self, query):
        filename = unquote(query.get("file", [""])[0])
        if not filename:
            self._send_json({"error": "Missing 'file' parameter"}, status=HTTPStatus.BAD_REQUEST)
            return

        audio_path = find_audio_file(filename)
        if not audio_path:
            self._send_json({"error": f"Audio file '{filename}' not found"}, status=HTTPStatus.NOT_FOUND)
            return

        # Check cached chords
        cache_path = CACHE_DIR / f"{audio_path.stem}_chords.json"
        if cache_path.exists() and cache_path.stat().st_mtime >= audio_path.stat().st_mtime:
            try:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                self._send_json(cached_data)
                return
            except Exception:
                pass

        # Detect chords
        t0 = time.time()
        timeline = chord_detector.detect_from_file(audio_path)
        print(f"[Chord] Analyzed {audio_path.name} in {time.time()-t0:.2f}s ({len(timeline.segments)} segments)")

        chord_data = {
            "file": filename,
            "duration": round(float(timeline.duration), 3),
            "segments": timeline.to_list(),
        }

        # Cache result
        try:
            cache_path.write_text(json.dumps(chord_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Warning] Failed to write cache: {e}")

        self._send_json(chord_data)

    def _handle_stems_status(self, query):
        filename = unquote(query.get("file", [""])[0])
        if not filename:
            self._send_json({"error": "Missing 'file' parameter"}, status=HTTPStatus.BAD_REQUEST)
            return

        audio_path = find_audio_file(filename)
        if not audio_path:
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return

        stems = check_stems_exist(audio_path)
        if stems:
            cache_name = stems["cache_dir_name"]
            self._send_json({
                "separated": True,
                "guitar_url": f"/audio/cache/{cache_name}/guitar.wav",
                "no_guitar_url": f"/audio/cache/{cache_name}/no_guitar.wav",
            })
        else:
            self._send_json({"separated": False})

    def _handle_separate_stems(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            filename = unquote(body.get("file", ""))

            audio_path = find_audio_file(filename)
            if not audio_path:
                self._send_json({"error": f"Audio file '{filename}' not found"}, status=HTTPStatus.NOT_FOUND)
                return

            print(f"[Demucs] Starting stem separation for {audio_path.name}...")
            t0 = time.time()
            from stem_extractor import GuitarStemExtractor
            extractor = GuitarStemExtractor()
            stems = extractor.separate_full_stems(audio_path)
            elapsed = time.time() - t0

            cache_name = stems["cache_dir_name"]
            self._send_json({
                "separated": True,
                "guitar_url": f"/audio/cache/{cache_name}/guitar.wav",
                "no_guitar_url": f"/audio/cache/{cache_name}/no_guitar.wav",
                "elapsed_time": round(elapsed, 2),
                "message": f"Стемы успешно созданы за {elapsed:.1f} сек!",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": f"Ошибка Demucs: {str(e)}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_stream_audio(self, filename: str):
        audio_path = find_audio_file(filename)
        if not audio_path:
            print(f"[404 Audio] File not found: {filename}")
            self.send_error(HTTPStatus.NOT_FOUND, f"File '{filename}' not found")
            return

        file_size = audio_path.stat().st_size
        mime_type, _ = mimetypes.guess_type(str(audio_path))
        if not mime_type:
            if audio_path.suffix.lower() == ".mp3":
                mime_type = "audio/mpeg"
            else:
                mime_type = "audio/wav"

        range_header = self.headers.get("Range")
        if range_header:
            # Handle byte range requests (Range: bytes=start-end)
            try:
                range_match = range_header.strip().replace("bytes=", "").split("-")
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", mime_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()

                with open(audio_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return
            except Exception as e:
                print(f"[Range Error] {e}")

        # Full file transfer
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(audio_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_len = int(self.headers.get("Content-Length", 0))

        if content_len == 0:
            self._send_json({"error": "Empty upload"}, status=HTTPStatus.BAD_REQUEST)
            return

        raw_data = self.rfile.read(content_len)

        # Detect filename or fallback to timestamped name
        upload_name = f"upload_{int(time.time())}.mp3"
        if "boundary=" in content_type:
            boundary_str = content_type.split("boundary=")[-1].strip().strip('"').strip("'")
            boundary = boundary_str.encode()
            parts = raw_data.split(b"--" + boundary)
            for part in parts:
                if b'filename="' in part:
                    fn_start = part.find(b'filename="') + 10
                    fn_end = part.find(b'"', fn_start)
                    extracted_name = part[fn_start:fn_end].decode("utf-8", errors="ignore")
                    if extracted_name:
                        upload_name = os.path.basename(extracted_name)

                    # Extract file payload between double CRLF and end
                    body_start = part.find(b"\r\n\r\n")
                    if body_start != -1:
                        body_start += 4
                        # Check for trailing \r\n before delimiter
                        body_end = len(part)
                        if part.endswith(b"\r\n"):
                            body_end -= 2
                        elif part.endswith(b"\r\n--"):
                            body_end -= 4
                        file_bytes = part[body_start:body_end]
                        target_path = UPLOADS_DIR / upload_name
                        target_path.write_bytes(file_bytes)
                        break
        else:
            target_path = UPLOADS_DIR / upload_name
            target_path.write_bytes(raw_data)

        print(f"[Upload] Saved uploaded file: {upload_name} ({target_path.stat().st_size} bytes)")
        self._send_json({"filename": upload_name, "message": "Uploaded successfully"})

    def _handle_generate_tabs(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))

            filename = unquote(body.get("file", ""))
            start_time = float(body.get("start", 0.0))
            end_time = float(body.get("end", 0.0))
            use_demucs = bool(body.get("use_demucs", False))
            guitar_volume = float(body.get("guitar_volume", 1.0))

            # Restriction: if guitar volume is zero/muted, tab extraction is not allowed
            if guitar_volume <= 0.01:
                self._send_json({
                    "error": "Громкость гитары на нуле или гитара заглушена! Включите звук гитары, чтобы снять табулатуру."
                }, status=HTTPStatus.BAD_REQUEST)
                return

            if not filename:
                self._send_json({"error": "Missing 'file' parameter"}, status=HTTPStatus.BAD_REQUEST)
                return

            audio_path = find_audio_file(filename)
            if not audio_path:
                self._send_json({"error": f"Audio file '{filename}' not found"}, status=HTTPStatus.NOT_FOUND)
                return

            if end_time <= start_time:
                self._send_json({"error": "Неверный интервал времени (конец должен быть больше начала)"}, status=HTTPStatus.BAD_REQUEST)
                return

            # Cap max snippet length to 30 seconds for fast responsive UX
            if end_time - start_time > 30.0:
                end_time = start_time + 30.0

            # Step 1: Check if separated guitar stem is available
            stems = check_stems_exist(audio_path)
            target_slice = CACHE_DIR / f"{audio_path.stem}_slice_{start_time:.2f}_{end_time:.2f}.wav"

            t0 = time.time()
            if stems:
                print(f"[Tabs] Slicing directly from pre-separated guitar stem: {stems['guitar'].name}")
                slice_audio(stems["guitar"], target_slice, start_time, end_time)
            elif use_demucs:
                print(f"[Tabs] Running Demucs guitar extraction for [{start_time:.2f}s - {end_time:.2f}s]...")
                from stem_extractor import GuitarStemExtractor
                extractor = GuitarStemExtractor()
                target_slice = extractor.get_guitar_snippet(audio_path, start_time, end_time)
            else:
                slice_audio(audio_path, target_slice, start_time, end_time)

            # Step 2: Transcribe via Basic Pitch & Solve Fretboard
            res = get_tab_extractor().transcribe(target_slice)
            elapsed = time.time() - t0

            notes_payload = [
                {
                    "pitch": n.pitch,
                    "start": round(n.start_time + start_time, 2),  # absolute song time
                    "end": round(n.end_time + start_time, 2),
                    "confidence": round(n.velocity, 2),
                }
                for n in res.notes
            ]

            self._send_json(
                {
                    "tab": res.ascii_tab,
                    "notes": notes_payload,
                    "num_notes": len(res.notes),
                    "snippet_start": round(start_time, 2),
                    "snippet_end": round(end_time, 2),
                    "duration": round(end_time - start_time, 2),
                    "elapsed_time": round(elapsed, 2),
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def run_server(port: int = 8080, on_ready=None):
    server = None
    for candidate_port in [port, 8081, 8082, 8000, 5000]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate_port), StudioRequestHandler)
            port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        print("Error: Could not bind to any port.")
        sys.exit(1)

    print("=" * 65)
    print(f"🎸 Guitar AI Studio running at: http://127.0.0.1:{port}")
    print(f"📁 Web UI files : {WEB_DIR}")
    print(f"📁 Demo tracks  : {ROOT_DIR}")
    print("=" * 65)

    if on_ready:
        on_ready(port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
