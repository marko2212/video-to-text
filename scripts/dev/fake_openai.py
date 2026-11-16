"""A stand-in for the OpenAI API on localhost, for watching a run on the page.

Answers every transcription request after a chosen delay with made-up speech,
and everything else with 404, so an OpenAI run can be watched in a browser
without the real key or any credit, at a known pace. It stops by itself after
20 minutes.

Usage (the page must be an isolated copy: scratch ``DATA_DIR`` / ``TEMP_DIR`` /
``UPLOAD_DIR``, see CLAUDE.md)::

    python scripts/dev/fake_openai.py 8596 20
    set OPENAI_BASE_URL=http://127.0.0.1:8596/v1
    set OPENAI_API_KEY=sk-not-a-real-key
    set SEGMENT_DURATION_MINUTES=1
    streamlit run app.py --server.port 8597

Run it from a folder without the project's ``.env``, or the real key is sent
to this server (it reads no headers, but there is no reason to send it).

The tests start it in a thread through :func:`make_handler`, with a list that
collects the form fields of every request (all but the audio file), to see
what the app really sends on the wire.
"""

import json
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LIFETIME_SECONDS = 1200
# About 170 words: enough for a one-minute part not to look thin.
_SPEECH = " ".join(
    ["We agreed to send the offer on Monday and to review the plan on Friday."] * 12
)


def form_fields(content_type: str, body: bytes) -> dict[str, str]:
    """Return the plain fields of a multipart form (the file is left out).

    Args:
        content_type: The request's ``Content-Type`` header, with its boundary.
        body: The request body.

    Returns:
        Field name to value, for every part that is not a file.
    """
    message = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_content()
        for part in message.iter_parts()
        if part.get_filename() is None
    }


def make_handler(
    delay: float, seen: list[dict[str, str]] | None = None, text: str = _SPEECH
) -> type[BaseHTTPRequestHandler]:
    """Return a request handler that answers transcriptions after ``delay``.

    Args:
        delay: Seconds to wait before each transcription answer.
        seen: When given, the form fields of each request are appended to it.
        text: What every transcription answer says.

    Returns:
        The handler class for the server.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self.path.endswith("/audio/transcriptions"):
                self.send_response(404)
                self.end_headers()
                return
            if seen is not None:
                seen.append(form_fields(self.headers.get("Content-Type", ""), body))
            time.sleep(delay)
            body = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    return Handler


def main() -> None:
    """Serve on the port given first, answering after the delay given second."""
    port, delay = int(sys.argv[1]), float(sys.argv[2])
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(delay))
    threading.Timer(_LIFETIME_SECONDS, server.shutdown).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
