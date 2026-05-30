# VideoToText Transcription App 📝

A simple Streamlit web application to transcribe speech into text using the OpenAI Whisper API. It accepts both video files (it extracts the audio track) and audio files directly.

## Features ✨

* Upload **video** files in MKV, MP4, MOV, AVI, WebM, M4V, WMV, FLV, MPEG/MPG, 3GP, TS/MTS/M2TS, OGV, or VOB format.
* Upload **audio** files directly in MP3, WAV, M4A, AAC, FLAC, OGG, Opus, WMA, AIFF, or AMR format (audio is sent straight to transcription — no extraction step).
* Extract audio from a video and save it as a WAV file (with an option to download it).
* Transcribe the extracted audio using OpenAI's Whisper model.
* Handles large audio files by splitting them into smaller segments for transcription.
* Displays the transcription progress in the interface.
* Shows a preview of the final transcript.
* Option to download the full transcript as a TXT file.
* Button to clean up temporary files generated during the process.

## Requirements 🛠️

* **Python:** Version 3.10 or higher. (`uv` will fetch a matching interpreter automatically if you don't have one.)
* **uv:** Used to manage the virtual environment and dependencies. Install from [the uv docs](https://docs.astral.sh/uv/getting-started/installation/).
* **ffmpeg:** This external tool **must be installed and accessible** for the application to work. See installation instructions below.
* **OpenAI API Key:** You need an API key from OpenAI to use the Whisper transcription service. You might incur costs depending on your usage.
* **Python Packages:** Declared in `pyproject.toml` and locked in `uv.lock` — installed via `uv sync`.

## Installation ⚙️

1. **Clone or Download the Repository:**

    ```bash
    git clone https://github.com/marko2212/video-to-text.git # Or download the ZIP and extract
    cd video-to-text
    ```

2. **Install ffmpeg:** `ffmpeg` must be installed system-wide and accessible via your system's `PATH`.

    * **Windows:**
        1. Go to the official FFmpeg download page: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
        2. Navigate to the Windows builds section (often linked under "Windows EXE Files"). Recommended sources are `gyan.dev` or `BtbN`.
        3. Download one of the builds (e.g., the "essentials" build from gyan.dev is usually sufficient). It will likely be a `.zip` or `.7z` archive.
        4. Extract the downloaded archive. You'll get a folder (e.g., `ffmpeg-6.1.1-essentials_build`).
        5. Move this extracted folder to a permanent location, for example, `C:\ffmpeg`.
        6. **Add FFmpeg to PATH:**
            * Search for "Environment Variables" in the Windows Start Menu and select "Edit the system environment variables".
            * In the System Properties window, click the "Environment Variables..." button.
            * In the "System variables" section (or "User variables" if you prefer), find the `Path` variable, select it, and click "Edit...".
            * Click "New".
            * Enter the **full path to the `bin` folder** inside your ffmpeg directory (e.g., `C:\ffmpeg\bin`).
            * Click "OK" on all open windows to save the changes.
        7. **Verify:** Open a **new** Command Prompt or PowerShell window (important!) and type `ffmpeg -version`. If it shows version information, you're set.

    * **macOS (using Homebrew):**
        If you don't have Homebrew, install it first from [https://brew.sh/](https://brew.sh/). Then, open Terminal and run:

        ```bash
        brew install ffmpeg
        ```

        Homebrew will handle adding it to your PATH. Verify with `ffmpeg -version`.

    * **Linux (using package manager):**
        Open your terminal and use your distribution's package manager:
        * Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
        * Fedora: `sudo dnf install ffmpeg` (You might need to enable the RPM Fusion repository first if it's not found).
        * Arch Linux: `sudo pacman -S ffmpeg`

        Verify with `ffmpeg -version`.

3. **Install `uv`** (if you don't have it already):
    See the official [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/). On Windows, the one-liner is:

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    On macOS/Linux:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

4. **Install Python Dependencies:**
    A single command creates the virtual environment (`.venv/`), fetches a matching Python interpreter if needed, and installs every dependency from `uv.lock`:

    ```bash
    uv sync
    ```

    You do **not** need to manually create or activate a virtual environment — `uv run` (see below) handles that for you. Re-run `uv sync` any time `pyproject.toml` or `uv.lock` changes.

5. **Set up Environment Variables:**
    * Copy the `.env.example` file to a new file named `.env`
    * Replace `'your_openai_api_key_here'` with your actual key.

## Running the Application 🚀

1. Ensure the `.env` file with your API key is present in the project root.
2. Run the Streamlit application from your terminal:

    ```bash
    uv run streamlit run app.py
    ```

    `uv run` automatically uses the project's `.venv` — no manual activation step needed.

3. Streamlit will provide local and network URLs (usually `http://localhost:8501` or similar). Open one of these URLs in your web browser.

## Usage 🖱️

1. **Upload File:** Use the file uploader to select a video file (MKV, MP4, etc.) or an audio file (MP3, WAV, etc.).
2. **Extract Audio (video only):** For a video upload, click the "Extract Audio" button. Wait for the process to complete. You'll see a success message and an option to download the `.wav` audio file. *Audio uploads skip this step — they are ready for transcription immediately.*
3. **Start Transcription:** Once the audio is ready, click the "Start Transcription" button.
    * The application will show progress information (processing segments). This might take several minutes depending on the audio length.
    * You might see status updates like "Info", "Start", "Progress", "Complete", or "Error".
4. **View & Download Transcript:** After successful transcription, a preview of the text will appear in a text area, and a "Download Transcript" button will become available for the `.txt` file.
5. **Clean Up:** Click the "Clean temporary files" button to remove files from the `temp` and `uploads` directories.

## Configuration 🔑

* **OpenAI API Key:** Must be set in the `.env` file as `OPENAI_API_KEY`. The application uses `python-dotenv` to load this key.

## Troubleshooting ⚠️

* **`ffmpeg not found` Error / Runtime Warning:** This is the most common issue. Double-check that `ffmpeg` is correctly installed using **one** of the methods described in the "Installation" section. If you installed it manually (Windows non-Conda), ensure the **correct `bin` folder** path is added to your system's PATH and **restart your terminal/VS Code** afterwards. Verify by running `ffmpeg -version` in a new terminal.
* **`openai.AuthenticationError`:** Double-check your API key in the `.env` file. Make sure the `.env` file is in the same directory where you run `streamlit run app.py`. Verify your OpenAI account status and billing information.

## Author 👨‍💻

* Marko A

---
