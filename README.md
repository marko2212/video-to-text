# VideoToText Transcription App 📝

A simple Streamlit web application to extract audio from video files (specifically MKV) and transcribe the audio into text using the OpenAI Whisper API.

## Features ✨

* Upload video files in MKV format.
* Extract audio from the video and save it as a WAV file.
* Option to download the extracted WAV audio file.
* Transcribe the extracted audio using OpenAI's Whisper model.
* Handles large audio files by splitting them into smaller segments for transcription.
* Displays the transcription progress in the interface.
* Shows a preview of the final transcript.
* Option to download the full transcript as a TXT file.
* Button to clean up temporary files generated during the process.

## Requirements 🛠️

* **Python:** Version 3.8 or higher.
* **ffmpeg:** This external tool **must be installed and accessible** for the application to work. See installation instructions below.
* **OpenAI API Key:** You need an API key from OpenAI to use the Whisper transcription service. You might incur costs depending on your usage.
* **Python Packages:** Listed in `requirements.txt`.

## Installation ⚙️

1. **Clone or Download the Repository:**

    ```bash
    git clone https://github.com/marko2212/video-to-text.git # Or download the ZIP and extract
    cd video-to-text
    ```

2. **Install ffmpeg:** Choose **one** of the following methods based on your setup:

    * **(A) If using Conda (Recommended for Conda users):**
        If you plan to use a Conda environment (see next step), the easiest way to install `ffmpeg` is within that environment:

        ```bash
        # Make sure you activate your conda environment first if already created
        # conda activate videototext
        conda install ffmpeg -c conda-forge
        ```

        This command installs `ffmpeg` specifically for the active Conda environment.

    * **(B) If NOT using Conda:** You need to install `ffmpeg` system-wide or make it accessible via your system's PATH.

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

3. **Create and Activate a Virtual Environment:** Choose **one** method:

    * **(A) Using Conda:**

        ```bash
        # Create the environment (if you haven't already for ffmpeg)
        conda create --name videototext python=3.11 # Or your preferred Python 3.x
        # Activate the environment
        conda activate videototext
        # Install ffmpeg now if you didn't in the previous step
        # conda install ffmpeg -c conda-forge
        ```

    * **(B) Using `uv` (Recommended if not using Conda):**
        Requires `uv` to be installed (`pip install uv` or see [uv installation docs](https://github.com/astral-sh/uv#installation)).

        ```bash
        # Create the virtual environment (named .venv by default)
        uv venv
        # Activate the environment
        # Windows: .\.venv\Scripts\Activate
        # macOS/Linux: source .venv/bin/activate
        ```

    * **(C) Using standard `venv`:**

        ```bash
        # Create the virtual environment
        python -m venv .venv
        # Activate the environment
        # Windows: .\.venv\Scripts\Activate
        # macOS/Linux: source .venv/bin/activate
        ```

4. **Install Python Dependencies:**
    Make sure your chosen virtual environment is **active** first!
    * Using `uv` (recommended for speed):

        ```bash
        uv pip install -r requirements.txt
        ```

    * Or using standard `pip`:

        ```bash
        pip install -r requirements.txt
        ```

5. **Set up Environment Variables:**
   * Copy the `.env.example` file to a new file named `.env`
   * Replace `'your_openai_api_key_here'` with your actual key.

## Running the Application 🚀

1. Make sure your virtual environment (`videototext` or `.venv`) is **activated**.
2. Ensure the `.env` file with your API key is present in the project root.
3. Run the Streamlit application from your terminal:
    * If using `uv` and your environment is *not* active (optional way):

        ```bash
        uv run streamlit run app.py
        ```

    * Standard way (with environment active):

        ```bash
        streamlit run app.py
        ```

4. Streamlit will provide local and network URLs (usually `http://localhost:8501` or similar). Open one of these URLs in your web browser.

## Usage 🖱️

1. **Upload File:** Use the file uploader to select a video file (MKV, MP4, etc.).
2. **Extract Audio:** Click the "Extract Audio" button. Wait for the process to complete. You'll see a success message and an option to download the `.wav` audio file.
3. **Start Transcription:** Once the audio is extracted, click the "Start Transcription" button.
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
