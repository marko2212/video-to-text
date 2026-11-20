# Petlje ponavljanja u transkriptima: šta drugi prijavljuju (19.11.2026)

Izveštaj Codex-a (ChatGPT, živa pretraga Reddita, OpenAI foruma i GitHub-a), neizmenjen. To je ChatGPT-ova tvrdnja: ključne niti proveriti pre oslanjanja. Povod: sastanak 19.11.2026 07:28, deo 4:58-10:00 sa jednom rečenicom ponovljenom deset puta (`gpt-4o-transcribe`); u 154 transkripta još tri takva slučaja, svi `gpt-4o-transcribe`.

**Switching to `whisper-1` does not eliminate repetition or hallucination loops.** It avoided one reported 4o duplication incident, but there are separate, explicit reports of the Whisper API replacing speech with unrelated or repeated output. I found no controlled comparison establishing which API loops more often on **five-minute, multi-speaker meeting chunks**.

I searched the requested subreddits, OpenAI’s forum, and GitHub for **May 19, 2025–November 19, 2026**. Older threads are included where they have relevant replies within that window; other older material is marked as background. Searches of r/MachineLearning and r/singularity did not yield sufficiently specific, model-identified reports to add. ChatGPT app reports cannot reliably identify the underlying transcription model.

Below is the complete inventory of pages I opened, followed by selected snippet-only leads. **“Votes not visible” means the retrieved page did not expose a usable score.** Reddit scores shown by search may be cached.

## Opened: 4o transcription reports and comparisons

**1. [Gpt-4o-transcribe model transcribes the text twice](https://community.openai.com/t/gpt-4o-transcribe-model-transcribes-the-text-twice/1273864)**  
OpenAI forum · May 30–31, 2025 · Votes not visible.

- The author reported that **both `gpt-4o-transcribe` and mini duplicated the entire transcription**, while `whisper-1` did not.
- Two other users independently reported the same behavior; one said their code had not changed and it had worked 12 hours earlier.
- By May 31, two participants said the problem had stopped. This supports a transient API incident, rather than evidence that Whisper generally avoids loops.

**2. [Anyone else getting double transcription with gpt-4o-transcribe today?](https://www.reddit.com/r/OpenAI/comments/1kzlfnt/)**  
r/OpenAI · Exact date unavailable in retrieved page; displayed “1y ago” · Votes not visible. **Exact inclusion in the date window could not be verified.**

- The author reported suddenly receiving the same text twice without changing their code.
- The one substantive reply visible confirmed the issue and offered no solution.
- No audio length, speaker count, parameters, or reproduction was provided. This is corroboration of duplication, not a detailed meeting-loop case.

**3. [Overtrained transcription model?](https://community.openai.com/t/overtrained-transcription-model/1290945)**  
OpenAI forum · June 17, 2025 · Votes not visible.

- A user repeatedly transcribed a recording of 24 Harvard sentences with `gpt-4o-transcribe`, **`temperature=0`, and no prompt**. One result expanded into about 250 sentences, including material absent from the recording.
- Their reported WER jumped from 17% to 845%; they said this extreme behavior had happened twice.
- A commenter suggested trying `whisper-1` and described prompt-copying and truncated endings as other 4o failure modes. Trailing silence and memorized sentence completion were hypotheses, **not demonstrated causes**.

**4. [GPT-4o Mini Transcribe repetition loop in V20 U9 / AI 1.4.48 – direct OpenAI API works correctly](https://www.3cx.com/community/threads/gpt-4o-mini-transcribe-repetition-loop-in-v20-u9-ai-1-4-48-%E2%80%93-direct-openai-api-works-correctly.137987/)**  
3CX forum · August 19–20, 2026 · No post vote count visible.

- Two Bulgarian call recordings reportedly began normally, then repeated an earlier conversation block dozens of times while losing subsequent speech.
- **The same WAV files transcribed correctly when submitted directly to OpenAI**, using mini and `language=bg`.
- The author suspected chunking, retries, or response assembly in 3CX; staff acknowledged the report and said the team would contact them. No confirmed root cause or published fix appeared.

This is an important counterexample to attributing every repeated transcript to the model itself.

**5. [Gpt-4o-mini-transcribe and gpt-4o-transcribe not as good as whisper](https://community.openai.com/t/gpt-4o-mini-transcribe-and-gpt-4o-transcribe-not-as-good-as-whisper/1153905)**  
OpenAI forum · Started March 27, 2025; relevant reply January 10, 2026 · Votes not visible.

- **In-window:** a developer using Whisper-1 in production for two years preferred it on informal two-person conversations. They said 4o missed or mistranscribed mumbled speech, sentence edges, and speech interrupted by pauses.
- **Older background:** another participant showed both 4o models omitting the beginning and ending of a short message that Whisper retained.
- A contrasting older reply preferred 4o because it did not transcribe background noise as broken language. This thread demonstrates workload-dependent accuracy, not comparative loop rates.

**6. [Best STT?](https://www.reddit.com/r/speechtech/comments/1nrf6b0/best_stt/)**  
r/speechtech · Exact date unavailable; relevant comments displayed “1y ago” · Votes not visible. **Exact date-window membership unverified.**

- One user strongly preferred 4o-transcribe for dictation involving medical and technical terms after trying hosted Whisper variants, Deepgram, and Voxtral.
- They ranked full 4o above mini, while noting greater latency.
- Other comments preferred Whisper or questioned whether more readable LLM output actually improves WER. No shared evaluation corpus or loop measurements were supplied.

## Opened: Whisper API and open-source reports

**7. [Whisper Transcription Inconsistencies and Guidance Needed for Production Use](https://community.openai.com/t/whisper-transcription-inconsistencies-and-guidance-needed-for-production-use/1367609)**  
OpenAI forum · November 26, 2025 · Votes not visible.

- The author explicitly identified **`whisper-1` through the API**, used in a VR Unity game.
- Reported failures included unrelated multilingual output, repeated fire emoji, and a lone “you” replacing a full spoken sentence.
- Failures were intermittent and difficult to reproduce. The retrieved thread contained no substantive reply or verified remedy.

**8. [Why it keep saying same words? — Whisper-WebUI #522](https://github.com/jhj0517/Whisper-WebUI/issues/522)**  
GitHub · Opened March 23, 2025; relevant replies May–November 2025; closed January 9, 2026 · Reactions not visible.

- **May 23:** a Danish-audio user said switching from large-v3 to **large-v2 plus VAD** avoided the problem. **September 24:** another user reported repetition in four of five Dutch files with large-v3/turbo—an individual workload, not a prevalence estimate.
- A June reply reported better results after removing background audio. Older background in the thread described large-v3 repeating from approximately minute 40 through the rest of a 90-minute recording.
- A November reply quoted an upgrade-to-faster-whisper-1.2.0 success report and a missing previous-text-setting problem. However, Purfview disputed the attribution to the cited prompt-reset bug, saying that bug had already been fixed in 1.0.0. **The upgrade anecdote is not a settled diagnosis.**

**9. [We collected 135 phrases Whisper hallucinates during silence — here's what it says when nobody's talking and how we stopped it](https://www.reddit.com/r/LocalLLaMA/comments/1rlqfd7/we_collected_135_phrases_whisper_hallucinates/)**  
r/LocalLLaMA · March 5, 2026 · **+353**, visible in search metadata.

- A meeting-bot developer described production hallucinations over silence, including entire paragraphs of repeated phrases. They advocated **VAD before transcription and `condition_on_previous_text=False`**.
- Their other measures included per-language phrase blocklists and repeated-output detection. They claimed VAD alone addressed roughly 90% of their problem; this was an operational claim without a controlled evaluation in the thread.
- A commenter warned that blocklists delete legitimate speech and that `beam_size=1` can worsen WER. The author accepted those tradeoffs and agreed that **compression-ratio detection followed by retry** was the standard repetition remedy.

**10. [Whisper.cpp is underwhelming](https://www.reddit.com/r/LocalLLaMA/comments/1tsas63/whispercpp_is_underwhelming/)**  
r/LocalLLaMA · May 30, 2026 · **+2**; leading chunking reply **+9**, visible in search metadata.

- The author reported large-v3 transcribing normally for about 20 minutes, then repeating one invented sentence through the remainder.
- Replies recommended smaller chunks, preferably ending at silence; one suggested **5–10 minutes**, along with VAD for silence/noise/music.
- Other suggestions were adjusting context and temperature fallback, testing another Whisper implementation, or switching engines. These were recommendations; the author did not publish a verified successful fix.

**11. [Anyone else struggling to detect fluent hallucinations in long-form ASR TTS workflows?](https://www.reddit.com/r/speechtech/comments/1tukyxt/anyone_else_struggling_to_detect_fluent/)**  
r/speechtech · June 2, 2026 · **+2**, visible in search metadata.

- The author described large-v3 producing plausible invented text on meeting recordings and support calls, especially with overlapping speakers, noise, or poor microphones.
- They were experimenting with timestamps, repetition detection, confidence, and multi-pass comparison, without finding a fully reliable detector.
- A commenter challenged the apparent one-hour threshold: Whisper already slices audio internally, and hallucinations can occur in 30-second clips. They distinguished easily detected repetitions from fluent inventions.

**12. [Proposal: reduce repetition hallucinations in long-form decoding — whisper.cpp #3744](https://github.com/ggml-org/whisper.cpp/issues/3744)**  
GitHub · April 5, 2026; substantive discussion August 25, 2026 · Reactions not visible.

- The proposal describes a wrong phrase being carried into subsequent chunks as prompt history, reinforcing itself until later speech is replaced by repetitions.
- Proposed remedies were limiting context across VAD-separated speech islands and **retrying repetitive chunks without dynamic previous-text context**. These are proposal-specific controls, not universally available Whisper options.
- Another developer independently supported fresh-context retries for missing speech. Participants disagreed about beam settings; one found beam size five induced a loop. They also distinguished keeping a useful static initial prompt from carrying accumulated transcript history.

**13. [[Bug] Repetitions in transcription (Whisper artifact) — Vexa #104](https://github.com/Vexa-ai/vexa/issues/104)**  
GitHub · February 4, 2026; update April 25; closed “not planned” July 4 · Reactions not visible.

- A meeting-bot issue describes repeated or near-duplicate text and overlapping transcript segments.
- Proposed measures include repetition penalties, `no_repeat_ngram_size`, and downstream overlap/deduplication rules.
- **An April update surfaced in search metadata, rather than the opened page’s extracted body:** repetition penalties reduced but did not eliminate artifacts; compression-ratio heuristics caught some cases but missed others. Closure does not establish resolution.

**14. [Whisper hallucination — how to recognize and solve? — reply #27](https://community.openai.com/t/whisper-hallucination-how-to-recognize-and-solve/218307/27)**  
OpenAI forum · August 19, 2026 reply in an older thread · Votes not visible.

- The commenter recommended counting repeated **four-word windows**, suggesting approximately 5% of all windows as a flagging threshold.
- For Chinese, Japanese, and Korean, they recommended character n-grams because whitespace tokenization fails.
- They warned that fluent hallucinations can evade LLM cleanup and that phrase lists need filtering to avoid flagging ordinary speech. The threshold and cleanup claims were community advice, not a validated universal detector.

**15. [Whisper being weird](https://www.reddit.com/r/ChatGPT/comments/1olpbna/)**  
r/ChatGPT · November 1, 2025 · **+3**, visible in search metadata.

- The author said a spoken question about recursion was replaced with a capitalized invitation to like, comment, share, and subscribe.
- One commenter reported similar multilingual subtitle/outro phrases and suggested training-data associations or background sounds.
- Another disputed the background-sound explanation and described unrelated boilerplate appearing without an audible source. **The model was identified by the chatbot itself, so this is not reliable evidence specifically against `whisper-1`.**

## Opened: documentation and older background

**16. [Audio Transcription API chunking_strategy option](https://community.openai.com/t/audio-transcription-api-chunking-strategy-option/1260804)**  
OpenAI forum · May 14–16, 2025 · Votes not visible · **Just outside the requested window.**

- Participants initially reported empty transcripts and inconsistent handling of `auto`/`server_vad`.
- Another participant demonstrated a working `server_vad` request and explained that `auto` normalizes loudness and selects VAD boundaries.
- A lecture test reportedly repeated supplied prompt text after chunking. This thread supports **testing prompt and chunking interactions**, not treating server VAD as a proven universal cure.

**17. [Create transcription — OpenAI API reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create/)**  
Living documentation · Publication date not shown; accessed November 19, 2026 · Votes not applicable.

- Documents `chunking_strategy="auto"` as loudness normalization followed by VAD boundary selection, with a `server_vad` object for customization.
- Describes `prompt` as guidance for style or continuation of a previous segment; `temperature` defaults to zero, with automatic temperature increases under documented fallback behavior.
- **It does not expose open-source Whisper’s `condition_on_previous_text` or compression-ratio controls as `whisper-1` request parameters.** Local Whisper fixes cannot simply be copied into API calls.

**18. [openai/whisper — whisper/transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)**  
Living source file · Accessed November 19, 2026; current revision date not established · Reactions not applicable.

- Its own documentation explicitly says disabling `condition_on_previous_text` makes repetition loops less likely, with a possible loss of cross-window consistency.
- It implements temperature fallback when compression ratio or average log probability indicates failure; the displayed default compression-ratio threshold is **2.4**.
- Silence rejection combines no-speech probability with low average log probability. Consequently, adjusting the no-speech threshold alone does not guarantee rejection of confidently hallucinated text.

**19. [Introducing next-generation audio models in the API](https://openai.com/index/introducing-our-next-generation-audio-models/)**  
OpenAI · March 20, 2025 · Votes not applicable · **Outside-window benchmark background.**

- OpenAI reports lower WER for 4o transcription models than original Whisper models across established benchmarks.
- It reports better recognition under noise, accents, and varying speech speeds.
- These are **aggregate accuracy claims**, not published loop-frequency measurements for five-minute meeting audio.

**20. [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models)**  
OpenAI Developers · December 22, 2025 · Votes not applicable.

- OpenAI reports that `gpt-4o-mini-transcribe-2025-12-15` improves WER and reliability relative to prior models.
- On its **internal noise/silence hallucination evaluation**, it reports approximately **90% fewer hallucinations than Whisper v2 and 70% fewer than previous GPT-4o-transcribe models**.
- This is meaningful comparative evidence, but it concerns a particular mini snapshot and an internal evaluation. It does **not** establish loop rates for meeting chunks or guarantee the same reduction in production.

## Selected leads seen only as search snippets

These were **not opened**, and I have not treated their findings as independently verified.

| Page | Date / votes visible | Snippet-only finding |
|---|---|---|
| [Transcriptions with repeated sentences — faster-whisper #1619](https://github.com/SYSTRAN/faster-whisper/discussions/1619) | Exact date not verified; search labels it older than the window; votes unavailable | Repeated sentences absent from the audio; suggestions include repeated-word/segment filtering. |
| [Sentences repeating in text — openai/whisper #1059](https://github.com/openai/whisper/discussions/1059) | March–June 2023; outside window | A **medium** user described an extended phrase loop. Replies distinguish an implementation duplication bug from longstanding hallucination loops and suggest VAD. |
| [Troubleshooting Whisper’s Repetition of Sentences in Silent Intervals](https://note.com/ennodes/n/n384d380e2d4c?hl=en) | September 26, 2026; votes unavailable | Faster-whisper large-v3 reportedly repeated an agenda sentence nearly 40 times during actual speech, then filled a meeting break with outro text. |
| [Stop Whisper from appending repetition loops to local dictation](https://notes.saurav.io/whisper-repetition-loop-cleanup/) | September 2, 2026; votes unavailable | Author reports 118 high-confidence loops in 3,484 history entries and discovers that their purported streaming path bypassed VAD. This is one application’s audit. |
| [New Audio Model Snapshots in the Realtime-API](https://community.openai.com/t/new-audio-model-snapshots-in-the-realtime-api/1369374) | December 15, 2025; votes unavailable | Forum announcement reports an **89% hallucination reduction versus `whisper-1`** for the December mini snapshot. The opened official article instead describes approximately 90% versus Whisper v2; I would preserve that labeling difference. |

## What the evidence supports

**How common, and which is better?** Whisper has more detailed, longstanding public reports in this search, but thread counts cannot establish relative frequency: its user base, deployment history, implementations, and decoding settings differ. The evidence supports **better aggregate benchmark accuracy for 4o**, alongside real workloads where `whisper-1` preserves speech better. Neither establishes a universal winner for multi-speaker meetings. See the contrasting [production-user comparisons](https://community.openai.com/t/gpt-4o-mini-transcribe-and-gpt-4o-transcribe-not-as-good-as-whisper/1153905) and [OpenAI benchmark claims](https://openai.com/index/introducing-our-next-generation-audio-models/).

**Causes with the clearest support:** silence/non-speech triggers, previous-text feedback in local Whisper, and application-level processing errors. Overlap and weak speech are reported risk factors, but the sources do not isolate them as causes. A fixed five-minute boundary is not established as a specific trigger. [Whisper’s source](https://github.com/openai/whisper/blob/main/whisper/transcribe.py) directly acknowledges feedback loops; the [3CX reproduction](https://www.3cx.com/community/threads/gpt-4o-mini-transcribe-repetition-loop-in-v20-u9-ai-1-4-48-%E2%80%93-direct-openai-api-works-correctly.137987/) shows why the surrounding pipeline also needs scrutiny.

**Temperature and prompts:** zero temperature is not a guarantee—the Harvard-sentence failure occurred at zero without a prompt. Higher-temperature retries have a concrete basis in Whisper’s fallback implementation, but I found no controlled evidence for a universally effective 4o temperature. Likewise, server VAD is a documented segmentation feature, not a proven blanket fix. [Failure report](https://community.openai.com/t/overtrained-transcription-model/1290945), [Whisper implementation](https://github.com/openai/whisper/blob/main/whisper/transcribe.py), [API reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create/).

**Does `whisper-1` avoid the problem? No.** It can avoid a particular 4o failure while still hallucinating or replacing real speech itself. The clearest in-window API-specific example is the [November 2025 report](https://community.openai.com/t/whisper-transcription-inconsistencies-and-guidance-needed-for-production-use/1367609).

The **three most recurring fixes in the reviewed material**, qualitatively rather than by a formal citation count, were:

1. **VAD before transcription and speech-aware chunk boundaries:** skip non-speech and split at natural pauses.
2. **Break contaminated context:** for local Whisper, use `condition_on_previous_text=False`, or retry only the failed excerpt without accumulated transcript history.
3. **Detect and retry bad chunks:** compression ratio, repeated n-grams, and duplicate-segment checks; resend from clean boundaries, then use another model if needed. Removing repeated text alone cannot recover the speech it replaced. [Production discussion](https://www.reddit.com/r/LocalLLaMA/comments/1rlqfd7/we_collected_135_phrases_whisper_hallucinates/), [fresh-context retry discussion](https://github.com/ggml-org/whisper.cpp/issues/3744).