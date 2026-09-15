# Demo

This repository does not include video or model files. Use any local video file to try the workflow.

## 1. Check environment

~~~bash
python3 scripts/transcribe_video_long.py doctor
~~~

## 2. Run transcription and translation

~~~bash
python3 scripts/transcribe_video_long.py ./demo.mp4 \
  --language en \
  --translate \
  --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

## 3. Expected output

By default, output is written to:

~~~text
.teaching-video-runtime/outputs/whisper-video-translator/demo-<hash>/
~~~

Visible files:

~~~text
demo.sentences.md
demo.srt
manifest.json
~~~

Hidden resumable cache:

~~~text
.segments/
~~~

## 4. Example sentence draft

~~~markdown
# 逐句双语稿

| # | 时间 | 原文 | 中文 |
|---:|---|---|---|
| 1 | 00:00:00.000 --> 00:00:03.200 | Today we are going to talk about attention. | 今天我们来谈谈注意力。 |
| 2 | 00:00:03.200 --> 00:00:07.100 | It is not just about working harder. | 这并不只是更努力地工作。 |
~~~

## 5. Example SRT cue

~~~srt
1
00:00:00,000 --> 00:00:03,200
Today we are going to talk about attention.
今天我们来谈谈注意力。
~~~

## 6. Validate output

~~~bash
python3 scripts/transcribe_video_long.py validate OUTPUT_DIR \
  --stem demo \
  --expect-bilingual
~~~
