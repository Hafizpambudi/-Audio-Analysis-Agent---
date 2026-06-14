# Agentic Audio Extraction

Audio analysis system using ffmpeg/ffprobe + LLM agent for court deposition recordings.

## Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│  LLM Agent (Cursor)  │◄────────┤ FastMCP Server       │
│                      │  JSON   │ (server.py)          │
└──────────────────────┘         └───────────┬──────────┘
                                            │
                         Subprocess calls    │
                                            ▼
                               ┌─────────────────────┐
                               │  ffmpeg / ffprobe   │
                               │  (Audio Analysis)   │
                               └─────────────────────┘
```

## Tech Stack
- Python 3.10+ with UV package manager
- OpenRouter as LLM provider
- Pydantic for schema validation
- LangGraph for agent orchestration
- FastMCP for tool exposure via MCP

## Setup

### 1. System Dependencies
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows - Choose one option:

# Option A: Download static binaries (recommended for project-local)
# 1. Download: https://github.com/FFmpeg/FFmpeg/releases/latest
# 2. Extract ffmpeg.exe and ffprobe.exe to ./ffmpeg/
# 3. Add to .env: FFMPEG_PATH=./ffmpeg/

# Option B: Manual installation
# Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH
```

### 2. Python Environment
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### 3. Environment Variables
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Start MCP Server
```bash
uv run python -m src.server
```

### Analyze Audio File
```bash
uv run python -m src.agent "path/to/audio.mp3"
uv run python -m src.agent "path/to/audio.mp3" --skip-llm  # Without LLM (for testing)
uv run python -m src.agent "path/to/audio.mp3" --with-save # With Output Saved
```

## Output Format

```json
{
  "file_name": "deposition_001.mp3",
  "duration_seconds": 3600,
  "metadata": {
    "sample_rate": 44100,
    "bit_rate": 128000,
    "channels": 2
  },
  "audio_quality": {
    "silence_ratio": 0.12,
    "clipping_detected": false,
    "clipping_segments": [],
    "avg_volume_db": -18.5,
    "dc_offset_db": 0.1
  },
  "issues": [
    {
      "issue_type": "clipping",
      "description": "...",
      "severity": "high"
    }
  ],
  "executive_summary": {
    "overall_quality": "good",
    "asr_viability": "high",
    "transcription_viable": true,
    "summary": "...",
    "blocking_issues": []
  },
  "mitigation_strategies": [
    {
      "issue_type": "clipping",
      "recommended_action": "ffmpeg -i input.mp3 -af 'de-click' output.mp3",
      "priority": "immediate"
    }
  ]
}
```

## Agent Processing Phases

1. **Phase A**: Signal diagnostics (metadata extraction, amplitude analysis)
   - Conditional routing: Invalid file → error_handler, Low sample rate (<8kHz) → critical_quality_report
2. **Phase B**: Heuristic processing (clipping detection, silence detection)
   - Conditional routing: Clean audio → lightweight_compilation, Issues detected → structured_compilation
3. **Phase C**: Structured JSON compilation (Pydantic validation, issue detection)
4. **Phase D**: LLM synthesis (executive summary + mitigation matrix)

### Quality Thresholds
- Sample rate < 8kHz → immediate "unusable" rating (critical quality path)
- Silence ratio > 40% → downgrades quality one level
- Clipping detected → automatic "poor" minimum rating
- RMS level < -30dB → low volume issue flagged
- DC offset > -40dB → preprocessing required

## Available Tools

- `inspect_metadata(file_path)` - Extract audio metadata via ffprobe
- `detect_silence(file_path, noise_threshold_db, min_duration)` - Detect silent segments
- `analyze_amplitude_stats(file_path)` - Get RMS, peak, DC offset metrics
- `detect_clipping(file_path)` - Identify potential clipping with flat-factor analysis

## Development

### Lint/Check
```bash
uv run python -m py_compile src/*.py  # Syntax check
```