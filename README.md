# Agentic Audio Extraction

Audio analysis system using ffmpeg/ffprobe + LLM agent for court deposition recordings.

## Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│       LLM Agent      │◄────────┤ FastMCP Server       │
│   (LangGraph)        │  JSON   │ (server.py)          │
└───────────┬──────────┘         └─────────────┬────────┘
            │                                  │
            │    Phase A → B → C → D1 → D2     │
            │    (LLM only in D1/D2)           │
            ▼                                  ▼
┌─────────────────────┐              ┌─────────────────────┐
│ ffmpeg / ffprobe    │◄──────────── │   Tool calls (MCP)  │
│ (Subprocess)        │              │                     │
└─────────────────────┘              └─────────────────────┘
```

### Why Multi-Step LLM?

**Two-phase design (not one, not three):** Step 1 classifies recording context from metadata only — cheap, small input (~100 tokens). Step 2 synthesizes quality assessment grounded in Step 1 output. The classification step is separated because ASR thresholds vary dramatically by environment: phone recordings (8kHz) use different standards than in-person depositions (44.1kHz). Merging analysis into Step 2 keeps costs low while enabling context-aware quality judgments.

### Processing Flow

```
signal_diagnostics
    │
    ├── error ──→ error_handler → END
    │
    └── sample_rate < 8kHz ──→ critical_quality_report → END
                        (bypasses LLM - unusable by definition)
                         
    └── normal ──→ heuristic_processing
                  │
                  ├── clean audio ──→ lightweight_compilation
                  │                              ↓
                  └── issues ──→ structured_compilation
                                      ↓
                               classify_context (D1)
                                      ↓
                                 synthesize (D2) → END
```

- **Phase A**: Signal diagnostics (ffprobe metadata + amplitude stats)
- **Phase B**: Heuristic processing (silence/clipping detection)
- **Phase C**: Structured JSON compilation (Pydantic models, issue detection)
- **Phase D1**: LLM classifies recording context from metadata only
- **Phase D2**: LLM synthesizes quality assessment using context-aware thresholds

## Tech Stack
- Python 3.10+ with UV package manager
- OpenRouter as LLM provider
- Pydantic for schema validation
- LangGraph for agent orchestration
- FastMCP for tool exposure via MCP

## Setup

### 1. System Dependencies
```bash
# ffmpeg/ffprobe binaries are included in repo (ffmpegBase/bin/)
# Windows: No installation needed - binaries auto-detected
# Linux/macOS: Uses system ffmpeg, or set FFMPEG_PATH to custom location

# Linux/macOS (optional):
# sudo apt install ffmpeg  # If system ffmpeg not available

# Windows (manual override - only if needed):
# FFMPEG_PATH=./ffmpegBase/bin  # Add to .env if auto-detection fails
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
# Copy .env.example and configure
cp .env.example .env
# Edit .env to add your OpenRouter API key
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
  "recording_context": {
    "environment": "in_person",
    "expected_noise_profile": "clean",
    "recording_era": "modern",
    "context_notes": "44.1kHz stereo with high bitrate suggests professional setup"
  },
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

1. **Phase A**: Signal diagnostics (ffprobe metadata, amplitude analysis)
   - Route: Invalid file → error_handler
   - Route: Sample rate < 8kHz → critical_quality_report (unusable, LLM bypassed)

2. **Phase B**: Heuristic processing (clipping detection, silence detection)
   - Route: Clean audio → lightweight_compilation
   - Route: Issues detected → structured_compilation

3. **Phase C**: Structured JSON compilation (Pydantic validation, issue detection)

4. **Phase D1**: LLM classifies recording context (phone/in_person/conference_room)
   - Input: Metadata only (duration, sample_rate, channels, bit_rate)
   - Output: RecordingContext model for threshold calibration
   - Failure: Falls back to "unknown" context, continues to D2

5. **Phase D2**: LLM synthesizes quality assessment with context-aware thresholds
   - Uses RecordingContext to calibrate ASR viability standards
   - Phone (8kHz) → relaxed thresholds vs In-person (44kHz) → strictest standards

### Context-Aware Thresholds

| Environment | Sample Rate Expectation | Noise Tolerance | Notes |
|-------------|------------------------|-----------------|-------|
| phone | >= 8kHz acceptable | Relaxed RMS | G.711 telephone spec |
| in_person | >= 16kHz required | Strictest bar | Studio quality expected |
| conference_room | >= 16kHz | Moderate noise OK | Room tone expected |
| unknown | >= 16kHz required | Strictest bar | Fallback - assume worst |

### Quality Standards

- Sample rate < 8kHz → immediate "unusable" rating (critical path, no LLM)
- Silence ratio > 40% → downgrades quality one level
- Clipping + flat_factor > 0.1 → automatic "poor" minimum
- RMS level < -30dB → low volume issue flagged
- DC offset > -40dB → preprocessing required (high-pass filter)

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