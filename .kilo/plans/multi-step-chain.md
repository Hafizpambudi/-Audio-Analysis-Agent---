# Multi-Step LLM Chain Plan

**Design choice:** 2-step, not 3. Step 1 classifies recording context from metadata only
(cheap, small input). Step 2 synthesizes quality assessment grounded in Step 1 output.
The "analyze" step is merged into Step 2 because they share the same input and answer
the same question — separating them adds cost with no clear boundary.

**Files touched:** `src/schema.py` · `src/agent.py`

---

## Step 1 — What Changes Where

### 1.1 · Add `RecordingContext` model → `schema.py`

Step 1 output must be a typed, validated model — not a raw dict — so Step 2
can reference fields safely.

```python
# schema.py

from typing import Literal

class RecordingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Literal["phone", "in_person", "conference_room", "unknown"]
    expected_noise_profile: Literal["clean", "moderate", "noisy", "unknown"]
    recording_era: Literal["modern", "legacy", "unknown"]
    # Derived inference: affects what silence ratio and RMS thresholds mean
    context_notes: str   # 1 sentence — e.g. "Phone call audio; 8kHz limits ASR"
```

Add `recording_context` field to `AudioAnalysisReport`:

```python
# schema.py — inside AudioAnalysisReport:
recording_context: Optional[RecordingContext] = None
```

Add `recording_context` to `AudioAgentState`:

```python
# agent.py — inside AudioAgentState TypedDict:
recording_context: Optional[RecordingContext]
```

---

### 1.2 · Add Step 1 Node: `phase_d1_classify_context` → `agent.py`

**Input:** metadata only — duration, sample_rate, channels, bit_rate.
No report data. This keeps the call cheap and focused.

**Failure contract:** If Step 1 fails for any reason, set context to
`RecordingContext(environment="unknown", ...)` and continue to Step 2.
Step 2 must never be blocked by Step 1 failure.

```python
# agent.py

CLASSIFY_SYSTEM_PROMPT = """You are an audio forensics expert classifying court deposition recordings.

Classify the recording based ONLY on the technical metadata provided.
Return ONLY valid JSON matching this exact shape — no markdown, no extra keys:
{
  "environment": "<phone|in_person|conference_room|unknown>",
  "expected_noise_profile": "<clean|moderate|noisy|unknown>",
  "recording_era": "<modern|legacy|unknown>",
  "context_notes": "<one sentence explaining your classification>"
}

CLASSIFICATION RULES:
- sample_rate <= 8000Hz                   → environment: "phone", era: "legacy"
- sample_rate 8001–16000Hz, channels == 1 → environment: "phone" or "unknown"
- sample_rate >= 32000Hz, channels == 2   → environment: "in_person" or "conference_room"
- bit_rate < 64000                        → era: "legacy"
- bit_rate >= 128000                      → era: "modern"
- When uncertain: use "unknown", do not guess"""


def phase_d1_classify_context(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase D1: Classifying recording context")

    metadata = state.get("metadata", {})

    user_prompt = f"""Classify this court deposition recording:

- Duration    : {metadata.get('duration_seconds', 0):.1f}s
- Sample Rate : {metadata.get('sample_rate', 0)} Hz
- Channels    : {metadata.get('channels', 1)}
- Bit Rate    : {metadata.get('bit_rate', 'unknown')} bps"""

    fallback = RecordingContext(
        environment="unknown",
        expected_noise_profile="unknown",
        recording_era="unknown",
        context_notes="Classification failed — applying default thresholds."
    )

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        data = _extract_json(response.content)
        context = RecordingContext(**data)
        logger.info(f"Phase D1: Classified as {context.environment} / "
                    f"{context.expected_noise_profile} / {context.recording_era}")

    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        logger.warning(f"Phase D1: Classification failed ({e}) — using fallback")
        context = fallback

    return {**state, "recording_context": context}
```

---

### 1.3 · Update Step 2 Node: `phase_d2_synthesize` → `agent.py`

**Rename** existing `phase_d_llm_synthesis` → `phase_d2_synthesize`.
**Add** `recording_context` to the system prompt so thresholds adjust per environment.

```python
# agent.py

SYNTHESIZE_SYSTEM_PROMPT = """You are a forensic audio quality analyst for legal transcription systems.

REFERENCE STANDARDS:
- EBU R128 : target -23 LUFS integrated loudness for speech (ITU-R BS.1770)
- ASR floor : 16kHz minimum sample rate (Whisper, wav2vec2 training spec)
- Hard floor : 8kHz (ITU-T G.711 telephone — expect high WER)
- Clipping   : true peak >= -1.0 dBFS with flat_factor > 0.1 = saturation distortion

CONTEXT-AWARE RULES:
- environment "phone"          → treat sample_rate >= 8kHz as acceptable, RMS threshold relaxed
- environment "conference_room"→ moderate background noise is expected, not flagged as critical
- environment "in_person"      → highest quality bar applies
- environment "unknown"        → apply strictest thresholds

OUTPUT RULES — use ONLY these exact values:
- overall_quality      : excellent | good | acceptable | poor | unusable
- asr_viability        : high | medium | low | not_viable
- transcription_viable : true only if overall_quality in [excellent, good, acceptable]
- priority             : immediate | before_transcription | optional
Return ONLY valid JSON. No markdown fences."""


def phase_d2_synthesize(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase D2: Synthesizing quality assessment")

    if state.get("error") or not state.get("report"):
        return state

    context: RecordingContext = state.get("recording_context") or RecordingContext(
        environment="unknown",
        expected_noise_profile="unknown",
        recording_era="unknown",
        context_notes="No context available."
    )

    report = state["report"]
    report_dict = report.model_dump()
    metadata = report_dict.get("metadata", {})
    quality  = report_dict.get("audio_quality", {})
    issues   = report_dict.get("issues", [])

    user_prompt = f"""RECORDING CONTEXT (from Step 1 classification):
- Environment         : {context.environment}
- Expected noise      : {context.expected_noise_profile}
- Recording era       : {context.recording_era}
- Context notes       : {context.context_notes}

SIGNAL FACTS:
- Duration            : {report_dict.get('duration_seconds', 0):.1f}s
- Sample Rate         : {metadata.get('sample_rate', 0)} Hz
- Channels            : {metadata.get('channels', 1)}
- Bit Rate            : {metadata.get('bit_rate', 'unknown')} bps
- Peak Level          : {report_dict.get('amplitude_stats', {}).get('peak_level_db', 'N/A')} dBFS
- RMS Level           : {report_dict.get('amplitude_stats', {}).get('rms_level_db', 'N/A')} dB
- DC Offset           : {report_dict.get('amplitude_stats', {}).get('dc_offset_db', 'N/A')} dB
- Silence Ratio       : {quality.get('silence_ratio', 0):.1%}
- Clipping Detected   : {quality.get('clipping_detected', False)}
- Issues Found        : {len(issues)}

DETECTED ISSUES:
{json.dumps(issues, indent=2) if issues else "None"}

Using the context above to calibrate thresholds, return this exact JSON:
{{
  "executive_summary": {{
    "overall_quality": "<excellent|good|acceptable|poor|unusable>",
    "asr_viability": "<high|medium|low|not_viable>",
    "transcription_viable": <true|false>,
    "summary": "<2-3 sentences specific to this recording type and its ASR impact>",
    "blocking_issues": ["<issue that must be fixed before transcription>"]
  }},
  "mitigation_strategies": [
    {{
      "issue_type": "<type>",
      "recommended_action": "<concrete ffmpeg command or tool>",
      "priority": "<immediate|before_transcription|optional>"
    }}
  ]
}}"""

    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        try:
            llm = _get_llm()
            response = llm.invoke([
                SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            synthesis = _extract_json(response.content)

            exec_summary = ExecutiveSummary(**synthesis["executive_summary"])
            mitigations = [
                MitigationStrategy(**m)
                for m in synthesis.get("mitigation_strategies", [])
            ]
            report.executive_summary = exec_summary
            report.mitigation_strategies = mitigations
            report.recording_context = context
            logger.info(f"Phase D2: Synthesis OK on attempt {attempt + 1} — "
                        f"quality={exec_summary.overall_quality}")
            break

        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            logger.warning(f"Phase D2: Parse failed attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES:
                report.executive_summary = ExecutiveSummary(
                    overall_quality="acceptable",
                    asr_viability="medium",
                    transcription_viable=True,
                    summary="Automated synthesis failed after retries. Manual review recommended.",
                    blocking_issues=[]
                )

    return {**state, "report": report}
```

---

## Step 2 — Wire Into LangGraph → `agent.py`

Replace the single `llm_synthesis` node with two chained nodes.
D1 always runs before D2. No conditional edge between them —
D1 failure is handled internally with a fallback, not by graph routing.

```python
# agent.py — inside create_agent_graph():

# Add nodes
graph.add_node("classify_context", phase_d1_classify_context)   # new
graph.add_node("synthesize",       phase_d2_synthesize)          # renamed from llm_synthesis

# Wire: both compilation paths feed D1 → D2
graph.add_edge("structured_compilation",  "classify_context")
graph.add_edge("lightweight_compilation", "classify_context")
graph.add_edge("classify_context",        "synthesize")
graph.add_edge("synthesize",              END)

# Error / critical paths bypass both LLM nodes entirely
graph.add_edge("critical_quality_report", END)
graph.add_edge("error_handler",           END)
```

**Resulting graph flow:**

```
signal_diagnostics
    │
    ├─(error / sr < 8kHz)──→ critical_quality_report → END
    │                         error_handler           → END
    │
    └─(normal)──→ heuristic_processing
                      │
                      ├─(clipping / silence > 40% / peak > -1dB)──→ structured_compilation
                      │                                                       │
                      └─(clean)──────────────────────────────────→ lightweight_compilation
                                                                             │
                                                                    classify_context  ← D1
                                                                             │
                                                                        synthesize    ← D2
                                                                             │
                                                                            END
```

---

## Step 3 — Remove Old Code → `agent.py`

```
DELETE: phase_d_llm_synthesis()
DELETE: graph.add_node("llm_synthesis", phase_d_llm_synthesis)
DELETE: graph.add_edge("structured_compilation", "llm_synthesis")
RENAME: all references to "llm_synthesis" in tests/logs → "synthesize"
```

---

## Execution Checklist

| # | Task | File | Done |
|---|------|------|------|
| 1.1 | Add `RecordingContext` model | `schema.py` | ☑ |
| 1.2 | Add `recording_context` to `AudioAnalysisReport` | `schema.py` | ☑ |
| 1.3 | Add `recording_context` to `AudioAgentState` | `agent.py` | ☑ |
| 2.1 | Add `phase_d1_classify_context()` with fallback | `agent.py` | ☑ |
| 2.2 | Rename + rewrite `phase_d_llm_synthesis` → `phase_d2_synthesize()` | `agent.py` | ☑ |
| 2.3 | Add `context` block to D2 user prompt | `agent.py` | ☑ |
| 3.1 | Replace `llm_synthesis` node with `classify_context` + `synthesize` | `agent.py` | ☑ |
| 3.2 | Rewire both compilation paths through D1 → D2 | `agent.py` | ☑ |
| 3.3 | Delete old `phase_d_llm_synthesis` | `agent.py` | ☑ |

---

> **Cost impact:** 2 LLM calls per file instead of 1. At batch scale, acceptable
> because D1 is a small-input call (metadata only, ~100 tokens input).
> D1 never blocks D2 — fallback to `unknown` context keeps the pipeline running.