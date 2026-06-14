import json
import os
import logging
import re as _re
from typing import List, Literal, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from .schema import (
    AudioAnalysisReport, AudioMetadata, AmplitudeStats,
    AudioQualityMetrics, SilenceSegment, ClippingSegment, ProcessingIssue, MitigationStrategy, ExecutiveSummary
)
from .server import inspect_metadata, analyze_amplitude_stats, detect_silence, detect_clipping

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_llm: Optional[ChatOpenAI] = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = create_llm_client()
    return _llm


class AudioAgentState(TypedDict, total=False):
    file_path: str
    current_phase: str
    metadata: dict
    amplitude_stats: dict
    silence_segments: List[dict]
    clipping_segments: List[dict]
    silence_ratio: float
    report: Optional[AudioAnalysisReport]
    error: Optional[str]


def create_llm_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    
    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3-opus")
    logger.info(f"Creating LLM client with model: {model}")
    
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.1
    )


def phase_a_signal_diagnostics(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase A: Starting signal diagnostics")
    
    file_path = state.get("file_path")
    if not file_path:
        logger.error("Phase A: No file_path in state")
        return {"error": "No file_path provided", "file_path": None}
    
    try:
        metadata = inspect_metadata(file_path)
        logger.info(f"Phase A: Extracted metadata - duration: {metadata.get('duration_seconds', 0)}s, sample_rate: {metadata.get('sample_rate', 0)}Hz")
    except Exception as e:
        logger.error(f"Phase A: Metadata inspection failed: {e}")
        return {"error": f"Metadata inspection failed: {str(e)}", "file_path": file_path}
    
    amplitude_stats = {}
    if metadata["duration_seconds"] > 0:
        try:
            amplitude_stats = analyze_amplitude_stats(file_path)
            logger.info(f"Phase A: Amplitude stats - peak: {amplitude_stats.get('peak_level_db', 'N/A')}dB, RMS: {amplitude_stats.get('rms_level_db', 'N/A')}dB")
        except Exception as e:
            logger.error(f"Phase A: Amplitude analysis failed: {e}")
            amplitude_stats = {"error": str(e)}
    
    return {
        **state,
        "metadata": metadata,
        "amplitude_stats": amplitude_stats,
        "current_phase": "Phase A Complete"
    }


def phase_b_heuristic_processing(state: AudioAgentState) -> AudioAgentState:
    logger.info(f"Phase B: Starting heuristic processing - phase: {state.get('current_phase', 'unknown')}")
    
    if "error" in state:
        logger.warning("Phase B: Skipping due to previous error")
        return state
    
    silence_segments = []
    clipping_segments = []
    file_path = state.get("file_path")
    
    if file_path:
        try:
            silence_segments = detect_silence(file_path)
            logger.info(f"Phase B: Found {len(silence_segments)} silence segments")
        except Exception as e:
            logger.warning(f"Phase B: Silence detection failed: {e}")
        
        try:
            clipping_segments = detect_clipping(file_path)
            logger.info(f"Phase B: Found {len(clipping_segments)} clipping segments")
        except Exception as e:
            logger.warning(f"Phase B: Clipping detection failed: {e}")
    
    metadata = state.get("metadata", {})
    duration = metadata.get("duration_seconds", 0)
    silence_ratio = sum(seg.get("duration", 0) for seg in silence_segments) / duration if duration > 0 else 0
    
    logger.info(f"Phase B: Silence ratio: {silence_ratio:.2%}")
    
    return {
        **state,
        "silence_segments": silence_segments,
        "clipping_segments": clipping_segments,
        "silence_ratio": silence_ratio,
        "current_phase": "Phase B Complete"
    }


def phase_c_structured_compilation(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase C: Starting structured compilation")
    
    if "error" in state:
        return state
    
    issues: List[ProcessingIssue] = []
    
    amplitude_stats = state.get("amplitude_stats", {})
    rms_level = amplitude_stats.get("rms_level_db", -20)
    
    if rms_level < -30.0:
        issues.append(ProcessingIssue(
            issue_type="low_volume",
            description=f"Low volume detected: RMS level at {rms_level:.1f}dB. Recommend checking input gain levels.",
            severity="medium"
        ))
        logger.info(f"Phase C: Low volume issue detected ({rms_level:.1f}dB)")
    
    dc_offset_db = amplitude_stats.get("dc_offset_db", 0)
    if dc_offset_db > -40.0:
        issues.append(ProcessingIssue(
            issue_type="dc_offset",
            description=f"DC offset detected: {dc_offset_db:.1f}dB. Apply high-pass filter.",
            severity="high"
        ))
        logger.info(f"Phase C: DC offset issue detected ({dc_offset_db:.2f}dB)")
    
    for seg in state.get("silence_segments", []):
        if seg.get("duration", 0) > 5.0:
            issues.append(ProcessingIssue(
                issue_type="long_silence",
                description=f"Long silence detected between {seg.get('start_time', 0):.1f}s and {seg.get('end_time', 0):.1f}s",
                start_time=seg.get("start_time"),
                end_time=seg.get("end_time"),
                severity="medium"
            ))
    
    for seg in state.get("clipping_segments", []):
        if not state.get("skip_deep"):
            issues.append(ProcessingIssue(
                issue_type="clipping",
                description=f"Clipping detected. Recommend de-clipping or gain-attenuation preprocessing.",
                severity="high"
            ))
    
    quality_metrics = AudioQualityMetrics(
        silence_ratio=state.get("silence_ratio", 0),
        clipping_detected=len(state.get("clipping_segments", [])) > 0,
        clipping_segments=[ClippingSegment(**seg) for seg in state.get("clipping_segments", [])],
        avg_volume_db=amplitude_stats.get("rms_level_db", -20),
        dc_offset_db=dc_offset_db
    )
    
    metadata = state.get("metadata", {})
    report = AudioAnalysisReport(
        file_name=metadata.get("file_name", "unknown"),
        duration_seconds=metadata.get("duration_seconds", 0),
        metadata=AudioMetadata(**metadata),
        amplitude_stats=AmplitudeStats(**amplitude_stats),
        silence_segments=[SilenceSegment(**seg) for seg in state.get("silence_segments", [])],
        audio_quality=quality_metrics,
        issues=issues
    )
    
    logger.info(f"Phase C: Compiled report with {len(issues)} issues identified")
    
    return {
        **state,
        "report": report,
        "current_phase": "Phase C Complete"
    }


def phase_c_lightweight_compilation(state: AudioAgentState) -> AudioAgentState:
    """For clean audio: compile report without deep issue analysis. Skip to LLM fast-path."""
    return phase_c_structured_compilation({**state, "skip_deep": True})


def phase_critical_report(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase C: Critical quality report")
    
    sr = state.get("metadata", {}).get("sample_rate", 0)
    report = AudioAnalysisReport(
        file_name=state.get("metadata", {}).get("file_name", "unknown"),
        duration_seconds=state.get("metadata", {}).get("duration_seconds", 0),
        metadata=AudioMetadata(**state.get("metadata", {
            "file_name": "unknown",
            "duration_seconds": 0,
            "sample_rate": 0,
            "bit_rate": None,
            "channels": 1
        })),
        amplitude_stats=AmplitudeStats(
            peak_level_db=None,
            flat_factor=0.0,
            rms_level_db=-20.0,
            dc_offset_db=0.0
        ),
        audio_quality=AudioQualityMetrics(
            silence_ratio=0,
            clipping_detected=False,
            avg_volume_db=-99,
            dc_offset_db=0
        ),
        issues=[ProcessingIssue(
            issue_type="critical_sample_rate",
            description=f"Sample rate {sr}Hz is below 8kHz minimum. File is not viable for ASR.",
            severity="critical"
        )],
        executive_summary=ExecutiveSummary(
            overall_quality="unusable",
            asr_viability="not_viable",
            transcription_viable=False,
            summary=f"Sample rate {sr}Hz destroys speech frequencies. Do not transcribe.",
            blocking_issues=[f"Sample rate {sr}Hz < 8kHz minimum"]
        )
    )
    return {**state, "report": report}


def phase_error_handler(state: AudioAgentState) -> AudioAgentState:
    logger.error(f"Error handler reached: {state.get('error')}")
    return state


SYSTEM_PROMPT = """You are a forensic audio quality analyst for legal transcription systems.

REFERENCE STANDARDS:
- EBU R128: target -23 LUFS integrated loudness for speech (ITU-R BS.1770)
- ASR minimum sample rate: 16kHz (wav2vec2, Whisper, Conformer training spec)
- Absolute floor sample rate: 8kHz (ITU-T G.711 telephone — high WER expected)
- Clipping: any true peak ≥ -1.0 dBFS with flat_factor > 0.1 indicates saturation distortion

INSTRUCTIONS:
- Apply the above standards when assigning quality grades
- For silence ratio: reason about it in context of deposition length, do not apply a fixed threshold
- For DC offset: flag if linear ratio > 0.01 (>1%), recommend high-pass filter
- Justify your overall_quality grade with specific reference to the signal facts provided

OUTPUT RULES:
- overall_quality  : exactly one of [excellent, good, acceptable, poor, unusable]
- asr_viability    : exactly one of [high, medium, low, not_viable]
- transcription_viable: true only if overall_quality is excellent/good/acceptable
- priority in mitigation_strategies: exactly one of [immediate, before_transcription, optional]
- blocking_issues  : list issues that PREVENT transcription if not fixed (empty list if none)

Return ONLY valid JSON. No markdown fences. No explanation outside the JSON object."""


def _extract_json(text: str) -> dict:
    cleaned = _re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    return json.loads(cleaned)


def build_user_prompt(report_dict: dict) -> str:
    metadata = report_dict.get("metadata", {})
    quality = report_dict.get("audio_quality", {})
    issues = report_dict.get("issues", [])
    
    return f"""Analyze this court deposition audio report and return structured JSON.

SIGNAL FACTS:
- Duration       : {report_dict.get('duration_seconds', 0):.1f}s
- Sample Rate    : {metadata.get('sample_rate', 0)} Hz
- Channels       : {metadata.get('channels', 1)}
- Bit Rate       : {metadata.get('bit_rate', 'unknown')} bps
- Peak Level     : {report_dict.get('amplitude_stats', {}).get('peak_level_db', 'N/A')} dBFS
- RMS Level      : {report_dict.get('amplitude_stats', {}).get('rms_level_db', 'N/A')} dB
- DC Offset      : {report_dict.get('amplitude_stats', {}).get('dc_offset_db', 'N/A')} dB
- Silence Ratio  : {quality.get('silence_ratio', 0):.1%}
- Clipping       : {quality.get('clipping_detected', False)}
- Issues Found   : {len(issues)}

DETECTED ISSUES:
{json.dumps(issues, indent=2) if issues else "None"}

Respond with this exact JSON shape:
{{
  "executive_summary": {{
    "overall_quality": "<one of: excellent|good|acceptable|poor|unusable>",
    "asr_viability": "<one of: high|medium|low|not_viable>",
    "transcription_viable": <true|false>,
    "summary": "<2-3 sentences. Be specific about what will hurt transcription accuracy.>",
    "blocking_issues": ["<issue>", ...]
  }},
  "mitigation_strategies": [
    {{
      "issue_type": "<type>",
      "recommended_action": "<concrete ffmpeg command or tool recommendation>",
      "priority": "<one of: immediate|before_transcription|optional>"
    }}
  ]
}}"""


def phase_d_llm_synthesis(state: AudioAgentState) -> AudioAgentState:
    logger.info("Phase D: Starting LLM synthesis")
    
    if "error" in state:
        return state
    
    llm = _get_llm()
    
    report_dict = state.get("report", {}).model_dump() if state.get("report") else {}
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(report_dict))
    ]
    
    MAX_RETRIES = 2
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            synthesis = _extract_json(response.content)
            
            exec_summary = ExecutiveSummary(**synthesis["executive_summary"])
            mitigations = [MitigationStrategy(**m) for m in synthesis.get("mitigation_strategies", [])]
            
            state["report"].executive_summary = exec_summary
            state["report"].mitigation_strategies = mitigations
            logger.info(f"Phase D: LLM synthesis OK (attempt {attempt + 1})")
            break
        
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            logger.warning(f"Phase D: LLM parse failed attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES:
                state["report"].executive_summary = ExecutiveSummary(
                    overall_quality="acceptable",
                    asr_viability="medium",
                    transcription_viable=True,
                    summary="Automated LLM synthesis failed. Manual review recommended.",
                    blocking_issues=[]
                )
    
    return {
        **state,
        "current_phase": "Phase D Complete"
    }


def route_after_diagnostics(state: AudioAgentState) -> str:
    if state.get("error"):
        return "error_handler"
    metadata = state.get("metadata", {})
    if metadata.get("duration_seconds", 0) == 0:
        return "error_handler"
    if metadata.get("sample_rate", 0) < 8000:
        return "critical_quality_report"
    return "heuristic_processing"


def route_after_heuristics(state: AudioAgentState) -> str:
    has_clipping = len(state.get("clipping_segments", [])) > 0
    silence_ratio = state.get("silence_ratio", 0.0)
    amplitude = state.get("amplitude_stats", {})
    peak_db = amplitude.get("peak_level_db") or -99.0
    
    if has_clipping or silence_ratio > 0.40 or peak_db > -1.0:
        return "structured_compilation"
    
    return "lightweight_compilation"


def create_agent_graph():
    graph = StateGraph(AudioAgentState)
    
    graph.add_node("signal_diagnostics", phase_a_signal_diagnostics)
    graph.add_node("heuristic_processing", phase_b_heuristic_processing)
    graph.add_node("structured_compilation", phase_c_structured_compilation)
    graph.add_node("lightweight_compilation", phase_c_lightweight_compilation)
    graph.add_node("critical_quality_report", phase_critical_report)
    graph.add_node("llm_synthesis", phase_d_llm_synthesis)
    graph.add_node("error_handler", phase_error_handler)
    
    graph.set_entry_point("signal_diagnostics")
    
    graph.add_conditional_edges(
        "signal_diagnostics",
        route_after_diagnostics,
        {
            "heuristic_processing": "heuristic_processing",
            "critical_quality_report": "critical_quality_report",
            "error_handler": "error_handler",
        }
    )
    
    graph.add_conditional_edges(
        "heuristic_processing",
        route_after_heuristics,
        {
            "structured_compilation": "structured_compilation",
            "lightweight_compilation": "lightweight_compilation",
        }
    )
    
    graph.add_edge("structured_compilation", "llm_synthesis")
    graph.add_edge("lightweight_compilation", "llm_synthesis")
    graph.add_edge("critical_quality_report", END)
    graph.add_edge("llm_synthesis", END)
    graph.add_edge("error_handler", END)
    
    return graph.compile()


def create_basic_agent_graph():
    graph = StateGraph(AudioAgentState)
    
    graph.add_node("signal_diagnostics", phase_a_signal_diagnostics)
    graph.add_node("heuristic_processing", phase_b_heuristic_processing)
    graph.add_node("structured_compilation", phase_c_structured_compilation)
    
    graph.set_entry_point("signal_diagnostics")
    graph.add_edge("signal_diagnostics", "heuristic_processing")
    graph.add_edge("heuristic_processing", "structured_compilation")
    graph.add_edge("structured_compilation", END)
    
    return graph.compile()


def analyze_audio_file(file_path: str, skip_llms: bool = False) -> AudioAnalysisReport:
    initial_state: AudioAgentState = {"file_path": file_path, "current_phase": "Starting"}
    
    logger.info(f"Starting audio analysis for: {file_path}")
    
    agent = create_basic_agent_graph() if skip_llms else create_agent_graph()
    
    result = agent.invoke(initial_state)
    
    if "error" in result:
        raise RuntimeError(result["error"])
    
    report = result.get("report")
    if not report:
        report = AudioAnalysisReport(
            file_name=file_path.split("/")[-1] if file_path else "unknown",
            duration_seconds=0,
            metadata=AudioMetadata(
                file_name="unknown",
                duration_seconds=0,
                sample_rate=0,
                bit_rate=None,
                channels=1
            ),
            amplitude_stats=AmplitudeStats(
                peak_level_db=None,
                flat_factor=0,
                rms_level_db=-20,
                dc_offset_db=0
            ),
            audio_quality=AudioQualityMetrics(
                silence_ratio=0,
                clipping_detected=False,
                avg_volume_db=-20,
                dc_offset_db=0
            )
        )
    
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze audio file quality")
    parser.add_argument("audio_file", help="Path to the audio file to analyze")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM synthesis phase")
    parser.add_argument("--with-save", action="store_true", help="Save JSON analysis output to file")
    
    args = parser.parse_args()
    
    file_path = args.audio_file
    skip_llm = args.skip_llm
    
    if skip_llm:
        logger.info("Running in skip-LLM mode (no OPENROUTER_API_KEY required)")
    
    report = analyze_audio_file(file_path, skip_llms=skip_llm)
    
    if skip_llm:
        report.executive_summary = ExecutiveSummary(
            overall_quality="acceptable",
            asr_viability="medium",
            transcription_viable=False,
            summary="LLM synthesis was skipped. Manual review recommended.",
            blocking_issues=["LLM synthesis was skipped - manual review recommended"]
        )
    
    json_output = report.to_json()
    print(json_output)
    
    if args.with_save:
        output_path = os.path.join(
            os.path.dirname(file_path),
            os.path.splitext(os.path.basename(file_path))[0] + "_analysis.json"
        )
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)
            logger.info(f"Analysis saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis file: {e}")