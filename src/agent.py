import json
import os
import logging
from typing import List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .schema import (
    AudioAnalysisReport, AudioMetadata, AmplitudeStats,
    AudioQualityMetrics, SilenceSegment, ProcessingIssue, MitigationStrategy, ExecutiveSummary
)
from .server import inspect_metadata, analyze_amplitude_stats, detect_silence, detect_clipping

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def phase_a_signal_diagnostics(state: dict) -> dict:
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


def phase_b_heuristic_processing(state: dict) -> dict:
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


def phase_c_structured_compilation(state: dict) -> dict:
    logger.info("Phase C: Starting structured compilation")
    
    if "error" in state:
        return state
    
    issues: List[ProcessingIssue] = []
    
    amplitude_stats = state.get("amplitude_stats", {})
    dc_offset = amplitude_stats.get("dc_offset_db", 0)
    if abs(dc_offset) > 1.0:
        issues.append(ProcessingIssue(
            issue_type="dc_offset",
            description=f"DC offset detected: {dc_offset:.2f}dB. Recommend high-pass filter preprocessing.",
            severity="high"
        ))
        logger.info(f"Phase C: DC offset issue detected ({dc_offset:.2f}dB)")
    
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
        issues.append(ProcessingIssue(
            issue_type="clipping",
            description=f"Clipping detected. Recommend de-clipping or gain-attenuation preprocessing.",
            severity="high"
        ))
    
    quality_metrics = AudioQualityMetrics(
        silence_ratio=state.get("silence_ratio", 0),
        clipping_detected=len(state.get("clipping_segments", [])) > 0,
        clipping_segments=state.get("clipping_segments", []),
        avg_volume_db=amplitude_stats.get("rms_level_db", -20),
        dc_offset_db=dc_offset
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


def phase_d_llm_synthesis(state: dict) -> dict:
    logger.info("Phase D: Starting LLM synthesis")
    
    if "error" in state:
        return state
    
    llm = create_llm_client()
    
    report_dict = state.get("report", {}).model_dump() if state.get("report") else {}
    
    system_prompt = """You are an audio quality analyst for court deposition recordings.
    Analyze the structured audio analysis data and provide:
    1. An executive summary assessing overall quality and ASR viability
    2. A mitigation strategy matrix with actionable recommendations
    
    Be concise but thorough. Focus on practical advice for improving speech recognition quality.
    Output in JSON format with 'executive_summary' and 'mitigation_strategies' keys."""
    
    user_prompt = f"""Analyze this audio analysis report:
    {json.dumps(report_dict, indent=2)}
    
    Return only JSON with:
    - executive_summary: {{overall_quality, asr_viability, summary}}
    - mitigation_strategies: [{{issue_type, recommended_action, priority}}]"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        synthesis = json.loads(response.content)
        
        exec_summary = ExecutiveSummary(
            overall_quality=synthesis["executive_summary"]["overall_quality"],
            asr_viability=synthesis["executive_summary"]["asr_viability"],
            summary=synthesis["executive_summary"]["summary"]
        )
        
        mitigations = [
            MitigationStrategy(**m) for m in synthesis.get("mitigation_strategies", [])
        ]
        
        if state.get("report"):
            state["report"].executive_summary = exec_summary
            state["report"].mitigation_strategies = mitigations
            logger.info("Phase D: LLM synthesis completed successfully")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Phase D: LLM response parsing failed: {e}")
        if state.get("report"):
            state["report"].executive_summary = ExecutiveSummary(
                overall_quality="unknown",
                asr_viability="requires_manual_review",
                summary=f"Automatic analysis failed: {str(e)}"
            )
    
    return {
        **state,
        "current_phase": "Phase D Complete"
    }


def create_agent_graph():
    graph = StateGraph(dict)
    
    graph.add_node("signal_diagnostics", phase_a_signal_diagnostics)
    graph.add_node("heuristic_processing", phase_b_heuristic_processing)
    graph.add_node("structured_compilation", phase_c_structured_compilation)
    graph.add_node("llm_synthesis", phase_d_llm_synthesis)
    
    graph.set_entry_point("signal_diagnostics")
    graph.add_edge("signal_diagnostics", "heuristic_processing")
    graph.add_edge("heuristic_processing", "structured_compilation")
    graph.add_edge("structured_compilation", "llm_synthesis")
    graph.add_edge("llm_synthesis", END)
    
    return graph.compile()


def create_basic_agent_graph():
    graph = StateGraph(dict)
    
    graph.add_node("signal_diagnostics", phase_a_signal_diagnostics)
    graph.add_node("heuristic_processing", phase_b_heuristic_processing)
    graph.add_node("structured_compilation", phase_c_structured_compilation)
    
    graph.set_entry_point("signal_diagnostics")
    graph.add_edge("signal_diagnostics", "heuristic_processing")
    graph.add_edge("heuristic_processing", "structured_compilation")
    graph.add_edge("structured_compilation", END)
    
    return graph.compile()


def analyze_audio_file(file_path: str, skip_llms: bool = False) -> AudioAnalysisReport:
    initial_state = {"file_path": file_path, "current_phase": "Starting"}
    
    logger.info(f"Starting audio analysis for: {file_path}")
    
    agent = create_basic_agent_graph() if skip_llms else create_agent_graph()
    
    result = agent.invoke(initial_state)
    
    if "error" in result:
        raise RuntimeError(result["error"])
    
    report = result.get("report")
    if not report:
        # Create a basic report if something went wrong
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
            overall_quality="not_analyzed",
            asr_viability="requires_llm",
            summary="LLM synthesis was skipped"
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