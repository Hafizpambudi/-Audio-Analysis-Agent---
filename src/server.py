import subprocess
import json
import re
import os
import shutil
import math
from pathlib import Path
from typing import List, Optional
from fastmcp import FastMCP
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

mcp = FastMCP("audio-analysis")


def get_ffmpeg_bin() -> tuple:
    """Get path to ffmpeg and ffprobe binaries.
    Returns tuple of (ffmpeg_path, ffprobe_path).
    Checks FFMPEG_PATH env var first, then ffmpegBase directory, then falls back to system PATH.
    """
    custom_path = os.environ.get("FFMPEG_PATH")
    if custom_path:
        custom_path = custom_path.strip().strip('"\'')
        if '\x0c' in custom_path or '\x08' in custom_path:
            custom_path = custom_path.encode('utf-8').decode('unicode_escape')
        ffmpeg_exe = Path(custom_path) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffprobe_exe = Path(custom_path) / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if ffmpeg_exe.exists():
            return (str(ffmpeg_exe), str(ffprobe_exe) if ffprobe_exe.exists() else None)
    
    # Check for bundled ffmpegBase directory
    if os.name == "nt":
        base_dir = Path(__file__).parent.parent / "ffmpegBase" / "bin"
        ffmpeg_exe = base_dir / "ffmpeg.exe"
        ffprobe_exe = base_dir / "ffprobe.exe"
        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            return (str(ffmpeg_exe), str(ffprobe_exe))
    
    return ("ffmpeg", "ffprobe")


def check_ffmpeg_available() -> bool:
    ffmpeg_cmd, ffprobe_cmd = get_ffmpeg_bin()
    return shutil.which(ffmpeg_cmd) is not None if ffmpeg_cmd else False


def _parse_db(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    val = m.group(1).strip()
    return float("-inf") if val == "-inf" else float("inf") if val == "inf" else float(val)


def run_ffmpeg_command(args: List[str]) -> str:
    ffmpeg_cmd, ffprobe_cmd = get_ffmpeg_bin()
    
    # Use local binary if available, otherwise check system
    if ffmpeg_cmd != "ffmpeg":
        # Custom path - use directly (already validated to exist)
        cmd = [ffmpeg_cmd, "-hide_banner", "-y"] + args
    else:
        # System path
        resolved = shutil.which("ffmpeg")
        if not resolved:
            raise RuntimeError("ffmpeg not found. Install from ffmpeg.org or set FFMPEG_PATH in .env")
        cmd = [resolved, "-hide_banner", "-y"] + args
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return result.stderr


def run_ffprobe_command(args: List[str]) -> dict:
    ffmpeg_cmd, ffprobe_cmd = get_ffmpeg_bin()
    
    if ffprobe_cmd and ffprobe_cmd != "ffprobe" and Path(ffprobe_cmd).exists():
        # Custom path
        cmd = [ffprobe_cmd, "-v", "quiet", "-print_format", "json"] + args
    elif ffprobe_cmd == "ffprobe":
        # System path
        resolved = shutil.which("ffprobe")
        if not resolved:
            raise RuntimeError("ffprobe not found. Install from ffmpeg.org or set FFMPEG_PATH in .env")
        cmd = [resolved, "-v", "quiet", "-print_format", "json"] + args
    else:
        raise RuntimeError("ffprobe not found. Install from ffmpeg.org or set FFMPEG_PATH in .env")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


@mcp.tool()
def inspect_metadata(file_path: str) -> dict:
    if not check_ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg and ensure it's in your PATH. See README.md for installation instructions.")
    
    args = [
        "-show_format",
        "-show_streams",
        "-select_streams", "a:0",
        file_path
    ]
    data = run_ffprobe_command(args)
    
    format_info = data.get("format", {})
    stream = data["streams"][0] if data.get("streams") else {}
    
    return {
        "file_name": format_info.get("filename", "").split("/")[-1].split("\\")[-1],
        "duration_seconds": float(format_info.get("duration", 0)),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "bit_rate": int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
        "channels": int(stream.get("channels", 1))
    }


@mcp.tool()
def detect_silence(file_path: str, noise_threshold_db: float = -50.0, min_duration: float = 0.5) -> List[dict]:
    if not check_ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg and ensure it's in your PATH.")
    
    args = [
        "-i", file_path,
        "-af", f"silencedetect=noise={noise_threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    
    output = run_ffmpeg_command(args)
    
    # Parse silence segments - format is multi-line entries
    start_pattern = r"silence_start: ([0-9.]+)"
    end_pattern = r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)"
    
    starts = re.findall(start_pattern, output)
    ends = re.findall(end_pattern, output)
    
    segments = []
    for i, start in enumerate(starts):
        if i < len(ends):
            end, duration = ends[i]
            segments.append({
                "start_time": float(start),
                "end_time": float(end),
                "duration": float(duration)
            })
    
    return segments


@mcp.tool()
def analyze_amplitude_stats(file_path: str) -> dict:
    if not check_ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg and ensure it's in your PATH.")
    
    args = [
        "-i", file_path,
        "-af", "astats=measure_perchannel=all",
        "-f", "null", "-",
    ]
    
    output = run_ffmpeg_command(args)
    
    peak_db = _parse_db(r"Peak level dB:\s*([-\d.inf]+)", output)
    flat_factor = _parse_db(r"Flat factor:\s*([-\d.inf]+)", output)
    rms_db = _parse_db(r"RMS level dB:\s*([-\d.inf]+)", output)
    dc_linear = _parse_db(r"DC offset:\s*([-\d.inf]+)", output) or 0.0
    
    dc_offset_db = 20 * math.log10(abs(dc_linear) + 1e-9)
    
    return {
        "peak_level_db": peak_db,
        "flat_factor": flat_factor or 0.0,
        "rms_level_db": rms_db or -20.0,
        "dc_offset_db": round(dc_offset_db, 2)
    }


@mcp.tool()
def detect_clipping(file_path: str, threshold_db: float = 0.1) -> List[dict]:
    """Detect potential clipping.
    
    Clips are detected when peak level is near 0dBFS AND flat_factor > 0
    (indicating samples at maximum value).
    """
    if not check_ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg and ensure it's in your PATH.")
    
    args = [
        "-i", file_path,
        "-af", f"astats=measure_perchannel=all",
        "-f", "null", "-",
    ]
    
    output = run_ffmpeg_command(args)
    
    peak_db = _parse_db(r"Peak level dB:\s*([-\d.inf]+)", output)
    flat_factor = _parse_db(r"Flat factor:\s*([-\d.inf]+)", output)
    
    segments = []
    if peak_db is not None:
        if abs(peak_db) <= threshold_db:
            if flat_factor is not None and flat_factor > 0:
                segments.append({
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "peak_dB": peak_db,
                    "confidence": "high" if flat_factor > 0.1 else "potential"
                })
    
    return segments


if __name__ == "__main__":
    mcp.run()