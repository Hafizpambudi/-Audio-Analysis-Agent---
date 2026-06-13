import json
from typing import List, Optional
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    file_name: str
    duration_seconds: float
    sample_rate: int
    bit_rate: Optional[int]
    channels: int


class AmplitudeStats(BaseModel):
    peak_level_db: Optional[float] = None
    flat_factor: float = 0.0
    rms_level_db: float
    dc_offset_db: float


class SilenceSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float


class ClippingSegment(BaseModel):
    start_time: float
    end_time: float
    peak_dB: float


class AudioQualityMetrics(BaseModel):
    silence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    clipping_detected: bool
    clipping_segments: List[ClippingSegment] = Field(default_factory=list)
    avg_volume_db: float
    noise_floor_db: Optional[float] = None
    dc_offset_db: float


class ProcessingIssue(BaseModel):
    issue_type: str
    description: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    severity: str = Field(default="medium", pattern="^(low|medium|high|critical)$")


class MitigationStrategy(BaseModel):
    issue_type: str
    recommended_action: str
    priority: str = Field(pattern="^(low|medium|high|critical)$")


class ExecutiveSummary(BaseModel):
    overall_quality: str
    asr_viability: str
    summary: str


class AudioAnalysisReport(BaseModel):
    file_name: str
    duration_seconds: float
    metadata: AudioMetadata
    amplitude_stats: AmplitudeStats
    silence_segments: List[SilenceSegment] = Field(default_factory=list)
    audio_quality: AudioQualityMetrics
    issues: List[ProcessingIssue] = Field(default_factory=list)
    executive_summary: Optional[ExecutiveSummary] = None
    mitigation_strategies: List[MitigationStrategy] = Field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, default=str)