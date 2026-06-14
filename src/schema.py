import json
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class AudioMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    file_name: str
    duration_seconds: float
    sample_rate: int
    bit_rate: Optional[int]
    channels: int


class AmplitudeStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    peak_level_db: Optional[float] = None
    flat_factor: float = 0.0
    rms_level_db: float
    dc_offset_db: float


class SilenceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    start_time: float
    end_time: float
    duration: float


class ClippingSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    start_time: float
    end_time: float
    peak_dB: float
    confidence: Literal["high", "potential"] = "potential"


class AudioQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    silence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    clipping_detected: bool
    clipping_segments: List[ClippingSegment] = Field(default_factory=list)
    avg_volume_db: float
    noise_floor_db: Optional[float] = None
    dc_offset_db: float


class ProcessingIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    issue_type: str
    description: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class MitigationStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    issue_type: str
    recommended_action: str
    priority: Literal["immediate", "before_transcription", "optional"]


class ExecutiveSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    overall_quality: Literal["excellent", "good", "acceptable", "poor", "unusable"]
    asr_viability: Literal["high", "medium", "low", "not_viable"]
    transcription_viable: bool
    summary: str
    blocking_issues: List[str] = Field(default_factory=list)


class AudioAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
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