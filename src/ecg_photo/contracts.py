import hashlib
import json
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"


class LeadStatus(StrEnum):
    confirmed = "confirmed"
    proposed = "proposed"
    unknown = "unknown"


class RepresentationKind(StrEnum):
    rhythm_segment = "rhythm_segment"
    representative_beat = "representative_beat"
    unknown = "unknown"


class TimeRelation(StrEnum):
    simultaneous = "simultaneous"
    sequential = "sequential"
    unknown = "unknown"


class ScaleStatus(StrEnum):
    confirmed = "confirmed"
    proposed = "proposed"
    manual = "manual"
    review_required = "review_required"
    unknown = "unknown"


class MeasurementStatus(StrEnum):
    available = "available"
    unavailable = "unavailable"
    not_applicable = "not_applicable"
    review_required = "review_required"


class QualityLabel(StrEnum):
    good = "good"
    acceptable = "acceptable"
    insufficient = "insufficient"


class HashStatus(StrEnum):
    computed = "computed"
    example_only = "example_only"


class SourceKind(StrEnum):
    image = "image"
    pdf = "pdf"
    native_signal = "native_signal"


class ScopeStatus(StrEnum):
    supported = "supported"
    unsupported = "unsupported"
    unknown = "unknown"


class TemporalTraceUnits(StrEnum):
    px = "px"
    arbitrary = "arbitrary"


class ReasonCode(StrEnum):
    TIME_SCALE_UNKNOWN = "TIME_SCALE_UNKNOWN"
    GAIN_UNKNOWN = "GAIN_UNKNOWN"
    LEAD_ID_UNCERTAIN = "LEAD_ID_UNCERTAIN"
    LAYOUT_UNSUPPORTED = "LAYOUT_UNSUPPORTED"
    LOCAL_WARP_UNCERTAIN = "LOCAL_WARP_UNCERTAIN"
    TRACE_OCCLUDED = "TRACE_OCCLUDED"
    TRACE_OVERLAP = "TRACE_OVERLAP"
    SEGMENT_TRUNCATED = "SEGMENT_TRUNCATED"
    TOO_FEW_BEATS = "TOO_FEW_BEATS"
    P_ONSET_UNCERTAIN = "P_ONSET_UNCERTAIN"
    AV_ASSOCIATION_UNCERTAIN = "AV_ASSOCIATION_UNCERTAIN"
    QRS_BOUNDARY_UNCERTAIN = "QRS_BOUNDARY_UNCERTAIN"
    T_END_UNCERTAIN = "T_END_UNCERTAIN"
    T_U_FUSION = "T_U_FUSION"
    WIDE_QRS_CONTEXT = "WIDE_QRS_CONTEXT"
    RR_STRATEGY_UNVALIDATED = "RR_STRATEGY_UNVALIDATED"
    PACED_RHYTHM_UNVALIDATED = "PACED_RHYTHM_UNVALIDATED"
    POPULATION_UNSUPPORTED = "POPULATION_UNSUPPORTED"
    INPUT_REVISION_STALE = "INPUT_REVISION_STALE"
    CALIBRATION_CONFLICT = "CALIBRATION_CONFLICT"
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    PIXEL_LIMIT = "PIXEL_LIMIT"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    PAGE_LIMIT = "PAGE_LIMIT"
    FILE_SIZE_LIMIT = "FILE_SIZE_LIMIT"


REASON_TEXT: dict[ReasonCode, tuple[str, str]] = {
    ReasonCode.TIME_SCALE_UNKNOWN: (
        "Escala temporal desconocida",
        "speed_mm_s ausente o no confirmado; no hay rejilla temporal",
    ),
    ReasonCode.GAIN_UNKNOWN: (
        "Ganancia desconocida",
        "gain_mm_mV ausente o no confirmado; voltajes absolutos bloqueados",
    ),
    ReasonCode.LEAD_ID_UNCERTAIN: (
        "Derivación no identificada",
        "lead_status no confirmado; la etiqueta puede ser incorrecta",
    ),
    ReasonCode.LAYOUT_UNSUPPORTED: (
        "Formato no soportado",
        "la plantilla de la página no está cubierta por el analizador",
    ),
    ReasonCode.LOCAL_WARP_UNCERTAIN: (
        "Deformación local incierta",
        "la escala puede variar espacialmente dentro de la región",
    ),
    ReasonCode.TRACE_OCCLUDED: (
        "Trazado ocluido",
        "una región del trazado no tiene soporte visible",
    ),
    ReasonCode.TRACE_OVERLAP: (
        "Cruce entre trazados",
        "dos trazados se superponen y la atribución es ambigua",
    ),
    ReasonCode.SEGMENT_TRUNCATED: (
        "Segmento truncado",
        "el fenómeno medido termina fuera del recorte",
    ),
    ReasonCode.TOO_FEW_BEATS: (
        "Latidos insuficientes",
        "no hay suficientes QRS para el cálculo solicitado",
    ),
    ReasonCode.P_ONSET_UNCERTAIN: (
        "Inicio de P incierto",
        "el límite inicial de la onda P no está delineado con soporte",
    ),
    ReasonCode.AV_ASSOCIATION_UNCERTAIN: (
        "Asociación AV incierta",
        "la relación P-QRS no está demostrada",
    ),
    ReasonCode.QRS_BOUNDARY_UNCERTAIN: (
        "Límite QRS incierto",
        "inicio o fin del QRS sin soporte suficiente",
    ),
    ReasonCode.T_END_UNCERTAIN: (
        "Final de T incierto",
        "el final de la onda T no está delineado con soporte",
    ),
    ReasonCode.T_U_FUSION: (
        "Fusión T/U",
        "ondas T y U no separables con el método vigente",
    ),
    ReasonCode.WIDE_QRS_CONTEXT: (
        "Contexto de QRS ancho",
        "la medida es descriptiva; requiere revisión de repolarización",
    ),
    ReasonCode.RR_STRATEGY_UNVALIDATED: (
        "Estrategia RR no validada",
        "los QRS no son consecutivos verificados en un tramo continuo",
    ),
    ReasonCode.PACED_RHYTHM_UNVALIDATED: (
        "Ritmo estimulado no validado",
        "espigas de marcapasos requieren alcance explícito",
    ),
    ReasonCode.POPULATION_UNSUPPORTED: (
        "Población fuera de alcance",
        "la medida no está validada para esta población",
    ),
    ReasonCode.INPUT_REVISION_STALE: (
        "Revisión de entrada obsoleta",
        "la configuración o entrada cambió desde que se lanzó el trabajo",
    ),
    ReasonCode.CALIBRATION_CONFLICT: (
        "Calibración contradictoria",
        "las evidencias de calibración discrepan más allá de la tolerancia",
    ),
    ReasonCode.CALIBRATION_MISSING: (
        "Calibración ausente",
        "no hay evidencia registrada para la magnitud solicitada",
    ),
    ReasonCode.PIXEL_LIMIT: (
        "Límite de píxeles excedido",
        "la página supera max_pixels_per_page y se rechaza antes de decodificar",
    ),
    ReasonCode.UNSUPPORTED_TYPE: (
        "Tipo no soportado",
        "los bytes mágicos del archivo no corresponden a un tipo admitido",
    ),
    ReasonCode.PAGE_LIMIT: (
        "Límite de páginas excedido",
        "el PDF supera max_pdf_pages",
    ),
    ReasonCode.FILE_SIZE_LIMIT: (
        "Límite de tamaño excedido",
        "el archivo supera max_file_mib",
    ),
}


def _check_rel_path(v: str) -> str:
    if not v:
        raise ValueError("path must be non-empty")
    if "\\" in v:
        raise ValueError("backslash not allowed; use POSIX separators")
    if v.startswith("/"):
        raise ValueError("absolute paths not allowed")
    if any(part == ".." for part in v.split("/")):
        raise ValueError("'..' components not allowed")
    return v


RelPath = Annotated[str, AfterValidator(_check_rel_path)]

SupportValue = str | int | float | None

_LOAD_ERRORS = (OSError, ValueError, EOFError, KeyError)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ResolutionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_frame_id: str
    method: str
    limitations: str


class ProcessingStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    implementation: str
    parameters: dict[str, str | float | int | bool | None] = {}
    weights_sha256: str | None = None
    code_version: str | None = None


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: Sha256 | None = None
    hash_status: HashStatus
    kind: SourceKind
    page_count: int = Field(ge=1)
    original_filename: str | None = None


def _positive_finite(name: str, value: float | None) -> None:
    if value is not None and not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite")


class TransformStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["exif_orientation", "crop", "rotate90", "affine", "homography"]
    parameters: dict[str, float | int | str | list[float]]
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    procedure_version: str


class TransformChain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transform_id: str
    source_frame_id: str
    target_frame_id: str
    steps: list[TransformStep] = []


class Page(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    index: int = Field(ge=0)
    width_px: int
    height_px: int
    dpi_declared: float | None
    raster_path: RelPath
    exif_orientation: int | None
    extraction: Literal["image_file", "pdf_embedded_raster", "pdf_rendered"]
    render_dpi: float | None = None
    grid_estimate_path: RelPath | None = None


class CalibrationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    kind: Literal["grid_period", "declared_text", "calibration_pulse", "manual"]
    quantity: Literal["px_per_mm_x", "px_per_mm_y", "speed_mm_s", "gain_mm_mV"]
    value: float | None
    unit: str
    region: list[tuple[float, float]] | None = None
    frame_id: str
    method: str
    limitations: str
    author: str | None = None
    reason: str | None = None
    previous_value: float | None = None


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    page_id: str
    lead_label: str
    lead_status: LeadStatus
    lead_evidence: str
    representation_kind: RepresentationKind
    representation_evidence: str
    source_region: list[tuple[float, float]] = Field(min_length=3)
    transform_id: str
    raw_path: RelPath
    raw_coordinate_frame: str
    raw_support_path: RelPath
    time_relation: TimeRelation
    sync_group_id: str | None = None
    study_start_s: float | None = None
    working_fs_hz: float | None = None
    original_fs_hz: float | None = None
    n_samples: int | None = None
    duration_s: float | None = None
    observed_duration_s: float | None = None
    discarded_tail_s: float | None = None
    units: Literal["mV"] | None = None
    speed_mm_s: float | None = None
    speed_status: ScaleStatus
    gain_mm_mV: float | None = None
    gain_status: ScaleStatus
    effective_dt_ms: float | None = None
    effective_dv_mV: float | None = None
    resolution_evidence: ResolutionEvidence | None = None
    signal_path: RelPath | None = None
    temporal_trace_path: RelPath | None = None
    temporal_trace_units: TemporalTraceUnits | None = None
    observed_mask_path: RelPath | None = None
    valid_mask_path: RelPath | None = None
    gap_fill_mask_path: RelPath | None = None
    calibration_evidence: list[CalibrationEvidence] = []
    px_per_mm_x: float | None = None
    px_per_mm_y: float | None = None
    processing: list[ProcessingStep] = []

    def measurements_allowed(self) -> bool:
        return self.representation_kind != RepresentationKind.unknown

    def rhythm_measurements_allowed(self) -> bool:
        return self.representation_kind == RepresentationKind.rhythm_segment

    def time_known(self) -> bool:
        return self.speed_mm_s is not None and self.speed_status != ScaleStatus.unknown

    def gain_known(self) -> bool:
        return self.gain_mm_mV is not None and self.gain_status != ScaleStatus.unknown

    @model_validator(mode="after")
    def _check_invariants(self) -> "Segment":
        tol = 1e-9
        time_known = self.time_known()
        gain_known = self.gain_known()

        if self.speed_status == ScaleStatus.unknown and self.speed_mm_s is not None:
            raise ValueError("speed_status unknown requires speed_mm_s null")
        if self.gain_status == ScaleStatus.unknown and self.gain_mm_mV is not None:
            raise ValueError("gain_status unknown requires gain_mm_mV null")

        _positive_finite("speed_mm_s", self.speed_mm_s)
        _positive_finite("gain_mm_mV", self.gain_mm_mV)
        _positive_finite("working_fs_hz", self.working_fs_hz)
        _positive_finite("original_fs_hz", self.original_fs_hz)
        _positive_finite("effective_dt_ms", self.effective_dt_ms)
        _positive_finite("effective_dv_mV", self.effective_dv_mV)

        if not time_known:
            forbidden = {
                "working_fs_hz": self.working_fs_hz,
                "n_samples": self.n_samples,
                "duration_s": self.duration_s,
                "observed_duration_s": self.observed_duration_s,
                "discarded_tail_s": self.discarded_tail_s,
                "signal_path": self.signal_path,
                "observed_mask_path": self.observed_mask_path,
                "valid_mask_path": self.valid_mask_path,
                "gap_fill_mask_path": self.gap_fill_mask_path,
                "effective_dt_ms": self.effective_dt_ms,
                "units": self.units,
            }
            bad = [k for k, v in forbidden.items() if v is not None]
            if bad:
                raise ValueError(f"time scale unknown requires null: {bad}")
        else:
            assert self.working_fs_hz is not None
            fs = self.working_fs_hz
            for name in (
                "n_samples",
                "duration_s",
                "observed_duration_s",
                "discarded_tail_s",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"time known requires {name}")
            assert self.n_samples is not None
            assert self.duration_s is not None
            assert self.observed_duration_s is not None
            assert self.discarded_tail_s is not None
            if self.n_samples < 0:
                raise ValueError("n_samples must be >= 0")
            if abs(self.duration_s - self.n_samples / fs) > tol:
                raise ValueError("duration_s != n_samples / working_fs_hz")
            if not (0.0 <= self.discarded_tail_s < 1.0 / fs + tol):
                raise ValueError("discarded_tail_s out of bounds")
            if abs(self.observed_duration_s - self.duration_s - self.discarded_tail_s) > tol:
                raise ValueError("observed_duration_s != duration_s + discarded_tail_s")
            if self.n_samples > 0:
                for p in (
                    self.observed_mask_path,
                    self.valid_mask_path,
                    self.gap_fill_mask_path,
                ):
                    if p is None:
                        raise ValueError("mask paths required when n_samples > 0")

        if (self.signal_path is not None) != (time_known and gain_known):
            raise ValueError("signal_path present iff time and gain are both known")
        if (self.units == "mV") != (self.signal_path is not None):
            raise ValueError("units 'mV' iff signal_path present")
        if not gain_known and self.effective_dv_mV is not None:
            raise ValueError("effective_dv_mV requires known gain")
        if self.temporal_trace_path is not None and self.temporal_trace_units is None:
            raise ValueError("temporal_trace_units required with temporal_trace_path")
            # an arbitrary/px trace is allowed while time is unknown; it only
            # becomes temporal evidence after an explicit scale confirmation

        if (
            time_known
            and not gain_known
            and self.n_samples
            and self.n_samples > 0
            and self.temporal_trace_path is None
        ):
            raise ValueError("temporal_trace_path required when gain unknown and n_samples > 0")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    study_id: str
    revision: int = Field(ge=1)
    input_revision: int = Field(ge=1)
    run_id: str
    config_hash: str | None
    source: Source
    segments: list[Segment]
    pages: list[Page] = []
    transforms: list[TransformChain] = []

    @model_validator(mode="after")
    def _check_manifest(self) -> "Manifest":
        ids = [s.segment_id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment_id")
        groups: dict[str, list[Segment]] = {}
        for s in self.segments:
            if s.sync_group_id is not None:
                groups.setdefault(s.sync_group_id, []).append(s)
        for gid, members in groups.items():
            if any(m.time_relation != TimeRelation.simultaneous for m in members):
                raise ValueError(
                    f"sync_group_id {gid} requires time_relation simultaneous on all members"
                )
        return self


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_id: str
    revision: int = Field(ge=1)
    name: str
    value: float | None
    unit: Literal["ms", "bpm", "mV", "s"]
    status: MeasurementStatus
    reason_codes: list[ReasonCode]
    method: str
    support: dict[str, SupportValue]
    quality_label: QualityLabel
    reviewed_by_human: bool = False
    scope_status: ScopeStatus = ScopeStatus.unknown

    @model_validator(mode="after")
    def _check_measurement(self) -> "Measurement":
        if self.status == MeasurementStatus.available and (
            self.value is None or not np.isfinite(self.value)
        ):
            raise ValueError("available measurement requires a finite value")
        if self.status in (MeasurementStatus.unavailable, MeasurementStatus.not_applicable):
            if self.value is not None:
                raise ValueError(f"{self.status} measurement requires null value")
            if self.status == MeasurementStatus.unavailable and not self.reason_codes:
                raise ValueError("unavailable measurement requires reason_codes")
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("measurement value must be finite")
        return self


def nan_to_none(x: float) -> float | None:
    v = float(x)
    if np.isnan(v):
        return None
    if np.isinf(v):
        raise ValueError("non-finite value cannot be serialized")
    return v


def to_jsonable(obj: object) -> object:
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump(mode="python"))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return nan_to_none(obj)
    if isinstance(obj, StrEnum):
        return str(obj)
    return obj


def dump_json(model_or_dict: object, path: Path | str) -> None:
    p = Path(path)
    data = to_jsonable(model_or_dict)
    text = json.dumps(data, allow_nan=False, indent=2, ensure_ascii=False)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def load_manifest(path: Path | str) -> Manifest:
    return Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_array_file(
    problems: list[str], label: str, arr: np.ndarray, expect_ndim: int = 1
) -> np.ndarray | None:
    if arr.ndim != expect_ndim:
        problems.append(f"{label}: expected {expect_ndim}-D array, got {arr.ndim}-D")
        return None
    return arr


def validate_revision_dir(root: Path, manifest: Manifest, strict: bool = True) -> list[str]:
    problems, _warnings = validate_revision_dir_report(root, manifest, strict=strict)
    return problems


def validate_revision_dir_report(
    root: Path, manifest: Manifest, strict: bool = True
) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    root = Path(root).resolve()

    def resolve(rel: str, label: str) -> Path | None:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            problems.append(f"{label}: {rel} escapes revision root")
            return None
        if not p.exists():
            problems.append(f"{label}: {rel} does not exist")
            return None
        if p.stat().st_size == 0:
            problems.append(f"{label}: {rel} is empty")
            return None
        return p

    if strict:
        if manifest.source.hash_status != HashStatus.computed:
            problems.append("source.hash_status must be 'computed' in strict mode")
        if manifest.source.sha256 is None:
            problems.append("source.sha256 required in strict mode")
        elif not re.fullmatch(r"[0-9a-f]{64}", manifest.source.sha256):
            problems.append("source.sha256 is not 64 lowercase hex")

    for seg in manifest.segments:
        sid = seg.segment_id
        n_samples = seg.n_samples

    transform_ids = {t.transform_id for t in manifest.transforms}
    if manifest.transforms:
        for seg in manifest.segments:
            if seg.transform_id not in transform_ids:
                problems.append(
                    f"{seg.segment_id}.transform_id: {seg.transform_id} not in manifest.transforms"
                )
    for page in manifest.pages:
        resolve(page.raster_path, f"{page.page_id}.raster_path")
        if page.grid_estimate_path is not None:
            resolve(page.grid_estimate_path, f"{page.page_id}.grid_estimate_path")

    for seg in manifest.segments:
        sid = seg.segment_id
        n_samples = seg.n_samples

        if seg.time_known():
            has_speed_evidence = any(
                ev.quantity == "speed_mm_s" and (ev.kind == "manual" or ev.value is not None)
                for ev in seg.calibration_evidence
            )
            if not has_speed_evidence and seg.calibration_evidence:
                warnings.append(f"{sid}: speed_status known without speed_mm_s evidence")
        if not seg.calibration_evidence:
            warnings.append(f"{sid}: no calibration evidence recorded")

        raw_p = resolve(seg.raw_path, f"{sid}.raw_path")
        n_raw = 0
        if raw_p is not None:
            try:
                raw = np.load(raw_p, allow_pickle=False)
                if raw.ndim != 2 or raw.shape[1] != 2:
                    problems.append(f"{sid}.raw_path: expected (N,2), got {raw.shape}")
                elif not np.issubdtype(raw.dtype, np.floating):
                    problems.append(f"{sid}.raw_path: expected float dtype")
                else:
                    n_raw = raw.shape[0]
            except _LOAD_ERRORS as e:
                problems.append(f"{sid}.raw_path: load failed: {e}")

        sup_p = resolve(seg.raw_support_path, f"{sid}.raw_support_path")
        if sup_p is not None:
            try:
                with np.load(sup_p, allow_pickle=False) as z:
                    if "support" not in z.files:
                        problems.append(f"{sid}.raw_support_path: missing 'support' key")
                    else:
                        sup = z["support"]
                        if sup.dtype != np.bool_:
                            problems.append(f"{sid}.raw_support_path: support not bool")
                        if sup.shape != (n_raw,):
                            problems.append(
                                f"{sid}.raw_support_path: len {sup.shape} != raw len {n_raw}"
                            )
            except _LOAD_ERRORS as e:
                problems.append(f"{sid}.raw_support_path: load failed: {e}")

        masks: dict[str, np.ndarray] = {}
        for attr in ("observed_mask_path", "valid_mask_path", "gap_fill_mask_path"):
            rel = getattr(seg, attr)
            if rel is None:
                continue
            p = resolve(rel, f"{sid}.{attr}")
            if p is None:
                continue
            try:
                m = np.load(p, allow_pickle=False)
                if m.dtype != np.bool_ or m.ndim != 1:
                    problems.append(f"{sid}.{attr}: expected 1-D bool")
                    continue
                if n_samples is not None and m.shape[0] != n_samples:
                    problems.append(f"{sid}.{attr}: len {m.shape[0]} != n_samples {n_samples}")
                masks[attr] = m
            except _LOAD_ERRORS as e:
                problems.append(f"{sid}.{attr}: load failed: {e}")

        observed = masks.get("observed_mask_path")
        valid = masks.get("valid_mask_path")
        gap = masks.get("gap_fill_mask_path")
        if (
            observed is not None
            and valid is not None
            and valid.shape == observed.shape
            and not np.all(valid <= observed)
        ):
            problems.append(f"{sid}: valid mask not subset of observed")
        if (
            observed is not None
            and gap is not None
            and gap.shape == observed.shape
            and np.any(gap & observed)
        ):
            problems.append(f"{sid}: gap_fill true where observed is true")

        for attr in ("signal_path", "temporal_trace_path"):
            rel = getattr(seg, attr)
            if rel is None:
                continue
            p = resolve(rel, f"{sid}.{attr}")
            if p is None:
                continue
            try:
                a = np.load(p, allow_pickle=False)
                if _check_array_file(problems, f"{sid}.{attr}", a) is None:
                    continue
                if not np.issubdtype(a.dtype, np.floating):
                    problems.append(f"{sid}.{attr}: expected float dtype")
                if n_samples is not None and a.shape[0] != n_samples:
                    problems.append(f"{sid}.{attr}: len {a.shape[0]} != n_samples {n_samples}")
                if observed is not None and a.shape == observed.shape:
                    if not np.all(np.isfinite(a[observed])):
                        problems.append(f"{sid}.{attr}: non-finite where observed")
                    if not np.all(np.isnan(a[~observed])):
                        problems.append(f"{sid}.{attr}: not NaN where not observed")
            except _LOAD_ERRORS as e:
                problems.append(f"{sid}.{attr}: load failed: {e}")

    return problems, warnings
