from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence


OBJECTIVE_POWER_KEYS = (
    "openslide.objective-power",
    "aperio.AppMag",
    "hamamatsu.SourceLens",
)
MPP_X_KEY = "openslide.mpp-x"
MPP_Y_KEY = "openslide.mpp-y"


@dataclass(frozen=True)
class MagnificationInfo:
    base_magnification: float | None
    source: str | None
    raw_value: str | None
    mpp_x: float | None
    mpp_y: float | None
    mpp: float | None
    mpp_derived_magnification: float | None
    objective_magnification: float | None
    objective_source: str | None
    warning: str | None


@dataclass(frozen=True)
class AdaptivePatchPlan:
    base_magnification: float
    target_magnification: float
    patch_level: int
    custom_downsample: int
    level_downsample: float
    effective_downsample: float
    effective_magnification: float
    reason: str
    note: str = ""
    target_mpp: float | None = None
    effective_mpp: float | None = None
    base_mpp: float | None = None
    rescale_factor: float | None = None


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_downsample(level_downsample: float | Sequence[float]) -> float:
    if isinstance(level_downsample, (tuple, list)):
        values = [float(value) for value in level_downsample]
        return sum(values) / len(values)
    return float(level_downsample)


def magnification_from_mpp(mpp: float | None) -> float | None:
    if mpp is None or mpp <= 0:
        return None
    return 10.0 / mpp


def inspect_slide_magnification(slide) -> MagnificationInfo:
    properties = slide.properties
    mpp_x = parse_optional_float(properties.get(MPP_X_KEY))
    mpp_y = parse_optional_float(properties.get(MPP_Y_KEY))
    positive_mpps = [value for value in (mpp_x, mpp_y) if value is not None and value > 0]
    primary_mpp = sum(positive_mpps) / len(positive_mpps) if positive_mpps else None
    mpp_derived_magnification = magnification_from_mpp(primary_mpp)

    objective_magnification = None
    objective_source = None
    raw_objective_value = None

    for key in OBJECTIVE_POWER_KEYS:
        raw_value = properties.get(key)
        parsed_value = parse_optional_float(raw_value)
        if parsed_value is not None and parsed_value > 0:
            objective_magnification = parsed_value
            objective_source = key
            raw_objective_value = str(raw_value)
            break

    warning = None
    if (
        mpp_derived_magnification is not None
        and objective_magnification is not None
        and objective_magnification > 0
    ):
        relative_difference = (
            abs(mpp_derived_magnification - objective_magnification) / objective_magnification
        )
        if relative_difference > 0.15:
            warning = (
                f"mpp-derived magnification {mpp_derived_magnification:.2f} "
                f"differs from {objective_source}={objective_magnification:.2f}"
            )

    if mpp_derived_magnification is not None:
        return MagnificationInfo(
            base_magnification=mpp_derived_magnification,
            source="mpp",
            raw_value=f"{primary_mpp:.5f}",
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            mpp=primary_mpp,
            mpp_derived_magnification=mpp_derived_magnification,
            objective_magnification=objective_magnification,
            objective_source=objective_source,
            warning=warning,
        )

    if objective_magnification is not None:
        return MagnificationInfo(
            base_magnification=objective_magnification,
            source=objective_source,
            raw_value=raw_objective_value,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            mpp=primary_mpp,
            mpp_derived_magnification=mpp_derived_magnification,
            objective_magnification=objective_magnification,
            objective_source=objective_source,
            warning=warning,
        )

    return MagnificationInfo(
        base_magnification=None,
        source=None,
        raw_value=None,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        mpp=primary_mpp,
        mpp_derived_magnification=mpp_derived_magnification,
        objective_magnification=objective_magnification,
        objective_source=objective_source,
        warning=warning,
    )


def choose_adaptive_patch_plan(
    base_magnification: float,
    level_downsamples: Iterable[float | Sequence[float]],
    target_magnification: float = 20.0,
    custom_downsample_choices: Sequence[int] = (1, 2),
    native_tolerance: float = 0.15,
) -> AdaptivePatchPlan:
    if base_magnification <= 0:
        raise ValueError("base_magnification must be positive")
    if target_magnification <= 0:
        raise ValueError("target_magnification must be positive")

    normalized_downsamples = [
        normalize_downsample(level_downsample)
        for level_downsample in level_downsamples
    ]

    if not normalized_downsamples:
        raise ValueError("level_downsamples cannot be empty")

    desired_downsample = base_magnification / target_magnification

    if desired_downsample <= 1:
        return AdaptivePatchPlan(
            base_magnification=base_magnification,
            target_magnification=target_magnification,
            patch_level=0,
            custom_downsample=1,
            level_downsample=normalized_downsamples[0],
            effective_downsample=normalized_downsamples[0],
            effective_magnification=base_magnification / normalized_downsamples[0],
            reason=(
                "base magnification is already at or below the target; "
                "using the highest available resolution"
            ),
        )

    def relative_error(effective_downsample: float) -> float:
        return abs(effective_downsample - desired_downsample) / desired_downsample

    native_candidates: list[tuple[float, int, float]] = []
    all_candidates: list[tuple[float, int, int, float]] = []

    for level, level_downsample in enumerate(normalized_downsamples):
        native_candidates.append(
            (relative_error(level_downsample), level, level_downsample)
        )
        for custom_downsample in custom_downsample_choices:
            effective_downsample = level_downsample * custom_downsample
            all_candidates.append(
                (
                    relative_error(effective_downsample),
                    level,
                    custom_downsample,
                    level_downsample,
                )
            )

    native_error, native_level, native_level_downsample = min(
        native_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )

    if native_error <= native_tolerance:
        effective_downsample = native_level_downsample
        return AdaptivePatchPlan(
            base_magnification=base_magnification,
            target_magnification=target_magnification,
            patch_level=native_level,
            custom_downsample=1,
            level_downsample=native_level_downsample,
            effective_downsample=effective_downsample,
            effective_magnification=base_magnification / effective_downsample,
            reason=(
                f"native level {native_level} matched the desired downsample "
                f"{desired_downsample:.2f} within tolerance"
            ),
        )

    best_error, best_level, best_custom_downsample, best_level_downsample = min(
        all_candidates,
        key=lambda candidate: (candidate[0], candidate[2] != 1, candidate[1]),
    )
    effective_downsample = best_level_downsample * best_custom_downsample

    return AdaptivePatchPlan(
        base_magnification=base_magnification,
        target_magnification=target_magnification,
        patch_level=best_level,
        custom_downsample=best_custom_downsample,
        level_downsample=best_level_downsample,
        effective_downsample=effective_downsample,
        effective_magnification=base_magnification / effective_downsample,
        reason=(
            f"used level {best_level} with custom_downsample={best_custom_downsample} "
            f"to approximate the desired downsample {desired_downsample:.2f} "
            f"(relative error {best_error:.3f})"
        ),
    )


# ---------------------------------------------------------------------------
# High-level API used by create_patches_*.py
# ---------------------------------------------------------------------------

def resolve_magnification_strategy(
    strategy: str | None,
    target_magnification: float | None,
    target_mpp: float | None,
) -> str:
    if strategy is not None:
        return strategy
    if target_mpp is not None:
        return "rescale"
    if target_magnification is not None:
        return "bin"
    return "manual"


def _choose_read_level_for_rescale(
    base_mpp: float,
    target_mpp: float,
    level_downsamples: list[float | Sequence[float]],
) -> int:
    """Pick the coarsest level whose mpp is still <= target_mpp.

    This minimizes the amount of data read while ensuring we downsample
    (rather than upsample) to the target resolution.  If no level satisfies
    the constraint (target is finer than the slide's finest level), level 0
    is returned (best available, will require slight upsampling).
    """
    best = 0
    for level, ld in enumerate(level_downsamples):
        level_mpp = base_mpp * normalize_downsample(ld)
        if level_mpp <= target_mpp * 1.02:
            best = level
    return best


def choose_patch_sampling_plan(
    slide,
    level_downsamples: Iterable[float | Sequence[float]],
    patch_size: int,
    step_size: int,
    magnification_strategy: str,
    target_magnification: float | None,
    target_mpp: float | None,
    patch_level: int,
    custom_downsample: int,
    unknown_magnification_policy: str,
) -> tuple[AdaptivePatchPlan | None, MagnificationInfo, str]:
    """High-level wrapper: inspect the slide and choose a patch sampling plan.

    Returns ``(plan, magnification_info, status)`` where *status* is
    ``'ok'`` or ``'skip'``.
    """
    info = inspect_slide_magnification(slide)
    ds_list = list(level_downsamples)

    # ---- manual: use explicit CLI args, no adaptation ----
    if magnification_strategy == "manual":
        ld = normalize_downsample(ds_list[patch_level])
        eff_ds = ld * custom_downsample
        base_mag = info.base_magnification or 0.0
        eff_mag = base_mag / eff_ds if base_mag and eff_ds else 0.0
        plan = AdaptivePatchPlan(
            base_magnification=base_mag,
            target_magnification=base_mag,
            patch_level=patch_level,
            custom_downsample=custom_downsample,
            level_downsample=ld,
            effective_downsample=eff_ds,
            effective_magnification=eff_mag,
            reason="manual",
            note="manual mode: using specified patch_level and custom_downsample",
            base_mpp=info.mpp,
        )
        return plan, info, "ok"

    # ---- bin / rescale: need magnification info ----

    eff_target_mag = target_magnification
    if eff_target_mag is None and target_mpp is not None:
        eff_target_mag = magnification_from_mpp(target_mpp)

    if info.base_magnification is None:
        if unknown_magnification_policy == "error":
            raise ValueError("no magnification metadata found for slide")

        if unknown_magnification_policy == "assume-target":
            if eff_target_mag is None:
                return None, info, "skip"
            ld = normalize_downsample(ds_list[0])
            plan = AdaptivePatchPlan(
                base_magnification=eff_target_mag,
                target_magnification=eff_target_mag,
                patch_level=0,
                custom_downsample=1,
                level_downsample=ld,
                effective_downsample=ld,
                effective_magnification=eff_target_mag / ld,
                reason="assumed target magnification (no metadata)",
                note=f"assumed base={eff_target_mag:.1f}x (no metadata); using level 0",
                target_mpp=target_mpp,
                effective_mpp=target_mpp,
                base_mpp=target_mpp,
            )
            return plan, info, "ok"

        return None, info, "skip"

    if eff_target_mag is None:
        raise ValueError(
            "target_magnification or target_mpp required for bin/rescale strategy"
        )

    # ---- bin: snap to nearest native level (+ optional cd=2), no resize ----
    if magnification_strategy == "bin":
        plan = choose_adaptive_patch_plan(
            base_magnification=info.base_magnification,
            level_downsamples=ds_list,
            target_magnification=eff_target_mag,
        )
        base_mpp = info.mpp
        eff_mpp = base_mpp * plan.effective_downsample if base_mpp else None

        note_parts: list[str] = [f"bin: {plan.reason}"]
        if target_mpp is not None and eff_mpp is not None:
            deviation = abs(eff_mpp - target_mpp) / target_mpp
            note_parts.append(
                f"target_mpp={target_mpp:.4f}; effective_mpp={eff_mpp:.4f}"
                + (f"; deviation={deviation:.1%}" if deviation > 0.01 else "")
            )

        plan = replace(
            plan,
            note="; ".join(note_parts),
            target_mpp=target_mpp,
            effective_mpp=eff_mpp,
            base_mpp=base_mpp,
        )
        return plan, info, "ok"

    # ---- rescale: exact target_mpp via post-read resize ----
    assert magnification_strategy == "rescale"

    base_mpp = info.mpp
    if base_mpp is None and info.base_magnification is not None:
        base_mpp = 10.0 / info.base_magnification

    if base_mpp is None or target_mpp is None:
        raise ValueError(
            "rescale strategy requires MPP metadata (or objective power) "
            "and --target_mpp"
        )

    best_level = _choose_read_level_for_rescale(base_mpp, target_mpp, ds_list)
    ld = normalize_downsample(ds_list[best_level])
    read_mpp = base_mpp * ld
    rf = target_mpp / read_mpp

    note = (
        f"target_mpp={target_mpp:.4f}; read_level={best_level}; "
        f"read_mpp={read_mpp:.4f}; rescale_factor={rf:.4f}"
    )
    if rf < 1.0:
        note += (
            f"; WARNING: target finer than best available level "
            f"(upsampling from mpp={read_mpp:.4f})"
        )

    plan = AdaptivePatchPlan(
        base_magnification=info.base_magnification,
        target_magnification=eff_target_mag,
        patch_level=best_level,
        custom_downsample=1,
        level_downsample=ld,
        effective_downsample=ld * rf,
        effective_magnification=info.base_magnification / (ld * rf),
        reason=f"rescale from level {best_level} (mpp={read_mpp:.4f}) with factor {rf:.4f}",
        note=note,
        target_mpp=target_mpp,
        effective_mpp=target_mpp,
        base_mpp=base_mpp,
        rescale_factor=rf,
    )
    return plan, info, "ok"


def build_magnification_record(
    strategy: str,
    info: MagnificationInfo,
    plan: AdaptivePatchPlan | None,
    target_magnification: float | None,
    target_mpp: float | None,
    note_override: str | None = None,
) -> dict:
    """Build a flat dict for writing into the process-list CSV."""
    record: dict = {
        "magnification_strategy": strategy,
        "base_magnification": info.base_magnification,
        "magnification_source": info.source,
        "mpp": info.mpp,
        "mpp_x": info.mpp_x,
        "mpp_y": info.mpp_y,
        "objective_magnification": info.objective_magnification,
        "target_magnification": target_magnification,
        "target_mpp": target_mpp,
    }
    if plan is not None:
        record.update(
            {
                "effective_magnification": plan.effective_magnification,
                "effective_mpp": plan.effective_mpp,
                "patch_level_used": plan.patch_level,
                "custom_downsample_used": plan.custom_downsample,
                "effective_downsample": plan.effective_downsample,
                "rescale_factor": plan.rescale_factor,
            }
        )
    record["magnification_note"] = note_override or (plan.note if plan else None)
    return record


def format_patch_plan(plan: AdaptivePatchPlan) -> str:
    parts = [
        f"level={plan.patch_level}",
        f"cd={plan.custom_downsample}",
        f"eff_ds={plan.effective_downsample:.2f}",
        f"eff_mag={plan.effective_magnification:.1f}x",
    ]
    if plan.effective_mpp is not None:
        parts.append(f"eff_mpp={plan.effective_mpp:.4f}")
    if plan.rescale_factor is not None:
        parts.append(f"rf={plan.rescale_factor:.4f}")
    return " | ".join(parts)
