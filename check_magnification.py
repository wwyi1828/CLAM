from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import openslide


DEFAULT_PATTERNS = ("*.svs", "*.tif", "*.tiff")
OBJECTIVE_POWER_PROPERTY = "openslide.objective-power"
MPP_X_PROPERTY = "openslide.mpp-x"
MPP_Y_PROPERTY = "openslide.mpp-y"


@dataclass
class SlideReport:
    path: Path
    objective_power: str | None
    mpp_x: float | None
    mpp_y: float | None
    estimated_from_mpp_x: float | None
    estimated_from_mpp_y: float | None
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_magnification_from_mpp(mpp: float | None) -> float | None:
    if mpp is None or mpp <= 0:
        return None
    return 10.0 / mpp


def resolve_slide_paths(
    inputs: Iterable[str | Path],
    patterns: Iterable[str],
    recursive: bool = False,
) -> list[Path]:
    resolved_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        candidates: list[Path] = []

        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            for pattern in patterns:
                iterator = path.rglob(pattern) if recursive else path.glob(pattern)
                candidates.extend(sorted(item for item in iterator if item.is_file()))
        else:
            print(f"Skipping missing path: {path}", file=sys.stderr)
            continue

        for candidate in candidates:
            normalized = candidate.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved_paths.append(candidate)

    return resolved_paths


def inspect_slide(slide_path: str | Path) -> SlideReport:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        objective_power = slide.properties.get(OBJECTIVE_POWER_PROPERTY)
        mpp_x = parse_optional_float(slide.properties.get(MPP_X_PROPERTY))
        mpp_y = parse_optional_float(slide.properties.get(MPP_Y_PROPERTY))

        return SlideReport(
            path=Path(slide_path),
            objective_power=objective_power,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            estimated_from_mpp_x=estimate_magnification_from_mpp(mpp_x),
            estimated_from_mpp_y=estimate_magnification_from_mpp(mpp_y),
            level_count=slide.level_count,
            level_dimensions=tuple(slide.level_dimensions),
        )
    finally:
        slide.close()


def select_magnification(report: SlideReport) -> tuple[float | None, str]:
    if report.estimated_from_mpp_x is not None:
        return report.estimated_from_mpp_x, "mpp-x"
    if report.estimated_from_mpp_y is not None:
        return report.estimated_from_mpp_y, "mpp-y"

    objective_power = parse_optional_float(report.objective_power)
    if objective_power is not None:
        return objective_power, "objective-power"

    return None, "unavailable"


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.2f}"


def format_report(report: SlideReport, show_levels: bool = False) -> str:
    parts = [
        f"path={report.path}",
        f"objective_power={report.objective_power or 'None'}",
        f"mpp_x={format_optional_float(report.mpp_x)}",
        f"mpp_y={format_optional_float(report.mpp_y)}",
        f"estimated_from_mpp_x={format_optional_float(report.estimated_from_mpp_x)}",
        f"estimated_from_mpp_y={format_optional_float(report.estimated_from_mpp_y)}",
        f"level_count={report.level_count}",
    ]

    if report.level_dimensions:
        parts.append(f"base_dimensions={report.level_dimensions[0]}")

    if show_levels:
        parts.append(f"level_dimensions={report.level_dimensions}")

    return ", ".join(parts)


def move_slide(slide_path: Path, destination_dir: Path, dry_run: bool = False) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / slide_path.name

    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")

    if not dry_run:
        shutil.move(str(slide_path), str(destination_path))

    return destination_path


def export_level_image(slide_path: Path, level: int, output_path: Path) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        if level < 0 or level >= slide.level_count:
            raise ValueError(
                f"Level {level} is out of range for {slide_path} "
                f"(available: 0-{slide.level_count - 1})"
            )

        dimensions = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, dimensions).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        region.save(output_path)
    finally:
        slide.close()


def run(
    inputs: Iterable[str | Path],
    patterns: Iterable[str] | None = None,
    recursive: bool = False,
    show_levels: bool = False,
    move_outside_range: tuple[float, float] | None = None,
    move_dest: str | Path | None = None,
    dry_run: bool = False,
    export_level: int | None = None,
    export_output: str | Path | None = None,
) -> int:
    patterns = tuple(patterns or DEFAULT_PATTERNS)
    slide_paths = resolve_slide_paths(inputs, patterns, recursive=recursive)

    if not slide_paths:
        print("No slide files found.", file=sys.stderr)
        return 1

    if move_outside_range is not None and move_dest is None:
        print("--move-dest is required when --move-outside-range is used.", file=sys.stderr)
        return 1

    if export_level is not None and len(slide_paths) != 1:
        print(
            "Export requires exactly one slide input after resolving directories.",
            file=sys.stderr,
        )
        return 1

    had_error = False

    for slide_path in slide_paths:
        try:
            report = inspect_slide(slide_path)
        except openslide.OpenSlideError as error:
            had_error = True
            print(f"{slide_path}: cannot open slide ({error})", file=sys.stderr)
            continue

        print(format_report(report, show_levels=show_levels))

        if move_outside_range is None:
            continue

        selected_magnification, source = select_magnification(report)
        min_magnification, max_magnification = move_outside_range

        if selected_magnification is None:
            print(
                f"Skipping move for {slide_path}: no usable magnification metadata.",
                file=sys.stderr,
            )
            continue

        if min_magnification <= selected_magnification <= max_magnification:
            continue

        try:
            moved_to = move_slide(
                report.path,
                Path(move_dest),
                dry_run=dry_run,
            )
        except OSError as error:
            had_error = True
            print(f"{slide_path}: failed to move slide ({error})", file=sys.stderr)
            continue

        action = "Would move" if dry_run else "Moved"
        print(
            f"{action} {slide_path} -> {moved_to} "
            f"because {source} magnification {selected_magnification:.2f} "
            f"is outside [{min_magnification:.2f}, {max_magnification:.2f}]"
        )

    if export_level is not None:
        export_path = Path(export_output) if export_output else slide_paths[0].with_name(
            f"{slide_paths[0].stem}_level_{export_level}.png"
        )
        try:
            export_level_image(slide_paths[0], export_level, export_path)
        except (OSError, ValueError, openslide.OpenSlideError) as error:
            print(f"Failed to export level image: {error}", file=sys.stderr)
            return 1
        print(f"Saved level {export_level} as {export_path}")

    return 1 if had_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect slide magnification metadata and optionally move or export slides.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["."],
        help="Slide files or directories to inspect. Defaults to the current directory.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob pattern used when an input is a directory. Repeat to add more patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when an input is a directory.",
    )
    parser.add_argument(
        "--show-levels",
        action="store_true",
        help="Print full level dimensions for each slide.",
    )
    parser.add_argument(
        "--move-outside-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Move slides whose selected magnification is outside the inclusive range.",
    )
    parser.add_argument(
        "--move-dest",
        help="Destination directory used with --move-outside-range.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would move without modifying the filesystem.",
    )
    parser.add_argument(
        "--export-level",
        type=int,
        help="Export a full slide pyramid level to a PNG image.",
    )
    parser.add_argument(
        "--export-output",
        help="Output path used with --export-level. Defaults to <slide>_level_<n>.png.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return run(
        inputs=args.inputs,
        patterns=args.patterns,
        recursive=args.recursive,
        show_levels=args.show_levels,
        move_outside_range=tuple(args.move_outside_range) if args.move_outside_range else None,
        move_dest=args.move_dest,
        dry_run=args.dry_run,
        export_level=args.export_level,
        export_output=args.export_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
