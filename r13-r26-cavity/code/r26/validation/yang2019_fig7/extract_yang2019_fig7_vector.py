#!/usr/bin/env python3
"""Extract the published R26 marker coordinates from Yang et al. (2019) Fig. 7.

The input is the publisher PDF (DOI 10.3390/app9132733).  Figure 7 is vector
artwork on PDF page 12.  Poppler converts that page to SVG without rasterising
the curves; this script then reads the blue triangular R26 markers and applies
axis calibrations obtained from the vector tick marks.  Legend markers are
excluded by fixed page-coordinate boxes.  The output is therefore a traceable
digitisation of the published curve, not a fitted or hand-redrawn target.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BLUE = "rgb(16.078186%, 16.078186%, 100%)"
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
SVG_NS = "{http://www.w3.org/2000/svg}"


@dataclass(frozen=True)
class Panel:
    name: str
    quantity: str
    coordinate: str
    bounds: tuple[float, float, float, float]
    legend_box: tuple[float, float, float, float]
    x_ticks: tuple[tuple[float, float], ...]
    y_ticks: tuple[tuple[float, float], ...]


PANELS = (
    Panel(
        "a", "ux", "y", (115.4609, 272.1132, 227.5469, 370.5352),
        (170.0, 205.0, 344.0, 358.0),
        ((137.934, 0.0), (193.848, 0.01), (249.758, 0.02)),
        ((356.211, -0.4), (327.625, -0.2), (299.043, 0.0),
         (270.457, 0.2), (241.871, 0.4)),
    ),
    Panel(
        "b", "uy", "x", (335.7539, 492.9453, 227.0078, 370.4727),
        (360.0, 395.0, 341.0, 355.0),
        ((351.457, -0.4), (382.918, -0.2), (414.320, 0.0),
         (445.781, 0.2), (477.184, 0.4)),
        ((361.188, -0.004), (330.023, -0.002), (298.922, 0.0),
         (267.758, 0.002), (236.656, 0.004)),
    ),
    Panel(
        "c", "qx", "y", (114.9102, 273.0586, 414.1250, 558.4883),
        (145.0, 178.0, 518.0, 532.0),
        ((139.301, -0.0005), (216.484, 0.0)),
        ((544.105, -0.4), (515.164, -0.2), (486.336, 0.0),
         (457.453, 0.2), (428.566, 0.4)),
    ),
    Panel(
        "d", "qy", "x", (342.1523, 484.1211, 417.0625, 560.3477),
        (420.0, 450.0, 528.0, 542.0),
        ((356.297, -0.4), (384.699, -0.2), (413.105, 0.0),
         (441.512, 0.2), (469.859, 0.4)),
        ((560.348, -0.0002), (524.512, -0.0001), (488.676, 0.0),
         (452.898, 0.0001), (417.062, 0.0002)),
    ),
)


def affine(ticks: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    pixel = np.asarray([p for p, _ in ticks])
    value = np.asarray([v for _, v in ticks])
    slope, intercept = np.polyfit(pixel, value, 1)
    residual = np.max(np.abs(slope * pixel + intercept - value))
    # Tick positions are transcribed from the publisher SVG at three decimal
    # places.  A 5e-4 tolerance in axis units is below 0.3% of the coordinate
    # tick interval and avoids a false failure from that transcription rounding.
    if residual > 5.0e-4:
        raise RuntimeError(f"axis calibration is not affine: residual={residual}")
    return float(slope), float(intercept)


def transform_points(element: ET.Element) -> np.ndarray:
    match = re.search(r"matrix\(([^)]+)\)", element.attrib.get("transform", ""))
    if match is None:
        return np.empty((0, 2))
    a, b, c, d, tx, ty = map(float, NUMBER.findall(match.group(1)))
    values = list(map(float, NUMBER.findall(element.attrib.get("d", ""))))
    points = []
    for i in range(0, len(values) - 1, 2):
        x, y = values[i : i + 2]
        points.append((a * x + c * y + tx, b * x + d * y + ty))
    return np.asarray(points)


def marker_centres(svg: Path) -> list[tuple[float, float]]:
    root = ET.parse(svg).getroot()
    centres: list[tuple[float, float]] = []
    for element in root.iter(SVG_NS + "path"):
        if element.attrib.get("stroke") != BLUE:
            continue
        points = transform_points(element)
        if len(points) != 4 or np.linalg.norm(points[0] - points[-1]) > 0.1:
            continue
        unique = points[:3]
        width = np.ptp(unique[:, 0])
        height = np.ptp(unique[:, 1])
        if 3.8 <= width <= 4.8 and 3.2 <= height <= 4.3:
            centres.append(tuple(np.mean(unique, axis=0)))
    return centres


def extract(pdf: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="yang2019_svg_") as tmp:
        svg = Path(tmp) / "page.svg"
        subprocess.run(
            ["pdftocairo", "-svg", "-f", "12", "-l", "12", str(pdf), str(svg)],
            check=True,
        )
        centres = marker_centres(svg)

    rows: list[dict[str, object]] = []
    for panel in PANELS:
        x0, x1, y0, y1 = panel.bounds
        lx0, lx1, ly0, ly1 = panel.legend_box
        sx, bx = affine(panel.x_ticks)
        sy, by = affine(panel.y_ticks)
        selected = []
        for px, py in centres:
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                continue
            if lx0 <= px <= lx1 and ly0 <= py <= ly1:
                continue
            x_value = sx * px + bx
            y_value = sy * py + by
            if panel.coordinate == "y":
                coordinate, value = y_value, x_value
            else:
                coordinate, value = x_value, y_value
            selected.append((coordinate, value, px, py))
        selected.sort()
        if len(selected) < 15:
            raise RuntimeError(f"too few R26 markers in panel {panel.name}: {len(selected)}")
        for coordinate, value, px, py in selected:
            rows.append(
                {
                    "panel": panel.name,
                    "quantity": panel.quantity,
                    "coordinate": panel.coordinate,
                    "coordinate_over_L0": f"{coordinate:.10g}",
                    "published_R26_value": f"{value:.12g}",
                    "svg_x_pt": f"{px:.6f}",
                    "svg_y_pt": f"{py:.6f}",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.pdf, args.output)


if __name__ == "__main__":
    main()
