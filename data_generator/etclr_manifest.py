# data_generator/etclr_manifest.py

from pathlib import Path
import csv
import numpy as np


def _read_metadata_csv(csv_path):
    """
    Reads DCASE-style metadata:
        frame_idx, class_id, event_idx, azi, ele, radius, mid

    Returns:
        dict[int, list[dict]]
    """
    frame_events = {}

    if not Path(csv_path).exists():
        return frame_events

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue

            frame_idx = int(row[0])
            class_id = int(row[1])
            event_idx = int(row[2])
            azi = float(row[3])
            ele = float(row[4])
            radius = float(row[5])
            mid = row[6].strip('"')

            frame_events.setdefault(frame_idx, []).append({
                "class_id": class_id,
                "event_idx": event_idx,
                "azi": azi,
                "ele": ele,
                "radius": radius,
                "mid": mid,
            })

    return frame_events


def _window_stats(frame_events, start_frame, end_frame):
    active_frames = 0
    max_polyphony = 0
    class_ids = set()
    azis = []
    eles = []
    radii = []

    for frame_idx in range(start_frame, end_frame):
        events = frame_events.get(frame_idx, [])
        n_events = len(events)

        if n_events > 0:
            active_frames += 1
            max_polyphony = max(max_polyphony, n_events)

            for ev in events:
                class_ids.add(ev["class_id"])
                azis.append(ev["azi"])
                eles.append(ev["ele"])
                radii.append(ev["radius"])

    n_frames = max(1, end_frame - start_frame)
    active_fraction = active_frames / n_frames

    return {
        "active_frames": active_frames,
        "active_fraction": active_fraction,
        "max_polyphony": max_polyphony,
        "class_ids": sorted(class_ids),
        "mean_azi": float(np.mean(azis)) if azis else np.nan,
        "mean_ele": float(np.mean(eles)) if eles else np.nan,
        "mean_radius": float(np.mean(radii)) if radii else np.nan,
    }


def build_etclr_manifest(params):
    """
    Builds a TSV file listing valid 3-second ET-CLR crops.

    Output:
        <mixturepath>/etclr_segments.tsv

    Columns:
        audio_path
        mixture_id
        start_sec
        duration_sec
        active_fraction
        max_polyphony
        class_ids
        mean_azi
        mean_ele
        mean_radius
    """
    mixturepath = Path(params["mixturepath"])
    audio_format = params.get("audio_format", "foa")

    if audio_format == "both":
        audio_dir = mixturepath / "foa"
    else:
        audio_dir = mixturepath / audio_format

    metadata_dir = mixturepath / "metadata"
    manifest_path = mixturepath / "etclr_segments.tsv"

    crop_duration = float(params.get("etclr_crop_duration", 3.0))
    crop_hop = float(params.get("etclr_crop_hop", 1.5))
    frame_rate = float(params.get("etclr_frame_rate", 10))
    min_active_fraction = float(params.get("etclr_min_active_fraction", 0.6))
    max_allowed_polyphony = int(params.get("etclr_max_polyphony", 1))
    require_single_source = bool(params.get("etclr_require_single_source", False))

    mixture_duration = float(params["mixture_duration"])
    crop_frames = int(round(crop_duration * frame_rate))
    hop_frames = int(round(crop_hop * frame_rate))
    total_frames = int(round(mixture_duration * frame_rate))

    rows = []

    for audio_path in sorted(audio_dir.glob("*.flac")):
        # Expected generated name:
        # fold0_room0_mix123.flac
        mixture_stem = audio_path.stem
        metadata_path = metadata_dir / f"{mixture_stem}.csv"

        frame_events = _read_metadata_csv(metadata_path)

        for start_frame in range(0, total_frames - crop_frames + 1, hop_frames):
            end_frame = start_frame + crop_frames
            stats = _window_stats(frame_events, start_frame, end_frame)

            if stats["active_fraction"] < min_active_fraction:
                continue

            if stats["max_polyphony"] > max_allowed_polyphony:
                continue

            if require_single_source and len(stats["class_ids"]) > 1:
                continue

            start_sec = start_frame / frame_rate

            rows.append({
                "audio_path": str(audio_path),
                "mixture_id": mixture_stem,
                "start_sec": f"{start_sec:.3f}",
                "duration_sec": f"{crop_duration:.3f}",
                "active_fraction": f"{stats['active_fraction']:.3f}",
                "max_polyphony": stats["max_polyphony"],
                "class_ids": ";".join(map(str, stats["class_ids"])),
                "mean_azi": f"{stats['mean_azi']:.3f}",
                "mean_ele": f"{stats['mean_ele']:.3f}",
                "mean_radius": f"{stats['mean_radius']:.3f}",
            })

    with open(manifest_path, "w", newline="") as f:
        fieldnames = [
            "audio_path",
            "mixture_id",
            "start_sec",
            "duration_sec",
            "active_fraction",
            "max_polyphony",
            "class_ids",
            "mean_azi",
            "mean_ele",
            "mean_radius",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nET-CLR manifest written to: {manifest_path}")
    print(f"Number of ET-CLR crops: {len(rows)}\n")