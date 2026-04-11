from __future__ import annotations

from pathlib import Path

from open_match_lca.io_utils import dump_json, ensure_directory


PUBLIC_DATASET_MANIFEST = {
    "amazon": {
        "description": "Place the publicly released CaML-style product data here.",
        "manual_download": True,
    },
    "epa": {
        "description": "Place EPA/USEEIO factor tables here.",
        "manual_download": True,
    },
    "naics": {
        "description": "Place official NAICS tables here.",
        "manual_download": True,
    },
    "uslci": {
        "description": "Optional extension. Place public USLCI JSON-LD or tabular exports here.",
        "manual_download": True,
        "optional_extension": True,
    },
}


def scaffold_download_targets(target: str, out_dir: str | Path, overwrite: bool = False) -> dict[str, dict]:
    out_root = Path(out_dir)
    ensure_directory(out_root)
    selected = PUBLIC_DATASET_MANIFEST.keys() if target == "all" else [target]
    manifest: dict[str, dict] = {}
    for item in selected:
        target_dir = out_root / item
        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            manifest[item] = {
                **PUBLIC_DATASET_MANIFEST[item],
                "path": str(target_dir),
                "status": "exists",
            }
            continue
        ensure_directory(target_dir)
        manifest[item] = {
            **PUBLIC_DATASET_MANIFEST[item],
            "path": str(target_dir),
            "status": "ready_for_manual_drop",
        }
    dump_json(manifest, out_root / "download_manifest.json")
    return manifest
