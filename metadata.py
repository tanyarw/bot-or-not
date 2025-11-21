import json
from datetime import datetime


def write_metadata(cfg, out_dir):
    meta_entry = {
        "version_name": cfg["version_name"],
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
    }

    path = out_dir / "metadata.json"

    # Case 1 — File exists → append
    if path.exists():
        try:
            with open(path, "r") as f:
                existing = json.load(f)

            # Ensure existing is a list
            if not isinstance(existing, list):
                existing = [existing]

            existing.append(meta_entry)

        except Exception:
            # If file is corrupted or not a list, reset it
            existing = [meta_entry]
    else:
        # Case 2 — file does not exist
        existing = [meta_entry]

    # Write back
    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
