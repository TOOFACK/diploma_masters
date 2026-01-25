#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import trimesh

def extract_ply_from_glb(glb_path: str, out_path: str = None):
    glb_path = Path(glb_path)
    scene = trimesh.load(glb_path.as_posix(), force="scene")

    # geometry keys indicate embedded files
    if not scene.geometry:
        print("[ERROR] No geometry found in GLB")
        return

    for name, geom in scene.geometry.items():
        meta = geom.metadata or {}
        file_path = meta.get("file_path", "")

        if file_path.lower().endswith(".ply"):
            print(f"[INFO] Found embedded PLY: {file_path}")

            # This is raw PLY data inside GLB
            data = meta.get("file_obj", None)
            if data is None:
                print("[ERROR] Embedded PLY found but no file_obj data")
                return

            if out_path is None:
                out_path = glb_path.with_suffix(".ply")
            out_path = Path(out_path)

            print(f"[INFO] Saving extracted PLY to: {out_path}")
            with open(out_path, "wb") as f:
                f.write(data.read())  # write raw bytes exactly as stored

            print("[DONE] Extracted successfully.")
            return

    print("[WARN] No embedded PLY found inside GLB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_embedded_ply_from_glb.py file.glb [output.ply]")
        sys.exit(1)

    glb = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    extract_ply_from_glb(glb, out)
