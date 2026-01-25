#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

SUPPORTED_IN = {".glb", ".gltf"}
PLY_ASCII_ENCODING = "ascii"   # else binary_little_endian by default


def is_glb_like(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IN


def find_inputs(path: Path, recursive: bool, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    globber = "**/" + pattern if recursive else pattern
    return [p for p in path.glob(globber) if is_glb_like(p)]


def dump_scene_meshes(scene: trimesh.Scene, apply_xforms: bool = True) -> List[trimesh.Trimesh]:
    """
    Получить списком все меши из сцены.
    scene.dump() возвращает уже разложенные по нодам меши.
    """
    meshes = scene.dump(concatenate=False)
    out = []
    for m in meshes:
        # На всякий случай приводим к Trimesh (вдруг Path3D и пр.)
        if isinstance(m, trimesh.Trimesh):
            out.append(m)
        else:
            try:
                t = trimesh.Trimesh(**m.to_dict())
                out.append(t)
            except Exception:
                pass
    if apply_xforms:
        # Обычно scene.dump уже учитывает трансформы; если нет — можно умножить вручную.
        # Здесь оставляем как есть, т.к. dump() в trimesh обычно применяет трансформы нод.
        pass
    return out


def ensure_vertex_colors(mesh: trimesh.Trimesh, prefer_texture_to_vertex: bool = True) -> trimesh.Trimesh:
    """
    Конвертирует визуалы в вершинные цвета, если возможно.
    - Если есть текстуры (uv+image), сэмплируем их в цвета вершин.
    - Если есть face_colors — переносим в vertex_colors.
    - Если ничего нет — задаём белые.
    """
    mesh = mesh.copy()

    try:
        # 1) Текстура → vertex_color (если есть uv и image)
        if prefer_texture_to_vertex and hasattr(mesh.visual, "to_color"):
            # TextureVisuals -> ColorVisuals
            mesh.visual = mesh.visual.to_color()

        # 2) Если всё ещё нет vertex_colors — пробуем face_colors → vertex_colors
        if (not hasattr(mesh.visual, "vertex_colors")) or mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) == 0:
            if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None and len(mesh.visual.face_colors) == len(mesh.faces):
                # Раскладываем face_colors на вершины
                vc = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
                counts = np.zeros((len(mesh.vertices),), dtype=np.int32)
                for f_idx, face in enumerate(mesh.faces):
                    color = mesh.visual.face_colors[f_idx]
                    for vid in face:
                        vc[vid] += color
                        counts[vid] += 1
                nonzero = counts > 0
                vc[nonzero] = (vc[nonzero].astype(np.float32) / counts[nonzero, None]).astype(np.uint8)
                vc[~nonzero] = np.array([255, 255, 255, 255], dtype=np.uint8)
                mesh.visual.vertex_colors = vc
            else:
                # 3) Если нет ничего — белые цвета
                mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)

    except Exception as e:
        # В случае любой нештатной ситуации — хотя бы белые
        mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)

    return mesh


def weld_and_repair(mesh: trimesh.Trimesh, weld_tol: float, do_repair: bool) -> trimesh.Trimesh:
    mesh = mesh.copy()
    try:
        if weld_tol is not None and weld_tol > 0:
            mesh.merge_vertices(weld_tol)
        if do_repair:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.remove_degenerate_faces(mesh)
            trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass
    return mesh


def export_ply(mesh: trimesh.Trimesh, out_path: Path, ascii_mode: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if ascii_mode:
        kwargs["encoding"] = PLY_ASCII_ENCODING
    mesh.export(out_path.as_posix(), file_type="ply", **kwargs)


def convert_file(
    in_path: Path,
    out_dir: Path,
    merge_meshes: bool,
    separate_meshes: bool,
    ascii_mode: bool,
    weld_tol: float,
    do_repair: bool,
    keep_normals: bool,
    no_colors: bool,
) -> List[Path]:
    """
    Конвертирует один .glb/.gltf файл.
    Возвращает список путей на сохранённые .ply.
    """
    print(f"[INFO] Loading: {in_path}")
    loaded = trimesh.load(in_path.as_posix(), force="scene")  # всегда как сцена
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded

    meshes = dump_scene_meshes(scene, apply_xforms=True)
    if not meshes:
        print(f"[WARN] No meshes found in: {in_path}")
        return []

    processed = []
    for i, m in enumerate(meshes):
        # Нормали можно сохранить, но PLY обычно воссоздаёт их при импорте;
        # оставим геометрию как есть — trimesh сам экспортнёт нормали если они заданы.
        if not keep_normals and hasattr(m, "vertex_normals"):
            # Удалить нормали, чтобы уменьшить размер (по желанию)
            pass

        if not no_colors:
            m = ensure_vertex_colors(m, prefer_texture_to_vertex=True)

        m = weld_and_repair(m, weld_tol=weld_tol, do_repair=do_repair)
        processed.append(m)

    outputs: List[Path] = []
    stem = in_path.stem

    # Сохранить раздельно
    if separate_meshes:
        for idx, m in enumerate(processed):
            out_path = out_dir / f"{stem}_part{idx:02d}.ply"
            export_ply(m, out_path, ascii_mode=ascii_mode)
            outputs.append(out_path)

    # Сохранить объединённым мешем
    if merge_meshes:
        merged = trimesh.util.concatenate(processed)
        out_path = out_dir / f"{stem}.ply"
        export_ply(merged, out_path, ascii_mode=ascii_mode)
        outputs.append(out_path)

    return outputs


def main():
    ap = argparse.ArgumentParser(
        description="Convert GLB/GLTF to PLY with multi-mesh handling and vertex-color preservation."
    )
    ap.add_argument("input", type=str, help="Входной .glb/.gltf файл или папка")
    ap.add_argument("-o", "--out", type=str, default=None, help="Папка для вывода .ply (по умолчанию рядом с входом)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Рекурсивный обход папки")
    ap.add_argument("--pattern", type=str, default="*.glb", help="Паттерн для поиска файлов (например, *.gltf)")
    ap.add_argument("--merge", action="store_true", help="Экспорт объединённого меша (по умолчанию включено)", default=True)
    ap.add_argument("--no-merge", dest="merge", action="store_false", help="Отключить объединение мешей")
    ap.add_argument("--separate", action="store_true", help="Также сохранить каждый меш отдельно (partXX)")
    ap.add_argument("--ascii", action="store_true", help="Сохранять в ASCII PLY (иначе бинарный)")
    ap.add_argument("--weld", type=float, default=1e-6, help="Сварка вершин (tolerance). 0 — отключить")
    ap.add_argument("--repair", action="store_true", help="Попытаться отремонтировать геометрию")
    ap.add_argument("--keep-normals", action="store_true", help="Не трогать нормали (если есть)")
    ap.add_argument("--no-colors", action="store_true", help="Не писать вершинные цвета (если включено)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out) if args.out else (in_path.parent if in_path.is_file() else in_path)

    inputs = find_inputs(in_path, recursive=args.recursive, pattern=args.pattern)
    if not inputs:
        print("[ERROR] Nothing to convert. Check path/pattern.")
        sys.exit(2)

    total_out = []
    for p in inputs:
        rel_parent = p.parent.relative_to(in_path) if in_path.is_dir() else Path(".")
        target_out = out_dir / rel_parent
        outs = convert_file(
            in_path=p,
            out_dir=target_out,
            merge_meshes=args.merge,
            separate_meshes=args.separate,
            ascii_mode=args.ascii,
            weld_tol=args.weld if args.weld > 0 else None,
            do_repair=args.repair,
            keep_normals=args.keep_normals,
            no_colors=args.no_colors,
        )
        for o in outs:
            print(f"[OK] Saved: {o}")
        total_out.extend(outs)

    print(f"[DONE] Converted {len(inputs)} file(s). Produced {len(total_out)} PLY file(s).")


if __name__ == "__main__":
    main()
