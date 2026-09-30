"""tools/colorqc_colorstrength.py — ColorStrength из журнала Color QC2 (CHNspec).

Назначение: распаковать журнал измерений ``*.colors`` программы Color QC2
(v4.0.6, CHNspec) и посчитать силу окрашивания образцов относительно эталона:

    ColorStrength = 100 · mean_λ(K/S образца) / mean_λ(K/S эталона)

Формат ``*.colors`` (установлен разбором файла, не по документации CHNspec):
  * ASCII-строка base64, затем ``\\r\\n`` и хвостовой маркер (у нас ``1``);
  * base64 → UTF-8 JSON: список записей измерений. Эталон — запись без
    ``target_uid``; образцы ссылаются на него полем ``target_uid``;
  * ``spectral_info`` — вложенная JSON-строка: ``wave_start``, ``wave_interval``,
    ``wave_number``, ``spectral_data`` — спектр ОТРАЖЕНИЯ R в процентах
    (проверено: L*≈35 ↔ Y≈8.6 % ↔ R≈8–9 %); нули = длина волны не измерена.

K/S считается по Кубелке–Мунку: K/S = (1 − R)² / (2R), R — доля (0..1).
Усреднение — по измеренным длинам волн внутри заданного диапазона.

Запуск:
    .venv\\Scripts\\python.exe tools\\colorqc_colorstrength.py "DOE 561\\Файл.colors"
    [--out результат.xlsx] [--range 400 700]
"""
import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RANGE = (400, 700)
ALT_RANGE = (400, 780)


def decode_colors_file(path):
    """Прочитать ``*.colors`` → список записей измерений (dict)."""
    raw = Path(path).read_bytes()
    body = raw.split(b"\r\n", 1)[0].strip()
    return json.loads(base64.b64decode(body).decode("utf-8"))


def spectrum_of(record):
    """Запись → (длины волн, нм; R, %)."""
    info = json.loads(record["spectral_info"])
    start = float(info["wave_start"])
    step = float(info["wave_interval"])
    values = np.asarray(info["spectral_data"], dtype=float)
    waves = start + step * np.arange(len(values))
    return waves, values


def kubelka_munk(reflectance_pct):
    """R, % → K/S = (1 − R)² / (2R); неизмеренные (R ≤ 0) → NaN."""
    r = np.asarray(reflectance_pct, dtype=float) / 100.0
    ks = np.full_like(r, np.nan)
    ok = r > 0
    ks[ok] = (1.0 - r[ok]) ** 2 / (2.0 * r[ok])
    return ks


def mean_ks(waves, ks, wave_range):
    """Среднее K/S по измеренным точкам в [lo, hi] нм."""
    lo, hi = wave_range
    mask = (waves >= lo) & (waves <= hi) & np.isfinite(ks)
    if not mask.any():
        raise ValueError(f"нет измеренных точек в диапазоне {lo}–{hi} нм")
    return float(np.mean(ks[mask]))


def split_reference(records):
    """Разделить записи на (эталон, образцы) по ``target_uid``."""
    refs = [r for r in records if not r.get("target_uid")]
    if len(refs) != 1:
        raise ValueError(f"ожидался ровно один эталон, найдено {len(refs)}")
    ref = refs[0]
    samples = [r for r in records if r.get("target_uid") == ref["uid"]]
    return ref, samples


def run_number(name):
    """«Образец 561-26_2_2026-09-30» → 26 (номер опыта), иначе None."""
    m = re.search(r"\d+-(\d+)", name)
    return int(m.group(1)) if m else None


def color_strength_table(records, wave_range=DEFAULT_RANGE, alt_range=ALT_RANGE):
    """Сводная таблица ColorStrength + спектры R и K/S (DataFrame'ы)."""
    ref, samples = split_reference(records)
    waves, ref_r = spectrum_of(ref)
    ref_ks = kubelka_munk(ref_r)
    ref_mean = mean_ks(waves, ref_ks, wave_range)
    ref_mean_alt = mean_ks(waves, ref_ks, alt_range)

    rows, r_cols, ks_cols = [], {}, {}
    for rec in [ref] + samples:
        w, r = spectrum_of(rec)
        if not np.array_equal(w, waves):
            raise ValueError(f"сетка длин волн не совпадает с эталоном: {rec['name']}")
        ks = kubelka_munk(r)
        m = mean_ks(w, ks, wave_range)
        m_alt = mean_ks(w, ks, alt_range)
        is_ref = rec is ref
        rows.append({
            "Образец": rec["name"],
            "№ опыта": run_number(rec["name"]),
            "Роль": "эталон" if is_ref else "образец",
            "Время измерения": rec["created_at"],
            "L*": float(rec["lab_data_l"]),
            "a*": float(rec["lab_data_a"]),
            "b*": float(rec["lab_data_b"]),
            f"mean K/S {wave_range[0]}–{wave_range[1]}": m,
            f"ColorStrength {wave_range[0]}–{wave_range[1]}, %": 100.0 * m / ref_mean,
            f"mean K/S {alt_range[0]}–{alt_range[1]}": m_alt,
            f"ColorStrength {alt_range[0]}–{alt_range[1]}, %": 100.0 * m_alt / ref_mean_alt,
        })
        r_cols[rec["name"]] = r
        ks_cols[rec["name"]] = ks

    summary = pd.DataFrame(rows)
    idx = pd.Index(waves.astype(int), name="λ, нм")
    r_df = pd.DataFrame(r_cols, index=idx).replace(0.0, np.nan)
    ks_df = pd.DataFrame(ks_cols, index=idx)
    return summary, r_df, ks_df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("colors_file")
    ap.add_argument("--out", default=None, help="xlsx (по умолчанию рядом с файлом)")
    ap.add_argument("--range", nargs=2, type=float, default=DEFAULT_RANGE,
                    metavar=("LO", "HI"), help="диапазон усреднения, нм")
    args = ap.parse_args(argv)

    src = Path(args.colors_file)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_ColorStrength.xlsx")
    rng = tuple(int(v) for v in args.range)

    summary, r_df, ks_df = color_strength_table(decode_colors_file(src), wave_range=rng)
    with pd.ExcelWriter(out) as xw:
        summary.to_excel(xw, sheet_name="ColorStrength", index=False)
        ks_df.to_excel(xw, sheet_name="K_S спектры")
        r_df.to_excel(xw, sheet_name="R спектры, %")

    cs_col = f"ColorStrength {rng[0]}–{rng[1]}, %"
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(summary[["Образец", "№ опыта", "L*", cs_col]].round(2).to_string(index=False))
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
