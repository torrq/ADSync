"""Refine uncertain edit boundaries with independent short waveform windows."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.align.fingerprint import _terminal_waveform_support

log = logging.getLogger("adsync")


def _nudge_to_silence(y_ad, sr, boundary, left_start, right_end, search_radius=20.0, threshold=0.008, silence_sec=0.15, jump=0.0, target_anchor=None):
    """Nudge boundary to the center of the nearest sustained silence gap in AD narration."""
    if y_ad is None or sr <= 0:
        return boundary
    ad_len = y_ad.shape[-1]
    origin = int(boundary * sr)
    if origin < 0 or origin >= ad_len:
        return boundary
    mono = y_ad.mean(axis=0) if y_ad.ndim == 2 else y_ad
    win = int(0.05 * sr)
    need_silent = max(2, int(silence_sec / 0.05))
    rad = int(search_radius * sr)
    lo = max(int(left_start * sr) + win, origin - rad)
    hi = min(int(right_end * sr) - win, origin + rad, ad_len - win)
    if hi <= lo + need_silent * win:
        return boundary

    # Check if origin is already centered inside a sustained silence run
    half_check = (need_silent * win) // 2
    c_start = max(0, origin - half_check)
    c_end = min(ad_len, origin + half_check)

    n_wins = (hi - lo) // win
    chunks = mono[lo:lo + n_wins * win].reshape(n_wins, win)
    rms_arr = np.sqrt(np.mean(chunks**2, axis=1))

    # Adaptive noise floor thresholding based on local 15th percentile
    local_floor = float(np.percentile(rms_arr, 15))
    eff_thresh = max(threshold, min(0.045, local_floor * 1.5))

    if float(np.sqrt(np.mean(mono[c_start:c_end]**2))) < eff_thresh and target_anchor is None:
        return boundary

    # Find all consecutive silence runs >= silence_sec
    silent_runs = []
    in_run = False
    run_start = lo
    for p in range(lo, hi, win):
        chunk = mono[p:p + win]
        r = float(np.sqrt(np.mean(chunk**2)))
        if r < eff_thresh:
            if not in_run:
                in_run = True
                run_start = p
        else:
            if in_run:
                in_run = False
                if (p - run_start) >= need_silent * win:
                    silent_runs.append((run_start, p))
    if in_run and (hi - run_start) >= need_silent * win:
        silent_runs.append((run_start, hi))

    # Fallback to shorter silence run (100ms) if no gap exists
    if not silent_runs and need_silent > 2:
        need_short = 2  # 100ms
        in_run = False
        run_start = lo
        for p in range(lo, hi, win):
            chunk = mono[p:p + win]
            r = float(np.sqrt(np.mean(chunk**2)))
            if r < eff_thresh:
                if not in_run:
                    in_run = True
                    run_start = p
            else:
                if in_run:
                    in_run = False
                    if (p - run_start) >= need_short * win:
                        silent_runs.append((run_start, p))
        if in_run and (hi - run_start) >= need_short * win:
            silent_runs.append((run_start, hi))

    if not silent_runs:
        return boundary

    # Select the optimal silence run
    if target_anchor is not None:
        anchor_idx = int(target_anchor * sr)
        best_run = min(silent_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - anchor_idx))
    elif jump > 0.4:
        # Forward jump in video: trace backwards from boundary through preceding speech
        runs_before = [r for r in silent_runs if 0.5 * (r[0] + r[1]) <= origin]
        if runs_before:
            best_run = runs_before[-1]
        else:
            best_run = min(silent_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - origin))
    elif jump < -0.4:
        # Backward jump: trace forwards from boundary
        runs_after = [r for r in silent_runs if 0.5 * (r[0] + r[1]) >= origin]
        if runs_after:
            best_run = runs_after[0]
        else:
            best_run = min(silent_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - origin))
    else:
        best_run = min(silent_runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - origin))

    best_mid = 0.5 * (best_run[0] + best_run[1]) / sr
    return float(best_mid)



def refine_warp_boundaries(warp_fns, segment_ranges, warp_path, y_vid, y_ad, sr) -> list[dict]:
    """Measure both sides of each cut; mutate ranges and report points together.

    Windowed correlation and ten-second fingerprint votes can put a jump too
    early inside a quiet gap. A bounded local search compares both fitted
    hypotheses using two disjoint waveform subwindows. The midpoint between
    last-left and first-right support is explicit uncertainty, not a claimed
    sample-exact cut. Unsupported boundaries are nudged to the nearest silence
    gap in AD narration so active speech is never split or truncated.
    """
    diagnostics = []
    for index in range(len(segment_ranges) - 1):
        left_fn, right_fn = warp_fns[index:index + 2]
        left_start, boundary = segment_ranges[index]
        right_start, right_end = segment_ranges[index + 1]
        if abs(right_start - boundary) > 1e-5:
            continue
        jump = float(right_fn(boundary) - left_fn(boundary))
        if abs(jump) < .25:
            continue
        low = max(left_start, boundary - 30)
        high = min(right_end - 1, boundary + 30)
        left_support, right_support = [], []
        for start in np.arange(np.ceil(low), np.floor(high) + .1, 1):
            scores = []
            for fn in (left_fn, right_fn):
                offset = float(fn(start + .5) - start - .5)
                scores.append(_terminal_waveform_support(y_vid, y_ad, sr, start, start + 1, offset))
            a, b = scores
            # Ambiguous repeated music cannot pick a boundary. One hypothesis
            # must be clearly supported and beat the other at the same time.
            if a is not None and (b is None or a > b + .15):
                left_support.append((float(start), float(a)))
            if b is not None and (a is None or b > a + .15):
                right_support.append((float(start), float(b)))
        record = {"original_boundary_sec": boundary, "jump_sec": jump,
                  "left_support": left_support, "right_support": right_support,
                  "status": "unresolved"}
        diagnostics.append(record)

        refined = None
        cand = None
        target_anchor = None
        if len(left_support) >= 2 and len(right_support) >= 2:
            last_left = max(t for t, _ in left_support) + 1
            first_right = min(t for t, _ in right_support)
            if first_right >= last_left - .01 and first_right - last_left <= 30:
                cand = (last_left + first_right) / 2
                target_anchor = cand
        elif len(right_support) >= 2 and jump > 0.4:
            first_right = min(t for t, _ in right_support)
            cand = first_right
            target_anchor = first_right
        elif len(left_support) >= 2 and jump > 0.4:
            last_left = max(t for t, _ in left_support) + 1
            cand = (last_left + boundary) / 2
            target_anchor = cand

        # Nudge candidate boundary to nearest silence gap in AD speech
        target_b = cand if cand is not None else boundary
        silence_b = _nudge_to_silence(
            y_ad, sr, target_b, left_start, right_end,
            search_radius=20.0, silence_sec=0.15, jump=jump, target_anchor=target_anchor,
        )
        if left_start < silence_b < right_end and abs(silence_b - boundary) > 0.05:
            refined = silence_b

        if refined is None:
            continue

        # PCHIP extrapolation is not trustworthy far outside its last knot.
        # Extend with the measured local slope, preserving every existing knot.
        updated_functions = {}
        for slot, endpoint, after in ((index, refined, True), (index + 1, refined, False)):
            fn = warp_fns[slot]
            if (after and endpoint > fn.x[-1]) or (not after and endpoint < fn.x[0]):
                edge = fn.x[-1] if after else fn.x[0]
                slope = float(fn.derivative()(edge))
                if not .5 <= slope <= 2:
                    break
                target = float(fn(edge)) + (endpoint - edge) * slope
                xs, ys = fn.x.copy(), fn(fn.x)
                xs = np.r_[xs, endpoint] if after else np.r_[endpoint, xs]
                ys = np.r_[ys, target] if after else np.r_[target, ys]
                updated_functions[slot] = PchipInterpolator(xs, ys)
        else:
            for slot, fn in updated_functions.items():
                warp_fns[slot] = fn
            segment_ranges[index] = (left_start, refined)
            segment_ranges[index + 1] = (refined, right_end)
            for attr in ("points", "anchor_points"):
                revised = []
                for point in getattr(warp_path, attr):
                    if min(boundary, refined) <= point.source_time <= max(boundary, refined):
                        fn = warp_fns[index if point.source_time < refined else index + 1]
                        point = point.model_copy(update={"target_time": float(fn(point.source_time))})
                    revised.append(point)
                setattr(warp_path, attr, revised)
            record.update(status="refined", boundary_sec=refined)
            log.info("Refined edit boundary %.2f -> %.2f s (silence nudged)",
                     boundary, refined)
    return diagnostics
