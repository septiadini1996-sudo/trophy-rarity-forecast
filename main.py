#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input), and for each
case writes <id>.out in $MAKEFUN_OUTPUT_DIR (else /output): five quantiles
(5/25/50/75/95%) of every trophy's earn rate at the case's target, as JSON
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Model, per trophy, fitted fresh to whatever case it is given:

  The page only changes when Exophase refreshes it, so the history is first
  compressed to its distinct states (a new state when the players-tracked
  count or any rate changes).  Two views of each trophy's series are fitted
  against x = ln(1 + hours since release), a clock on which both player
  growth and trophy earning decelerate:

  A. Rate view:  a least-squares line of logit(rate) on x fitted over the
     last day and a half of states (the recent trend is the honest one).
     The prediction is anchored on the mean of the last few states and
     extended by the fitted change to the target, discounted the farther
     the extrapolation runs beyond the window's own span — growth on this
     clock decelerates — and shrunk toward the mean change of the trophy's
     tier group.
  B. Earners view: the same for ln(earners + 1/2), where earners = rate *
     players_tracked / 100, together with a like fit of ln(players_tracked).
     Dividing predicted earners by predicted players captures dilution: when
     the tracked pool grows faster than a trophy's earners, its rate falls,
     as happened at Exophase's first refresh on launch night.

  The median is the average of the two views on the logit scale.  The spread
  is a per-trophy sigma on the logit scale combining a horizon random-walk
  term, the slope's standard error over the extrapolation distance, the
  fit's residuals and a small-count term for trophies few players hold; the
  five quantiles are inv_logit(mu + z_tau * sigma).  The platinum's earners
  are a subset of every other trophy's, so its quantiles are capped at the
  elementwise minimum of the other trophies' quantiles.

Deterministic: pure arithmetic on the input, no randomness, no clock.
Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RATE_FLOOR = 0.01     # the scorer clamps rates to [0.01, 99.99] before logit
RATE_CAP = 99.99

# Spread constants (logit units), calibrated by simulation holdout.
SIGMA_BASE = 0.14          # irreducible level uncertainty
SIGMA_WALK = 0.20          # random walk per sqrt(horizon in days)
RESID_WEIGHT = 0.25        # share of the fit's residual sigma added in quadrature
SHRINK_K = 3.0             # slope shrinkage strength toward the tier mean
SLOPE_HALF_LIFE_H = 24.0   # recency half-life within the slope window
SLOPE_WINDOW_H = 36.0      # slopes are fitted on the last this-many hours
MAX_MOVE = 2.0             # cap on the median's logit move over the horizon


def parse_ts(text):
    """ISO 8601 timestamp -> epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp_rate(r):
    return min(max(r, RATE_FLOOR), RATE_CAP)


def logit_pct(r):
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x.

    Returns (intercept, slope, slope_se, resid_sigma).  With fewer than two
    usable points or no x-spread, slope is 0 and its error is infinite.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, float("inf"), 0.0
    sw = sum(ws)
    if sw <= 0:
        return ys[0], 0.0, float("inf"), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if n < 2 or sxx <= 1e-12:
        return my, 0.0, float("inf"), 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    dof = max(n - 2, 1)
    sse = sum(w * (y - (intercept + slope * x)) ** 2
              for x, y, w in zip(xs, ys, ws))
    resid = math.sqrt(sse / dof) if n > 2 else 0.35
    slope_se = resid / math.sqrt(sxx)
    return intercept, slope, slope_se, resid


def fit_move(xs, ys, ws, x_last, x_target):
    """Linear WLS on the window, extrapolated with self-scaling damping.

    The fitted slope is trusted over a distance comparable to the window's
    own x-span and discounted beyond it: move = slope * dx * span/(span+dx),
    where span is the window's x-range and dx the extrapolation distance.
    A short extrapolation from a long window is barely damped; a leap far
    beyond the evidence is heavily damped, because growth on the log-time
    clock decelerates.  Returns (move, move_se, resid_sigma).
    """
    _, slope, slope_se, resid = wls(xs, ys, ws)
    dx = max(0.0, x_target - x_last)
    span = (max(xs) - min(xs)) if xs else 0.0
    if span <= 0.0 or dx <= 0.0:
        return 0.0, slope_se * dx, resid
    factor = span / (span + dx)
    return slope * dx * factor, slope_se * dx * factor, resid


def recency_weights(times_h, half_life_h):
    t_last = times_h[-1]
    return [0.5 ** ((t_last - t) / half_life_h) for t in times_h]


def shrink(value, group_mean, n_eff):
    """Pull a per-trophy slope toward its tier group's mean slope."""
    if group_mean is None:
        return value
    return (n_eff * value + SHRINK_K * group_mean) / (n_eff + SHRINK_K)


def compress_states(history):
    """Merge consecutive reads that show the identical page.

    Returns (times, pools, rates_matrix): times in epoch seconds, pools the
    players_tracked of each distinct state, rates_matrix[s][i] the rate of
    trophy i in state s (or None).
    """
    times, pools, rates_matrix = [], [], []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        times.append(parse_ts(read.get("at") or read.get("slot")))
        pools.append(read.get("players_tracked") or 0)
        rates_matrix.append(list(rates))
    return times, pools, rates_matrix


def forecast_case(case):
    trophies = case["trophies"]
    n_t = len(trophies)
    hist = case.get("history") or []

    # Time anchor: midnight UTC on the release date, as the house curve
    # baseline uses; x = ln(1 + hours since the anchor).
    rel = case.get("game", {}).get("release_date", "2026-11-19")
    anchor = parse_ts(rel + "T00:00:00Z")
    target_t = parse_ts(case["target"])
    x_target = math.log1p(max(0.0, (target_t - anchor) / 3600.0))
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)

    times, pools, rates_matrix = compress_states(hist)
    if not times:
        # No history at all: a flat, wide prior around low rates.
        rows = []
        for tr in trophies:
            rows.append({"id": tr["id"], "q": [0.01, 0.05, 0.25, 1.2, 6.0]})
        return {"trophies": rows}

    t_h = [max(0.0, (t - anchor) / 3600.0) for t in times]
    xs = [math.log1p(t) for t in t_h]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)
    w_slope = recency_weights(t_h, SLOPE_HALF_LIFE_H)
    n_states = len(times)

    # Pool growth: one fit for the whole case (view B's denominator), on the
    # recent window only, like the per-trophy fits.
    ln_pool = [math.log(max(p, 1)) for p in pools]
    win = [s for s in range(n_states) if t_h[s] >= t_h[-1] - SLOPE_WINDOW_H]
    if len(win) < 3:
        win = list(range(n_states))
    pool_move, _, _ = fit_move(
        [xs[s] for s in win], [ln_pool[s] for s in win],
        [w_slope[s] for s in win], x_last, x_target)
    pool_move = min(max(pool_move, -0.2), 3.5)
    ln_pool_target = ln_pool[-1] + pool_move
    pool_target = math.exp(ln_pool_target)

    # First pass: per-trophy raw fits, grouped by tier for shrinkage.
    raw = []
    for i in range(n_t):
        idx = [s for s in range(n_states)
               if i < len(rates_matrix[s]) and rates_matrix[s][i] is not None]
        if not idx:
            raw.append(None)
            continue
        sr = [rates_matrix[s][i] for s in idx]
        sp = [pools[s] for s in idx]
        logit_r = [logit_pct(r) for r in sr]
        earners = [r * p / 100.0 for r, p in zip(sr, sp)]
        ln_e = [math.log(e + 0.5) for e in earners]
        # The level is the mean of the last few states: a longer average
        # lags behind a fast-rising series, and a fitted line's endpoint is
        # dragged around by early states whose rate was still clamped at 0.
        tail = min(3, len(idx))
        level_a = sum(logit_r[-tail:]) / tail
        level_b = sum(ln_e[-tail:]) / tail
        # Slopes come from the recent window only: growth on the log-time
        # clock decelerates, so a line through the whole history points too
        # steeply at the target.  Points at exactly 0% or 100% are censored
        # (their logit is a clamp, not a measurement), so the line is fitted
        # on the strictly-inside points when there are enough of them.
        t_end = t_h[idx[-1]]
        win = [k for k, s in enumerate(idx)
               if t_h[s] >= t_end - SLOPE_WINDOW_H]
        if len(win) < 3:
            win = list(range(len(idx)))
        inside = [k for k in win if 0.0 < sr[k] < 100.0]
        if len(inside) >= 3:
            win = inside
        wx = [xs[idx[k]] for k in win]
        ww = [0.5 ** ((t_end - t_h[idx[k]]) / SLOPE_HALF_LIFE_H) for k in win]
        move_a, move_a_se, resid_a = fit_move(
            wx, [logit_r[k] for k in win], ww, xs[idx[-1]], x_target)
        move_b, move_b_se, resid_b = fit_move(
            wx, [ln_e[k] for k in win], ww, xs[idx[-1]], x_target)
        raw.append({
            "move_a": move_a, "move_a_se": move_a_se, "resid_a": resid_a,
            "level_a": level_a,
            "move_b": move_b,
            "move_b_se": move_b_se, "resid_b": resid_b,
            "level_b": level_b,
            "e_last": earners[-1], "r_last": sr[-1],
            "n": len(idx),
        })

    # Tier-group mean moves (weighted by each fit's information).
    def group_mean(key):
        groups = {}
        for i, fit in enumerate(raw):
            if fit is None:
                continue
            tier = trophies[i].get("tier", "bronze")
            w = fit["n"] / (1.0 + (fit[key + "_se"] ** 2 if fit[key + "_se"] != float("inf") else 25.0))
            num, den = groups.get(tier, (0.0, 0.0))
            groups[tier] = (num + w * fit[key], den + w)
        return {t: (num / den if den > 0 else None) for t, (num, den) in groups.items()}

    mean_a = group_mean("move_a")
    mean_b = group_mean("move_b")
    all_a = [f["move_a"] for f in raw if f is not None]
    all_b = [f["move_b"] for f in raw if f is not None]
    glob_a = sum(all_a) / len(all_a) if all_a else 0.0
    glob_b = sum(all_b) / len(all_b) if all_b else 0.0
    resids = sorted(f["resid_a"] for f in raw if f is not None)
    pooled_resid = resids[len(resids) // 2] if resids else 0.35

    # Tier medians of last rates, for trophies with no usable series.
    tier_last = {}
    for i, fit in enumerate(raw):
        if fit is not None:
            tier_last.setdefault(trophies[i].get("tier", "bronze"), []).append(fit["r_last"])

    def tier_median(tier):
        vals = sorted(tier_last.get(tier, []))
        if not vals:
            vals = sorted(v for vs in tier_last.values() for v in vs)
        return vals[len(vals) // 2] if vals else 1.0

    mus, sigmas = [], []
    for i in range(n_t):
        tier = trophies[i].get("tier", "bronze")
        fit = raw[i]
        if fit is None:
            med = max(0.02, 0.3 * tier_median(tier))
            mus.append(logit_pct(med))
            sigmas.append(1.25)
            continue
        ga = mean_a.get(tier)
        gb = mean_b.get(tier)
        move_a = shrink(fit["move_a"], ga if ga is not None else glob_a, fit["n"])
        move_b = shrink(fit["move_b"], gb if gb is not None else glob_b, fit["n"])
        move_a = min(max(move_a, -MAX_MOVE), MAX_MOVE)
        move_b = min(max(move_b, -1.0), 4.5)

        if fit["e_last"] <= 0.0 and fit["r_last"] <= 0.0:
            # Never earned in any read: a small prior rate with a wide,
            # mostly upward spread.
            med = max(0.02, 0.1 * tier_median(tier))
            mus.append(logit_pct(med))
            sigmas.append(1.3)
            continue

        mu_a = fit["level_a"] + move_a
        ln_e_target = fit["level_b"] + move_b
        r_b = 100.0 * math.exp(ln_e_target) / max(pool_target, 1.0)
        mu_b = logit_pct(min(r_b, RATE_CAP))
        mu = 0.5 * (mu_a + mu_b)

        se = fit["move_a_se"]
        slope_var = se ** 2 if se != float("inf") else (0.6 * dx) ** 2
        resid = fit["resid_a"] if fit["resid_a"] > 0 else pooled_resid
        count_var = min(0.8, 1.0 / max(fit["e_last"], 0.5))
        sigma = math.sqrt(SIGMA_BASE ** 2
                          + (SIGMA_WALK * math.sqrt(horizon_days)) ** 2
                          + slope_var
                          + RESID_WEIGHT * resid ** 2
                          + count_var)
        mus.append(mu)
        sigmas.append(min(max(sigma, 0.15), 3.0))

    rows = []
    for i in range(n_t):
        qs = []
        for z in Z:
            q = inv_logit_pct(mus[i] + z * sigmas[i])
            qs.append(round(min(max(q, 0.0), 100.0), 4))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophies[i]["id"], "q": qs})

    # The platinum is earned only by players who hold every other trophy:
    # cap its quantiles at the other trophies' elementwise minimum.
    plat = [i for i in range(n_t) if trophies[i].get("tier") == "platinum"]
    if plat and n_t > 1:
        p = plat[0]
        for k in range(len(Z)):
            cap = min(rows[j]["q"][k] for j in range(n_t) if j != p)
            if rows[p]["q"][k] > cap:
                rows[p]["q"][k] = cap
        for k in range(1, len(Z)):
            if rows[p]["q"][k] < rows[p]["q"][k - 1]:
                rows[p]["q"][k] = rows[p]["q"][k - 1]

    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        return 0
    for name in names:
        path = os.path.join(in_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
