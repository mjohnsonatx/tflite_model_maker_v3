#!/usr/bin/env python3.9
"""
world_ema_viz.py — schema-flexible WORLD_* log review (lazy-rendered).

Views: Group / Individual / 3D.  Only the visible plots are mounted; hidden
ones are Plotly.purge()'d so WebGL contexts are released.
"""

import argparse
import codecs
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

AXES = ("X", "Y", "Z")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _decode_bytes(raw: bytes) -> str:
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16")
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path, tag_re: str = r"WORLD_\w+", extra_names=None) -> pd.DataFrame:
    """
    Contract:  <TAG>|<frameNo>|<landmark>|<g0>|<g1>|...|<gN>
      g0 = raw x,y,z (required)   g1 = filt x,y,z (required)
      gN(last) = alpha if scalar (else 0.0)
      g2..gN-1 = extras (CSV-only)
    """
    payload_re = re.compile(
        rf"(?P<tag>{tag_re})\|(?P<frame>\d+)\|(?P<lm>\w+)\|(?P<rest>\S.*)"
    )
    extra_names = extra_names or []
    text = _decode_bytes(path.read_bytes())

    records = []
    for line in text.splitlines():
        m = payload_re.search(line)
        if not m:
            continue
        tag, frame, lm = m["tag"], int(m["frame"]), m["lm"]
        groups = []
        for chunk in m["rest"].strip().split("|"):
            nums = NUM_RE.findall(chunk)
            if nums:
                groups.append([float(x) for x in nums])
        if len(groups) < 2 or len(groups[0]) != 3 or len(groups[1]) != 3:
            continue
        rec = {"tag": tag, "frameNo": frame, "landmark": lm,
               "rawX": groups[0][0], "rawY": groups[0][1], "rawZ": groups[0][2],
               "filtX": groups[1][0], "filtY": groups[1][1], "filtZ": groups[1][2]}
        tail = len(groups)
        if tail >= 3 and len(groups[-1]) == 1:
            rec["alpha"] = groups[-1][0]; tail -= 1
        else:
            rec["alpha"] = 0.0
        for j, gi in enumerate(range(2, tail)):
            name = extra_names[j] if j < len(extra_names) else f"extra{j}"
            vals = groups[gi]
            if len(vals) == 3:
                rec[f"{name}X"], rec[f"{name}Y"], rec[f"{name}Z"] = vals
            elif len(vals) == 1:
                rec[name] = vals[0]
            else:
                for k, v in enumerate(vals):
                    rec[f"{name}_{k}"] = v
        records.append(rec)

    if not records:
        raise ValueError(f"No lines matching tag /{tag_re}/ in {path}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["frameNo", "landmark", "tag", "alpha"], keep="last")
    df["variant"] = [f"{t} α={a:g}" for t, a in zip(df["tag"], df["alpha"])]
    df = df.sort_values(["landmark", "variant", "frameNo"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def enrich(df, iqr_k, min_delta):
    df = df.copy()
    for ax in AXES:
        df[f"res{ax}"]    = df[f"filt{ax}"] - df[f"raw{ax}"]
        df[f"absRes{ax}"] = df[f"res{ax}"].abs()

    def _per_group(g):
        g = g.sort_values("frameNo").copy()
        any_flag = pd.Series(False, index=g.index)
        for ax in AXES:
            d_raw  = g[f"raw{ax}"].diff().abs()
            d_filt = g[f"filt{ax}"].diff().abs()
            g[f"dRaw{ax}"], g[f"dFilt{ax}"] = d_raw, d_filt
            q1, q3 = d_raw.quantile(0.25), d_raw.quantile(0.75)
            thr = q3 + iqr_k * (q3 - q1)
            flag = (d_raw > thr) & (d_raw > min_delta) & d_raw.notna()
            g[f"outlier{ax}"] = flag; any_flag |= flag
        g["outlier"] = any_flag
        return g

    return (df.groupby(["landmark", "variant"], group_keys=False)
              .apply(_per_group)
              .sort_values(["landmark", "variant", "frameNo"])
              .reset_index(drop=True))


def summarize(df):
    recs = []
    for (lm, var), g in df.groupby(["landmark", "variant"]):
        rec = {"landmark": lm, "variant": var,
               "tag": g["tag"].iloc[0], "alpha": g["alpha"].iloc[0],
               "nFrames": len(g), "nOutliers": int(g["outlier"].sum())}
        jit = []
        for ax in AXES:
            ar = g[f"absRes{ax}"]
            rec[f"std{ax}"]  = g[f"res{ax}"].std()
            rec[f"mean{ax}"] = ar.mean()
            rec[f"p25{ax}"]  = ar.quantile(0.25)
            rec[f"p50{ax}"]  = ar.quantile(0.50)
            rec[f"p75{ax}"]  = ar.quantile(0.75)
            sr, sf = g[f"dRaw{ax}"].std(), g[f"dFilt{ax}"].std()
            if sr and sr > 0:
                jit.append(max(0.0, 1.0 - sf / sr))
        for k in ("std", "mean", "p25", "p50", "p75"):
            rec[f"{k}Avg"] = sum(rec[f"{k}{ax}"] for ax in AXES) / 3.0
        rec["jitterReduction"] = sum(jit) / len(jit) if jit else 0.0
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["rankInLandmark"] = out.groupby("landmark")["p50Avg"].rank(method="min")
    out["bestInLandmark"] = out["rankInLandmark"] == 1.0
    return out


# ---------------------------------------------------------------------------
# 2D plotting
# ---------------------------------------------------------------------------
AXIS_COLORS = {"X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c"}


def _line_traces(g, landmark, showlegend):
    frames = g["frameNo"].values; out = []
    for ax in AXES:
        out.append(go.Scatter(x=frames, y=g[f"raw{ax}"],
            name=f"raw {ax}", legendgroup=f"raw{ax}", showlegend=showlegend,
            mode="lines", line=dict(color=AXIS_COLORS[ax], width=1.0), opacity=0.35,
            hovertemplate=f"{landmark} raw{ax}: %{{y:.4f}}<extra></extra>"))
        out.append(go.Scatter(x=frames, y=g[f"filt{ax}"],
            name=f"filt {ax}", legendgroup=f"filt{ax}", showlegend=showlegend,
            mode="lines", line=dict(color=AXIS_COLORS[ax], width=2.2),
            hovertemplate=f"{landmark} filt{ax}: %{{y:.4f}}<extra></extra>"))
    return out


def _outlier_traces(g, landmark, showlegend):
    out, first = [], showlegend
    for ax in AXES:
        mask = g[f"outlier{ax}"]
        if not mask.any():
            continue
        out.append(go.Scatter(x=g.loc[mask, "frameNo"], y=g.loc[mask, f"raw{ax}"],
            name="outlier", legendgroup="outlier", showlegend=first,
            mode="markers", marker=dict(color="red", size=7, symbol="x"),
            hovertemplate=f"{landmark} outlier raw{ax} @ f%{{x}}: %{{y:.4f}}<extra></extra>"))
        first = False
    return out


def build_group_figure(df, pairs):
    n = len(pairs); ncols = min(4, n); nrows = math.ceil(n / ncols)
    vspace = min(0.05, 0.5 / max(nrows - 1, 1))
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"{lm} — {v}" for lm, v in pairs],
                        shared_xaxes=True,
                        horizontal_spacing=0.04, vertical_spacing=vspace)
    for idx, (lm, var) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        g = df[(df["landmark"] == lm) & (df["variant"] == var)]
        for tr in _line_traces(g, lm, showlegend=(idx == 0)):
            fig.add_trace(tr, row=r + 1, col=c + 1)
        for tr in _outlier_traces(g, lm, showlegend=(idx == 0)):
            fig.add_trace(tr, row=r + 1, col=c + 1)
    fig.update_layout(title="Group — all (landmark, variant) pairs",
                      height=max(280 * nrows, 600), hovermode="x unified",
                      margin=dict(t=80),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1.0))
    fig.update_annotations(font_size=10)
    return fig


def build_detail_figure(g, landmark, variant):
    fig = go.Figure()
    for tr in _line_traces(g, landmark, showlegend=True):
        fig.add_trace(tr)
    for tr in _outlier_traces(g, landmark, showlegend=True):
        fig.add_trace(tr)
    fig.update_layout(title=f"{landmark} — {variant}",
                      xaxis_title="frameNo", yaxis_title="world coord (m)",
                      hovermode="x unified", height=620,
                      margin=dict(t=60, b=40),
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ---------------------------------------------------------------------------
# 3D plotting
# ---------------------------------------------------------------------------
_CAM_PRESETS = {
    "Free":          dict(eye=dict(x=1.6, y=1.6, z=1.2), up=dict(x=0, y=0, z=1)),
    "Along X → YZ":  dict(eye=dict(x=2.2, y=0,   z=0  ), up=dict(x=0, y=0, z=1)),
    "Along Y → XZ":  dict(eye=dict(x=0,   y=2.2, z=0  ), up=dict(x=0, y=0, z=1)),
    "Along Z → XY":  dict(eye=dict(x=0,   y=0,   z=2.2), up=dict(x=0, y=1, z=0)),
}


def build_3d_figure(g, landmark, variant):
    g = g.sort_values("frameNo")
    frames = g["frameNo"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=g["rawX"], y=g["rawY"], z=g["rawZ"],
        mode="lines", name="raw",
        line=dict(color="gray", width=3), opacity=0.35, customdata=frames,
        hovertemplate="raw f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                      "Z=%{z:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=g["filtX"], y=g["filtY"], z=g["filtZ"],
        mode="lines", name="filt",
        line=dict(color=frames, colorscale="Viridis", width=6,
                  colorbar=dict(title="frame", thickness=14, len=0.6, x=1.0)),
        customdata=frames,
        hovertemplate="filt f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                      "Z=%{z:.4f}<extra></extra>"))
    mask = g["outlier"]
    if mask.any():
        fig.add_trace(go.Scatter3d(
            x=g.loc[mask, "rawX"], y=g.loc[mask, "rawY"], z=g.loc[mask, "rawZ"],
            mode="markers", name="outlier",
            marker=dict(color="red", size=4, symbol="x"),
            customdata=g.loc[mask, "frameNo"],
            hovertemplate="outlier f%{customdata}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>"
                          "Z=%{z:.4f}<extra></extra>"))
    buttons = [dict(label=k, method="relayout", args=[{"scene.camera": v}])
               for k, v in _CAM_PRESETS.items()]
    fig.update_layout(
        title=f"{landmark} — {variant}",
        height=620, margin=dict(t=60, b=10, l=10, r=10),
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data", camera=_CAM_PRESETS["Free"]),
        updatemenus=[dict(type="dropdown", buttons=buttons, direction="down",
                          x=0.0, xanchor="left", y=1.06, yanchor="bottom")],
        legend=dict(x=0.0, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.6)"))
    return fig


# ---------------------------------------------------------------------------
# HTML — lazy mount/purge
# ---------------------------------------------------------------------------
def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def _quick_glance_html(row):
    star = " ★" if row.get("bestInLandmark", False) else ""
    cells = [("α", f"{row['alpha']:g}{star}"), ("frames", int(row["nFrames"])),
             ("|res| p25", _fmt(row["p25Avg"])), ("p50", _fmt(row["p50Avg"])),
             ("p75", _fmt(row["p75Avg"])), ("mean", _fmt(row["meanAvg"])),
             ("std", _fmt(row["stdAvg"])),
             ("jitter↓", f"{row['jitterReduction']*100:.1f}%"),
             ("outlier frames", int(row["nOutliers"]))]
    head = "<tr>" + "".join(f"<th>{k}</th>" for k, _ in cells) + "</tr>"
    body = "<tr>" + "".join(f"<td>{v}</td>" for _, v in cells) + "</tr>"
    return f"<table class='quick'>{head}{body}</table>"


def _per_axis_table_html(row):
    head = "<tr><th></th>" + "".join(f"<th>{ax}</th>" for ax in AXES) + "<th>avg(XYZ)</th></tr>"
    body = ""
    for label, key in [("std(res)", "std"), ("mean |res|", "mean"),
                       ("p25 |res|", "p25"), ("p50 |res|", "p50"),
                       ("p75 |res|", "p75")]:
        body += f"<tr><td><b>{label}</b></td>"
        for ax in AXES:
            body += f"<td>{_fmt(row[key + ax], 5)}</td>"
        body += f"<td>{_fmt(row[key + 'Avg'], 5)}</td></tr>"
    return f"<table class='stats'>{head}{body}</table>"


def _ranking_table_html(summary):
    s = summary.sort_values("p50Avg").reset_index(drop=True)
    head = ("<tr><th>#</th><th>landmark</th><th>variant</th>"
            "<th>p50 |res|</th><th>p75 |res|</th><th>jitter↓</th>"
            "<th>outliers</th><th>frames</th></tr>")
    rows = []
    for i, r in s.iterrows():
        star = " ★" if r["bestInLandmark"] else ""
        rows.append(
            f"<tr><td>{i+1}</td>"
            f"<td><a href='#' onclick=\"setLandmark('{r['landmark']}');return false;\">"
            f"{r['landmark']}</a></td>"
            f"<td style='text-align:left'>{r['variant']}{star}</td>"
            f"<td>{_fmt(r['p50Avg'])}</td><td>{_fmt(r['p75Avg'])}</td>"
            f"<td>{r['jitterReduction']*100:.1f}%</td>"
            f"<td>{int(r['nOutliers'])}</td><td>{int(r['nFrames'])}</td></tr>")
    return ("<details class='ranking'><summary><b>Ranking — all (landmark, variant) "
            "by p50 |res| (★ = best variant for that landmark)</b></summary>"
            f"<table class='stats'>{head}{''.join(rows)}</table></details>")


def _plotlyjs_cdn_tag():
    try:
        from plotly.offline.offline import get_plotlyjs_version
        ver = get_plotlyjs_version()
    except Exception:
        ver = "2.35.2"
    return (f'<script src="https://cdn.plot.ly/plotly-{ver}.min.js" '
            f'charset="utf-8"></script>')


def _lazy_plot_html(fig, plot_id, keep=False):
    """Ship figure as JSON + empty placeholder div; mounted on demand in JS."""
    h = int(fig.layout.height or 450)
    fig_json = fig.to_json().replace("</", "<\\/")   # script-tag safety
    keep_attr = ' data-keep="1"' if keep else ""
    return (f'<div id="plot-{plot_id}" class="lazy-plot" data-figid="{plot_id}"'
            f'{keep_attr} style="min-height:{h}px"></div>'
            f'<script type="application/json" id="figdata-{plot_id}">'
            f'{fig_json}</script>')


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"/>
<title>WORLD Filter Review</title>
<style>
  body { font-family: sans-serif; margin: 0; padding: 12px; }
  .toggle-bar { position: sticky; top: 0; background: #fff; padding: 8px 0;
                border-bottom: 1px solid #ddd; z-index: 10;
                display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .toggle-bar button, .toggle-bar select {
      font-size: 14px; padding: 6px 12px; cursor: pointer; }
  .toggle-bar button.active { background: #1f77b4; color: #fff; border: none; }
  .nav { display: none; gap: 4px; align-items: center; }
  .nav.active { display: inline-flex; }
  .nav button { padding: 6px 10px; font-size: 16px; }
  .view { display: none; } .view.active { display: block; }
  .lm-block { display: none; } .lm-block.active { display: block; }
  .alpha-block { border: 1px solid #eee; border-radius: 6px;
                 margin: 14px 0; padding: 8px 8px 14px 8px; }
  .lazy-plot { width: 100%; }
  table.stats, table.quick { border-collapse: collapse; margin: 6px 4px; font-size: 12px; }
  table.stats th, table.stats td,
  table.quick th, table.quick td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; }
  table.stats th, table.quick th { background: #f3f3f3; }
  table.quick td, table.quick th { font-size: 13px; }
  .stats-caption { font-size: 12px; color: #555; margin: 2px 4px; }
  .params { font-size: 12px; color: #666; margin-left: auto; }
  details.ranking { margin: 10px 0 18px 0; }
  details.ranking summary { cursor: pointer; font-size: 14px; padding: 6px 0; }
</style></head><body>
"""

HTML_TAIL_TMPL = """
<script>
var LANDMARKS = __LANDMARKS__; var lmIndex = 0;
var VIEWS = ['group','detail','3d'];
var mounted = {};
var PLOT_CFG = {responsive: true, displaylogo: false};

function isLive(div){
  var v = div.closest('.view');
  if (!v || !v.classList.contains('active')) return false;
  var b = div.closest('.lm-block');
  return !b || b.classList.contains('active');
}
function refreshMounts(){
  document.querySelectorAll('.lazy-plot').forEach(function(div){
    var id = div.dataset.figid, live = isLive(div);
    if (live && !mounted[id]) {
      try {
        var j = JSON.parse(document.getElementById('figdata-'+id).textContent);
        Plotly.newPlot(div, j.data, j.layout, PLOT_CFG);
        mounted[id] = true;
      } catch(e){ console.error('mount '+id, e); }
    } else if (!live && mounted[id] && !div.dataset.keep) {
      try { Plotly.purge(div); } catch(e){}
      mounted[id] = false;
    }
  });
}
function setView(v){
  VIEWS.forEach(function(k){
    document.getElementById('view-'+k).classList.toggle('active', v===k);
    document.getElementById('btn-'+k).classList.toggle('active', v===k);
  });
  document.getElementById('nav').classList.toggle('active', v!=='group');
  refreshMounts();
}
function setLandmark(lm){
  var i = LANDMARKS.indexOf(lm); if (i<0) return; lmIndex=i;
  document.querySelectorAll('.lm-block').forEach(function(el){
    el.classList.toggle('active', el.dataset.lm === lm); });
  document.getElementById('lm-select').value = lm;
  document.getElementById('lm-pos').textContent = (i+1)+' / '+LANDMARKS.length;
  window.scrollTo({top:0});
  refreshMounts();
}
function stepLandmark(d){ var n=LANDMARKS.length;
  lmIndex=((lmIndex+d)%n+n)%n; setLandmark(LANDMARKS[lmIndex]); }
document.addEventListener('keydown', function(e){
  if(document.getElementById('view-group').classList.contains('active')) return;
  if(e.key==='ArrowRight'){stepLandmark(1);e.preventDefault();}
  if(e.key==='ArrowLeft'){stepLandmark(-1);e.preventDefault();}
});
refreshMounts();   // initial mount of the default (Individual) view
</script></body></html>
"""


def write_html(df, summary, pairs, landmarks, out_path,
               iqr_k, min_delta, extras_label, render_group=True):
    summary_idx = summary.set_index(["landmark", "variant"], drop=False)
    parts = [HTML_HEAD, _plotlyjs_cdn_tag()]

    pid = 0
    def nid():
        nonlocal pid; pid += 1; return f"p{pid}"

    opts = "".join(f"<option value='{lm}'>{lm}</option>" for lm in landmarks)
    parts.append(
        "<div class='toggle-bar'>"
        "<button id='btn-group'  onclick=\"setView('group')\">Group view</button>"
        "<button id='btn-detail' class='active' onclick=\"setView('detail')\">Individual view</button>"
        "<button id='btn-3d'     onclick=\"setView('3d')\">3D view</button>"
        "<span id='nav' class='nav active'>"
        "  <button onclick='stepLandmark(-1)' title='Previous (←)'>◀</button>"
        f" <select id='lm-select' onchange='setLandmark(this.value)'>{opts}</select>"
        "  <button onclick='stepLandmark(1)' title='Next (→)'>▶</button>"
        f" <span id='lm-pos'>1 / {len(landmarks)}</span></span>"
        f"<span class='params'>extras (CSV only): {extras_label} · "
        f"outlier: IQR·k={iqr_k:g}, minΔ={min_delta:g}</span></div>"
    )

    # ---- group (lazy, keep-once-mounted) -------------------------------
    parts.append("<div id='view-group' class='view'>")
    if render_group:
        parts.append(_lazy_plot_html(build_group_figure(df, pairs), nid(), keep=True))
    else:
        parts.append("<p style='color:#888;padding:20px'>Group view disabled "
                     "(<code>--no-group</code>).</p>")
    parts.append("</div>")

    # ---- individual (lazy, purge on hide) ------------------------------
    parts.append("<div id='view-detail' class='view active'>")
    parts.append(_ranking_table_html(summary))
    for i, lm in enumerate(landmarks):
        active = " active" if i == 0 else ""
        parts.append(f"<div class='lm-block{active}' data-lm='{lm}'>")
        for var in sorted(df.loc[df["landmark"] == lm, "variant"].unique()):
            g = df[(df["landmark"] == lm) & (df["variant"] == var)]
            srow = summary_idx.loc[(lm, var)]
            parts.append("<div class='alpha-block'>")
            parts.append(_quick_glance_html(srow))
            parts.append(_lazy_plot_html(build_detail_figure(g, lm, var), nid()))
            parts.append("<div class='stats-caption'>res = filt − raw · "
                         "jitter↓ = 1 − std(Δfilt)/std(Δraw), avg XYZ · "
                         "outliers: |Δraw| > Q3 + k·IQR and > minΔ</div>")
            parts.append(_per_axis_table_html(srow))
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")

    # ---- 3D (lazy, purge on hide — this is what frees WebGL) -----------
    parts.append("<div id='view-3d' class='view'>")
    for i, lm in enumerate(landmarks):
        active = " active" if i == 0 else ""
        parts.append(f"<div class='lm-block{active}' data-lm='{lm}'>")
        for var in sorted(df.loc[df["landmark"] == lm, "variant"].unique()):
            g = df[(df["landmark"] == lm) & (df["variant"] == var)]
            srow = summary_idx.loc[(lm, var)]
            parts.append("<div class='alpha-block'>")
            parts.append(_quick_glance_html(srow))
            parts.append(_lazy_plot_html(build_3d_figure(g, lm, var), nid()))
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")

    parts.append(HTML_TAIL_TMPL.replace("__LANDMARKS__", json.dumps(landmarks)))
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"[info] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Schema-flexible WORLD_* filter review (lazy-rendered 2D + 3D).")
    ap.add_argument("logfile", type=Path)
    ap.add_argument("--landmarks", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("world_filter_review.html"))
    ap.add_argument("--csv-frames", type=Path, default=None)
    ap.add_argument("--csv-summary", type=Path, default=None)
    ap.add_argument("--tag-re", default=r"WORLD_\w+")
    ap.add_argument("--schema", default=None,
                    help="Names for extra groups between filt and alpha, e.g. 'vel'.")
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--no-group", action="store_true",
                    help="Skip building the Group grid (fastest load, smallest HTML).")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    extra_names = [s.strip() for s in args.schema.split(",")] if args.schema else []
    df = parse_log(args.logfile, tag_re=args.tag_re, extra_names=extra_names)

    all_lms = sorted(df["landmark"].unique())
    if args.landmarks:
        lms = [lm for lm in args.landmarks if lm in all_lms]
        missing = set(args.landmarks) - set(lms)
        if missing:
            print(f"[warn] not in log: {sorted(missing)}", file=sys.stderr)
        df = df[df["landmark"].isin(lms)].reset_index(drop=True)
    else:
        lms = all_lms

    df = enrich(df, iqr_k=args.iqr_k, min_delta=args.min_delta)
    summary = summarize(df)
    pairs = sorted(set(zip(df["landmark"], df["variant"])))

    core = {"tag", "frameNo", "landmark", "variant", "alpha",
            "rawX", "rawY", "rawZ", "filtX", "filtY", "filtZ"}
    extras_found = sorted(c for c in df.columns
                          if c not in core and not c.startswith(
                              ("res", "absRes", "dRaw", "dFilt", "outlier")))
    extras_label = ", ".join(extras_found) if extras_found else "none"

    print(f"[info] {len(df)} rows, {len(lms)} landmarks, "
          f"{len(pairs)} (landmark,variant) pairs, "
          f"frames {df['frameNo'].min()}..{df['frameNo'].max()}")
    print(f"[info] extra columns captured (CSV only): {extras_label}")

    stem = args.out.with_suffix("")
    frames_csv  = args.csv_frames  or Path(f"{stem}_frames.csv")
    summary_csv = args.csv_summary or Path(f"{stem}_summary.csv")
    df.to_csv(frames_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"[info] wrote {frames_csv}")
    print(f"[info] wrote {summary_csv}")

    write_html(df, summary, pairs, lms, args.out,
               iqr_k=args.iqr_k, min_delta=args.min_delta,
               extras_label=extras_label,
               render_group=not args.no_group)

    if args.open:
        import webbrowser
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()

