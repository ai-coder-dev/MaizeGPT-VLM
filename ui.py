"""Presentation helpers for the MaizeGPT VLM Streamlit app.

Every function returns an HTML string (or injects CSS). No model logic lives here.
"""

import base64
import html
import io
import json
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"

TIER_LABEL = {
    "critical": "Critical",
    "moderate": "Moderate",
    "nutritional": "Nutritional",
    "healthy": "Healthy",
    "review": "Needs review",
}
TIER_DOT = {
    "critical": "#B42318",
    "moderate": "#B54708",
    "nutritional": "#175CD3",
    "healthy": "#067647",
    "review": "#475467",
}

LEAF_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#F3E3B8" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M5 20c0-8 5-14 15-16-1 10-7 15-15 16z"/>'
    '<path d="M5 20c3-4 6-7 10-9"/></svg>'
)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def load_css() -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def topbar() -> str:
    return f"""
<div class="cv-topbar">
  <div class="cv-brand">
    <div class="cv-mark">{LEAF_SVG}</div>
    <div class="cv-wordmark">MaizeGPT VLM<small>Maize pathology diagnostics</small></div>
  </div>
</div>"""


def hero() -> str:
    return """
<div class="cv-hero">
  <div class="cv-eyebrow">Vision-language decision support</div>
  <h1>Field diagnosis for <em>maize</em> diseases, pests and deficiencies.</h1>
  <p class="cv-lede">Upload photos of the affected leaf, stalk, cob or tassel. MaizeGPT VLM lists the most likely
  causes from the ICAR-IIMR list of maize problems, says how sure it is, and gives ICAR-IIMR management advice only
  when it is confident. Otherwise it tells you what to check in the field.</p>
</div>"""


def stat_strip(stats) -> str:
    cells = "".join(
        f'<div class="cv-stat"><div class="cv-stat-v">{esc(v)}</div><div class="cv-stat-l">{esc(l)}</div></div>'
        for v, l in stats
    )
    return f'<div class="cv-stats">{cells}</div>'


def section(num: str, title: str, desc: str = "") -> str:
    d = f'<div class="cv-sec-d">{esc(desc)}</div>' if desc else ""
    n = f'<span class="cv-sec-n">{esc(num)}</span>' if num else ""
    return f'<div class="cv-sec">{n}<span class="cv-sec-t">{esc(title)}</span></div>{d}'


def condition_groups(groups: dict) -> str:
    out = ['<div class="cv-empty">']
    for title, (tier, items) in groups.items():
        chips = "".join(
            f'<span class="cv-chip"><i style="background:{TIER_DOT[tier]}"></i>{esc(n)}</span>' for n in items
        )
        out.append(f'<div class="cv-group"><div class="cv-group-t">{esc(title)}</div><div class="cv-chips">{chips}</div></div>')
    out.append("</div>")
    return "".join(out)


def _data_uri(image, size: int, quality: int = 82) -> str:
    img = image.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def photo(image, name: str) -> str:
    return f'<div class="cv-photo"><img src="{_data_uri(image, 1000, 88)}" alt="{esc(name)}"/></div>'


def thumb(image, name: str, size: int = 320) -> str:
    return (
        f'<div class="cv-thumb"><img src="{_data_uri(image, size)}" alt="{esc(name)}"/></div>'
        f'<div class="cv-thumb-n" title="{esc(name)}">{esc(name)}</div>'
    )


def kpi(label: str, value: str, foot: str = "", suffix: str = "", alert: bool = False) -> str:
    s = f"<small>{esc(suffix)}</small>" if suffix else ""
    f = f'<div class="cv-kpi-f">{esc(foot)}</div>' if foot else ""
    cls = "cv-kpi alert" if alert else "cv-kpi"
    return f'<div class="{cls}"><div class="cv-kpi-l">{esc(label)}</div><div class="cv-kpi-v">{esc(value)}{s}</div>{f}</div>'


def badge(tier: str, text: str) -> str:
    return f'<span class="cv-badge {esc(tier)}"><i></i>{esc(text)}</span>'


def diagnosis_head(tier: str, severity: str, title: str, agent: str) -> str:
    sub = f"Causal agent · <em>{esc(agent)}</em>" if agent and agent != "None" else esc(severity)
    return f'<div class="cv-dx">{esc(title)}</div><div class="cv-agent">{sub}</div>'


STATUS_COPY = {
    "confident": ("ok", "Confident diagnosis", "Both checks agree. Follow the ICAR-IIMR management below."),
    "likely": ("warn", "Likely cause: confirm before spraying",
               "Check the signs below in the field, or ask your KVK, before using any chemical."),
    "uncertain": ("info", "Not sure yet: take a clearer photo",
                  "The photo does not show enough to decide. Possible causes are listed below."),
    "retake": ("info", "Photo not usable", "This does not look like a clear photo of maize."),
}


def status_banner(status: str, problem_id: str = "") -> str:
    if problem_id == "healthy" and status == "confident":
        cls, title, text = "ok", "No problem visible", "Both checks agree the plant looks healthy. Keep scouting regularly."
        return f'<div class="cv-status-banner {cls}"><b>{esc(title)}</b><span>{esc(text)}</span></div>'
    cls, title, text = STATUS_COPY.get(status, STATUS_COPY["uncertain"])
    return f'<div class="cv-status-banner {cls}"><b>{esc(title)}</b><span>{esc(text)}</span></div>'


def signs_block(signs: str, visible) -> str:
    seen = "".join(f"<li>{esc(s)}</li>" for s in (visible or [])[:4])
    seen_html = f'<div class="cv-label">What the model sees</div><ul class="cv-list">{seen}</ul>' if seen else ""
    key = f'<div class="cv-label">How ICAR-IIMR describes it</div><p class="cv-signs">{esc(signs)}</p>' if signs else ""
    return f'<div class="cv-signs-block">{seen_html}{key}</div>'


def advice_block(advice: dict, status: str) -> str:
    healthy = status == "confident" and not advice.get("chemical") and not advice.get("steps")
    steps = "".join(f"<li>{esc(s)}</li>" for s in advice.get("steps", []))
    chem = "".join(f"<li>{esc(s)}</li>" for s in advice.get("chemical", []))
    label = "What to do" if status != "confident" else "Recommended management"
    body = f'<ul class="cv-list">{steps}</ul>' if steps else ""
    if chem:
        body += f'<div class="cv-label" style="margin-top:10px">Chemical control (ICAR-IIMR)</div><ul class="cv-list">{chem}</ul>'
    sources = advice.get("source") or []
    if sources:
        body += f'<div class="cv-source">Source: {esc("; ".join(sources[:2]))}</div>'
    cls = "cv-rx healthy" if healthy else "cv-rx"
    return f'<div class="{cls}"><div class="cv-label">{label}</div>{body}</div>'


def confidence_meter(conf: float) -> str:
    pct = max(0.0, min(conf, 100.0))
    return f"""
<div class="cv-meter">
  <div class="cv-meter-row"><div class="cv-label">Model confidence</div><div class="cv-meter-v">{pct:.1f}%</div></div>
  <div class="cv-track"><div class="cv-fill" style="width:{pct:.1f}%"></div></div>
</div>"""


def differential(top3, label: str = "Differential diagnosis") -> str:
    rows = []
    for i, (name, p) in enumerate(top3):
        cls = "cv-diff-row top" if i == 0 else "cv-diff-row"
        rows.append(
            f'<div class="{cls}"><div class="cv-diff-n">{esc(name)}</div>'
            f'<div class="cv-diff-bar"><span style="width:{max(p, 1.5):.1f}%"></span></div>'
            f'<div class="cv-diff-p">{p:.1f}%</div></div>'
        )
    return f'<div class="cv-diff"><div class="cv-label">{esc(label)}</div>{"".join(rows)}</div>'


def rx(text: str, healthy: bool = False) -> str:
    cls = "cv-rx healthy" if healthy else "cv-rx"
    label = "Recommended practice" if healthy else "Recommended treatment"
    return f'<div class="{cls}"><div class="cv-label">{label}</div><p>{esc(text)}</p></div>'


def archive_link(url: str) -> str:
    if isinstance(url, str) and url.startswith("http"):
        return f'<a class="cv-link" href="{esc(url)}" target="_blank" rel="noopener">View archived image ↗</a>'
    return ""


def sidebar_about(splits) -> str:
    kv = "".join(f'<div class="cv-kv"><span>{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in splits)
    return f"""
<div class="cv-side">
  <h3>About the model</h3>
  <p>A SigLIP image model checked by a second vision model, grounded in ICAR-IIMR publications.</p>
  <div class="cv-steps">
    <div class="cv-step"><div class="cv-step-n">1</div><div><b>SigLIP classifier</b><span>Google SigLIP Base · 384px features with a calibrated head trained on agronomist, ICAR-IIMR and open field images.</span></div></div>
    <div class="cv-step"><div class="cv-step-n">2</div><div><b>Second check</b><span>When unsure, or for stalk, cob and tassel photos, a vision model reviews the photo against the ICAR-IIMR list of maize problems.</span></div></div>
    <div class="cv-step"><div class="cv-step-n">3</div><div><b>Safe advice</b><span>Spray advice only for confident diagnoses, using ICAR-IIMR recommendations; otherwise field checks and KVK referral.</span></div></div>
  </div>
  <div class="cv-label">Dataset</div>
  {kv}
  <div class="cv-disclaimer">Decision support only. Confirm with a qualified agronomist before any chemical application, and always follow product label rates.</div>
</div>"""


def scroll_to(anchor_id: str) -> None:
    """Smooth-scroll the main page to an element id once it has rendered."""
    import streamlit.components.v1 as components

    # The logic is injected into the parent page so it runs on the main
    # window's timers (hidden zero-size iframes get their timers throttled).
    parent_js = f"""
      (function () {{
        let tries = 0, lastH = -1, stable = 0;
        const tick = () => {{
          const el = document.getElementById("{anchor_id}");
          const main = document.querySelector('[data-testid="stMain"]');
          const h = main ? main.scrollHeight : document.body.scrollHeight;
          stable = (el && h === lastH) ? stable + 1 : 0;
          lastH = h;
          if (el && (stable >= 3 || tries > 40)) {{
            el.scrollIntoView({{ behavior: "smooth", block: "start" }});
            // Fallback: if smooth scrolling can't animate (background tab,
            // reduced motion), jump instantly.
            setTimeout(() => {{
              if (Math.abs(el.getBoundingClientRect().top) > 40) {{
                el.scrollIntoView({{ behavior: "auto", block: "start" }});
              }}
            }}, 900);
          }} else if (tries++ < 80) {{
            setTimeout(tick, 100);
          }}
        }};
        setTimeout(tick, 100);
      }})();
    """
    components.html(
        "<script>"
        "const s = window.parent.document.createElement('script');"
        f"s.textContent = {json.dumps(parent_js)};"
        "window.parent.document.body.appendChild(s);"
        "</script>",
        height=0,
    )


def footer() -> str:
    return '<div class="cv-foot"><span>MaizeGPT VLM · Maize pathology diagnostics</span><span>SigLIP classifier + second-check vision model · advice from ICAR-IIMR publications</span></div>'
