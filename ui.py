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
}
TIER_DOT = {
    "critical": "#B42318",
    "moderate": "#B54708",
    "nutritional": "#175CD3",
    "healthy": "#067647",
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


def hero() -> str:
    return """
<div class="cv-hero">
  <div class="cv-eyebrow">Vision-language decision support</div>
  <h1>Field-grade diagnosis for <em>maize</em> foliar disorders.</h1>
  <p class="cv-lede">Upload leaf photographs from the field. MaizeGPT VLM identifies the most likely disease,
  pest or nutrient deficiency, shows how sure it is, and returns a grounded treatment protocol in seconds.</p>
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
    agent_html = (
        f'<div class="cv-agent">Causal agent · <em>{esc(agent)}</em></div>'
        if agent and agent != "None"
        else '<div class="cv-agent">No pathogen or deficiency signature detected</div>'
    )
    return f'<div class="cv-dx">{esc(title)}</div>{agent_html}'


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
  <p>A two-stage hybrid pipeline built to run on modest cloud hardware without losing diagnostic precision.</p>
  <div class="cv-steps">
    <div class="cv-step"><div class="cv-step-n">1</div><div><b>Vision encoder</b><span>Google SigLIP Base · patch16 · 384px, with mean + max spatial token pooling.</span></div></div>
    <div class="cv-step"><div class="cv-step-n">2</div><div><b>Calibrated diagnostic head</b><span>Multi-class classifier with probability calibration across 12 conditions.</span></div></div>
    <div class="cv-step"><div class="cv-step-n">3</div><div><b>Agronomic grounding</b><span>Maize-specific protocols map each diagnosis to a treatment.</span></div></div>
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
    return '<div class="cv-foot"><span>MaizeGPT VLM · Maize pathology diagnostics</span><span>SigLIP vision encoder · calibrated diagnostic head</span></div>'
