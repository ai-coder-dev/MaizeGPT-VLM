import base64
import datetime
import json
import os

import gspread
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google.oauth2 import service_account
from PIL import Image
import ui
from vlm_pipeline import HEAD_DIR, MaizeDiagnoser, advice_text, load_problems

# --- Page Configuration ---
st.set_page_config(
    page_title="MaizeGPT VLM · Maize Pathology Diagnostics",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.load_css()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# tier drives badge colour and ordering: critical > moderate > nutritional > review > healthy
TIER_ORDER = {"critical": 0, "moderate": 1, "nutritional": 2, "review": 3, "healthy": 4}
STATUS_LABEL = {
    "confident": "Confident",
    "likely": "Likely · confirm in field",
    "uncertain": "Uncertain · retake photo",
    "retake": "Photo not usable",
}
PROBLEMS = load_problems()
HEAD_META = json.loads((HEAD_DIR / "meta.json").read_text())
TRAINED = set(HEAD_META["classes"])
CONDITION_GROUPS = {
    "Diseases": ("moderate", [p.name for p in PROBLEMS.values() if p.category == "disease"]),
    "Insect pests": ("critical", [p.name for p in PROBLEMS.values() if p.category == "pest"]),
    "Nutrient deficiencies": ("nutritional", [p.name for p in PROBLEMS.values() if p.category == "nutrient"]),
}


# --- Integrations (fail quietly; the UI reports status) ---
def _secret(*path):
    try:
        node = st.secrets
        for key in path:
            node = node[key]
        return node
    except Exception:
        return None


SHEETS_CONFIGURED = bool(_secret("gcp_service_account") and _secret("drive_config", "sheet_id"))
IMGBB_KEY = _secret("IMGBB_API_KEY") or _secret("drive_config", "IMGBB_API_KEY")


def get_google_sheet():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        raw_key = creds_dict.get("private_key", "")
        if "\\n" in raw_key:
            creds_dict["private_key"] = raw_key.replace("\\n", "\n")
        creds_dict["private_key"] = creds_dict["private_key"].strip()

        credentials = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        gc_client = gspread.authorize(credentials)
        sheet_id = st.secrets["drive_config"]["sheet_id"]
        return gc_client.open_by_key(sheet_id).sheet1
    except Exception:
        return None


def upload_image_to_cloud(image_bytes, filename):
    if not IMGBB_KEY:
        return "Not archived"
    try:
        payload = {
            "key": IMGBB_KEY,
            "name": filename,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
        }
        res = requests.post("https://api.imgbb.com/1/upload", data=payload, timeout=25)
        data = res.json()
        if data.get("status") == 200:
            return data["data"]["url"]
        return f"Upload error: {data.get('error', {}).get('message', 'failed')}"
    except Exception as e:
        return f"Upload error: {type(e).__name__}"


# --- Cache Model Pipeline ---
@st.cache_resource(show_spinner="Warming up the vision model. The first load takes about 30 seconds…")
def load_diagnoser():
    api_key = _secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return MaizeDiagnoser(openai_api_key=api_key, second_check_model=_secret("SECOND_CHECK_MODEL"))


# --- Session state ---
ss = st.session_state
ss.setdefault("uploader_key", 0)
ss.setdefault("results", None)
ss.setdefault("run_meta", None)
ss.setdefault("scroll_to_results", False)


def reset_batch():
    ss.uploader_key += 1
    ss.results = None
    ss.run_meta = None


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        ui.sidebar_about([
            ("Training images", f"{sum(HEAD_META['support'].values()):,}"),
            ("Classes trained", str(len(TRAINED))),
            ("IIMR problems covered", str(len(PROBLEMS))),
            ("Held-out top-3", f"{HEAD_META.get('eval', {}).get('top3', 0):.0%}"),
        ]),
        unsafe_allow_html=True,
    )

# --- Header ---
st.markdown(ui.topbar(), unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)
st.markdown(
    ui.stat_strip([
            (str(len(PROBLEMS)), "ICAR-IIMR maize problems"),
            (f"{HEAD_META.get('eval', {}).get('top3', 0):.0%}", "held-out top-3 accuracy"),
            (f"{sum(HEAD_META['support'].values()):,}", "labelled training images"),
            ("2-stage", "SigLIP + second check"),
        ]),
    unsafe_allow_html=True,
)

diagnoser = load_diagnoser()

# --- Workspace ---
col_session, col_upload = st.columns([1, 1.65], gap="large")

with col_session:
    with st.container(border=True, key="cvcard_session"):
        st.markdown(ui.section("01", "Session details", "Attached to every record in this batch."), unsafe_allow_html=True)
        field_agent = st.text_input("Agronomist / field officer", value="Field Officer")
        plot_id = st.text_input("Plot or farm identifier", value="Zone A · Block 4")
        notes = st.text_area("Field notes", placeholder="e.g. High humidity after monsoon rain", height=88)

        st.markdown('<div class="cv-divider"></div>', unsafe_allow_html=True)
        st.markdown(ui.section("", "Integrations"), unsafe_allow_html=True)
        sync_sheets = st.toggle("Log results to Google Sheets", value=SHEETS_CONFIGURED, disabled=not SHEETS_CONFIGURED)
        if not SHEETS_CONFIGURED:
            st.markdown('<div class="cv-integ-note">Not configured for this deployment</div>', unsafe_allow_html=True)
        archive_imgs = st.toggle("Archive images to cloud storage", value=bool(IMGBB_KEY), disabled=not IMGBB_KEY)
        if not IMGBB_KEY:
            st.markdown('<div class="cv-integ-note">Not configured for this deployment</div>', unsafe_allow_html=True)

with col_upload:
    with st.container(border=True, key="cvcard_upload"):
        st.markdown(ui.section("02", "Plant photos", "Close, well-lit photos of the affected leaf, stalk (split open if rotting), cob or tassel. JPG, PNG or WEBP."), unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Leaf images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"uploader_{ss.uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded_files:
            n = len(uploaded_files)
            shown = uploaded_files[:8]
            cells = []
            for f in shown:
                try:
                    cells.append(f"<div>{ui.thumb(Image.open(f).convert('RGB'), f.name)}</div>")
                except Exception:
                    cells.append(f'<div><div class="cv-thumb"></div><div class="cv-thumb-n">{ui.esc(f.name)}</div></div>')
            more = f" · showing first {len(shown)}" if n > len(shown) else ""
            st.markdown(
                f'<div class="cv-queue-meta"><b>{n} specimen{"s" if n != 1 else ""}</b> ready for analysis{more}</div>'
                f'<div class="cv-thumbs {"cv-thumbs-lg" if n <= 3 else ""}">{"".join(cells)}</div>',
                unsafe_allow_html=True,
            )
            # Keyed container lets CSS pin the actions to the bottom of the card
            with st.container(key="cvactions"):
                b1, b2 = st.columns([2.4, 1])
                with b1:
                    run_batch = st.button(
                        f"Analyse {n} specimen{'s' if n != 1 else ''}",
                        type="primary", icon=":material/biotech:", width="stretch",
                    )
                with b2:
                    st.button("Clear", icon=":material/close:", width="stretch", on_click=reset_batch)
        else:
            run_batch = False
            st.markdown(
                '<div class="cv-sec-d" style="margin:18px 0 0">MaizeGPT checks for these ICAR-IIMR maize problems:</div>'
                + ui.condition_groups(CONDITION_GROUPS),
                unsafe_allow_html=True,
            )

# --- Run analysis ---
if run_batch and uploaded_files:
    results, rows_for_sheet = [], []
    total = len(uploaded_files)
    with st.status(f"Analysing {total} specimen{'s' if total != 1 else ''}…", expanded=True) as status:
        progress = st.progress(0.0)
        for idx, file in enumerate(uploaded_files):
            st.write(f"**{idx + 1}/{total}** · `{file.name}`: extracting visual features and classifying")
            image_bytes = file.getvalue()
            image = Image.open(file).convert("RGB")
            dx = diagnoser.diagnose(image)
            dx["severity"] = STATUS_LABEL.get(dx["status"], dx["status"])
            dx["advice_view"] = advice_text(dx)
            dx["rx"] = " ".join(dx["advice_view"]["steps"] + dx["advice_view"]["chemical"])

            cloud_url = "Not archived"
            if archive_imgs:
                st.write(f"Archiving `{file.name}`")
                cloud_url = upload_image_to_cloud(image_bytes, file.name)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_for_sheet.append([
                now_str, file.name, cloud_url, dx["title"], f"{dx['conf']:.2f}%",
                dx["severity"], field_agent, plot_id, ", ".join(n for n, _ in dx["top3"]),
            ])
            results.append({**dx, "file": file.name, "url": cloud_url, "image": image, "time": now_str})
            progress.progress((idx + 1) / total)

        synced, sync_note = 0, "Off"
        if sync_sheets:
            st.write("Logging records to Google Sheets")
            sheet = get_google_sheet()
            if sheet is None:
                sync_note = "Unavailable"
            else:
                try:
                    sheet.append_rows(rows_for_sheet)
                    synced, sync_note = len(rows_for_sheet), "Synced"
                except Exception:
                    sync_note = "Failed"
        status.update(label="Analysis complete", state="complete", expanded=False)

    if sync_sheets and sync_note != "Synced":
        st.toast("Google Sheets sync is unavailable. Results are kept in this session.", icon=":material/cloud_off:")
    elif synced:
        st.toast(f"{synced} record{'s' if synced != 1 else ''} logged to Google Sheets", icon=":material/check_circle:")

    results.sort(key=lambda r: (TIER_ORDER[r["tier"]], -r["conf"]))
    ss.results = results
    ss.run_meta = {
        "synced": synced, "sync_note": sync_note, "sync_on": sync_sheets,
        "officer": field_agent, "plot": plot_id, "notes": notes,
        "time": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    ss.scroll_to_results = True

# --- Results ---
if ss.results:
    results, meta = ss.results, ss.run_meta
    n = len(results)
    need_action = sum(1 for r in results if r["status"] == "confident" and r["tier"] in ("critical", "moderate"))
    avg_conf = float(np.mean([r["conf"] for r in results]))

    st.markdown('<div id="cv-results" style="height:40px;scroll-margin-top:8px"></div>', unsafe_allow_html=True)
    if ss.scroll_to_results:
        ss.scroll_to_results = False
        ui.scroll_to("cv-results")
    h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
    with h1:
        st.markdown(
            f'<div class="cv-eyebrow">03 · Diagnostic report</div>'
            f'<div class="cv-h2">Batch results</div>'
            f'<div class="cv-sub">{ui.esc(meta["plot"])} · {ui.esc(meta["officer"])} · {ui.esc(meta["time"])}</div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.button("New batch", icon=":material/restart_alt:", width="stretch", on_click=reset_batch)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(ui.kpi("Specimens analysed", str(n), foot="Photos in this batch"), unsafe_allow_html=True)
    k2.markdown(
        ui.kpi("Need urgent action", str(need_action), foot="Confident pest or disease", alert=need_action > 0),
        unsafe_allow_html=True,
    )
    k3.markdown(ui.kpi("Mean confidence", f"{avg_conf:.1f}", suffix="%", foot="Average of top predictions"), unsafe_allow_html=True)
    if meta["sync_on"]:
        k4.markdown(ui.kpi("Logged to Sheets", str(meta["synced"]), suffix=f"/ {n}", foot=meta["sync_note"]), unsafe_allow_html=True)
    else:
        k4.markdown(ui.kpi("Logged to Sheets", "Off", foot="Sync disabled"), unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    tab_reports, tab_table, tab_export = st.tabs(["Specimen reports", "Summary table", "Export"])

    with tab_reports:
        for i, r in enumerate(results):
            with st.container(border=True, key=f"cvcard_report_{i}"):
                c_img, c_dx = st.columns([1, 1.75], gap="large")
                with c_img:
                    st.markdown(ui.photo(r["image"], r["file"]), unsafe_allow_html=True)
                    st.markdown(f'<div class="cv-file">{ui.esc(r["file"])}</div>', unsafe_allow_html=True)
                with c_dx:
                    st.markdown(
                        ui.diagnosis_head(r["tier"], r["severity"], r["title"], r["agent"])
                        + ui.status_banner(r["status"], r["problem_id"])
                        + ui.confidence_meter(r["conf"])
                        + ui.differential(r["top3"], "Possible causes (top 3)")
                        + ui.signs_block(r["signs"], r["visible_signs"])
                        + ui.advice_block(r["advice_view"], r["status"])
                        + ui.archive_link(r["url"]),
                        unsafe_allow_html=True,
                    )

    df_export = pd.DataFrame([{
        "Image Name": r["file"],
        "Diagnosis": r["title"],
        "Confidence": f"{r['conf']:.2f}%",
        "Severity": r["severity"],
        "Causal Factor": r["agent"],
        "Status": r["severity"],
        "Top 3": "; ".join(n for n, _ in r["top3"]),
        "Recommended Action": r["rx"],
        "Image URL": r["url"],
        "Plot": meta["plot"],
        "Officer": meta["officer"],
        "Timestamp": r["time"],
    } for r in results])

    with tab_table:
        df_view = pd.DataFrame([{
            "Specimen": r["file"],
            "Diagnosis": r["title"],
            "Severity": r["severity"],
            "Confidence": round(r["conf"], 1),
            "Causal agent": r["agent"],
            "Action": r["rx"],
            "Image": r["url"] if str(r["url"]).startswith("http") else None,
        } for r in results])
        st.dataframe(
            df_view,
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                "Image": st.column_config.LinkColumn("Image", display_text="Open ↗"),
                "Action": st.column_config.TextColumn("Action", width="large"),
            },
        )

    with tab_export:
        e1, e2 = st.columns([1.4, 1], gap="large")
        with e1:
            st.markdown(
                '<div class="cv-h2" style="font-size:1.5rem">Download the agronomic report</div>'
                '<div class="cv-sub">A CSV of every diagnosis in this batch, with severity, causal agent, '
                'treatment protocol and session details. Opens in Excel or Google Sheets.</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download report (CSV)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"maize_diagnostic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                icon=":material/download:",
                on_click="ignore",
            )
        with e2:
            sync_text = (
                f"{meta['synced']} of {n} records logged to Google Sheets." if meta["sync_on"] and meta["synced"]
                else "Google Sheets logging was unavailable for this batch." if meta["sync_on"]
                else "Google Sheets logging is turned off for this batch."
            )
            notes_html = f'<div class="cv-kv"><span>Notes</span><span>{ui.esc(meta["notes"])}</span></div>' if meta["notes"] else ""
            st.markdown(
                f'<div class="cv-label">Batch details</div>'
                f'<div class="cv-kv"><span>Plot</span><span>{ui.esc(meta["plot"])}</span></div>'
                f'<div class="cv-kv"><span>Officer</span><span>{ui.esc(meta["officer"])}</span></div>'
                f'<div class="cv-kv"><span>Specimens</span><span>{n}</span></div>'
                f'{notes_html}'
                f'<div class="cv-integ-note" style="margin:12px 0 0">{ui.esc(sync_text)}</div>',
                unsafe_allow_html=True,
            )

st.markdown(ui.footer(), unsafe_allow_html=True)
