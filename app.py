import base64
import datetime
import gc

import gspread
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from google.oauth2 import service_account
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import ui

# --- Page Configuration ---
st.set_page_config(
    page_title="MaizeGPT VLM · Maize Pathology Diagnostics",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.load_css()

device = "cuda" if torch.cuda.is_available() else "cpu"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
HEALTHY_THRESHOLD = 60.0

# --- Agronomic Protocols ---
# tier drives badge colour and ordering: critical > moderate > nutritional > healthy
MAIZE_AGRONOMIC_PROTOCOLS = {
    "Charcoal_Rot": {
        "name": "Charcoal Rot",
        "agent": "Macrophomina phaseolina",
        "severity": "High · stalk lodging",
        "tier": "critical",
        "chemical": "Carbendazim 50 WP (1 g/L) or Thiophanate-Methyl 70 WP (1 g/L).",
    },
    "Fall_Army_worm": {
        "name": "Fall Armyworm",
        "agent": "Spodoptera frugiperda",
        "severity": "Critical · whorl defoliation",
        "tier": "critical",
        "chemical": "Chlorantraniliprole 18.5 SC @ 0.4 ml/L OR Emamectin Benzoate 5 SG @ 0.4 g/L.",
    },
    "Grey_Leaf_Spot": {
        "name": "Grey Leaf Spot",
        "agent": "Cercospora zeae-maydis",
        "severity": "Moderate to high",
        "tier": "moderate",
        "chemical": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L water.",
    },
    "Maydis_Leaf_Blight": {
        "name": "Maydis Leaf Blight",
        "agent": "Bipolaris maydis",
        "severity": "Moderate",
        "tier": "moderate",
        "chemical": "Mancozeb 75 WP @ 2.5 g/L OR Azoxystrobin 23 SC @ 1 ml/L.",
    },
    "Nitrogen_Deficiency_in_Maize": {
        "name": "Nitrogen Deficiency",
        "agent": "Macronutrient deficiency (N)",
        "severity": "Nutritional stress",
        "tier": "nutritional",
        "chemical": "Top-dress Urea @ 40 kg/ha OR 2% foliar Urea spray.",
    },
    "Northern_Leaf_Blight": {
        "name": "Northern Leaf Blight",
        "agent": "Exserohilum turcicum",
        "severity": "High · foliar necrosis",
        "tier": "critical",
        "chemical": "Propiconazole 25 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L.",
    },
    "Phosphorus_Deficiency": {
        "name": "Phosphorus Deficiency",
        "agent": "Macronutrient deficiency (P)",
        "severity": "Nutritional stress",
        "tier": "nutritional",
        "chemical": "Foliar 12:61:0 @ 5 g/L OR Single Super Phosphate @ 100 kg/ha.",
    },
    "Potassium_Deficiency_in_Maize": {
        "name": "Potassium Deficiency",
        "agent": "Macronutrient deficiency (K)",
        "severity": "Nutritional stress",
        "tier": "nutritional",
        "chemical": "MOP @ 40 kg/ha OR 13:0:45 foliar spray @ 10 g/L.",
    },
    "Rust_in_Maize": {
        "name": "Common Rust",
        "agent": "Puccinia sorghi",
        "severity": "Moderate to high",
        "tier": "moderate",
        "chemical": "Tebuconazole 25.9 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L.",
    },
    "Stem_Borer": {
        "name": "Stem Borer",
        "agent": "Chilo partellus",
        "severity": "Critical · dead heart",
        "tier": "critical",
        "chemical": "Carbofuran 3G @ 5 kg/ha in whorls OR Chlorantraniliprole 18.5 SC @ 0.4 ml/L.",
    },
    "Trips": {
        "name": "Thrips",
        "agent": "Frankliniella spp.",
        "severity": "Moderate",
        "tier": "moderate",
        "chemical": "Thiamethoxam 25 WG @ 0.3 g/L OR Imidacloprid 17.8 SL @ 0.3 ml/L.",
    },
    "Zinc_Deficiency_in_Maize": {
        "name": "Zinc Deficiency",
        "agent": "Micronutrient deficiency (Zn)",
        "severity": "Nutritional stress · white bud",
        "tier": "nutritional",
        "chemical": "0.5% ZnSO4 (5 g/L) + 0.25% lime spray.",
    },
}
TIER_ORDER = {"critical": 0, "moderate": 1, "nutritional": 2, "healthy": 3}
CONDITION_GROUPS = {
    "Fungal disease": ("moderate", ["Charcoal Rot", "Grey Leaf Spot", "Maydis Leaf Blight", "Northern Leaf Blight", "Common Rust"]),
    "Insect pests": ("critical", ["Fall Armyworm", "Stem Borer", "Thrips"]),
    "Nutrient deficiencies": ("nutritional", ["Nitrogen", "Phosphorus", "Potassium", "Zinc"]),
}


def pretty_name(raw: str) -> str:
    return MAIZE_AGRONOMIC_PROTOCOLS.get(raw, {}).get("name", raw.replace("_", " "))


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
def load_vision_pipeline():
    clf = joblib.load("maize_vlm_classifier.joblib")
    scaler = joblib.load("maize_vlm_scaler.joblib")
    class_names = joblib.load("maize_class_names.joblib")
    vision_id = "google/siglip-base-patch16-384"
    processor = AutoImageProcessor.from_pretrained(vision_id)
    vision_model = AutoModel.from_pretrained(
        vision_id, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    vision_model.eval()
    gc.collect()
    return clf, scaler, class_names, processor, vision_model


def diagnose(image, clf, scaler, class_names, processor, vision_model):
    inputs = processor(images=[image], return_tensors="pt").to(device, dtype=torch.float32)
    with torch.no_grad():
        out = vision_model.vision_model(inputs["pixel_values"]).last_hidden_state
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat_scaled = scaler.transform(feat.cpu().float().numpy())

    probs = clf.predict_proba(feat_scaled)[0]
    top_idx = int(np.argmax(probs))
    max_conf = float(probs[top_idx]) * 100
    top3 = [(pretty_name(class_names[i]), float(probs[i]) * 100) for i in np.argsort(probs)[::-1][:3]]

    if max_conf < HEALTHY_THRESHOLD:
        return {
            "title": "No disorder detected",
            "severity": "Healthy · asymptomatic",
            "tier": "healthy",
            "agent": "None",
            "rx": "Maintain standard fertigation and periodic foliar scouting.",
            "conf": max_conf,
            "top3": top3,
        }

    raw_class = class_names[top_idx]
    meta = MAIZE_AGRONOMIC_PROTOCOLS.get(raw_class, {
        "name": raw_class.replace("_", " "), "agent": "Unknown", "severity": "Check field",
        "tier": "moderate", "chemical": "Consult the regional agricultural extension office.",
    })
    return {
        "title": meta["name"],
        "severity": meta["severity"],
        "tier": meta["tier"],
        "agent": meta["agent"],
        "rx": meta["chemical"],
        "conf": max_conf,
        "top3": top3,
    }


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
            ("Training", "1,394 images"),
            ("Validation", "398 · 98.99%"),
            ("Holdout test", "200 · 100.0%"),
            ("Classes", "12 conditions"),
        ]),
        unsafe_allow_html=True,
    )

# --- Header ---
st.markdown(ui.topbar(), unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)
st.markdown(
    ui.stat_strip([
        ("12", "maize conditions detected"),
        ("98.99%", "validation accuracy"),
        ("1,992", "curated field images"),
        ("384px", "SigLIP vision encoder"),
    ]),
    unsafe_allow_html=True,
)

clf, scaler, class_names, processor, vision_model = load_vision_pipeline()

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
        st.markdown(ui.section("02", "Leaf specimens", "Clear, well-lit photos of a single leaf work best. JPG, PNG or WEBP."), unsafe_allow_html=True)
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
                '<div class="cv-sec-d" style="margin:18px 0 0">The model recognises 12 conditions:</div>'
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
            dx = diagnose(image, clf, scaler, class_names, processor, vision_model)

            cloud_url = "Not archived"
            if archive_imgs:
                st.write(f"Archiving `{file.name}`")
                cloud_url = upload_image_to_cloud(image_bytes, file.name)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_for_sheet.append([
                now_str, file.name, cloud_url, dx["title"], f"{dx['conf']:.2f}%",
                dx["severity"], field_agent, plot_id,
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
    need_action = sum(1 for r in results if r["tier"] == "critical")
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
    k1.markdown(ui.kpi("Specimens analysed", str(n), foot="Leaf images in this batch"), unsafe_allow_html=True)
    k2.markdown(
        ui.kpi("Need urgent action", str(need_action), foot="Critical or high severity", alert=need_action > 0),
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
                        + ui.confidence_meter(r["conf"])
                        + ui.differential(
                            r["top3"],
                            f"Closest matches (below {HEALTHY_THRESHOLD:.0f}% threshold)" if r["tier"] == "healthy"
                            else "Differential diagnosis",
                        )
                        + ui.rx(r["rx"], healthy=r["tier"] == "healthy")
                        + ui.archive_link(r["url"]),
                        unsafe_allow_html=True,
                    )

    df_export = pd.DataFrame([{
        "Image Name": r["file"],
        "Diagnosis": r["title"],
        "Confidence": f"{r['conf']:.2f}%",
        "Severity": r["severity"],
        "Causal Factor": r["agent"],
        "Recommended Prescription": r["rx"],
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
            "Treatment": r["rx"],
            "Image": r["url"] if str(r["url"]).startswith("http") else None,
        } for r in results])
        st.dataframe(
            df_view,
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                "Image": st.column_config.LinkColumn("Image", display_text="Open ↗"),
                "Treatment": st.column_config.TextColumn("Treatment", width="large"),
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
