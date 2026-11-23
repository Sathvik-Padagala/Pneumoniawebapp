"""
PneumoScan: Explainable, Uncertainty-Aware Pneumonia Detection Web App
Built with Streamlit, OpenCV, DenseNet121, U-Net, Grad-CAM++, and Monte Carlo Dropout.
Department of Computer Science & Engineering, MANIT Bhopal
"""

import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import pipeline
import sample_data

# ---------------- Streamlit Page Configuration ----------------
st.set_page_config(
    page_title="PneumoScan | Uncertainty-Aware Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS for Professional Medical UI ----------------
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Status Banners */
    .banner-stable {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 5px solid #10B981;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0;
        color: #D1FAE5;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .banner-uncertain {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0;
        color: #FEF3C7;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .banner-fallback {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%);
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0;
        color: #FEE2E2;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Uncertainty Gauge Bar */
    .gauge-container {
        margin: 16px 0 24px 0;
        padding: 14px 18px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    .gauge-track {
        height: 18px;
        background: linear-gradient(to right, #10B981 0%, #10B981 40%, #F59E0B 65%, #EF4444 100%);
        border-radius: 9px;
        position: relative;
        margin: 18px 0 10px 0;
    }
    .gauge-marker {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 30px;
        background: #FFFFFF;
        border-radius: 3px;
        box-shadow: 0 0 8px rgba(255,255,255,0.9);
        transform: translateX(-50%);
        z-index: 2;
    }
    .gauge-threshold {
        position: absolute;
        top: -10px;
        width: 2px;
        height: 38px;
        background: #F8FAFC;
        border-left: 2px dashed #000000;
        transform: translateX(-50%);
        z-index: 1;
    }

    /* Badges */
    .badge-normal {
        color: #10B981;
        background: rgba(16, 185, 129, 0.15);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-pneumonia {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.15);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Details panel */
    .details-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Sidebar: Settings & Controls ----------------
with st.sidebar:
    st.title("🩺 PneumoScan")
    st.markdown("**Explainable, uncertainty-aware pneumonia detection from chest X-rays.**")
    st.markdown("---")

    st.subheader("⚙️ Analysis Settings")

    t_mc = st.slider(
        "Monte Carlo Passes (T)",
        min_value=10,
        max_value=100,
        value=pipeline.DEFAULT_T_MC,
        step=5,
        help="Number of stochastic forward passes performed through the dropout head to measure predictive spread."
    )

    review_threshold = st.slider(
        "Review Threshold (Entropy)",
        min_value=0.10,
        max_value=0.90,
        value=pipeline.DEFAULT_ENTROPY_THRESH,
        step=0.05,
        help="Information-theoretic entropy limit in bits (0.0 to 1.0). Cases with entropy above this threshold are flagged for expert review."
    )

    heatmap_opacity = st.slider(
        "Heatmap Opacity (α)",
        min_value=0.10,
        max_value=0.90,
        value=pipeline.DEFAULT_HEATMAP_ALPHA,
        step=0.05,
        help="Grad-CAM++ overlay blend weight on the segmented lung crop."
    )

    st.markdown("---")

    # Disclaimer Box
    st.warning(
        "⚠️ **Clinical Disclaimer**\n\n"
        "Educational prototype for research and academic demonstration only. "
        "Not a certified medical device and not intended for primary clinical diagnosis or medical treatment decisions.",
        icon="⚠️"
    )

    st.markdown("---")
    st.caption("🏛️ **MANIT Bhopal — Dept of CSE**")
    st.caption("Balaka Laluth Vardhan · Padagala Sathvik · Shashank Agarwal")


# ---------------- Header ----------------
st.title("PneumoScan: AI-Assisted Chest X-ray Triage")
st.markdown(
    "**End-to-end framework integrating U-Net lung segmentation, DenseNet121 transfer learning, "
    "Grad-CAM++ visual interpretability, and Monte Carlo Dropout for calibrated uncertainty quantification.**"
)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "🩺 Single X-ray Analysis",
    "📁 Batch Analysis",
    "📊 Architecture & Performance"
])


# ==============================================================================
# TAB 1: SINGLE X-RAY ANALYSIS
# ==============================================================================
with tab1:
    st.subheader("Single Chest X-ray Analysis")

    col_upload, col_samples = st.columns([1.8, 1.2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray image (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            help="Upload an AP/PA chest radiograph in PNG or JPG format."
        )

    with col_samples:
        st.markdown("**Or load a pre-evaluated sample case:**")
        sample_options = {
            "Select a demo sample...": None,
            "1️⃣ Normal Case (Confident Clear Lungs)": "normal_confident",
            "2️⃣ Pneumonia Case (Dense Consolidation)": "pneumonia_high_confidence",
            "3️⃣ Borderline Case (High Uncertainty Flag)": "uncertain_case",
            "4️⃣ Diffuse Pneumonia (Bilateral Infiltrate)": "pneumonia_diffuse",
            "5️⃣ Non-Chest Artifact (Safety Fallback Test)": "failed_segmentation"
        }
        selected_sample_label = st.selectbox("Demo presets", list(sample_options.keys()))
        selected_sample_key = sample_options[selected_sample_label]

    # Resolve active image
    active_bytes = None
    image_source_name = ""

    if uploaded_file is not None:
        active_bytes = uploaded_file.getvalue()
        image_source_name = uploaded_file.name
    elif selected_sample_key is not None:
        sample_item = sample_data.SAMPLES[selected_sample_key]
        active_bytes = base64.b64decode(sample_item['b64'])
        image_source_name = sample_item['name']
    else:
        # Default prompt
        st.info("👆 Please upload a chest X-ray image (PNG or JPG) or pick one of the demo samples above to begin analysis.")

    if active_bytes is not None:
        with st.spinner("Analyzing radiograph: segmenting lungs, evaluating DenseNet121 features, running MC Dropout passes..."):
            result = pipeline.analyze_xray(
                active_bytes,
                T=t_mc,
                entropy_thresh=review_threshold,
                alpha=heatmap_opacity
            )

        # ---------------- Status Banners ----------------
        # 1. Fallback / Lungs Not Found Warning
        if result['fallback']:
            st.markdown("""
            <div class="banner-fallback">
                <span style="font-size:1.4rem;">⚠️</span>
                <div>
                    <strong>Lungs Could Not Be Reliably Located (Safety Fallback Triggered)</strong><br>
                    <span style="font-size:0.88rem;">
                        The predicted lung parenchyma covers under 5% of the frame (or severe artifact/non-chest image).
                        The classification was executed on the full uncropped image. Expert radiographic review is strictly required.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Uncertainty Status Banner
        if result['needs_review']:
            st.markdown(f"""
            <div class="banner-uncertain">
                <span style="font-size:1.4rem;">⚠️</span>
                <div>
                    <strong>Uncertain, refer for expert review</strong><br>
                    <span style="font-size:0.88rem;">
                        Predictive entropy (<strong>{result['entropy']:.3f} bits</strong>) exceeds the review threshold ({review_threshold:.2f} bits).
                        Model output exhibits significant stochastic variance; autonomous diagnosis is withheld in favor of clinician review.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="banner-stable">
                <span style="font-size:1.4rem;">✅</span>
                <div>
                    <strong>Prediction is stable</strong><br>
                    <span style="font-size:0.88rem;">
                        Predictive entropy (<strong>{result['entropy']:.3f} bits</strong>) is well below the threshold ({review_threshold:.2f} bits).
                        Monte Carlo passes yielded tightly clustered agreement across stochastic subnetworks.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- Four Result Cards ----------------
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            is_pneu = (result['prediction'] == 'PNEUMONIA')
            badge_html = f"<span class='badge-pneumonia'>PNEUMONIA</span>" if is_pneu else f"<span class='badge-normal'>NORMAL</span>"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Class</div>
                <div style="margin: 8px 0;">{badge_html}</div>
                <div class="metric-sub">Binary decision at P = 0.50</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color: #38BDF8;">{result['confidence']*100:.1f}%</div>
                <div class="metric-sub">P(pneumonia) = {result['prob_mean']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MC Spread (Std Dev)</div>
                <div class="metric-value" style="color: #A78BFA;">±{result['spread']:.4f}</div>
                <div class="metric-sub">Across {t_mc} stochastic passes</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            ent_color = "#F59E0B" if result['entropy'] > review_threshold else "#10B981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predictive Entropy</div>
                <div class="metric-value" style="color: {ent_color};">{result['entropy']:.3f} <span style="font-size:0.9rem; font-weight:normal;">bits</span></div>
                <div class="metric-sub">Threshold: {review_threshold:.2f} bits</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- Uncertainty Bar with Marked Threshold ----------------
        entropy_val = np.clip(result['entropy'], 0.0, 1.0)
        marker_left = f"{entropy_val * 100:.1f}%"
        threshold_left = f"{review_threshold * 100:.1f}%"

        st.markdown(f"""
        <div class="gauge-container">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:600; font-size:0.9rem; color:#CBD5E1;">Predictive Uncertainty Scale (Entropy)</span>
                <span style="font-size:0.85rem; color:#94A3B8;">Current: <strong>{result['entropy']:.3f}</strong> | Threshold: <strong>{review_threshold:.2f}</strong></span>
            </div>
            <div class="gauge-track">
                <div class="gauge-marker" style="left: {marker_left};" title="Current Entropy: {result['entropy']:.3f}"></div>
                <div class="gauge-threshold" style="left: {threshold_left};" title="Threshold: {review_threshold:.2f}"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748B;">
                <span>0.0 (High Certainty / Confident)</span>
                <span style="color:#F59E0B;">◀ Review Cutoff ({review_threshold:.2f}) ▶</span>
                <span>1.0 (Maximum Ambiguity / Equivocal)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- Three Images Side-by-Side ----------------
        st.markdown("### Radiographic Inspection")
        col_img1, col_img2, col_img3 = st.columns(3)

        with col_img1:
            st.markdown("**1. Original X-ray + Lung Outline**")
            st.image(
                result['img_outline'],
                caption=f"U-Net Segmented Contour (Area: {result['lung_frac']:.1%})",
                use_container_width=True
            )

        with col_img2:
            st.markdown("**2. Segmented Lung Crop**")
            st.image(
                result['img_crop'],
                caption="320×320 CLAHE Normalized (Classifier Input)",
                use_container_width=True
            )

        with col_img3:
            st.markdown("**3. Grad-CAM++ Overlay**")
            st.image(
                result['img_heatmap'],
                caption=f"DenseNet121 Attention Map (Opacity: {heatmap_opacity:.2f})",
                use_container_width=True
            )

        # ---------------- Monte Carlo Histogram & Details Panel ----------------
        st.markdown("---")
        col_hist, col_details = st.columns([1.4, 1.0])

        with col_hist:
            st.markdown("### Monte Carlo Probability Distribution")
            fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#0E1726')
            ax.set_facecolor('#0E1726')

            mc_data = result['mc_samples']
            bins = np.linspace(0.0, 1.0, 21)

            # Shaded background risk zones
            ax.axvspan(0.0, 0.4, color='#10B981', alpha=0.10, label='Normal Region')
            ax.axvspan(0.4, 0.6, color='#F59E0B', alpha=0.15, label='Equivocal / Borderline')
            ax.axvspan(0.6, 1.0, color='#EF4444', alpha=0.10, label='Pneumonia Region')

            # Histogram bars
            n, _, patches_list = ax.hist(
                mc_data,
                bins=bins,
                color='#38BDF8',
                edgecolor='#0284C7',
                alpha=0.85,
                rwidth=0.85
            )

            # Decision Boundary & Mean
            ax.axvline(0.50, color='#F8FAFC', linestyle='--', linewidth=1.5, label='Decision Boundary (0.5)')
            ax.axvline(result['prob_mean'], color='#F43F5E', linestyle='-', linewidth=2.0,
                       label=f"Mean P = {result['prob_mean']:.3f}")

            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("P(Pneumonia)", color='#E2E8F0', fontsize=10)
            ax.set_ylabel("Pass Count", color='#E2E8F0', fontsize=10)
            ax.set_title(f"Monte Carlo Dropout Histogram (T = {t_mc} Passes)", color='#F8FAFC', fontsize=11, fontweight='bold')
            ax.tick_params(colors='#94A3B8')
            for spine in ax.spines.values():
                spine.set_color('#334155')

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3,
                      frameon=False, fontsize=8, labelcolor='#E2E8F0')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_details:
            st.markdown("### Diagnostic Metrics & Export")
            st.markdown(f"""
            <div class="details-box">
                <table style="width:100%; font-size:0.88rem; color:#CBD5E1;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Source File:</strong></td>
                        <td style="text-align:right; color:#38BDF8;">{image_source_name}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Mean Probability P̄:</strong></td>
                        <td style="text-align:right;">{result['prob_mean']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Predictive Spread (Std):</strong></td>
                        <td style="text-align:right;">±{result['spread']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Total Entropy H(p):</strong></td>
                        <td style="text-align:right;">{result['entropy']:.4f} bits</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Model Uncertainty (MI):</strong></td>
                        <td style="text-align:right;">{result['mutual_info']:.4f} bits</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 0;"><strong>Lung Field Area:</strong></td>
                        <td style="text-align:right;">{result['lung_frac']:.1%}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;"><strong>Clinical Triage:</strong></td>
                        <td style="text-align:right; font-weight:bold; color:{'#F59E0B' if result['needs_review'] else '#10B981'};">
                            {'REFER FOR REVIEW' if result['needs_review'] else 'AUTO-STABLE'}
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # Download Heatmap Button
            # Convert RGB heatmap to PNG buffer
            buf = io.BytesIO()
            Image.fromarray(result['img_heatmap']).save(buf, format="PNG")
            btn = st.download_button(
                label="📥 Download Grad-CAM++ Heatmap (PNG)",
                data=buf.getvalue(),
                file_name=f"pneumoscan_heatmap_{image_source_name}.png",
                mime="image/png",
                use_container_width=True
            )

        # ---------------- "How to read this" Expander ----------------
        with st.expander("ℹ️ How to read this analysis (Plain Language Guide)"):
            st.markdown("""
            ### Understanding the PneumoScan Results

            **1. Lung Outline & Segmented Crop:**
            * Standard deep learning models frequently learn unintended shortcuts from text labels, radiopaque markers, clavicles, or machine borders.
            * PneumoScan uses an encoder-decoder U-Net to segment the anatomical lung boundaries, eliminating background artifacts and presenting the classifier with only the relevant lung parenchyma.
            * If the lung coverage falls below 5%, the system automatically flags a safety warning and falls back to full-field analysis.

            **2. Grad-CAM++ Heatmap Overlay:**
            * Grad-CAM++ visualizes the weighted gradients of the final DenseNet121 convolutional layer.
            * **Red and yellow regions** highlight areas of high diagnostic attention—typically alveolar consolidations, patchy opacities, or bronchovascular infiltrates.
            * **Blue and purple regions** indicate background lung parenchyma that did not trigger pneumonia activations.

            **3. Monte Carlo Dropout & Spread:**
            * Traditional neural networks produce overconfident predictions, even when wrong.
            * PneumoScan keeps dropout active during inference and samples 30 stochastic forward passes.
            * **Spread (Standard Deviation):** Measures how much the predictions fluctuate across passes. Confident cases exhibit tiny spread (±0.005), whereas ambiguous borderline cases show wide disagreement (±0.08+).

            **4. Predictive Entropy & Review Threshold:**
            * **Entropy** is an information-theoretic measure of doubt, ranging from 0.0 (total certainty) to 1.0 (50/50 coin-flip ambiguity).
            * When predictive entropy exceeds the adjustable review threshold (default 0.50 bits), the system presents an **amber triage banner**, recommending that a human radiologist review the scan before any clinical action.
            """)


# ==============================================================================
# TAB 2: BATCH ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("Batch Chest X-ray Triage")
    st.markdown("Upload multiple chest radiographs to screen cohorts, compute selective triage flags, and export tabular reports.")

    col_batch_upload, col_batch_demo = st.columns([2.0, 1.0])

    with col_batch_upload:
        batch_files = st.file_uploader(
            "Upload multiple chest X-ray images (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Select several files at once to run batch analysis."
        )

    with col_batch_demo:
        st.markdown("**Quick Demonstration:**")
        load_demo_batch = st.button("⚡ Load All 5 Pre-evaluated Demo Scans", use_container_width=True)

    # Prepare batch items
    batch_items = []
    if load_demo_batch:
        for k, v in sample_data.SAMPLES.items():
            batch_items.append((v['name'], base64.b64decode(v['b64'])))
    elif batch_files:
        for f in batch_files:
            batch_items.append((f.name, f.getvalue()))

    if batch_items:
        st.markdown(f"**Loaded {len(batch_items)} scans for batch processing.**")
        run_batch = st.button("🚀 Run Batch Analysis", type="primary")

        if run_batch or load_demo_batch:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            batch_results = []
            detailed_results = {}

            for idx, (fname, raw_data) in enumerate(batch_items):
                status_text.text(f"Processing scan {idx + 1} of {len(batch_items)}: {fname}...")
                res = pipeline.analyze_xray(
                    raw_data,
                    T=t_mc,
                    entropy_thresh=review_threshold,
                    alpha=heatmap_opacity
                )

                batch_results.append({
                    "File Name": fname,
                    "Prediction": res['prediction'],
                    "Probability": round(res['prob_mean'], 4),
                    "Spread": round(res['spread'], 4),
                    "Entropy": round(res['entropy'], 4),
                    "Needs Review": "⚠️ YES" if res['needs_review'] else "✅ NO",
                    "Lungs Not Found": "⚠️ FALLBACK" if res['fallback'] else "✅ FOUND"
                })
                detailed_results[fname] = res
                progress_bar.progress((idx + 1) / len(batch_items))

            status_text.success(f"Successfully processed all {len(batch_items)} radiographs!")

            # ---------------- Batch Summary Metrics ----------------
            df_results = pd.DataFrame(batch_results)

            total_cases = len(df_results)
            pneumonia_count = int((df_results['Prediction'] == 'PNEUMONIA').sum())
            normal_count = int((df_results['Prediction'] == 'NORMAL').sum())
            review_count = int(df_results['Needs Review'].str.contains('YES').sum())
            fallback_count = int(df_results['Lungs Not Found'].str.contains('FALLBACK').sum())

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            with sm1:
                st.metric("Total Scans", total_cases)
            with sm2:
                st.metric("Normal", normal_count)
            with sm3:
                st.metric("Pneumonia", pneumonia_count)
            with sm4:
                st.metric("Flagged for Review", f"{review_count} ({review_count/total_cases:.0%})")
            with sm5:
                st.metric("Lung Fallbacks", fallback_count)

            # ---------------- Styled Results Table ----------------
            st.markdown("### Results Table")

            def highlight_row(val):
                if val == 'PNEUMONIA':
                    return 'color: #EF4444; font-weight: bold;'
                elif val == 'NORMAL':
                    return 'color: #10B981; font-weight: bold;'
                elif 'YES' in str(val) or 'FALLBACK' in str(val):
                    return 'color: #F59E0B; font-weight: bold;'
                return ''

            styled_df = df_results.style.map(highlight_row, subset=['Prediction', 'Needs Review', 'Lungs Not Found'])
            st.dataframe(styled_df, use_container_width=True)

            # ---------------- CSV Export ----------------
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results as CSV",
                data=csv_data,
                file_name="pneumoscan_batch_results.csv",
                mime="text/csv",
                use_container_width=False
            )

            # ---------------- Inspect Specific Case from Batch ----------------
            st.markdown("---")
            st.markdown("### Detailed Inspection of Batch Scan")
            selected_inspect_name = st.selectbox(
                "Select a radiograph to inspect visual overlays and MC distribution:",
                df_results['File Name'].tolist()
            )

            if selected_inspect_name in detailed_results:
                case_res = detailed_results[selected_inspect_name]
                bi_col1, bi_col2, bi_col3 = st.columns(3)
                with bi_col1:
                    st.image(case_res['img_outline'], caption="Lung Contour Outline", use_container_width=True)
                with bi_col2:
                    st.image(case_res['img_crop'], caption="Segmented Lung Crop", use_container_width=True)
                with bi_col3:
                    st.image(case_res['img_heatmap'], caption="Grad-CAM++ Heatmap Overlay", use_container_width=True)
    else:
        st.info("Upload multiple images or click 'Load All 5 Pre-evaluated Demo Scans' to test batch analysis.")


# ==============================================================================
# TAB 3: ARCHITECTURE & EVALUATION
# ==============================================================================
with tab3:
    st.subheader("System Architecture & Test Set Evaluation")
    st.markdown(
        "Empirical findings and methodology from the PneumoScan study on 5,856 pediatric and adult chest X-rays."
    )

    # ---------------- 1. Pipeline Architecture ----------------
    st.markdown("### 1. End-to-End Processing Architecture")
    st.markdown("""
    ```
    ┌────────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌────────────────┐     ┌──────────────────────┐
    │  Raw Chest     │     │ U-Net Lung   │     │ Morphological Clean  │     │ DenseNet121    │     │ Monte Carlo Dropout  │
    │  Radiograph    │ ──▶ │ Segmentation │ ──▶ │ & CLAHE (320x320)    │ ──▶ │ Feature Model  │ ──▶ │ Head (T passes)      │
    │  (PNG / JPG)   │     │ (256x256)    │     │ (Safety Net Fallback)│     │ (GAP Layer)    │     │ Predictive Spread    │
    └────────────────┘     └──────────────┘     └──────────────────────┘     └────────────────┘     └──────────┬───────────┘
                                                                                                               │
                                                                   ┌───────────────────────────────────────────┴───────────┐
                                                                   ▼                                                       ▼
                                                     ┌────────────────────────────┐                         ┌────────────────────────────┐
                                                     │ Label: NORMAL / PNEUMONIA  │                         │ Grad-CAM++ Explainability  │
                                                     │ Status: Stable vs Review   │                         │ Visual Attention Overlay   │
                                                     └────────────────────────────┘                         └────────────────────────────┘
    ```
    """)

    # ---------------- 2. Benchmark Test Set Metrics ----------------
    st.markdown("### 2. Test Set Evaluation Performance (586 Images)")
    st.caption("Stratified 80/10/10 split: 4,684 training, 586 validation, and 586 holdout test images.")

    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value" style="color:#10B981;">96.59%</div>
            <div class="metric-sub">566 of 586 correct</div>
        </div>
        """, unsafe_allow_html=True)
    with pm2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pneumonia Precision</div>
            <div class="metric-value" style="color:#38BDF8;">97.22%</div>
            <div class="metric-sub">Only 12 false positives</div>
        </div>
        """, unsafe_allow_html=True)
    with pm3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pneumonia Recall</div>
            <div class="metric-value" style="color:#A78BFA;">98.13%</div>
            <div class="metric-sub">420 of 428 detected</div>
        </div>
        """, unsafe_allow_html=True)
    with pm4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pneumonia F1-Score</div>
            <div class="metric-value" style="color:#F59E0B;">0.9767</div>
            <div class="metric-sub">Macro F1: 0.9563</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 3. Confusion Matrix & Selective Prediction ----------------
    col_cm, col_selective = st.columns([1.1, 1.4])

    with col_cm:
        st.markdown("#### Test Set Confusion Matrix")
        cm_data = pd.DataFrame(
            [[146, 12], [8, 420]],
            columns=["Predicted NORMAL", "Predicted PNEUMONIA"],
            index=["True NORMAL", "True PNEUMONIA"]
        )
        st.table(cm_data)
        st.caption("Only 8 false negatives out of 428 pneumonia cases (miss rate < 1.9%).")

    with col_selective:
        st.markdown("#### Selective Prediction & Uncertainty Review")
        st.markdown("""
        * **19 of 20 total misclassifications** in the test set fall into the group flagged by the entropy review threshold ($H > 0.50$).
        * **Error rate on confident cases:** Only **0.25%** (near-zero error for autonomous decisions).
        * **Error rate on flagged cases:** **10.22%** (concentrated risk redirected to human radiologists).
        * **Mean MC Spread (σ):** 0.029 on correct predictions vs **0.076** on incorrect predictions (~2.6× higher spread).
        """)

    # ---------------- 4. Academic Attribution ----------------
    st.markdown("---")
    st.markdown("### 3. Academic & Project Attribution")
    st.markdown("""
    * **Project Name:** PneumoScan — Explainable, Uncertainty-Aware Pneumonia Detection
    * **Institution:** Department of Computer Science & Engineering, Maulana Azad National Institute of Technology (MANIT), Bhopal
    * **Team Members:**
      * **Balaka Laluth Vardhan** (Scholar No. 24112011116)
      * **Padagala Sathvik** (Scholar No. 24112011159)
      * **Shashank Agarwal** (Scholar No. 24112011114)
    * **Key References:**
      * Chattopadhay et al., *Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks*, IEEE WACV 2018.
      * Gal & Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016.
      * Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
    """)
