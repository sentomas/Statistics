import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="GenEpi Analytics Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2c3e50;
    }
    
    /* Custom Containers */
    .css-1r6slb0 {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Table Styling */
    .dataframe {
        font-size: 14px; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (HELPER FUNCTIONS) ---

def robust_read(file):
    """Tries to read a file as CSV (various encodings) or Excel."""
    # Reset pointer
    file.seek(0)
    
    if file.name.endswith('.xlsx'):
        try: return pd.read_excel(file)
        except: pass
    
    # CSV Attempts
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    separators = [',', '\t', ';', None]
    
    for enc in encodings:
        for sep in separators:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=enc, sep=sep, engine='python' if sep is None else 'c')
            except:
                continue
                
    st.error(f"❌ Failed to read {file.name}. Please ensure it's a valid CSV or Excel file.")
    return pd.DataFrame()

def calculate_or_stats(a, b, c, d, model_label):
    """Calculates Odds Ratio, CI, and P-values for 2x2 tables."""
    # Haldane Correction
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    try:
        or_val = (a * d) / (b * c)
        log_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo, ci_up = np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)
        
        # P-value (Z-test)
        p_val = 2 * (1 - stats.norm.cdf(abs(log_or / se)))
        
        # Chi-Square & Fisher
        obs = [[a, b], [c, d]]
        chi2, p_chi2, df, _ = stats.chi2_contingency(obs, correction=False)
        _, p_fisher = stats.fisher_exact(obs)

        return {
            "Model": model_label, "OR": or_val, "Lower": ci_lo, "Upper": ci_up,
            "P_Chi2": p_chi2, "P_Fisher": p_fisher
        }
    except: return None

def eggers_test(effect, se):
    """Performs Egger's regression test for publication bias."""
    try: 
        # Weighted linear regression of standard normal deviate on precision
        # y = effect/se, x = 1/se
        y = effect / se
        x = 1 / se
        return sm.OLS(y, sm.add_constant(x)).fit().pvalues[0] # Intercept P-value
    except: return 0.0

# --- 3. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=80)
    st.title("Study Config")
    
    st.subheader("📋 Study Details")
    study_name = st.text_input("Study ID", "PCOS_Gen_2025")
    investigator = st.text_input("Investigator", "Dr. S. Thomas")
    
    st.divider()
    
    st.subheader("⚙️ Analysis Settings")
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
    
    st.info("💡 **Tip:** Use the tabs on the right to navigate between Clinical, Genetic, and Meta-Analysis modules.")
    st.caption(f"v2.1 | Powered by GenEpi")

# --- 4. MAIN INTERFACE ---
st.title(f"🧬 GenEpi Analytics: {study_name}")
st.markdown(f"**Investigator:** {investigator} | **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")

# Tabs with Icons
tabs = st.tabs([
    "🏥 Clinical Stats", 
    "🧬 Genetic Models", 
    "📈 Advanced Analysis", 
    "🌍 Meta-Analysis", 
    "📝 Final Report"
])

# ==========================================
# TAB 1: CLINICAL STATS
# ==========================================
with tabs[0]:
    st.markdown("### 📊 Descriptive Statistics")
    st.markdown("Compare baseline characteristics between Case and Control groups.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1️⃣ Upload Case Data")
        f_case = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="clin_case")
    with col2:
        st.markdown("#### 2️⃣ Upload Control Data")
        f_ctrl = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="clin_ctrl")

    if f_case and f_ctrl:
        with st.spinner("Processing Clinical Data..."):
            df_case = robust_read(f_case)
            df_ctrl = robust_read(f_ctrl)
            
            # Identify Numeric Columns
            num_cols = list(set(df_case.select_dtypes(include=np.number).columns) & 
                            set(df_ctrl.select_dtypes(include=np.number).columns))
            
            if num_cols:
                # Results Calculation
                res_clin = []
                for col in num_cols:
                    vc, vt = df_case[col].dropna(), df_ctrl[col].dropna()
                    if len(vc) > 1 and len(vt) > 1:
                        t, p = stats.ttest_ind(vc, vt, equal_var=False)
                        sig = "⭐⭐" if p < 0.001 else ("⭐" if p < 0.05 else "")
                        res_clin.append({
                            "Variable": col,
                            "Case (Mean ± SD)": f"{vc.mean():.2f} ± {vc.std():.2f}",
                            "Control (Mean ± SD)": f"{vt.mean():.2f} ± {vt.std():.2f}",
                            "P-Value": p,
                            "Sig": sig
                        })
                
                df_res = pd.DataFrame(res_clin).sort_values("P-Value")
                st.session_state['clin_results'] = df_res

                # Display Logic
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader("Statistical Comparison (T-Test)")
                    st.dataframe(
                        df_res.style.format({"P-Value": "{:.4f}"})
                        .background_gradient(subset=['P-Value'], cmap="Reds_r", vmin=0, vmax=0.1),
                        use_container_width=True, height=400
                    )
                with c2:
                    st.subheader("Correlation Heatmap (Cases)")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(df_case[num_cols].corr(), annot=False, cmap='coolwarm', cbar=False, ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ No common numeric columns found between files.")

# ==========================================
# TAB 2: GENETIC MODELS
# ==========================================
with tabs[1]:
    st.markdown("### 🧬 Genotype Association Models")
    
    # Dataset Selector
    DEFAULT_GENO = {
        "YAP1 (rs11225161)": {"Case": [89, 29, 82], "Ctrl": [132, 17, 51], "Alleles": ["C", "T"]},
        "YAP2 (rs11225138)": {"Case": [52, 16, 132], "Ctrl": [109, 27, 64], "Alleles": ["G", "C"]}
    }
    
    col_sel, col_kpi = st.columns([1, 3])
    with col_sel:
        snp = st.selectbox("Select Target SNP", list(DEFAULT_GENO.keys()))
        d = DEFAULT_GENO[snp]
    
    # Interactive Input Grid
    with st.expander("✏️ Edit Genotype Counts (Observed)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🤒 Cases (n={sum(d['Case'])})**")
            ca_hom1 = st.number_input("Hom Ref", value=d["Case"][0], key="c1")
            ca_het = st.number_input("Het", value=d["Case"][1], key="c2")
            ca_hom2 = st.number_input("Hom Risk", value=d["Case"][2], key="c3")
        with c2:
            st.markdown(f"**🛡️ Controls (n={sum(d['Ctrl'])})**")
            co_hom1 = st.number_input("Hom Ref", value=d["Ctrl"][0], key="ct1")
            co_het = st.number_input("Het", value=d["Ctrl"][1], key="ct2")
            co_hom2 = st.number_input("Hom Risk", value=d["Ctrl"][2], key="ct3")

    # QC Metrics
    with col_kpi:
        n_ctrl = co_hom1 + co_het + co_hom2
        p_freq = (2*co_hom1 + co_het) / (2*n_ctrl)
        exp = [n_ctrl*p_freq**2, n_ctrl*2*p_freq*(1-p_freq), n_ctrl*(1-p_freq)**2]
        hwe_p = stats.chisquare([co_hom1, co_het, co_hom2], exp)[1]
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Control MAF", f"{min(p_freq, 1-p_freq):.3f}")
        k2.metric("HWE P-Value", f"{hwe_p:.4f}", delta="Pass" if hwe_p > 0.05 else "Fail", delta_color="normal")
        k3.metric("Risk Allele", d["Alleles"][1])

    # Run Models
    models = [
        calculate_or_stats(2*ca_hom2+ca_het, 2*co_hom2+co_het, 2*ca_hom1+ca_het, 2*co_hom1+co_het, "Allelic (A vs a)"),
        calculate_or_stats(ca_hom2, co_hom2, ca_hom1, co_hom1, "Genotypic (aa vs AA)"),
        calculate_or_stats(ca_hom2+ca_het, co_hom2+co_het, ca_hom1, co_hom1, "Dominant (aa+Aa vs AA)"),
        calculate_or_stats(ca_hom2, co_hom2, ca_het+ca_hom1, co_het+co_hom1, "Recessive (aa vs Aa+AA)")
    ]
    df_res = pd.DataFrame([m for m in models if m])
    st.session_state['geno_results'] = df_res
    
    st.table(df_res.set_index("Model").style.format("{:.4f}"))

# ==========================================
# TAB 3: ADVANCED ANALYSIS
# ==========================================
with tabs[2]:
    st.markdown("### 📈 Multivariate Analysis")
    st.info("ℹ️ **Requirement:** Upload a SINGLE merged dataset containing both Clinical variables and Genotype columns.")
    
    f_merged = st.file_uploader("Upload Merged Dataset (CSV)", type="csv")
    
    if f_merged:
        df_m = robust_read(f_merged)
        st.dataframe(df_m.head(3), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Two-Way ANOVA")
            dep_var = st.selectbox("Dependent Variable (e.g., BMI)", df_m.select_dtypes(include=np.number).columns)
            if st.button("Run ANOVA"):
                try:
                    model = ols(f'{dep_var} ~ C(Group) + C(Genotype)', data=df_m).fit()
                    st.write(sm.stats.anova_lm(model, typ=2))
                except Exception as e: st.error(str(e))
                
        with c2:
            st.subheader("Logistic Regression (Adjusted)")
            covars = st.multiselect("Confounders", df_m.columns)
            if st.button("Run Regression") and covars:
                try:
                    df_m['y'] = df_m['Group'].apply(lambda x: 1 if 'Case' in str(x) else 0)
                    res = logit(f"y ~ C(Genotype) + {'+'.join(covars)}", df_m).fit()
                    st.write(res.summary())
                except Exception as e: st.error(str(e))

# ==========================================
# TAB 4: META-ANALYSIS
# ==========================================
with tabs[3]:
    st.markdown("### 🌍 Meta-Analysis & Bias Detection")
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("Data Input")
        current_or = st.session_state.get('geno_results', pd.DataFrame({'OR':[1.0]})).iloc[0]['OR']
        
        lit_data = pd.DataFrame([
            {"Study": "Current Study", "Year": 2024, "OR": current_or, "SE": 0.2},
            {"Study": "Zhang et al.", "Year": 2020, "OR": 1.5, "SE": 0.15},
            {"Study": "Smith et al.", "Year": 2018, "OR": 1.1, "SE": 0.18},
        ])
        edited_df = st.data_editor(lit_data, num_rows="dynamic")
        
        st.subheader("Quality Assessment (NOS)")
        nos = st.slider("Newcastle-Ottawa Score", 0, 9, 7)
        st.session_state['nos'] = nos
        if nos >= 7: st.success("Quality: High")
        elif nos >= 4: st.warning("Quality: Moderate")
        else: st.error("Quality: Low")

    with col_viz:
        if len(edited_df) > 1:
            edited_df['Log_OR'] = np.log(edited_df['OR'])
            edited_df['W'] = 1 / (edited_df['SE']**2)
            
            # Pooled Stats
            pooled_log = np.average(edited_df['Log_OR'], weights=edited_df['W'])
            pooled_or = np.exp(pooled_log)
            
            # Heterogeneity
            q = np.sum(edited_df['W'] * (edited_df['Log_OR'] - pooled_log)**2)
            i2 = max(0, (q - (len(edited_df)-1))/q) * 100
            
            # Dashboard Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Pooled OR", f"{pooled_or:.2f}")
            m2.metric("Heterogeneity (I²)", f"{i2:.1f}%")
            m3.metric("Egger's P", f"{eggers_test(edited_df['Log_OR'], edited_df['SE']):.3f}")

            # Forest Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(edited_df['OR'], np.arange(len(edited_df)), xerr=1.96*edited_df['SE'], fmt='o', color='black')
            ax.axvline(1, color='red', linestyle='--')
            ax.set_yticks(np.arange(len(edited_df)))
            ax.set_yticklabels(edited_df['Study'])
            ax.set_xlabel("Odds Ratio (95% CI)")
            ax.set_title("Forest Plot")
            st.pyplot(fig)

# ==========================================
# TAB 5: REPORT GENERATION
# ==========================================
with tabs[4]:
    st.markdown("### 📝 Generate Professional Report")
    st.markdown("Download a summarized report of all analyses performed in this session.")
    
    if st.button("📄 Generate PDF Report Preview"):
        st.markdown("---")
        st.markdown(f"## **Genetic Association Study Report**")
        st.markdown(f"**Study ID:** {study_name} | **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        
        st.markdown("#### 1. Study Quality")
        st.write(f"- **NOS Score:** {st.session_state.get('nos', 'N/A')}/9")
        st.write(f"- **HWE Status:** {'Pass' if hwe_p > 0.05 else 'Fail (Caution)'}")
        
        st.markdown("#### 2. Clinical Characteristics")
        if 'clin_results' in st.session_state:
            st.dataframe(st.session_state['clin_results'])
        else:
            st.write("No clinical data processed.")
            
        st.markdown("#### 3. Genetic Association Results")
        if 'geno_results' in st.session_state:
            st.table(st.session_state['geno_results'])
            
        st.markdown("---")
        st.caption("Generated by GenEpi Analytics Suite. Confidential.")