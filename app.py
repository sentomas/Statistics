import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="GenEpi: Comprehensive Genetic Report", layout="wide")

# --- UTILITIES ---
def robust_read(file):
    """
    Tries to read a file as CSV (various encodings) or Excel.
    Reset file pointer before each attempt.
    """
    # 1. Try reading as Excel (.xlsx)
    try:
        file.seek(0)
        return pd.read_excel(file)
    except:
        pass # Not an Excel file, move on to CSV attempts

    # 2. Try Standard CSV (UTF-8)
    try:
        file.seek(0)
        return pd.read_csv(file)
    except:
        pass

    # 3. Try CSV with ISO-8859-1 (Common for Excel-saved CSVs)
    try:
        file.seek(0)
        return pd.read_csv(file, encoding='ISO-8859-1')
    except:
        pass

    # 4. Try Python Engine with auto-separator (Last resort)
    try:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine='python', encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Could not read file. Error: {e}")
        return pd.DataFrame() # Return empty DF to prevent crash

def calculate_or_stats(a, b, c, d, model_label):
    # Haldane
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    
    # OR
    try:
        or_val = (a * d) / (b * c)
        log_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(log_or - 1.96 * se)
        ci_up = np.exp(log_or + 1.96 * se)
        z_score = log_or / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
    except:
        return None

    # Chi2 & Fisher
    obs = [[a, b], [c, d]]
    chi2, p_chi2, df, _ = stats.chi2_contingency(obs, correction=False)
    _, p_fisher = stats.fisher_exact(obs)

    return {
        "Model": model_label,
        "OR": or_val, "Lower": ci_lo, "Upper": ci_up,
        "Log_OR": log_or, "SE": se,
        "Chi2": chi2, "P_Chi2": p_chi2, "df": df,
        "P_Fisher": p_fisher
    }

def eggers_test(effect, se):
    """Regress standard normal deviate against precision."""
    try:
        y = effect / se
        x = 1 / se
        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model.pvalues[0] # Intercept P-value
    except:
        return np.nan

def beggs_test(effect, se):
    """Kendall's Tau correlation between effect and variance."""
    try:
        var = se**2
        tau, p = stats.kendalltau(effect, var)
        return p
    except:
        return np.nan

# --- DATA PRESETS ---
DEFAULT_GENO = {
    "YAP1 (rs11225161)": {"Case": [89, 29, 82], "Ctrl": [132, 17, 51], "Alleles": ["C", "T"]},
    "YAP2 (rs11225138)": {"Case": [52, 16, 132], "Ctrl": [109, 27, 64], "Alleles": ["G", "C"]}
}

# --- APP LAYOUT ---
st.title("🧬 GenEpi: Advanced Genetic Association & Meta-Analysis")
st.markdown("### Interactive Statistical Reporting Tool")

tabs = st.tabs([
    "1. Clinical Stats", 
    "2. Genetic Models", 
    "3. Advanced (ANOVA/Reg)", 
    "4. Meta-Analysis", 
    "5. Bias (NOS)",
    "📝 GENERATE REPORT"
])

# ==========================================
# TAB 1: CLINICAL
# ==========================================
with tabs[0]:
    st.header("Clinical Descriptive Statistics")
    c1, c2 = st.columns(2)
    f_case = c1.file_uploader("Upload Case Clinical Data (CSV)", key="clin_case")
    f_ctrl = c2.file_uploader("Upload Control Clinical Data (CSV)", key="clin_ctrl")
    
    if f_case and f_ctrl:
        df_case = robust_read(f_case)
        df_ctrl = robust_read(f_ctrl)
        
        # Numeric Stats
        num_cols = list(set(df_case.select_dtypes(include=np.number).columns) & 
                        set(df_ctrl.select_dtypes(include=np.number).columns))
        
        res_clin = []
        for col in num_cols:
            vc = df_case[col].dropna()
            vt = df_ctrl[col].dropna()
            if len(vc) > 1 and len(vt) > 1:
                t, p = stats.ttest_ind(vc, vt, equal_var=False)
                res_clin.append({
                    "Variable": col,
                    "Case Mean±SD": f"{vc.mean():.2f}±{vc.std():.2f}",
                    "Ctrl Mean±SD": f"{vt.mean():.2f}±{vt.std():.2f}",
                    "P-Value": p
                })
        
        st.session_state['clin_results'] = pd.DataFrame(res_clin)
        st.dataframe(st.session_state['clin_results'])
        
        # Heatmap
        st.subheader("Correlation Heatmap (Cases)")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df_case[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.session_state['clin_heatmap'] = fig

# ==========================================
# TAB 2: GENETIC MODELS
# ==========================================
with tabs[1]:
    st.header("Genetic Association Models")
    
    snp = st.selectbox("Select SNP", list(DEFAULT_GENO.keys()))
    d = DEFAULT_GENO[snp]
    
    # Inputs allow override
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Case Genotypes**")
        ca_hom1 = st.number_input("Hom Ref", value=d["Case"][0], key="c1")
        ca_het = st.number_input("Het", value=d["Case"][1], key="c2")
        ca_hom2 = st.number_input("Hom Risk", value=d["Case"][2], key="c3")
    with col2:
        st.markdown("**Control Genotypes**")
        co_hom1 = st.number_input("Hom Ref", value=d["Ctrl"][0], key="ct1")
        co_het = st.number_input("Het", value=d["Ctrl"][1], key="ct2")
        co_hom2 = st.number_input("Hom Risk", value=d["Ctrl"][2], key="ct3")

    # 1. Frequencies & HWE
    n_ctrl = co_hom1 + co_het + co_hom2
    p_freq = (2*co_hom1 + co_het) / (2*n_ctrl)
    q_freq = 1 - p_freq
    exp = [n_ctrl*p_freq**2, n_ctrl*2*p_freq*q_freq, n_ctrl*q_freq**2]
    hwe_p = stats.chisquare([co_hom1, co_het, co_hom2], exp)[1]
    
    st.metric("Control HWE P-Value", f"{hwe_p:.4f}", delta="Deviates" if hwe_p < 0.05 else "Consistent", delta_color="inverse")
    st.write(f"**MAF (Control):** {min(p_freq, q_freq):.3f}")

    # 2. Models
    models = []
    # Allelic (2x2)
    models.append(calculate_or_stats(
        2*ca_hom2+ca_het, 2*co_hom2+co_het, 2*ca_hom1+ca_het, 2*co_hom1+co_het, "Allelic"
    ))
    # Genotypic (Hom vs Hom)
    models.append(calculate_or_stats(ca_hom2, co_hom2, ca_hom1, co_hom1, "Genotypic"))
    # Dominant
    models.append(calculate_or_stats(ca_hom2+ca_het, co_hom2+co_het, ca_hom1, co_hom1, "Dominant"))
    # Recessive
    models.append(calculate_or_stats(ca_hom2, co_hom2, ca_het+ca_hom1, co_het+co_hom1, "Recessive"))
    # Overdominant
    models.append(calculate_or_stats(ca_het, co_het, ca_hom1+ca_hom2, co_hom1+co_hom2, "Overdominant"))
    
    df_res = pd.DataFrame([m for m in models if m])
    st.session_state['geno_results'] = df_res
    st.table(df_res[['Model', 'OR', 'Lower', 'Upper', 'P_Chi2', 'P_Fisher']])

# ==========================================
# TAB 3: ADVANCED (ANOVA/REGRESSION)
# ==========================================
with tabs[2]:
    st.header("Two-Way ANOVA & Confounder Adjustment")
    st.warning("Requires a SINGLE merged CSV with columns: `Group` (Case/Control), `Genotype` (0/1/2), and Clinical Variables.")
    
    f_merged = st.file_uploader("Upload Merged Data", type="csv")
    
    if f_merged:
        df_m = robust_read(f_merged)
        st.write("Preview:", df_m.head())
        
        c_var = st.selectbox("Select Clinical Variable (Dependent)", df_m.select_dtypes(include=np.number).columns)
        
        # ANOVA
        if st.button("Run Two-Way ANOVA"):
            try:
                model = ols(f'{c_var} ~ C(Group) + C(Genotype) + C(Group):C(Genotype)', data=df_m).fit()
                st.write(sm.stats.anova_lm(model, typ=2))
                st.success("ANOVA Complete")
            except Exception as e:
                st.error(f"ANOVA Failed: {e}")

        # Logistic Regression
        st.divider()
        st.subheader("Confounder Adjusted Analysis")
        confounders = st.multiselect("Select Confounders (e.g., Age, BMI)", df_m.columns)
        
        if st.button("Run Logistic Regression"):
            try:
                # Assuming 'Group' is Case/Control strings, convert to 0/1
                df_m['Outcome_Bin'] = df_m['Group'].apply(lambda x: 1 if 'Case' in str(x) or 'Patient' in str(x) else 0)
                formula = f"Outcome_Bin ~ C(Genotype) + {' + '.join(confounders)}"
                log_reg = logit(formula, df_m).fit()
                st.write(log_reg.summary())
            except Exception as e:
                st.error(f"Regression Failed: {e}")
    else:
        st.info("Please upload merged data to unlock these tools.")

# ==========================================
# TAB 4: META-ANALYSIS
# ==========================================
with tabs[3]:
    st.header("Meta-Analysis & Bias Assessment")
    st.write("Combine your study with external data.")
    
    # Manual Entry or Upload
    st.subheader("Data Input")
    st.caption("Enter counts for: Case_Event, Case_Total, Ctrl_Event, Ctrl_Total")
    
    # Default with the current study's data added automatically
    current_study_or = st.session_state.get('geno_results', pd.DataFrame()).iloc[0]['OR'] if 'geno_results' in st.session_state else 1.0
    
    # Editable DataFrame for Literature
    lit_data = pd.DataFrame([
        {"Study": "Current Study", "Year": 2024, "OR": current_study_or, "SE": 0.2},
        {"Study": "Literature 1", "Year": 2020, "OR": 1.5, "SE": 0.15},
        {"Study": "Literature 2", "Year": 2018, "OR": 1.2, "SE": 0.18},
    ])
    
    edited_df = st.data_editor(lit_data, num_rows="dynamic")
    
    if len(edited_df) > 1:
        # Calculate weights (Inverse Variance)
        edited_df['Log_OR'] = np.log(edited_df['OR'])
        edited_df['Weight'] = 1 / (edited_df['SE']**2)
        
        # Pooled Effect (Fixed)
        pooled_log = np.sum(edited_df['Weight'] * edited_df['Log_OR']) / np.sum(edited_df['Weight'])
        pooled_se = np.sqrt(1 / np.sum(edited_df['Weight']))
        pooled_or = np.exp(pooled_log)
        
        # Heterogeneity
        q = np.sum(edited_df['Weight'] * (edited_df['Log_OR'] - pooled_log)**2)
        df_q = len(edited_df) - 1
        i2 = max(0, (q - df_q)/q) * 100 if q > 0 else 0
        p_het = 1 - stats.chi2.cdf(q, df_q)
        
        # Bias Tests
        p_egger = eggers_test(edited_df['Log_OR'], edited_df['SE'])
        p_begg = beggs_test(edited_df['Log_OR'], edited_df['SE'])
        
        # Report
        c1, c2, c3 = st.columns(3)
        c1.metric("Pooled OR", f"{pooled_or:.2f}")
        c2.metric("Heterogeneity (I²)", f"{i2:.1f}%")
        c3.metric("Egger's P-Value", f"{p_egger:.4f}")
        st.write(f"**Begg's P-Value:** {p_begg:.4f}")
        
        # Forest Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        y = np.arange(len(edited_df))
        ax.errorbar(edited_df['OR'], y, xerr=1.96*edited_df['SE']*edited_df['OR'], fmt='o')
        ax.set_yticks(y)
        ax.set_yticklabels(edited_df['Study'])
        ax.axvline(1, color='r', linestyle='--')
        ax.set_title(f"Forest Plot (I²={i2:.1f}%)")
        st.pyplot(fig)
        st.session_state['meta_plot'] = fig
        
        # Meta-Regression (Year vs Log OR)
        if len(edited_df) > 2:
            st.subheader("Meta-Regression (Year)")
            fig2, ax2 = plt.subplots()
            sns.regplot(x='Year', y='Log_OR', data=edited_df, ax=ax2)
            st.pyplot(fig2)

# ==========================================
# TAB 5: BIAS (NOS)
# ==========================================
with tabs[4]:
    st.header("Risk of Bias Assessment (NOS)")
    nos_score = st.slider("Select Total NOS Score (Calculated manually)", 0, 9, 7)
    if nos_score >= 7:
        st.success("Study Quality: HIGH")
    elif nos_score >= 4:
        st.warning("Study Quality: MODERATE")
    else:
        st.error("Study Quality: LOW")
    st.session_state['nos_score'] = nos_score

# ==========================================
# TAB 6: REPORT
# ==========================================
with tabs[5]:
    st.header("📝 Final Study Report")
    
    if st.button("Generate Report"):
        st.markdown("---")
        st.subheader("1. Clinical Characteristics")
        if 'clin_results' in st.session_state:
            st.dataframe(st.session_state['clin_results'])
        else:
            st.write("No clinical data analyzed.")
            
        st.subheader("2. Genetic Association")
        if 'geno_results' in st.session_state:
            st.table(st.session_state['geno_results'])
        else:
            st.write("No genetic models run.")
            
        st.subheader("3. Meta-Analysis & Bias")
        if 'nos_score' in st.session_state:
            st.write(f"**NOS Quality Score:** {st.session_state['nos_score']}/9")
        
        st.markdown("---")
        st.info("To save this report, use Ctrl+P (Print) and select 'Save as PDF'.")