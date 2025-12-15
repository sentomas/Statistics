import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Genetic Epi Suite", layout="wide")

# --- UTILITY FUNCTIONS ---

def calculate_hwe(aa, aa_het, aa_homo):
    """Calculates HWE P-value for Controls."""
    total = aa + aa_het + aa_homo
    if total == 0: return 1.0
    p = (2 * aa + aa_het) / (2 * total) # Major allele freq
    q = 1 - p # Minor allele freq
    exp_aa = (p ** 2) * total
    exp_het = (2 * p * q) * total
    exp_homo = (q ** 2) * total
    obs = [aa, aa_het, aa_homo]
    exp = [exp_aa, exp_het, exp_homo]
    # Simple Chi-square for HWE
    chi2, p_val = stats.chisquare(obs, f_exp=exp)
    return p_val

def genetic_model_calc(a, b, c, d, model_name):
    """
    Generic 2x2 Calculator for any model.
    Input: a (Case Exp), b (Ctrl Exp), c (Case Unexp), d (Ctrl Unexp)
    """
    # Haldane correction
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    try:
        odds_ratio = (a * d) / (b * c)
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)
        
        # Chi-Square & Fisher
        obs = np.array([[a, b], [c, d]])
        chi2, p_chi2, dof, _ = stats.chi2_contingency(obs, correction=False)
        _, p_fisher = stats.fisher_exact(obs)
        
        return {
            "Model": model_name,
            "OR": odds_ratio, "CI95": f"{ci_lower:.2f}-{ci_upper:.2f}",
            "Chi2": chi2, "P_Value": p_chi2, "Fisher_P": p_fisher,
            "Log_OR": log_or, "SE": se
        }
    except:
        return None

def eggers_test(effect_sizes, standard_errors):
    """
    Performs Egger's test for publication bias.
    Regress Standardized Effect (Estimate/SE) against Precision (1/SE).
    """
    try:
        y = np.array(effect_sizes) / np.array(standard_errors)
        x = 1 / np.array(standard_errors)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return model.pvalues[0] # P-value of the intercept
    except:
        return np.nan

# --- APP LAYOUT ---

st.title("🧬 Comprehensive Genetic Association Suite")
st.markdown("Supports: Descriptive Stats, 6 Genetic Models, Meta-Analysis & Bias Assessment")

tabs = st.tabs([
    "📋 1. Descriptive Stats", 
    "🧬 2. Genetic Models (YAP1/2)", 
    "📊 3. Meta-Analysis & Bias",
    "⚖️ 4. Confounder & Risk of Bias"
])

# ==========================================
# TAB 1: DESCRIPTIVE STATISTICS (Raw Data)
# ==========================================
with tabs[0]:
    st.header("Clinical Characteristics (Control vs Patient)")
    st.info("Upload RAW data. Columns required: `Group` (must contain 'Control' and 'Case'), and numeric columns (Age, BMI, etc.)")
    
    uploaded_clin = st.file_uploader("Upload Clinical Data (CSV)", key="clin")
    
    if uploaded_clin:
        df_clin = pd.read_csv(uploaded_clin)
        st.write("Preview:", df_clin.head())
        
        if 'Group' in df_clin.columns:
            controls = df_clin[df_clin['Group'] == 'Control']
            cases = df_clin[df_clin['Group'] == 'Case']
            
            st.subheader("Statistical Comparison (Student's t-test)")
            
            # Select numeric columns automatically
            numeric_cols = df_clin.select_dtypes(include=np.number).columns.tolist()
            
            results = []
            for col in numeric_cols:
                mean_ctrl = controls[col].mean()
                std_ctrl = controls[col].std()
                mean_case = cases[col].mean()
                std_case = cases[col].std()
                
                # T-test
                t_stat, p_val = stats.ttest_ind(controls[col].dropna(), cases[col].dropna())
                
                results.append({
                    "Variable": col,
                    "Control (Mean ± SD)": f"{mean_ctrl:.2f} ± {std_ctrl:.2f}",
                    "Case (Mean ± SD)": f"{mean_case:.2f} ± {std_case:.2f}",
                    "P-Value": p_val
                })
            
            st.table(pd.DataFrame(results))
        else:
            st.error("Column 'Group' not found in CSV.")

# ==========================================
# TAB 2: GENETIC MODELS (Summary Data)
# ==========================================
with tabs[1]:
    st.header("Genotypic Association Models")
    st.markdown("Upload 'genotypic ratio_YAP1' or 'YAP2'. Required cols: `Study`, `AA_case`, `Aa_case`, `aa_case`, `AA_ctrl`, `Aa_ctrl`, `aa_ctrl`")
    
    # Template download
    template = pd.DataFrame({
        'Study': ['Study1'], 
        'AA_case': [10], 'Aa_case': [20], 'aa_case': [5],
        'AA_ctrl': [15], 'Aa_ctrl': [25], 'aa_ctrl': [10]
    })
    st.download_button("📥 Download Template", template.to_csv(index=False), "genotype_template.csv")

    geno_file = st.file_uploader("Upload Genotype Data", key="geno")
    
    if geno_file:
        df_g = pd.read_csv(geno_file)
        
        # Calculate Allele Counts
        # Case Alleles
        df_g['A_case'] = 2*df_g['AA_case'] + df_g['Aa_case']
        df_g['a_case'] = 2*df_g['aa_case'] + df_g['Aa_case']
        # Control Alleles
        df_g['A_ctrl'] = 2*df_g['AA_ctrl'] + df_g['Aa_ctrl']
        df_g['a_ctrl'] = 2*df_g['aa_ctrl'] + df_g['Aa_ctrl']
        
        # Store all results for display
        model_results = []

        for idx, row in df_g.iterrows():
            study = row.get('Study', f'Study {idx+1}')
            
            # 1. HWE (Controls)
            hwe_p = calculate_hwe(row['AA_ctrl'], row['Aa_ctrl'], row['aa_ctrl'])
            
            # --- DEFINE MODELS ---
            models = [
                # Allelic: A vs a
                ("Allelic (A vs a)", row['A_case'], row['A_ctrl'], row['a_case'], row['a_ctrl']),
                
                # Dominant: (AA + Aa) vs aa
                ("Dominant (AA+Aa vs aa)", 
                 row['AA_case']+row['Aa_case'], row['AA_ctrl']+row['Aa_ctrl'], 
                 row['aa_case'], row['aa_ctrl']),
                 
                # Recessive: AA vs (Aa + aa)
                ("Recessive (AA vs Aa+aa)", 
                 row['AA_case'], row['AA_ctrl'], 
                 row['Aa_case']+row['aa_case'], row['Aa_ctrl']+row['aa_ctrl']),
                 
                # Overdominant: Aa vs (AA + aa)
                ("Overdominant (Aa vs AA+aa)", 
                 row['Aa_case'], row['Aa_ctrl'], 
                 row['AA_case']+row['aa_case'], row['AA_ctrl']+row['aa_ctrl'])
            ]
            
            for m_name, a, b, c, d in models:
                res = genetic_model_calc(a, b, c, d, m_name)
                if res:
                    res['Study'] = study
                    res['HWE_P_Ctrl'] = hwe_p
                    model_results.append(res)

        df_res = pd.DataFrame(model_results)
        
        st.subheader("Model Results per Study")
        st.dataframe(df_res)
        
        # Store in session state for Meta-Analysis
        st.session_state['genetics_results'] = df_res

# ==========================================
# TAB 3: META-ANALYSIS & BIAS
# ==========================================
with tabs[2]:
    st.header("Meta-Analysis & Bias Assessment")
    
    if 'genetics_results' in st.session_state:
        df_meta = st.session_state['genetics_results']
        
        # Filter by Model
        model_choice = st.selectbox("Select Genetic Model to Meta-Analyze", df_meta['Model'].unique())
        subset = df_meta[df_meta['Model'] == model_choice].copy()
        
        if not subset.empty:
            # Pooled Calculations (Fixed Effects - Inverse Variance)
            weights = 1 / (subset['SE'] ** 2)
            pooled_log_or = np.sum(weights * subset['Log_OR']) / np.sum(weights)
            pooled_se = np.sqrt(1 / np.sum(weights))
            pooled_or = np.exp(pooled_log_or)
            
            # Heterogeneity (I^2)
            q_stat = np.sum(weights * (subset['Log_OR'] - pooled_log_or)**2)
            df_q = len(subset) - 1
            p_het = 1 - stats.chi2.cdf(q_stat, df_q)
            i2 = max(0, (q_stat - df_q) / q_stat) * 100 if q_stat > df_q else 0
            
            # Egger's Test
            egger_p = eggers_test(subset['Log_OR'], subset['SE'])
            
            # Display Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pooled OR", f"{pooled_or:.2f}")
            c2.metric("Heterogeneity (I²)", f"{i2:.1f}%")
            c3.metric("Het P-Value", f"{p_het:.4f}")
            c4.metric("Egger's Bias P", f"{egger_p:.4f}")
            
            # Begg's Test Note
            st.caption("*Begg's test (Rank Correlation) requires larger sample sizes; Egger's is prioritized here.*")
            
            # Forest Plot
            st.subheader("Forest Plot")
            fig, ax = plt.subplots(figsize=(8, len(subset)*0.5 + 2))
            y_pos = np.arange(len(subset))
            ax.errorbar(subset['OR'], y_pos, xerr=[subset['OR']-(np.exp(subset['Log_OR']-1.96*subset['SE'])), 
                                                   (np.exp(subset['Log_OR']+1.96*subset['SE']))-subset['OR']],
                        fmt='o', color='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(subset['Study'])
            ax.axvline(x=1, color='red', linestyle='--')
            ax.set_xlabel("Odds Ratio (Log Scale)")
            ax.set_xscale('log')
            st.pyplot(fig)
            
            # Heatmap (Correlation of metrics)
            st.subheader("Parameter Heatmap")
            corr_data = subset[['OR', 'P_Value', 'HWE_P_Ctrl', 'Chi2']].astype(float)
            fig2, ax2 = plt.subplots()
            sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

    else:
        st.warning("Please calculate models in Tab 2 first.")

# ==========================================
# TAB 4: CONFOUNDER & RISK OF BIAS
# ==========================================
with tabs[3]:
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("Confounder Analysis (Logistic Reg)")
        st.info("Requires raw patient data with columns: Outcome (0/1), Genotype (0/1/2), Age, BMI")
        # Placeholder for Logistic Regression implementation
        st.text("Upload raw data to perform: \nlogit(Outcome) ~ Genotype + Age + BMI")
        
    with c2:
        st.header("Newcastle–Ottawa Scale (NOS)")
        st.write("Rate the quality of your studies manually:")
        
        selection = st.slider("Selection (0-4 stars)", 0, 4, 3)
        comparability = st.slider("Comparability (0-2 stars)", 0, 2, 1)
        exposure = st.slider("Exposure/Outcome (0-3 stars)", 0, 3, 2)
        
        total_score = selection + comparability + exposure
        st.metric("Total NOS Score", f"{total_score} / 9")
        
        if total_score >= 7:
            st.success("High Quality Study")
        else:
            st.warning("High Risk of Bias")