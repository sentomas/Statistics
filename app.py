import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="BioStats Meta-App", layout="wide")

# --- HELPER FUNCTIONS ---
def calculate_or_stats(a, b, c, d):
    """Calculates OR, SE, 95% CI, Chi2, and P-value."""
    # Haldane correction for zero cells
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    # Odds Ratio
    try:
        or_val = (a * d) / (b * c)
        log_or = np.log(or_val)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        lower_ci = np.exp(log_or - 1.96 * se_log_or)
        upper_ci = np.exp(log_or + 1.96 * se_log_or)
    except:
        return None

    # Chi-Square
    obs = np.array([[a, b], [c, d]])
    chi2, p_val, _, _ = stats.chi2_contingency(obs, correction=False)

    return {
        "OR": or_val, "Log_OR": log_or, "SE": se_log_or,
        "CI_Lower": lower_ci, "CI_Upper": upper_ci,
        "Chi2": chi2, "P_Value": p_val
    }

def calculate_hwe(aa, aa_het, aa_homo):
    """Calculates Hardy-Weinberg Equilibrium."""
    total = aa + aa_het + aa_homo
    if total == 0: return None
    
    # Alelle frequencies
    p = (2 * aa + aa_het) / (2 * total)
    q = 1 - p
    
    # Expected counts
    exp_aa = (p ** 2) * total
    exp_het = (2 * p * q) * total
    exp_homo = (q ** 2) * total
    
    # Chi-Square for HWE
    obs = [aa, aa_het, aa_homo]
    exp = [exp_aa, exp_het, exp_homo]
    chi2, p_val = stats.chisquare(obs, f_exp=exp)
    
    return {"Chi2": chi2, "P_Value": p_val, "Exp": exp}

# --- UI LAYOUT ---
st.title("🧬 BioStats: Meta-Analysis & Calculator")

tab1, tab2, tab3 = st.tabs(["🧮 Single Study Calculator", "📊 Meta-Analysis", "📈 Meta-Regression"])

# ==========================================
# TAB 1: SINGLE STUDY CALCULATOR
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Association Test (2x2 Table)")
        c1, c2 = st.columns(2)
        a = c1.number_input("Cases Exposed (a)", min_value=0, value=10)
        b = c2.number_input("Controls Exposed (b)", min_value=0, value=20)
        c = c1.number_input("Cases Unexposed (c)", min_value=0, value=15)
        d = c2.number_input("Controls Unexposed (d)", min_value=0, value=35)
        
        if st.button("Calculate Association"):
            res = calculate_or_stats(a, b, c, d)
            if res:
                st.success(f"**Odds Ratio:** {res['OR']:.4f} [{res['CI_Lower']:.4f} - {res['CI_Upper']:.4f}]")
                st.info(f"**P-Value:** {res['P_Value']:.4e} | **Chi²:** {res['Chi2']:.4f}")
            else:
                st.error("Invalid calculation (check for zeros).")

    with col2:
        st.subheader("Hardy-Weinberg Equilibrium (HWE)")
        g1 = st.number_input("Homozygous Major (AA)", min_value=0, value=50)
        g2 = st.number_input("Heterozygous (Aa)", min_value=0, value=30)
        g3 = st.number_input("Homozygous Minor (aa)", min_value=0, value=10)
        
        if st.button("Check HWE"):
            hwe = calculate_hwe(g1, g2, g3)
            st.write(f"**HWE P-Value:** {hwe['P_Value']:.4f}")
            if hwe['P_Value'] < 0.05:
                st.warning("⚠️ Deviation from HWE detected.")
            else:
                st.success("✅ Consistent with HWE.")

# ==========================================
# TAB 2: META-ANALYSIS
# ==========================================
with tab2:
    st.markdown("### Upload Data for Meta-Analysis")
    st.caption("CSV must have columns: `StudyName`, `a`, `b`, `c`, `d`")
    
    # Demo Data Creator
    demo_data = pd.DataFrame({
        'StudyName': ['Study 1', 'Study 2', 'Study 3', 'Study 4'],
        'a': [10, 20, 15, 40],
        'b': [20, 30, 25, 50],
        'c': [15, 25, 20, 45],
        'd': [35, 45, 40, 60]
    })
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using demo data. Upload your own CSV to override.")
        df = demo_data

    st.dataframe(df.head(), use_container_width=True)

    if st.button("Run Meta-Analysis"):
        # 1. Calculate Effects for all rows
        results = []
        for idx, row in df.iterrows():
            res = calculate_or_stats(row['a'], row['b'], row['c'], row['d'])
            if res:
                results.append({
                    'Study': row.get('StudyName', f'Study {idx+1}'),
                    'Log_OR': res['Log_OR'],
                    'SE': res['SE'],
                    'OR': res['OR'],
                    'Lower': res['CI_Lower'],
                    'Upper': res['CI_Upper']
                })
        
        meta_df = pd.DataFrame(results)
        
        # 2. Fixed Effects Model (Inverse Variance)
        weights = 1 / (meta_df['SE'] ** 2)
        pooled_log_or = np.sum(weights * meta_df['Log_OR']) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))
        
        pooled_or = np.exp(pooled_log_or)
        pooled_lower = np.exp(pooled_log_or - 1.96 * pooled_se)
        pooled_upper = np.exp(pooled_log_or + 1.96 * pooled_se)

        # Display Results
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pooled Odds Ratio", f"{pooled_or:.3f}")
        c2.metric("95% CI", f"{pooled_lower:.3f} - {pooled_upper:.3f}")
        
        # Heterogeneity (Cochran's Q)
        q_stat = np.sum(weights * (meta_df['Log_OR'] - pooled_log_or)**2)
        df_q = len(meta_df) - 1
        p_het = 1 - stats.chi2.cdf(q_stat, df_q)
        c3.metric("Heterogeneity P-value", f"{p_het:.4f}")

        # 3. Forest Plot (Matplotlib)
        st.subheader("Forest Plot")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        y_pos = np.arange(len(meta_df))
        ax.errorbar(meta_df['OR'], y_pos, xerr=[meta_df['OR']-meta_df['Lower'], meta_df['Upper']-meta_df['OR']], 
                    fmt='o', color='blue', label='Studies')
        
        # Add Pooled Diamond (approximated as a point for this demo)
        ax.errorbar(pooled_or, -1, xerr=[[pooled_or-pooled_lower], [pooled_upper-pooled_or]], 
                    fmt='D', color='red', label='Pooled Estimate')
        
        ax.set_yticks(list(y_pos) + [-1])
        ax.set_yticklabels(list(meta_df['Study']) + ['Pooled'])
        ax.axvline(x=1, color='gray', linestyle='--')
        ax.set_xlabel("Odds Ratio (log scale)")
        ax.set_xscale('log')
        ax.legend()
        
        st.pyplot(fig)

# ==========================================
# TAB 3: META-REGRESSION
# ==========================================
with tab3:
    st.markdown("### Meta-Regression")
    st.info("Requires a 'Year' column in your dataset.")
    
    # Add dummy year to demo data
    if 'Year' not in df.columns:
        df['Year'] = [2010, 2012, 2015, 2018][:len(df)]
        
    if st.checkbox("Show Regression Data"):
        st.dataframe(df)

    if st.button("Run Regression (Year vs Log OR)"):
        # Calculate Log ORs again
        y_vals = [] # Effect sizes (Log OR)
        w_vals = [] # Weights (1/variance)
        x_vals = [] # Moderator (Year)
        
        for idx, row in df.iterrows():
            res = calculate_or_stats(row['a'], row['b'], row['c'], row['d'])
            if res:
                y_vals.append(res['Log_OR'])
                w_vals.append(1/(res['SE']**2))
                x_vals.append(row['Year'])

        # Weighted Least Squares (WLS) using Statsmodels
        X = sm.add_constant(x_vals) # Add intercept
        model = sm.WLS(y_vals, X, weights=w_vals)
        reg_results = model.fit()
        
        st.write(reg_results.summary())
        
        # Plot
        fig2, ax2 = plt.subplots()
        ax2.scatter(x_vals, y_vals, s=[w*5 for w in w_vals], alpha=0.6)
        ax2.plot(x_vals, reg_results.predict(X), color='red', label='Regression Line')
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Log Odds Ratio")
        st.pyplot(fig2)