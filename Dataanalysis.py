# Data Cleaner & Analyzer — Streamlit App
# ------------------------------------------------------------
# Quick start:
#   pip install streamlit pandas numpy seaborn matplotlib openpyxl
#   streamlit run app.py
#
# Features:
#   - Upload CSV/Excel, preview head, and view DataFrame info (df.info()).
#   - Clean data: handle missing values, remove duplicates, convert dtypes, handle outliers (IQR).
#   - Analyze: descriptive stats, value counts, interactive plots, correlation heatmap.
#   - Organized UI with sidebar controls and helpful messages.

import io
from io import StringIO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------- Page setup --------------------------
st.set_page_config(page_title="Data Cleaner & Analyzer", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------- Session state init ----------------------
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []

# -------------------------- Utilities ---------------------------
@st.cache_data(show_spinner=False)
def _read_file_from_bytes(bytes_data: bytes, name: str) -> pd.DataFrame:
    name = name.lower()
    b = io.BytesIO(bytes_data)
    if name.endswith(".csv"):
        # Try common CSV settings; fall back to default
        try:
            return pd.read_csv(b)
        except Exception:
            b.seek(0)
            return pd.read_csv(b, sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            return pd.read_excel(b)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Excel file. Ensure 'openpyxl' is installed. Details: {e}"
            )
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

def df_info_to_str(df: pd.DataFrame) -> str:
    buf = StringIO()
    try:
        df.info(buf=buf, memory_usage="deep", show_counts=True)
    except TypeError:
        # For older pandas without show_counts
        df.info(buf=buf, memory_usage="deep")
    return buf.getvalue()

def impute_numeric(df: pd.DataFrame, strategy: str = "mean", columns=None):
    df2 = df.copy()
    if columns is None or len(columns) == 0:
        columns = df2.select_dtypes(include=np.number).columns.tolist()
    filled = {}
    for col in columns:
        if col not in df2.columns:
            continue
        n_missing = int(df2[col].isna().sum())
        if n_missing == 0:
            continue
        if strategy == "mean":
            value = df2[col].mean()
        elif strategy == "median":
            value = df2[col].median()
        elif strategy == "mode":
            mode_vals = df2[col].mode(dropna=True)
            value = mode_vals.iloc[0] if not mode_vals.empty else np.nan
        else:
            continue
        df2[col] = df2[col].fillna(value)
        filled[col] = n_missing
    return df2, filled

def impute_categorical(df: pd.DataFrame, columns=None):
    df2 = df.copy()
    if columns is None or len(columns) == 0:
        columns = df2.select_dtypes(include=["object", "category"]).columns.tolist()
    filled = {}
    for col in columns:
        if col not in df2.columns:
            continue
        n_missing = int(df2[col].isna().sum())
        if n_missing == 0:
            continue
        mode_vals = df2[col].mode(dropna=True)
        if not mode_vals.empty:
            df2[col] = df2[col].fillna(mode_vals.iloc[0])
            filled[col] = n_missing
    return df2, filled

def remove_duplicates(df: pd.DataFrame, subset=None, keep="first"):
    df2 = df.copy()
    before = len(df2)
    if subset and len(subset) > 0:
        df2 = df2.drop_duplicates(subset=subset, keep=keep)
    else:
        df2 = df2.drop_duplicates(keep=keep)
    removed = before - len(df2)
    return df2, removed

def convert_columns(df: pd.DataFrame, columns, target_type, datetime_format=None, downcast_numeric=False):
    df2 = df.copy()
    errors = {}
    for col in columns:
        try:
            if target_type == "numeric":
                df2[col] = pd.to_numeric(df2[col], errors="coerce", downcast="float" if downcast_numeric else None)
            elif target_type == "datetime":
                df2[col] = pd.to_datetime(df2[col], errors="coerce", format=datetime_format if datetime_format else None)
            elif target_type == "category":
                df2[col] = df2[col].astype("category")
            elif target_type == "string":
                df2[col] = df2[col].astype("string")
            else:
                errors[col] = f"Unsupported type: {target_type}"
        except Exception as e:
            errors[col] = str(e)
    return df2, errors

def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

def handle_outliers(df: pd.DataFrame, columns, method="cap", k: float = 1.5):
    df2 = df.copy()
    summary = {}
    for col in columns:
        if col not in df2.columns or not pd.api.types.is_numeric_dtype(df2[col]):
            continue
        s = df2[col]
        lower, upper = iqr_bounds(s, k=k)
        mask_low = s < lower
        mask_high = s > upper
        n_out = int((mask_low | mask_high).sum())

        action = "none"
        if n_out > 0:
            if method == "cap":
                df2.loc[mask_low, col] = lower
                df2.loc[mask_high, col] = upper
                action = "capped"
            elif method == "remove":
                df2 = df2[~(mask_low | mask_high)]
                action = "removed_rows"
            elif method == "mark":
                df2[f"{col}_is_outlier"] = (mask_low | mask_high)
                action = "marked_flag"

        summary[col] = {"outliers": n_out, "action": action, "lower": lower, "upper": upper}
    return df2, summary

# --------------------------- Plotters ---------------------------
def plot_hist(df: pd.DataFrame, column: str, bins: int = 30, kde: bool = False):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax)
    ax.set_title(f"Histogram of {column}")
    return fig

def plot_box(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Box plot of {column}")
    return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"Scatter: {x} vs {y}")
    return fig

def plot_bar_counts(df: pd.DataFrame, column: str, top_n: int = 20, normalize: bool = False):
    vc = df[column].value_counts(dropna=False, normalize=normalize).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h")
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(column)
    ax.set_title(f'{"Proportion" if normalize else "Count"} of {column} (Top {top_n})')
    return fig

def plot_corr_heatmap(df: pd.DataFrame, method: str = "pearson", annot: bool = False):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] == 0:
        return None
    corr = num_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=annot, cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation heatmap ({method})")
    return fig

# --------------------------- UI: Sidebar ---------------------------
st.title("🧼 Data Cleaning and Analysis App")

with st.sidebar:
    st.header("1) Upload Data")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            with st.spinner("Reading file..."):
                df_loaded = _read_file_from_bytes(uploaded.getvalue(), uploaded.name)
            st.session_state.df_original = df_loaded.copy()
            st.session_state.df = df_loaded.copy()
            st.session_state.cleaning_log = []
            st.success(f"Loaded: {df_loaded.shape[0]} rows × {df_loaded.shape[1]} columns")
        except Exception as e:
            st.error(str(e))

    if st.session_state.df is not None:
        st.divider()
        st.header("2) Cleaning Options")

        # Reset
        if st.button("Reset to original data", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.cleaning_log = []
            st.toast("Data reset to original.", icon="↩️")

        # Missing values
        with st.expander("Missing values"):
            df = st.session_state.df
            total_missing = int(df.isna().sum().sum())
            st.caption(f"Total missing values: {total_missing}")

            # Drops
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Drop rows with any NaNs", key="drop_rows_nans", use_container_width=True):
                    before = len(st.session_state.df)
                    with st.spinner("Dropping rows..."):
                        st.session_state.df = st.session_state.df.dropna(axis=0, how="any")
                    after = len(st.session_state.df)
                    removed = before - after
                    st.session_state.cleaning_log.append(f"Dropped {removed} rows with missing values.")
                    st.success(f"Removed {removed} rows.")
            with c2:
                if st.button("Drop columns with any NaNs", key="drop_cols_nans", use_container_width=True):
                    before = st.session_state.df.shape[1]
                    with st.spinner("Dropping columns..."):
                        st.session_state.df = st.session_state.df.dropna(axis=1, how="any")
                    after = st.session_state.df.shape[1]
                    removed = before - after
                    st.session_state.cleaning_log.append(f"Dropped {removed} columns with missing values.")
                    st.success(f"Removed {removed} columns.")

            st.markdown("---")
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Impute numeric
            st.subheader("Impute numeric columns")
            num_sel = st.multiselect("Select numeric columns to impute (optional, defaults to all numeric)", options=num_cols, key="imp_num_cols")
            strategy = st.radio("Strategy", options=["mean", "median", "mode"], horizontal=True, key="imp_num_strategy")
            if st.button("Impute numeric NaNs", use_container_width=True, key="btn_imp_num"):
                with st.spinner("Imputing numeric NaNs..."):
                    st.session_state.df, filled = impute_numeric(st.session_state.df, strategy=strategy, columns=num_sel)
                if filled:
                    total = sum(filled.values())
                    st.session_state.cleaning_log.append(f"Imputed {total} numeric NaNs using {strategy}.")
                    st.success(f"Imputed {total} numeric values across {len(filled)} column(s).")
                else:
                    st.info("No numeric NaNs to impute (in selected columns).")

            # Impute categorical
            st.subheader("Impute categorical columns")
            cat_sel = st.multiselect("Select categorical columns to impute (optional, defaults to all categorical)", options=cat_cols, key="imp_cat_cols")
            if st.button("Impute categorical NaNs (mode)", use_container_width=True, key="btn_imp_cat"):
                with st.spinner("Imputing categorical NaNs..."):
                    st.session_state.df, filled = impute_categorical(st.session_state.df, columns=cat_sel)
                if filled:
                    total = sum(filled.values())
                    st.session_state.cleaning_log.append(f"Imputed {total} categorical NaNs using mode.")
                    st.success(f"Imputed {total} categorical values across {len(filled)} column(s).")
                else:
                    st.info("No categorical NaNs to impute (in selected columns).")

        # Duplicates
        with st.expander("Duplicates"):
            df = st.session_state.df
            st.caption(f"Current duplicate rows (by all columns): {int(df.duplicated().sum())}")
            subset = st.multiselect("Subset of columns (optional, default all)", options=df.columns.tolist(), key="dup_subset")
            keep = st.radio("Keep which duplicate?", options=["first", "last", "none"], format_func=lambda x: "None (drop all duplicates)" if x == "none" else x.capitalize(), horizontal=True, key="dup_keep")
            keep_param = False if keep == "none" else keep
            if st.button("Remove duplicates", use_container_width=True, key="btn_rm_dups"):
                with st.spinner("Removing duplicates..."):
                    st.session_state.df, removed = remove_duplicates(st.session_state.df, subset=subset, keep=keep_param)
                st.session_state.cleaning_log.append(f"Removed {removed} duplicate rows.")
                st.success(f"Removed {removed} rows.")

        # Data type conversion
        with st.expander("Data type conversion"):
            df = st.session_state.df
            cols_to_convert = st.multiselect("Columns to convert", options=df.columns.tolist(), key="conv_cols")
            target_type = st.radio("Target type", options=["numeric", "datetime", "category", "string"], horizontal=True, key="conv_type")
            dt_fmt = None
            downcast_flag = False
            if target_type == "datetime":
                dt_fmt = st.text_input("Optional datetime format (e.g., %Y-%m-%d). Leave blank to auto-parse.", value="", key="conv_dt_fmt") or None
            if target_type == "numeric":
                downcast_flag = st.checkbox("Downcast numeric to reduce memory (float)", value=False, key="conv_downcast")
            if st.button("Convert", use_container_width=True, key="btn_convert"):
                if not cols_to_convert:
                    st.warning("Select at least one column to convert.")
                else:
                    with st.spinner("Converting types..."):
                        st.session_state.df, errors = convert_columns(
                            st.session_state.df, cols_to_convert, target_type, datetime_format=dt_fmt, downcast_numeric=downcast_flag
                        )
                    if errors:
                        st.warning(f"Some conversions had issues: {errors}")
                    changed = [c for c in cols_to_convert if c not in errors]
                    if changed:
                        st.session_state.cleaning_log.append(f"Converted {len(changed)} column(s) to {target_type}.")
                        st.success(f"Converted: {', '.join(changed)}")

        # Outliers (IQR)
        with st.expander("Outlier detection/handling (IQR)"):
            df = st.session_state.df
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for outlier handling.")
            else:
                cols = st.multiselect("Numeric columns", options=num_cols, key="out_cols")
                method = st.radio("Action", options=["detect_only", "cap", "remove", "mark"], format_func=lambda x: {
                    "detect_only":"Detect only",
                    "cap":"Cap (winsorize)",
                    "remove":"Remove rows with outliers",
                    "mark":"Mark with boolean flag"
                }[x], horizontal=True, key="out_method")
                k = st.slider("IQR multiplier (k)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="out_k")
                if st.button("Run outlier step", use_container_width=True, key="btn_outliers"):
                    if not cols:
                        st.warning("Select at least one numeric column.")
                    else:
                        if method == "detect_only":
                            with st.spinner("Detecting outliers..."):
                                _, summary = handle_outliers(df, cols, method="mark", k=k)  # create flags in a temp result
                            total_out = sum(s["outliers"] for s in summary.values())
                            st.info(f"Detected {total_out} outliers across {len(cols)} column(s). No changes made.")
                        else:
                            with st.spinner("Handling outliers..."):
                                st.session_state.df, summary = handle_outliers(st.session_state.df, cols, method=("cap" if method == "cap" else "remove" if method == "remove" else "mark"), k=k)
                            total_out = sum(s["outliers"] for s in summary.values())
                            st.session_state.cleaning_log.append(f"Outlier step ({method}) affected {total_out} values across {len(cols)} column(s).")
                            st.success(f"Completed outlier step: {method}. Affected {total_out} values.")

                st.caption("Quick box plot (pick a numeric column to preview distribution):")
                preview_col = st.selectbox("Box plot preview column", options=["(none)"] + num_cols, index=0, key="out_box_col")
                if preview_col != "(none)":
                    with st.spinner("Rendering box plot..."):
                        fig = plot_box(df, preview_col)
                    st.pyplot(fig, use_container_width=True)

        st.divider()
        st.header("3) Export")
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned data (CSV)", data=csv_bytes, file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)

# --------------------------- UI: Main ----------------------------
if st.session_state.df is None:
    st.info("Upload a CSV or Excel file to begin.")
else:
    df = st.session_state.df
    df_orig = st.session_state.df_original

    tabs = st.tabs(["Overview", "Descriptive stats", "Visualizations", "Correlation"])
    # Overview
    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Current data (preview)")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        with c2:
            st.subheader("DataFrame info")
            st.text(df_info_to_str(df))

        if st.session_state.cleaning_log:
            st.subheader("Cleaning log")
            for entry in st.session_state.cleaning_log[-10:]:
                st.write("• " + entry)

        with st.expander("Original data (preview)"):
            st.dataframe(df_orig.head(), use_container_width=True)
            st.caption(f"{df_orig.shape[0]} rows × {df_orig.shape[1]} columns")

    # Descriptive stats
    with tabs[1]:
        st.subheader("Numerical summary (df.describe)")
        if df.select_dtypes(include=np.number).shape[1] > 0:
            st.dataframe(df.select_dtypes(include=np.number).describe().T, use_container_width=True)
        else:
            st.info("No numerical columns found.")

        st.markdown("---")
        st.subheader("Value counts (categorical)")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            cat_col = st.selectbox("Select a categorical column", options=cat_cols)
            top_n = st.slider("Top N categories", min_value=5, max_value=50, value=20, step=1)
            norm = st.checkbox("Show proportions instead of counts", value=False)
            vc = df[cat_col].value_counts(dropna=False, normalize=norm).head(top_n)
            st.dataframe(vc.rename("value").to_frame(), use_container_width=True)

    # Visualizations
    with tabs[2]:
        st.subheader("Customizable plots")
        plot_type = st.selectbox(
            "Plot type",
            options=["Histogram/KDE", "Box plot", "Scatter plot", "Bar/Count"],
        )

        if plot_type in ["Histogram/KDE", "Box plot", "Scatter plot"]:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for this plot.")
        if plot_type == "Bar/Count":
            cat_cols_all = df.columns.tolist()

        if plot_type == "Histogram/KDE":
            col = st.selectbox("Numeric column", options=num_cols)
            bins = st.slider("Bins", min_value=5, max_value=100, value=30)
            kde = st.checkbox("Overlay KDE", value=False)
            with st.spinner("Generating histogram..."):
                fig = plot_hist(df, col, bins=bins, kde=kde)
            st.pyplot(fig, use_container_width=True)

        elif plot_type == "Box plot":
            col = st.selectbox("Numeric column", options=num_cols, key="box_col")
            with st.spinner("Generating box plot..."):
                fig = plot_box(df, col)
            st.pyplot(fig, use_container_width=True)

        elif plot_type == "Scatter plot":
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for a scatter plot.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    xcol = st.selectbox("X-axis", options=num_cols, key="scatter_x")
                with c2:
                    ycol = st.selectbox("Y-axis", options=[c for c in num_cols if c != xcol], key="scatter_y")
                hue_opts = ["(none)"] + df.columns.tolist()
                hue = st.selectbox("Color by (optional)", options=hue_opts, index=0)
                hue = None if hue == "(none)" else hue
                with st.spinner("Generating scatter plot..."):
                    fig = plot_scatter(df, xcol, ycol, hue=hue)
                st.pyplot(fig, use_container_width=True)

        elif plot_type == "Bar/Count":
            col = st.selectbox("Column", options=cat_cols_all, key="bar_col")
            top_n = st.slider("Top N", min_value=5, max_value=50, value=20)
            normalize = st.checkbox("Show proportions", value=False, key="bar_norm")
            with st.spinner("Generating bar chart..."):
                fig = plot_bar_counts(df, col, top_n=top_n, normalize=normalize)
            st.pyplot(fig, use_container_width=True)

    # Correlation
    with tabs[3]:
        st.subheader("Correlation heatmap")
        method = st.radio("Method", options=["pearson", "spearman", "kendall"], horizontal=True)
        annot = st.checkbox("Annotate cells", value=False)
        with st.spinner("Computing correlation..."):
            fig = plot_corr_heatmap(df, method=method, annot=annot)
        if fig is None:
            st.info("No numeric columns available for correlation.")
        else:
            st.pyplot(fig, use_container_width=True)