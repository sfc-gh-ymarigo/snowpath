import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from snowflake.snowpark.context import get_active_session
    from snowflake.snowpark.functions import col, listagg, count, avg, max as sf_max, min as sf_min
    from snowflake.snowpark.types import StringType, ArrayType, FloatType, IntegerType
    session = get_active_session()
except:
    st.error("Could not connect to Snowflake session", icon=":material/chat_error:")
    st.stop()

# --- Session state initialization (after imports, before UI) ---
# No session state needed for simplified approach

# Function to check if current parameters differ from cached ones
def parameters_changed():
    """Check if current parameters differ from the last cached analysis"""
    if 'last_behavioral_params' not in st.session_state:
        return True, []  # No cached params, so return True with empty changed list
    
    # Get current parameters
    current_params = {
        'database': database,
        'schema': schema, 
        'table': table,
        'uid': uid,
        'evt': evt,
        'tmstp': tmstp,
        'period_type': period_type,
        'clustering_method': clustering_method,
        'feature_weighting': feature_weighting,
        'min_sequence_length': min_sequence_length,
        'max_sequence_length': max_sequence_length,
        'min_count': min_count,
        'n_clusters': n_clusters if clustering_method != "Latent Dirichlet Allocation (LDA)" else n_topics,
        'max_customers': max_customers,
        'pca_option': pca_option,
        'normalize_vectors': normalize_vectors,
        'vector_size': vector_size,
        'window_size': window_size
    }
    
    # Add LDA-specific parameters if applicable
    if clustering_method == "Latent Dirichlet Allocation (LDA)":
        current_params.update({
            'alpha': alpha,
            'beta': beta,
            'n_topics': n_topics
        })
    elif clustering_method == "DBSCAN":
        current_params.update({
            'eps': eps,
            'min_samples': min_samples
        })
    
    # Add date range parameters if applicable
    if period_type == "Last N Days":
        current_params['days_back'] = days_back
    elif period_type == "Date Range":
        current_params.update({
            'start_date': str(start_date),
            'end_date': str(end_date)
        })
    
    # Compare with cached parameters
    cached_params = st.session_state['last_behavioral_params']
    
    # Find changed parameters
    changed_params = []
    for key, value in current_params.items():
        if key not in cached_params:
            changed_params.append(key)
        else:
            cached_value = cached_params[key]
            # Handle type mismatches (e.g., int vs float) and None values
            if cached_value != value:
                # Try string comparison as fallback for type mismatches
                if str(cached_value) != str(value):
                    changed_params.append(key)
    
    return len(changed_params) > 0, changed_params

# Function to determine optimal PCA components
def get_optimal_pca_components(X, clustering_method, max_components=50):
    """Determine optimal PCA components based on method and data"""
    n_features = X.shape[1] 
    n_samples = X.shape[0]
    
    if clustering_method == "Latent Dirichlet Allocation (LDA)":
        return {"recommended": 0, "reason": "PCA not applicable to LDA (uses discrete token counts)"}
    
    if n_features <= 5:
        return {"recommended": 0, "reason": f"Too few features ({n_features}) for meaningful PCA"}
    
    try:
        # Calculate PCA to analyze variance
        from sklearn.decomposition import PCA
        pca_analysis = PCA()
        pca_analysis.fit(X)
        
        # Find components for different variance thresholds
        cumvar = pca_analysis.explained_variance_ratio_.cumsum()
        comp_85 = int(np.argmax(cumvar >= 0.85)) + 1 if np.any(cumvar >= 0.85) else n_features
        comp_90 = int(np.argmax(cumvar >= 0.90)) + 1 if np.any(cumvar >= 0.90) else n_features
        comp_95 = int(np.argmax(cumvar >= 0.95)) + 1 if np.any(cumvar >= 0.95) else n_features
        
        # Method-specific recommendations
        recommendations = {
            "K-Means": min(comp_90, max_components, max(10, n_features//3)),
            "DBSCAN": min(comp_95, max_components, max(15, n_features//2)), 
            "Gaussian Mixture": min(comp_85, max_components, max(8, n_features//4)),
            "Hierarchical": min(comp_90, max_components, max(10, n_features//3))
        }
        
        optimal = recommendations.get(clustering_method, comp_90)
        optimal = min(optimal, n_features - 1)  # Can't exceed original dimensions
        
        return {
            "recommended": optimal,
            "variance_explained": float(cumvar[optimal-1]) if optimal > 0 else 0,
            "total_features": n_features,
            "options": {
                "conservative": comp_85,
                "balanced": comp_90, 
                "aggressive": comp_95
            },
            "reason": f"Optimal for {clustering_method} clustering"
        }
        
    except Exception as e:
        return {"recommended": 0, "reason": f"PCA analysis failed: {str(e)}"}

# Function to get available Cortex models (data only, no UI)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_cortex_models():
    """Fetch available Cortex models from Snowflake with optimized logic"""
    fallback_models = [
        "mixtral-8x7b",
        "mistral-large", 
        "mistral-7b",
        "llama3-8b",
        "llama3-70b",
        "gemma-7b"
    ]
    
    try:
        # First, check existing models without refreshing (faster)
        # Note: SHOW MODELS doesn't support WHERE clause, so we filter after
        models_result = session.sql("SHOW MODELS IN SNOWFLAKE.MODELS").collect()
        
        # Extract model names and filter for Cortex base models
        available_models = []
        for row in models_result:
            # Filter for Cortex base models (check model_type if available, or filter by name patterns)
            # Use dictionary-style access instead of attribute access for reliability
            try:
                # Try different possible column names for model type and name
                model_type = ''
                model_name = ''
                
                # Try to get model type (can be MODEL_TYPE, model_type, type, etc.)
                for type_col in ['MODEL_TYPE', 'model_type', 'type', 'TYPE']:
                    if hasattr(row, 'get') and type_col in row:
                        model_type = str(row[type_col]).lower()
                        break
                    elif hasattr(row, type_col):
                        model_type = str(getattr(row, type_col)).lower()
                        break
                
                # Try to get model name (can be name, NAME, model_name, etc.)
                for name_col in ['name', 'NAME', 'model_name', 'MODEL_NAME']:
                    if hasattr(row, 'get') and name_col in row:
                        model_name = str(row[name_col])
                        break
                    elif hasattr(row, name_col):
                        model_name = str(getattr(row, name_col))
                        break
                
                # Explicitly exclude Arctic models (text-to-SQL, not suitable for analysis)
                if 'arctic' in model_name.lower():
                    continue  # Skip Arctic models
                
                if model_type == 'cortex_base' and model_name:
                    available_models.append(model_name)
                elif model_name and any(keyword in model_name.lower() for keyword in ['mixtral', 'mistral', 'openai','llama','claude','gemma']):
                    available_models.append(model_name)
            except Exception:
                # If accessing row data fails, skip this row
                continue
        
        if available_models:
            # Sort and return existing models
            available_models.sort()
            return {"models": available_models, "status": "found"}
        else:
            # No models found, but don't show UI here
            return {"models": fallback_models, "status": "not_found"}
        
    except Exception as e:
        # Return fallback models with error status
        return {"models": fallback_models, "status": "error", "error": str(e)}

# Function to refresh models (separate from getting models)
def refresh_cortex_models():
    """Refresh Cortex models list"""
    try:
        # Refresh the models list (this can be slow)
        session.sql("CALL SNOWFLAKE.MODELS.CORTEX_BASE_MODELS_REFRESH()").collect()
        
        # Clear cache to get fresh models
        get_available_cortex_models.clear()
        
        # Get fresh models
        result = get_available_cortex_models()
        return result
        
    except Exception as e:
        return {"models": [], "status": "refresh_error", "error": str(e)}

#===================================================================================
# Cached Parameter Query Functions (to avoid repeated queries)
#===================================================================================
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_databases(_session):
    """Fetch list of databases - cached to avoid repeated queries"""
    sqldb = "SHOW DATABASES"
    databases = _session.sql(sqldb).collect()
    return pd.DataFrame(databases)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_schemas(_session, database):
    """Fetch schemas for a given database - cached"""
    if not database:
        return pd.DataFrame()
    sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
    schemas = _session.sql(sqlschemas).collect()
    return pd.DataFrame(schemas)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tables(_session, database, schema):
    """Fetch tables for a given database and schema - cached"""
    if not database or not schema:
        return pd.DataFrame()
    sqltables = f"""
        SELECT TABLE_NAME 
        FROM {database}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
    """
    tables = _session.sql(sqltables).collect()
    return pd.DataFrame(tables)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_columns(_session, database, schema, tbl):
    """Fetch columns for a given table - cached"""
    if not database or not schema or not tbl:
        return pd.DataFrame()
    cols = f"""
        SELECT COLUMN_NAME
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{tbl}'
        ORDER BY ORDINAL_POSITION;
    """
    colssql = _session.sql(cols).collect()
    return pd.DataFrame(colssql)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_column_type(_session, database, schema, tbl, column):
    """Fetch column data type - cached"""
    if not database or not schema or not tbl or not column:
        return None
    column_info_query = f"""
        SELECT DATA_TYPE
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{tbl}' AND COLUMN_NAME = '{column}';
    """
    column_info = _session.sql(column_info_query).collect()
    if column_info:
        return column_info[0]['DATA_TYPE']
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_distinct_values(_session, database, schema, tbl, column):
    """Fetch distinct values from a column - cached"""
    if not database or not schema or not tbl or not column:
        return pd.DataFrame()
    query = f"SELECT DISTINCT {column} FROM {database}.{schema}.{tbl} ORDER BY {column}"
    values = _session.sql(query).collect()
    return pd.DataFrame(values)

#===================================================================================

# Function to display AI insights UI with model selection and toggle
def display_ai_insights_section(key_prefix, help_text="Select the LLM model for AI analysis", ai_content_callback=None):
    """Display AI insights section with model selection UI and toggle in an expander"""
    
    with st.expander("AI Insights & Recommendations", expanded=False,icon=":material/network_intel_node:"):
        models_result = get_available_cortex_models()
        available_models = models_result["models"]
        status = models_result["status"]
        
        # Show status message if needed
        if status == "not_found":
            st.warning("No Cortex models found in your Snowflake account. Using default models.", icon=":material/warning:")
            
            # Add refresh option in expander to not clutter main UI
            with st.expander("Refresh Model List"):
                if st.button("Refresh Cortex Models", key=f"refresh_{key_prefix}", help="This may take a few moments to complete"):
                    with st.spinner("Refreshing Cortex models list..."):
                        refresh_result = refresh_cortex_models()
                        if refresh_result["status"] == "found":
                            st.success(f"Found {len(refresh_result['models'])} Cortex models after refresh!", icon=":material/check:")
                            st.rerun()  # Refresh the UI to show new models
                        elif refresh_result["status"] == "refresh_error":
                            st.error(f"Failed to refresh models: {refresh_result.get('error', 'Unknown error')}", icon=":material/chat_error:")
                        else:
                            st.error("No Cortex models found even after refresh.", icon=":material/chat_error:")
        
        elif status == "error":
            st.warning(f"Could not fetch model list: {models_result.get('error', 'Unknown error')}. Using default models.", icon=":material/warning:")
        
        # Always show model selection (with fallback models if needed)
        model_options = available_models + ["Enter custom model name"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_option = st.selectbox(
                "Choose AI Model",
                options=model_options,
                index=0,
                key=f"{key_prefix}_select",
                help=help_text
            )
            
            if selected_option == "Enter custom model name":
                selected_model = st.text_input(
                    "Enter Model Name",
                    value="mixtral-8x7b",
                    key=f"{key_prefix}_custom",
                    help="Enter the exact model name (e.g., mixtral-8x7b, mistral-large)"
                )
            else:
                selected_model = selected_option
            
            # Toggle placed under model selection
            ai_enabled = st.toggle("Expl**AI**n Me!", key=f"{key_prefix}_toggle", help="Generate AI insights and recommendations for your clustering results")
        
        # with col2 & col3: (left empty for alignment with other sections)
        
        # If AI is enabled and callback provided, execute the AI content within the expander
        if ai_enabled and ai_content_callback:
            ai_content_callback(selected_model)
        
        return selected_model, ai_enabled

# Function to display AI cluster interpretation UI with model selection and toggle
def display_cluster_interpretation_section(key_prefix, cluster_counts, cluster_percentages, cluster_event_analysis, clustering_method, feature_weighting, results_df):
    """Display AI cluster interpretation section with model selection UI and toggle in an expander"""
    
    with st.expander("AI Cluster Interpretation", expanded=False, icon=":material/network_intel_node:"):
        st.markdown("**Analyze cluster behaviors and business implications**")
        
        models_result = get_available_cortex_models()
        available_models = models_result["models"]
        status = models_result["status"]
        
        # Show status message if needed
        if status == "not_found":
            st.warning("No Cortex models found in your Snowflake account. Using default models.", icon=":material/warning:")
            
            # Add refresh option in expander to not clutter main UI
            with st.expander("Refresh Model List"):
                if st.button("Refresh Cortex Models", key=f"refresh_{key_prefix}", help="This may take a few moments to complete"):
                    with st.spinner("Refreshing Cortex models list..."):
                        refresh_result = refresh_cortex_models()
                        if refresh_result["status"] == "found":
                            st.success(f"Found {len(refresh_result['models'])} Cortex models after refresh!", icon=":material/check:")
                            st.rerun()  # Refresh the UI to show new models
                        elif refresh_result["status"] == "refresh_error":
                            st.error(f"Failed to refresh models: {refresh_result.get('error', 'Unknown error')}", icon=":material/chat_error:")
                        else:
                            st.error("No Cortex models found even after refresh.", icon=":material/chat_error:")
        
        elif status == "error":
            st.warning(f"Could not fetch model list: {models_result.get('error', 'Unknown error')}. Using default models.", icon=":material/warning:")
        
        # Always show model selection (with fallback models if needed)
        model_options = available_models + ["Enter custom model name"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_option = st.selectbox(
                "Choose AI Model for Interpretation",
                options=model_options,
                index=0,
                key=f"{key_prefix}_select",
                help="Select the LLM model for cluster behavior analysis"
            )
            
            if selected_option == "Enter custom model name":
                selected_model = st.text_input(
                    "Enter Model Name",
                    value="mixtral-8x7b",
                    key=f"{key_prefix}_custom",
                    help="Enter the exact model name (e.g., mixtral-8x7b, mistral-large)"
                )
            else:
                selected_model = selected_option
            
            # Toggle placed under model selection
            interpret_enabled = st.toggle("Interpret Clusters", key=f"{key_prefix}_toggle", help="Generate AI interpretation of cluster behaviors and business implications")
        
        # with col2 & col3: (left empty for alignment with other sections)
        
        # If interpretation is enabled, execute the AI analysis within the expander
        if interpret_enabled:
            with st.spinner("Analyzing cluster behaviors and business context..."):
                try:
                    # Prepare cluster analysis data for AI
                    cluster_analysis_text = ""
                    
                    for cluster_id in cluster_counts.index:
                        cluster_name = f"Cluster {cluster_id}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {cluster_id}"
                        cluster_count = cluster_counts[cluster_id]
                        cluster_percentage = cluster_percentages[cluster_id]
                        
                        cluster_analysis_text += f"\n**{cluster_name}:**\n"
                        cluster_analysis_text += f"- Size: {cluster_count:,} customers ({cluster_percentage}% of total)\n"
                        
                        if cluster_id in cluster_event_analysis:
                            top_events = cluster_event_analysis[cluster_id].head(10)  # Top 10 events for AI analysis
                            cluster_analysis_text += f"- Top Events:\n"
                            for event_name, event_count in top_events.items():
                                event_pct = (event_count / cluster_event_analysis[cluster_id].sum() * 100)
                                cluster_analysis_text += f"  • {event_name}: {event_count} occurrences ({event_pct:.1f}%)\n"
                        else:
                            cluster_analysis_text += "- No significant events identified\n"
                    
                    # Create AI prompt for cluster interpretation
                    interpretation_prompt = f"""
                    You are a business analyst specializing in customer behavioral segmentation. Analyze the following customer clusters and provide business insights.
                    
                    CLUSTERING ANALYSIS RESULTS:
                    - Method Used: {clustering_method}
                    - Feature Weighting: {feature_weighting}
                    - Total Customers Analyzed: {len(results_df):,}
                    - Number of Clusters/Topics: {len(cluster_counts)}
                    
                    CLUSTER BREAKDOWN:
                    {cluster_analysis_text}
                    
                    ANALYSIS INSTRUCTIONS:
                    Please provide a comprehensive interpretation for each cluster/topic with the following structure:
                    
                    For each cluster, provide:
                    1. **Behavioral Profile**: What type of customer behavior does this cluster represent?
                    2. **Business Significance**: What does this cluster mean from a business perspective?
                    3. **Key Characteristics**: What are the defining traits based on the top events?
                    4. **Business Implications**: How should the business approach this customer segment?
                    5. **Potential Actions**: What specific strategies or tactics would be most effective?
                    
                    CONTEXT CONSIDERATIONS:
                    - Focus on actionable business insights
                    - Consider the relative size of each cluster (larger clusters may be more strategically important)
                    - Interpret event patterns in terms of customer journey stages, preferences, or behaviors
                    - Suggest practical applications for marketing, product development, or customer experience
                    - Keep interpretations concise but insightful
                    
                    Format your response with clear headers for each cluster and bullet points for easy reading.
                    """
                    
                    # Call Snowflake Cortex for cluster interpretation
                    interpretation_query = f"""
                    SELECT SNOWFLAKE.CORTEX.COMPLETE(
                        '{selected_model}',
                        '{interpretation_prompt.replace("'", "''")}'
                    ) as interpretation
                    """
                    
                    interpretation_result = session.sql(interpretation_query).collect()
                    
                    if interpretation_result and len(interpretation_result) > 0:
                        interpretation = interpretation_result[0]['INTERPRETATION']
                        
                        st.markdown("**AI Cluster Interpretation & Business Insights**")
                        st.markdown(interpretation)
                    else:
                        st.error("Failed to generate cluster interpretation", icon=":material/chat_error:")
                        
                except Exception as e:
                    st.error(f"Error generating cluster interpretation: {str(e)}", icon=":material/chat_error:")
        
        return selected_model, interpret_enabled

# Improved writeback functionality with cluster selection
def add_writeback_functionality_segments_improved(results_df, clustering_method, customer_id_col, event_seq_col, seq_length_col, unique_events_col):
    """Enhanced writeback functionality for segments with cluster selection and Snowpark DataFrames"""
    
    # Add section header matching other sections
    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Writeback Segments</h2>
    <hr style='margin-top: -8px;margin-bottom: 10px;'>
    """, unsafe_allow_html=True)
    
    if st.toggle("Writeback the behavioral segments to Snowflake", key="writeback_toggle_segments"):
        
        # Cluster Selection

        available_clusters = sorted(results_df['cluster'].unique())
        
        selected_clusters = st.multiselect(
            "Choose clusters to include in the export:",
            options=available_clusters,
            format_func=lambda x: f"Cluster {x}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {x}",
            default=available_clusters,  # All clusters selected by default
            key="cluster_selection_segments"
        )
        
        if not selected_clusters:
            st.warning("Please select at least one cluster to export.", icon=":material/warning:")
            return
        
        # Filter results_df to selected clusters
        filtered_df = results_df[results_df['cluster'].isin(selected_clusters)]
        
        # Show preview
        total_customers = len(filtered_df)
        st.info(f"Export Preview: {total_customers:,} customers across {len(selected_clusters)} clusters",icon=":material/info:")
        
        # Fetch DBs using cached method
        db0 = fetch_databases(session)
        
        # Database, Schema, Table selection in better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wb_database = st.selectbox("Database", db0['name'].unique(), index=None, key="wb_db_segments", placeholder="Choose...")
        
        wb_schema = None
        if wb_database:
            schema0 = fetch_schemas(session, wb_database)
            
            with col2:
                wb_schema = st.selectbox("Schema", schema0['name'].unique(), index=None, key="wb_schema_segments", placeholder="Choose...")
        
        with col3:
            if wb_database and wb_schema:
                wb_table_name = st.text_input("Table Name", key="wb_tbl_segments", placeholder="e.g. behavioral_segments")
            else:
                wb_table_name = None
        
        # Write button and success message - left aligned
        if wb_database and wb_schema and wb_table_name:
            
            if st.button("Write Table", key="wb_btn_segments", type="primary"):
                with st.spinner("Writing data to Snowflake..."):
                    try:
                        # Convert pandas DataFrame to Snowpark DataFrame
                        from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
                        from snowflake.snowpark import Row
                            
                        # Define schema for Snowpark DataFrame
                        schema_def = StructType([
                            StructField("CUSTOMER_ID", StringType()),
                            StructField("CLUSTER_ID", IntegerType()),
                            StructField("SEQUENCE_LENGTH", IntegerType()),
                            StructField("UNIQUE_EVENTS", IntegerType()),
                            StructField("CLUSTERING_METHOD", StringType())
                        ])
                        
                        # Prepare data for Snowpark DataFrame
                        rows_data = []
                        for _, row in filtered_df.iterrows():
                            rows_data.append(Row(
                                CUSTOMER_ID=str(row['customer_id']),
                                CLUSTER_ID=int(row['cluster']),
                                SEQUENCE_LENGTH=int(row[seq_length_col]),
                                UNIQUE_EVENTS=int(row[unique_events_col]),
                                CLUSTERING_METHOD=clustering_method
                            ))
                        
                        # Create Snowpark DataFrame
                        snowpark_df = session.create_dataframe(rows_data, schema=schema_def)
                        
                        # Write to Snowflake table using Snowpark
                        snowpark_df.write.mode("overwrite").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                        
                        # Get actual row count from written table
                        written_count = session.sql(f"SELECT COUNT(*) as count FROM {wb_database}.{wb_schema}.{wb_table_name}").collect()[0]['COUNT']
                        
                        # Success message appears right after the button
                        st.success(f"Table {wb_database}.{wb_schema}.{wb_table_name} created successfully with {written_count:,} records", icon=":material/check:")
                        
                    except Exception as e:
                        st.error(f"Error writing to Snowflake: {str(e)}", icon=":material/chat_error:")
            

# Page config
st.set_page_config(page_title="Event2Vec Behavioral Segmentation", layout="wide")

# Custom CSS for consistent styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .custom-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .metric-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
    
    .segment-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .insight-box {
        background-color: rgba(41, 181, 232, 0.1);
        border-left: 4px solid #29B5E8;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .nlp-analogy {
        background: linear-gradient(135deg, rgba(41, 181, 232, 0.1), rgba(117, 205, 215, 0.05));
        border: 2px solid #29B5E8;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    /* Dark mode adaptations */
    @media (prefers-color-scheme: dark) {
        .custom-container {
            background-color: transparent;
            border: 1px solid #29B5E8 !important;
        }
        
        .segment-card {
            background-color: transparent;
            border: 1px solid #29B5E8 !important;
        }
        
        .metric-container {
            background-color: transparent;
            border: 1px solid #29B5E8;
        }
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for theme detection
st.markdown("""
<script>
function updateTheme() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const hasThemeAttr = document.querySelector('[data-theme="dark"]');
    
    if (isDark || hasThemeAttr) {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }
}

// Run immediately and on changes
updateTheme();
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addListener(updateTheme);
}

// Also run periodically to catch Streamlit theme changes
setInterval(updateTheme, 500);
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Override Streamlit theme variables for rounded corners */
:root {
    --baseRadius: 10px !important;
}

.stApp {
    --baseRadius: 10px !important;
}

/* Force rounded corners on all alert/message elements */
* {
    --baseRadius: 10px !important;
}

/* Subtle spacing adjustment for message elements */
div[data-testid="stAlert"],
div[data-testid="stInfo"], 
div[data-testid="stWarning"],
div[data-testid="stError"],
div[data-testid="stSuccess"] {
    margin-bottom: 0.5rem !important;
}

.custom-container-1 {
    background-color: #f0f2f6 !important;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

/* Custom styling for all message types - aggressive rounded corners and tight margins */
.stAlert,
.stAlert *,
.stAlert[data-baseweb="notification"],
.stAlert[data-baseweb="notification"] *,
div[data-testid="stAlert"],
div[data-testid="stAlert"] *,
div[data-testid="stInfo"],
div[data-testid="stInfo"] *,
div[data-testid="stWarning"],
div[data-testid="stWarning"] *,
div[data-testid="stError"],
div[data-testid="stError"] *,
div[data-testid="stSuccess"],
div[data-testid="stSuccess"] * {
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    -webkit-border-radius: 10px !important;
    -moz-border-radius: 10px !important;
}

.stAlert[data-baseweb="notification"] {
    background-color: #f0f2f6 !important;
    border: none !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

.stAlert[data-baseweb="notification"] > div {
    background-color: #f0f2f6 !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

/* Warning messages - orange/yellow tint */
.stAlert[data-baseweb="notification"][data-testid="stWarning"] {
    background-color: #fff3cd !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

.stAlert[data-baseweb="notification"][data-testid="stWarning"] > div {
    background-color: #fff3cd !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

/* Error messages - red tint */
.stAlert[data-baseweb="notification"][data-testid="stError"] {
    background-color: #f8d7da !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

.stAlert[data-baseweb="notification"][data-testid="stError"] > div {
    background-color: #f8d7da !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

/* Success messages - green tint */
.stAlert[data-baseweb="notification"][data-testid="stSuccess"] {
    background-color: #d1f2eb !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

.stAlert[data-baseweb="notification"][data-testid="stSuccess"] > div {
    background-color: #d1f2eb !important;
    --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
}

/* Remove default icons and customize */
.stAlert .stAlert-content::before {
    content: none !important;
}

/* Hide all default icons with multiple selectors */
.stAlert svg,
.stSuccess svg,
.stInfo svg,
div[data-testid="stAlert"] svg,
div[data-testid="stSuccess"] svg,
div[data-testid="stInfo"] svg,
.stAlert .stIcon,
.stSuccess .stIcon,
.stInfo .stIcon,
.stAlert::before,
.stSuccess::before,
.stInfo::before,
div[data-testid="stAlert"]::before,
div[data-testid="stSuccess"]::before,
div[data-testid="stInfo"]::before {
    display: none !important;
    visibility: hidden !important;
    content: "" !important;
}


@media (prefers-color-scheme: dark) {
    .custom-container-1 {
        background-color: transparent !important;
        border: 1px solid #4a4a4a !important;
    }
    
    .custom-container-1 h5 {
        color: #ffffff !important;
    }
    
    /* Custom styling for all message types in dark mode - Match Path Analysis */
    .stAlert[data-baseweb="notification"] {
        background-color: transparent !important;
        border: 1px solid #4a4a4a !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    .stAlert[data-baseweb="notification"] > div {
        background-color: transparent !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    /* Warning messages in dark mode */
    .stAlert[data-baseweb="notification"][data-testid="stWarning"] {
        background-color: transparent !important;
        border: 1px solid #ffc107 !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    .stAlert[data-baseweb="notification"][data-testid="stWarning"] > div {
        background-color: transparent !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    /* Error messages in dark mode */
    .stAlert[data-baseweb="notification"][data-testid="stError"] {
        background-color: transparent !important;
        border: 1px solid #dc3545 !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    .stAlert[data-baseweb="notification"][data-testid="stError"] > div {
        background-color: transparent !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    /* Success messages in dark mode */
    .stAlert[data-baseweb="notification"][data-testid="stSuccess"] {
        background-color: transparent !important;
        border: 1px solid #28a745 !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
    
    .stAlert[data-baseweb="notification"][data-testid="stSuccess"] > div {
        background-color: transparent !important;
        --baseRadius: 10px !important;
    baseRadius: 10px !important;
    border-radius: 10px !important;
    }
}

/* Fix for st.container(border=True) */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin: 8px 0 !important;
}

/* Dark mode support for bordered containers */
@media (prefers-color-scheme: dark) {
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1px solid #4a4a4a !important;
        background-color: transparent !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
        BEHAVIORAL SEGMENTATION
    </h5>
</div>
""", unsafe_allow_html=True)

with st.expander("Input Parameters",icon=":material/settings:"):
    # DATA SOURCE 
    st.markdown("""
<h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
<hr style='margin-top: -8px;margin-bottom: 5px;'>
""", unsafe_allow_html=True)

    # Database, Schema, Table Selection (cached)
    db0 = fetch_databases(session)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        database = st.selectbox('Select Database', key='behavdb', index=None, 
                                    placeholder="Choose from list...", options=db0['name'].unique())
        
    # **Schema Selection (Only if a database is selected)**
    if database:
        schema0 = fetch_schemas(session, database)
        
        with col2:
            schema = st.selectbox('Select Schema', key='behavsch', index=None, 
                                      placeholder="Choose from list...", options=schema0['name'].unique())
    else:
        schema = None  # Prevents SQL execution
        
    # **Table Selection (Only if a database & schema are selected)**
    if database and schema:
        table0 = fetch_tables(session, database, schema)
        
        with col3:
            table = st.selectbox('Select Table', key='behavtbl', index=None, 
                                     placeholder="Choose from list...", options=table0['TABLE_NAME'].unique())
    else:
        table = None  # Prevents SQL execution

    # **Column Selection (Only if database, schema & table are selected)**
    if database and schema and table:
        columns0 = fetch_columns(session, database, schema, table)
        
        st.markdown("""
    <h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Column Mapping</h2>
    <hr style='margin-top: -8px;margin-bottom: 5px;'>
    """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uid = st.selectbox('Select identifier column', key='behavuid', index=None,
                                   placeholder="Choose from list...", options=columns0['COLUMN_NAME'].unique(),
                                   help="Column containing customer identifiers")
        
        with col2:
            evt = st.selectbox('Select event column', key='behavevt', index=None,
                                   placeholder="Choose from list...", options=columns0['COLUMN_NAME'].unique(),
                                   help="Column containing event names")
        
        with col3:
            tmstp = st.selectbox('Select timestamp column', key='behavtmstp', index=None,
                                     placeholder="Choose from list...", options=columns0['COLUMN_NAME'].unique(),
                                     help="Column containing event timestamps")
    else:
        uid = None
        evt = None
        tmstp = None

    # Time Period & Analysis Parameters
    st.markdown("""
<h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Analysis Parameters</h2>
<hr style='margin-top: -8px;margin-bottom: 5px;'>
""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        period_type = st.selectbox("Time Period", ["Last N Days", "Date Range", "All Time"])
    
    with col2:
        if period_type == "Last N Days":
            days_back = st.number_input("Days Back", min_value=1, max_value=365, value=90)
            date_filter = f"AND {tmstp} >= CURRENT_DATE() - {days_back}" if tmstp else ""
        elif period_type == "Date Range":
            start_date = st.date_input("Start Date", value=datetime.date.today())
            date_filter = f"AND {tmstp} >= '{start_date}'" if tmstp else ""
        else:
            st.write("")  # Empty placeholder when "All Time" is selected
            date_filter = ""
    
    # Handle end date for date range in a separate row if needed
    if period_type == "Date Range":
        with col3:
            end_date = st.date_input("End Date", value=datetime.date.today())
            if 'start_date' in locals():
                date_filter = f"AND {tmstp} >= '{start_date}' AND {tmstp} <= '{end_date}'" if tmstp else ""
        with col4:
            st.write("")  # Empty placeholder
    else:
        with col3:
            min_sequence_length = st.number_input("Min Events per Customer", min_value=3, max_value=20, value=5,
                                                help="Minimum number of events required per customer")
        with col4:
            min_count = st.number_input("Min Event Frequency", min_value=1, max_value=100, value=2,
                                      help="Minimum frequency for events to be included in vocabulary")
    
    # If date range was selected, put the other parameters in a second row
    if period_type == "Date Range":
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.write("")  # Empty
        with col6:
            st.write("")  # Empty
        with col7:
            min_sequence_length = st.number_input("Min Events per Customer", min_value=3, max_value=20, value=5,
                                                help="Minimum number of events required per customer")
        with col8:
            min_count = st.number_input("Min Event Frequency", min_value=1, max_value=100, value=2,
                                      help="Minimum frequency for events to be included in vocabulary")
    
    # Additional row for Max Events per Customer (optional parameter)
    col_max1, col_max2, col_max3, col_max4 = st.columns(4)
    with col_max1:
        st.write("")  # Empty
    with col_max2:
        st.write("")  # Empty
    with col_max3:
        max_sequence_length = st.number_input("Max Events per Customer (Optional)", min_value=0, max_value=1000, value=0,
                                            help="Maximum number of most recent events to consider per customer. Set to 0 for no limit. Useful for focusing on recent behavior.")
    with col_max4:
        st.write("")  # Empty

    # Filters section
    if database and schema and table and uid and evt and tmstp:
        # Get all columns for filtering
        cols_query = f"""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{schema}' 
        AND TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION
        """
        colssql = session.sql(cols_query).collect()
        colsdf = pd.DataFrame(colssql)
        
        # Get events for exclusion
        events_query = f"SELECT DISTINCT {evt} FROM {database}.{schema}.{table} ORDER BY {evt}"
        events_result = session.sql(events_query).collect()
        events_df = pd.DataFrame(events_result)
        available_events = events_df[evt].tolist() if not events_df.empty else []
        
        # Get remaining columns (excluding uid, evt, tmstp)
        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
        
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([5, 2, 8])
        with col1:
            # Exclude Events
            excluded_events = st.multiselect('Exclude event(s) - optional', available_events, 
                                           placeholder="Select event(s)...", 
                                           help="Event(s) to be excluded from the analysis.")
        with col2:
            st.write("")  # Spacer
        with col3:
            st.write("")  # Spacer
        
        # Additional filters toggle
        if not remaining_columns.empty:
            col1, col2 = st.columns([5, 10])
            with col1:
                checkfilters = st.toggle("Additional filters", key="additional_filters_behavioral", help="Apply one or many conditional filters to the input data used in the behavioral segmentation.")
            with col2:
                st.write("")
        else:
            checkfilters = False
        
        # Initialize sql_where_clause outside the if block
        sql_where_clause = ""
        
        # Additional filters logic
        if checkfilters and not remaining_columns.empty:
            with st.container():
                # Helper to get cached distinct values as a Python list
                def get_distinct_values_list(column):
                    df_vals = fetch_distinct_values(session, database, schema, table, column)
                    return df_vals.iloc[:, 0].tolist() if not df_vals.empty else []
        
                # Helper function to display operator selection based on column data type
                def get_operator_input(col_name, col_data_type, filter_index):
                    """ Returns the operator for filtering based on column type """
                    operator_key = f"{col_name}_operator_{filter_index}"  # Ensure unique key
        
                    if col_data_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                        operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                    elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                        operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                    else:  # For string or categorical columns
                        operator = st.selectbox("Operator", ['=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                    return operator
        
                # Helper function to display value input based on column data type
                def get_value_input(col_name, col_data_type, operator, filter_index):
                    """ Returns the value for filtering based on column type """
                    value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
        
                    # Handle NULL operators - no value input needed
                    if operator in ['IS NULL', 'IS NOT NULL']:
                        return None
                    
                    # Handle IN and NOT IN operators
                    elif operator in ['IN', 'NOT IN']:
                        distinct_values = get_distinct_values_list(col_name)
                        value = st.multiselect(f"Values for {col_name}", distinct_values, key=value_key)
                        return value
                    
                    # Handle LIKE and NOT LIKE operators
                    elif operator in ['LIKE', 'NOT LIKE']:
                        value = st.text_input(f"Pattern for {col_name} (use % for wildcards)", key=value_key, placeholder="e.g., %text% or prefix%")
                        return value
                    
                    # Handle date/timestamp columns
                    elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                        value = st.date_input(f"Value for {col_name}", key=value_key)
                        return value
                    
                    # Handle numeric columns with accept_new_options for manual input
                    elif col_data_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                        distinct_values = get_distinct_values_list(col_name)
                        value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key, accept_new_options=True)
                        return value
                    
                    # Handle other data types (strings, etc.)
                    else:
                        distinct_values = get_distinct_values_list(col_name)
                        value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key)
                    return value
        
                # Initialize variables to store filters and logical conditions
                filters = []
                logic_operator = None
                filter_index = 0
        
                while True:
                    available_columns = remaining_columns  # Columns available for filtering
        
                    if available_columns.empty:
                        st.write("No more columns available for filtering.")
                        break  # Stop the loop if no columns remain
        
                    # Create 3 columns for column selection, operator, and value input
                    col1, col2, col3 = st.columns([2, 1, 2])
        
                    with col1:
                        selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns)
        
                    # Determine column data type (cached)
                    col_data_type = fetch_column_type(session, database, schema, table, selected_column)
        
                    with col2:
                        operator = get_operator_input(selected_column, col_data_type, filter_index)
        
                    with col3:
                        value = get_value_input(selected_column, col_data_type, operator, filter_index)
        
                    # Append filter if valid
                    if operator:
                        if operator in ['IS NULL', 'IS NOT NULL']:
                            filters.append((selected_column, operator, None))
                        elif operator in ['IN', 'NOT IN'] and value:
                            filters.append((selected_column, operator, value))
                        elif operator not in ['IS NULL', 'IS NOT NULL', 'IN', 'NOT IN'] and (value is not None and value != ''):
                            filters.append((selected_column, operator, value))
        
                    # Ask user if they want another filter
                    add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"add_filter_{filter_index}")
        
                    if add_filter == 'Yes':
                        col1, col2 = st.columns([2, 13])
                        with col1: 
                            logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"logic_operator_{filter_index}")
                            filter_index += 1
                        with col2:
                            st.write("")
                    else:
                        break
                
                # Generate SQL WHERE clause based on selected filters and logic
                if filters:
                    sql_where_clause = ""
                    
                    for i, (col, operator, value) in enumerate(filters):
                        if i > 0 and logic_operator:
                            sql_where_clause += f" {logic_operator} "
                        
                        # Handle NULL operators
                        if operator in ['IS NULL', 'IS NOT NULL']:
                            sql_where_clause += f"{col} {operator}"
                        
                        # Handle IN and NOT IN operators
                        elif operator in ['IN', 'NOT IN']:
                            if len(value) == 1:
                                # Single value - convert to = or != for better performance
                                single_op = '=' if operator == 'IN' else '!='
                                if isinstance(value[0], (int, float)):
                                    sql_where_clause += f"{col} {single_op} {value[0]}"
                                else:
                                    sql_where_clause += f"{col} {single_op} '{value[0]}'"
                            else:
                                # Multiple values - use proper IN/NOT IN with tuple
                                formatted_values = []
                                for v in value:
                                    if isinstance(v, (int, float)):
                                        formatted_values.append(str(v))
                                    else:
                                        formatted_values.append(f"'{v}'")
                                sql_where_clause += f"{col} {operator} ({', '.join(formatted_values)})"
                        
                        # Handle LIKE and NOT LIKE operators
                        elif operator in ['LIKE', 'NOT LIKE']:
                            sql_where_clause += f"{col} {operator} '{value}'"
                        
                        # Handle other operators (=, !=, <, <=, >, >=)
                        else:
                            if isinstance(value, (int, float)):
                                sql_where_clause += f"{col} {operator} {value}"
                            else:
                                # For non-numeric values (strings, dates), enclose the value in quotes
                                sql_where_clause += f"{col} {operator} '{value}'"        
                
                else:
                    # If no filters were created, ensure sql_where_clause is empty
                    sql_where_clause = ""
        
        # Add excluded events to where clause if any
        if excluded_events:
            excluded_events_str = ', '.join([f"'{event}'" for event in excluded_events])
            if sql_where_clause:
                sql_where_clause += f" AND {evt} NOT IN ({excluded_events_str})"
            else:
                sql_where_clause = f"{evt} NOT IN ({excluded_events_str})"
    else:
        excluded_events = []
        sql_where_clause = ""

    # Clustering Configuration
    st.markdown("""
<h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Clustering Configuration</h2>
<hr style='margin-top: -8px;margin-bottom: 5px;'>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        clustering_method = st.selectbox("Clustering Algorithm", 
                                       ["K-Means", "DBSCAN", "Gaussian Mixture", "Hierarchical", "Latent Dirichlet Allocation (LDA)"])

        # Customer limit options
        use_limit = st.checkbox("Limit number of customers", value=True, 
                               help="Uncheck to analyze all customers (may be slower for large datasets)")
        
        if use_limit:
            if clustering_method == "Hierarchical":
                default_max = min(5000, 50000)  # Limit for hierarchical
                max_limit = 10000
                help_text = "Limit for hierarchical clustering (max 10K due to memory constraints)"
            else:
                default_max = 50000
                max_limit = 1000000
                help_text = "Limit analysis to top N customers by activity"
            
            max_customers = st.number_input("Max Customers to Analyze", 
                                           min_value=1000, 
                                           max_value=max_limit, 
                                           value=default_max,
                                           help=help_text)
        else:
            max_customers = None  # No limit
    
    with col2:
        if clustering_method in ["K-Means", "Gaussian Mixture", "Hierarchical"]:
            n_clusters = st.slider("Number of Clusters", min_value=3, max_value=20, value=6)
        elif clustering_method == "Latent Dirichlet Allocation (LDA)":
            n_topics = st.slider("Number of Topics", min_value=3, max_value=20, value=8, 
                                help="Number of latent behavioral topics to discover")
            alpha = st.slider("Alpha (Topic Concentration)", min_value=0.01, max_value=2.0, value=0.1, step=0.01,
                             help="Lower values = fewer topics per customer (more focused behavior)")
            beta = st.slider("Beta (Word Concentration)", min_value=0.01, max_value=2.0, value=0.01, step=0.01,
                            help="Lower values = fewer events per topic (more specific topics)")
            n_clusters = n_topics  # For consistency with other algorithms
        else:
            n_clusters = 6  # Default value for other methods
        
        if clustering_method == "DBSCAN":
            eps = st.slider("DBSCAN Epsilon", min_value=0.1, max_value=5.0, value=1.5, step=0.1, 
                           help="Distance threshold - larger values create fewer, larger clusters")
            min_samples = st.slider("DBSCAN Min Samples", min_value=5, max_value=50, value=15,
                                   help="Minimum customers per cluster - higher values reduce noise")
        else:
            eps = 1.5  # Default values for other methods (more conservative)
            min_samples = 15
    
    # with col3: (left empty for alignment with data source section layout)

    # Advanced Options
    st.markdown("""
<h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Advanced Options</h2>
<hr style='margin-top: -8px;margin-bottom: 10px;'>
""", unsafe_allow_html=True)
    
    # Organize into logical groups with better spacing
    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 5px;'>Feature Engineering</h3>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Set intelligent default and contextual help based on clustering method
        if clustering_method == "Latent Dirichlet Allocation (LDA)":
            default_weighting = 1  # TF-IDF works best with LDA
            help_text = """Choose how to weight event importance. 
            
**Recommended for LDA**: TF-IDF (emphasizes distinctive events, standard for topic modeling)
**Alternative**: Raw Frequency (simple counts), Binary Presence (event occurrence)
**Avoid**: Smart Filtering (may over-filter for topic discovery)"""
        elif clustering_method == "K-Means":
            default_weighting = 4  # Smart Filtering 
            help_text = """Choose how to weight event importance.
            
**Recommended for K-Means**: Smart Filtering (removes noise, focuses on discriminative events)
**Alternative**: TF-IDF (good for distinctive patterns), Log Frequency (balanced weighting)
**Consider**: Binary Presence for sparse data, Raw Frequency for frequency-based patterns"""
        elif clustering_method == "DBSCAN":
            default_weighting = 1  # TF-IDF for density-based clustering
            help_text = """Choose how to weight event importance.
            
**Recommended for DBSCAN**: TF-IDF (highlights unique patterns for density detection)
**Alternative**: Smart Filtering (reduces noise), Binary Presence (focus on event combinations)
**Avoid**: Raw Frequency (may create artificial density due to high frequencies)"""
        elif clustering_method == "Gaussian Mixture":
            default_weighting = 4  # Smart Filtering
            help_text = """Choose how to weight event importance.
            
**Recommended for Gaussian Mixture**: Smart Filtering (clean features for probabilistic modeling)
**Alternative**: TF-IDF (distinctive features), Log Frequency (balanced distributions)
**Consider**: Raw Frequency if you want frequency-sensitive clusters"""
        elif clustering_method == "Hierarchical":
            default_weighting = 2  # Log Frequency for hierarchical
            help_text = """Choose how to weight event importance.
            
**Recommended for Hierarchical**: Log Frequency (balanced weighting for tree construction)
**Alternative**: Smart Filtering (clean hierarchy), TF-IDF (distinctive-based hierarchy)
**Consider**: Binary Presence for event co-occurrence patterns, Raw Frequency for frequency-based trees"""
        else:
            default_weighting = 4  # Smart Filtering default
            help_text = "Choose how to weight event importance in customer vectors"
            
        feature_weighting = st.selectbox("Feature Weighting Strategy", 
                                          ["Raw Frequency", "TF-IDF", "Log Frequency", "Binary Presence", "Smart Filtering"],
                                          index=default_weighting,
                                          help=help_text)
    
    with col2:
        if clustering_method == "Latent Dirichlet Allocation (LDA)":
            st.info("Vector Dimensions not used by LDA - LDA operates on discrete event counts", icon=":material/info:")
            vector_size = 100  # Default value, not used by LDA
        else:
            vector_size = st.slider("Vector Dimensions", min_value=50, max_value=300, value=100, 
                                   help="""Size of event embeddings created by Word2Vec:
                                   
• **50-100**: Faster, good for simple patterns
• **100-200**: Balanced performance and quality  
• **200-300**: Captures complex relationships, slower

Higher dimensions can capture more nuanced event relationships but require more data and computation.""")
    
    # with col3: (left empty for alignment with other sections)
    
    # Show method-specific recommendations in a compact format
    tip_message = None
    if clustering_method == "Latent Dirichlet Allocation (LDA)" and feature_weighting != "TF-IDF":
        tip_message = "TF-IDF typically works best with LDA as it emphasizes distinctive events"
    elif clustering_method == "K-Means" and feature_weighting not in ["Smart Filtering", "TF-IDF"]:
        tip_message = "Smart Filtering or TF-IDF typically work best with K-Means for cleaner clusters"
    elif clustering_method == "DBSCAN" and feature_weighting not in ["TF-IDF", "Smart Filtering"]:
        tip_message = "TF-IDF or Smart Filtering typically work best with DBSCAN for density detection"
    elif clustering_method == "Gaussian Mixture" and feature_weighting not in ["Smart Filtering", "TF-IDF"]:
        tip_message = "Smart Filtering or TF-IDF typically work best with Gaussian Mixture for probabilistic modeling"
    elif clustering_method == "Hierarchical" and feature_weighting not in ["Log Frequency", "Smart Filtering"]:
        tip_message = "Log Frequency or Smart Filtering typically work best with Hierarchical clustering"
    
    if tip_message:
        with st.expander("💡 Optimization Tip", expanded=False, icon=":material/lightbulb:"):
            st.markdown(tip_message)
    
    st.markdown("")  # Add some spacing
    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 5px;'>Data Processing</h3>""", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if clustering_method == "Latent Dirichlet Allocation (LDA)":
            st.info("Vector Normalization not used by LDA - LDA uses raw event counts", icon=":material/info:")
            normalize_vectors = True  # Default value, not used by LDA
        else:
            normalize_vectors = st.checkbox("Normalize Customer Vectors", value=True,
                                          help="""Scale all customer vectors to unit length:
                                          
• **Enabled**: Focus on behavioral patterns (recommended)
• **Disabled**: Consider absolute event volumes

Normalization ensures customers with different activity levels are compared fairly based on behavioral patterns rather than total volume.""")
    with col4:
        if clustering_method == "Latent Dirichlet Allocation (LDA)":
            st.info("Context Window not used by LDA - LDA analyzes event frequency patterns", icon=":material/info:")
            window_size = 5  # Default value, not used by LDA
        else:
            window_size = st.slider("Context Window", min_value=2, max_value=10, value=5,
                                   help="""Number of surrounding events Word2Vec considers when learning relationships:
                                   
• **2-3**: Captures immediate sequential patterns
• **4-6**: Balanced local and broader context (recommended)
• **7-10**: Captures long-term journey patterns

Larger windows help identify relationships across longer customer journeys but may dilute immediate sequential associations.""")
    with col5:
        # PCA selection with clear options - hide for LDA
        if clustering_method == "Latent Dirichlet Allocation (LDA)":
            st.info("PCA not applicable to LDA - LDA uses discrete token counts", icon=":material/info:")
            pca_option = "No PCA"
            pca_components = 0
        else:
            pca_option = st.radio(
                "PCA Dimensionality Reduction",
                options=["No PCA", "Auto (Recommended)", "Manual"],
                index=1,  # Default to Auto
                help="""Principal Component Analysis (PCA) reduces data complexity while preserving important patterns:

**No PCA**: Uses all original event features
• Pros: Preserves all information, no data loss
• Cons: High dimensionality may cause noise, slower processing
• Best for: Small datasets, when you need all features

**Auto (Recommended)**: Automatically selects optimal components
• K-Means/Gaussian: ~90% variance (balanced performance)
• DBSCAN/Hierarchical: ~95% variance (preserves local structure)
• Analyzes your data to find the best number of components
• Pros: Optimal balance of information vs. efficiency
• Best for: Most use cases, when unsure

**Manual**: You choose the exact number of components
• Full control over dimensionality reduction
• Slider lets you select 1-50 components
• Higher components = more information, more complexity
• Lower components = faster processing, potential information loss
• Best for: Experimentation, specific performance requirements

PCA transforms event features into uncorrelated components ordered by importance, helping clustering algorithms focus on the most meaningful patterns."""
            )
            
            if pca_option == "Manual":
                pca_components = st.slider("PCA Components", min_value=1, max_value=50, value=20,
                                         help="Number of principal components to retain")
            else:
                pca_components = 0  # Will be set later based on option


# Validation message
if not all([database, schema, table, uid, evt, tmstp]):
    st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 14px; font-weight: 200 ; margin-top: 0px; margin-bottom: -15px;">
            Please ensure all required inputs are selected before running the app.
        </h5>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Button-based approach - explicit control over clustering execution
run_button = st.button("Run Clustering Analysis", 
                      help="Click to run behavioral segmentation analysis and find customer segments based on journey patterns")

# Check if parameters have changed since last run (for use in tabs later)
params_changed = False
if all([database, schema, table, uid, evt, tmstp]):
    params_changed, _ = parameters_changed()
has_cached_results = 'behavioral_results' in st.session_state and st.session_state['behavioral_results'].get('completed')

# Don't show warning at top level - it will be shown in the tabs if needed

# Execute clustering when button is clicked
if run_button and all([database, schema, table, uid, evt, tmstp]):
    # Clear old results
    if 'behavioral_results' in st.session_state:
        del st.session_state['behavioral_results']
    
    with st.spinner("Running behavioral segmentation analysis..."):
        tab1, tab2 = st.tabs(["Analysis Process", "Results"])
        
        with tab1:
            # Run all analysis steps here
            
            # Step 1: Create Event Sequences
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 1: Creating Customer Event Sequences</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            try:
                # Build date filter condition
                sql_date_filter = ""
                if period_type == "Last N Days":
                    sql_date_filter = f"AND {tmstp} >= CURRENT_DATE() - {days_back}"
                elif period_type == "Date Range":
                    sql_date_filter = f"AND {tmstp} >= '{start_date}' AND {tmstp} <= '{end_date}'"
                
                # Build complete WHERE clause with filters
                where_conditions = ["1=1"]
                if sql_date_filter:
                    where_conditions.append(sql_date_filter.replace("AND ", ""))
                if sql_where_clause:
                    where_conditions.append(sql_where_clause)
                
                complete_where_clause = " AND ".join(where_conditions)
                
                # Generate clean, optimized SQL query
                limit_clause = f"LIMIT {max_customers}" if max_customers is not None else ""
                
                # Build HAVING clause with min and max sequence length
                if max_sequence_length and max_sequence_length > 0:
                    having_clause = f"HAVING COUNT(*) BETWEEN {min_sequence_length} AND {max_sequence_length}"
                else:
                    having_clause = f"HAVING COUNT(*) >= {min_sequence_length}"
                
                sequences_query = f"""
                SELECT 
                    {uid} AS CUSTOMER_ID,
                    LISTAGG({evt}, ' → ') WITHIN GROUP (ORDER BY {tmstp}) AS EVENT_SEQUENCE,
                    COUNT(*) AS SEQUENCE_LENGTH,
                    COUNT(DISTINCT {evt}) AS UNIQUE_EVENTS,
                    MIN({tmstp}) AS FIRST_EVENT,
                    MAX({tmstp}) AS LAST_EVENT
                FROM {database}.{schema}.{table}
                WHERE {complete_where_clause}
                GROUP BY {uid}
                {having_clause}
                ORDER BY SEQUENCE_LENGTH DESC
                {limit_clause}
                """
                
                # Execute query and get results
                sequences_df = session.sql(sequences_query).to_pandas()
                
                # Validate we have enough customers for clustering
                total_customers = len(sequences_df)
                
                # Check minimum customers requirement based on clustering method
                min_required = n_topics if clustering_method == "Latent Dirichlet Allocation (LDA)" else n_clusters
                method_name = "topics" if clustering_method == "Latent Dirichlet Allocation (LDA)" else "clusters"
                
                if total_customers < min_required:
                    st.error(f"Insufficient data for {clustering_method}", icon=":material/chat_error:")
                    st.warning(f"Found only {total_customers} customers, but need at least {min_required} customers for {min_required} {method_name}.", icon=":material/warning:")
                    
                    with st.expander("Suggestions to fix this issue", icon=":material/lightbulb:"):
                        param_name = "Number of Topics" if clustering_method == "Latent Dirichlet Allocation (LDA)" else "Number of Clusters"
                        st.markdown(f"""
                        **Option 1: Reduce number of {method_name}**
                        - Change "{param_name}" to **{max(2, total_customers - 1)}** or less
                        
                        **Option 2: Relax data filters**
                        - Reduce "Min Events per Customer" (currently {min_sequence_length})
                        - Reduce "Min Event Frequency" (currently {min_count})
                        - Remove event exclusions or additional filters
                        - Expand the time period
                        
                        **Option 3: Use all customers**
                        - Uncheck "Limit number of customers" to analyze the full dataset
                        
                        **Current settings:**
                        - Time period: {period_type}
                        - Min events per customer: {min_sequence_length}
                        - Min event frequency: {min_count}
                        - Customer limit: {max_customers if max_customers else "No limit"}
                        - Excluded events: {len(excluded_events) if 'excluded_events' in locals() else 0}
                        """)
                    st.stop()
                
            except Exception as e:
                st.error(f"Error creating sequences: {str(e)}", icon=":material/chat_error:")
                st.stop()
            
            # Handle potential column naming issues
            try:
                # Try to access columns with expected names
                seq_length_col = 'SEQUENCE_LENGTH' if 'SEQUENCE_LENGTH' in sequences_df.columns else sequences_df.columns[2]
                unique_events_col = 'UNIQUE_EVENTS' if 'UNIQUE_EVENTS' in sequences_df.columns else sequences_df.columns[3]
                event_seq_col = 'EVENT_SEQUENCE' if 'EVENT_SEQUENCE' in sequences_df.columns else sequences_df.columns[1]
                customer_id_col = 'CUSTOMER_ID' if 'CUSTOMER_ID' in sequences_df.columns else sequences_df.columns[0]
                
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)
                
                    col1.metric("Total Customers", f"{len(sequences_df):,}")
                    col2.metric("Avg Sequence Length", f"{sequences_df[seq_length_col].mean():.1f}")
                    col3.metric("Avg Unique Events", f"{sequences_df[unique_events_col].mean():.1f}")
                
                # Count total unique event types
                all_events_sample = ' '.join(sequences_df[event_seq_col].head(100))
                unique_event_types = len(set(all_events_sample.split(' → ')))
                col4.metric("Event Types (Sample)", f"{unique_event_types}", help="Number of distinct event types found in the first 100 customer sequences (sample)")
                
            except Exception as e:
                st.error(f"Error accessing sequence data: {str(e)}", icon=":material/chat_error:")
                st.write("**DataFrame shape:**", sequences_df.shape)
                st.write("**DataFrame columns:**", list(sequences_df.columns))
                st.stop()
            
            # Step 2: Event Vocabulary Analysis
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 2: Event Vocabulary Analysis</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            
            # Use server-side vocabulary analysis when possible
            try:
                # Build proper date filter for SQL
                if period_type == "Last N Days":
                    sql_date_filter = f"AND {tmstp} >= DATEADD(day, -{days_back}, CURRENT_DATE())"
                elif period_type == "Date Range":
                    sql_date_filter = f"AND {tmstp} >= '{start_date}' AND {tmstp} <= '{end_date}'"
                else:
                    sql_date_filter = ""
                
                # Server-side event frequency analysis
                vocab_query = f"""
                WITH event_frequencies AS (
                    SELECT 
                        {evt} as event_name,
                        COUNT(*) as frequency,
                        COUNT(DISTINCT {uid}) as unique_customers
                    FROM {database}.{schema}.{table}
                    WHERE 1=1 {sql_date_filter}
                    GROUP BY {evt}
                ORDER BY frequency DESC
            )
            SELECT * FROM event_frequencies
                """
                
                event_counts_df = session.sql(vocab_query).to_pandas()
                event_counts = pd.Series(event_counts_df['FREQUENCY'].values, index=event_counts_df['EVENT_NAME'])
                event_vocab = event_counts[event_counts >= min_count].index.tolist()
                vocab_size = len(event_counts)
                
            except Exception as e:
                # Fallback to client-side processing
                all_events = []
                for sequence in sequences_df['EVENT_SEQUENCE']:
                    all_events.extend(sequence.split(' → '))
                
                event_counts = pd.Series(all_events).value_counts()
                event_vocab = event_counts[event_counts >= min_count].index.tolist()
                vocab_size = len(event_counts)
            
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
            
                col1.metric("Vocabulary Size", f"{vocab_size:,}", help="Total number of unique event types found in the dataset")
                col2.metric("Total Event Instances", f"{event_counts.sum():,}", help="Total count of all individual events across all customers")
                col3.metric("Events Above Min Count", f"{len(event_counts[event_counts >= min_count]):,}", help="Number of event types that occur frequently enough to be included in analysis")
                
            # Show event frequency distribution
            fig_vocab = px.bar(
                x=event_counts.head(20).index,
                y=event_counts.head(20).values,
                labels={'x': 'Event Type', 'y': 'Frequency'},
                color_discrete_sequence=['#29B5E8']
            )
            fig_vocab.update_layout(
                height=400, 
                xaxis_tickangle=45,
                title={
                    'text': "Top 20 Most Frequent Events",
                    'font': {'size': 14},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            with st.container(border=True):
                st.plotly_chart(fig_vocab, use_container_width=True)
            
            # Step 3: Feature Engineering - conditional title based on clustering method
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 3: LDA Document Preparation</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                
                # LDA-specific feature engineering (no Event2Vec needed)
                st.info(f"Preparing TF-IDF document-term matrix for {len(sequences_df):,} customers using {feature_weighting} weighting", icon=":material/description:")
                
                try:
                    # LDA-specific preprocessing (no feature vectors needed!)
                    st.info(f"Preparing documents for LDA topic modeling...", icon=":material/info:")
                    
                    # Convert event sequences to documents (simple text format)
                    documents = []
                    customer_ids = []
                    
                    for _, row in sequences_df.iterrows():
                        customer_id = row[customer_id_col]
                        # Convert "A → B → C" to "A B C" for LDA
                        document = row[event_seq_col].replace(' → ', ' ')
                        
                        # Filter out very short sequences
                        if len(document.split()) >= 3:
                            documents.append(document)
                            customer_ids.append(customer_id)
                    
                    st.success(f"Prepared {len(documents):,} documents for LDA topic modeling", icon=":material/check:")
                    
                    # Set X to None to indicate we're using LDA approach
                    X = None
                    customer_ids = np.array(customer_ids)
                    
                    # Store documents in session state for LDA clustering
                    st.session_state['lda_documents'] = documents
                    st.session_state['lda_customer_ids'] = customer_ids
                    
                except Exception as e:
                    st.error(f"LDA preprocessing failed: {str(e)}", icon=":material/chat_error:")
                    st.error(f"Error type: {type(e).__name__}", icon=":material/bug_report:")
                    import traceback
                    st.error(f"Full traceback: {traceback.format_exc()}", icon=":material/code:")
                    st.info("Falling back to Event2Vec approach...", icon=":material/info:")
                    # Set variables to None so Event2Vec path will run
                    X = None
                    customer_ids = None
            else:
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 3: Training Event2Vec Embeddings</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                
                # Real Event2Vec Implementation (only for non-LDA methods)
                # Prepare sentences for Word2Vec (list of lists)
                sentences = [row[event_seq_col].split(' → ') for _, row in sequences_df.iterrows()]
                
                try:
                    # Train actual Word2Vec model
                    from gensim.models import Word2Vec
                    
                    # Train Event2Vec model
                    model = Word2Vec(
                        sentences=sentences,
                        vector_size=vector_size,
                        window=window_size,
                        min_count=min_count,
                        workers=4,
                        sg=1,  # Skip-gram model
                        epochs=20
                    )
                    
                    # Display training results in 2 columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"Trained Event2Vec model with {len(model.wv.key_to_index)} event embeddings", icon=":material/check:")
                    
                    # Generate customer vectors by averaging event embeddings
                    customer_vectors = []
                    customer_ids = []
                    
                    for _, row in sequences_df.iterrows():
                        customer_id = row[customer_id_col]
                        events = row[event_seq_col].split(' → ')
                        
                        # Get vectors for events that exist in vocabulary
                        event_vectors = []
                        for event in events:
                            if event in model.wv:
                                event_vectors.append(model.wv[event])
                        
                        if len(event_vectors) > 0:
                            # Average the event vectors to get customer vector
                            customer_vector = np.mean(event_vectors, axis=0)
                            customer_vectors.append(customer_vector)
                            customer_ids.append(customer_id)
                    
                    # Convert to arrays
                    X = np.array(customer_vectors)
                    customer_ids = np.array(customer_ids)
                    
                    with col2:
                        st.info(f"Generated {len(X)} customer vectors of dimension {X.shape[1]}", icon=":material/info:")
                    
                except ImportError:
                    st.warning("gensim not available. Using advanced feature engineering instead.", icon=":material/warning:")
                    
                    # Fallback to advanced feature engineering
                    customer_features = []
                    customer_ids = []
                    
                    # Calculate document frequencies for all events (needed for TF-IDF and Smart Filtering)
                    st.info(f"Event2Vec Fallback: Processing {len(sequences_df):,} customers for document frequency calculation...", icon=":material/info:")
                    customer_event_sets = []
                    for i, (_, row) in enumerate(sequences_df.iterrows()):
                        events = set(row[event_seq_col].split(' → '))
                        customer_event_sets.append(events)
                    st.info(f"Event2Vec Fallback: Document frequency calculation completed for {len(customer_event_sets):,} customers", icon=":material/check:")
                    
                    # Calculate how many customers have each event (document frequency)
                    event_document_frequency = {}
                    for event in event_vocab:
                        doc_freq = sum(1 for event_set in customer_event_sets if event in event_set)
                        event_document_frequency[event] = doc_freq
                    
                    # Apply smart filtering if selected
                    if feature_weighting == "Smart Filtering":
                        total_customers = len(sequences_df)
                        
                        # Apply smart filtering thresholds
                        max_df_threshold = 0.7  # Remove events in >70% of customers
                        min_df_threshold = max(3, int(total_customers * 0.02))  # At least 3 customers or 2%
                        
                        # Filter event vocabulary
                        filtered_vocab = []
                        excluded_common = []
                        excluded_rare = []
                        
                        for event in event_vocab:
                            doc_freq = event_document_frequency[event]
                            doc_pct = doc_freq / total_customers
                            
                            if doc_pct > max_df_threshold:
                                excluded_common.append((event, doc_freq, doc_pct))
                            elif doc_freq < min_df_threshold:
                                excluded_rare.append((event, doc_freq, doc_pct))
                            else:
                                filtered_vocab.append(event)
                        
                        st.info(f"Smart Filtering Applied: Using {len(filtered_vocab)} events (excluded {len(excluded_common)} common + {len(excluded_rare)} rare)",icon=":material/info:")
                        
                        # Show filtering results
                        with st.expander("Smart Filtering Results"):
                            st.markdown("**Events included in clustering:**")
                            for event in filtered_vocab[:10]:  # Show first 10
                                doc_freq = event_document_frequency[event]
                                doc_pct = (doc_freq / total_customers) * 100
                                st.markdown(f"• **{event}**: {doc_freq} customers ({doc_pct:.1f}%)")
                            if len(filtered_vocab) > 10:
                                st.markdown(f"... and {len(filtered_vocab) - 10} more events")
                            
                            if excluded_common:
                                st.markdown("**Excluded (too common):**")
                                for event, freq, pct in excluded_common[:5]:
                                    st.markdown(f"• ~~{event}~~: {freq} customers ({pct*100:.1f}%)")
                            
                            if excluded_rare:
                                st.markdown(f"**Excluded (too rare)**: {len(excluded_rare)} events appearing in <{min_df_threshold} customers")
                        
                        # Use filtered vocabulary
                        working_vocab = filtered_vocab
                    else:
                        # Use full vocabulary
                        working_vocab = event_vocab
                    
                    st.info(f"Event2Vec Fallback: Creating feature vectors for {len(sequences_df):,} customers...", icon=":material/info:")
                    for i, (_, row) in enumerate(sequences_df.iterrows()):
                        customer_id = row[customer_id_col]
                        events = row[event_seq_col].split(' → ')
                        
                        # Filter events by working vocabulary (filtered or full)
                        filtered_events = [e for e in events if e in working_vocab]
                        
                        if len(filtered_events) < 3:
                            continue
                        
                        features = []
                        
                        # Event frequency features with different weighting strategies
                        event_freq = pd.Series(filtered_events).value_counts()
                        total_events = len(filtered_events)
                        
                        for event in working_vocab:
                            raw_count = event_freq.get(event, 0)
                            tf = raw_count / total_events
                            
                            if feature_weighting == "Raw Frequency":
                                weight = tf
                            elif feature_weighting == "TF-IDF":
                                # Proper TF-IDF: use pre-calculated document frequency
                                customers_with_event = event_document_frequency.get(event, 0)
                                total_docs = len(sequences_df)
                                
                                # Calculate IDF: log(total_docs / (1 + doc_freq))
                                idf = np.log(total_docs / (1 + customers_with_event))
                                weight = tf * idf
                            elif feature_weighting == "Log Frequency":
                                # Log(1 + frequency) reduces dominance of very frequent events
                                weight = np.log(1 + raw_count) / np.log(1 + total_events)
                            elif feature_weighting == "Binary Presence":
                                # Binary: 1 if event occurs, 0 otherwise
                                weight = 1.0 if raw_count > 0 else 0.0
                            elif feature_weighting == "Smart Filtering":
                                # Use log frequency for smart filtered events
                                weight = np.log(1 + raw_count) / np.log(1 + total_events) if total_events > 0 else 0
                            else:
                                weight = tf  # Default to raw frequency
                            
                            features.append(weight)
                        
                        # Add behavioral features
                        features.extend([
                            len(filtered_events) / 100,  # Normalized sequence length
                            len(set(filtered_events)) / len(working_vocab) if len(working_vocab) > 0 else 0,  # Event diversity
                            len(set(filtered_events)) / len(filtered_events) if len(filtered_events) > 0 else 0  # Uniqueness ratio
                        ])
                        
                        customer_features.append(features)
                        customer_ids.append(customer_id)
                    
                    X = np.array(customer_features)
                    customer_ids = np.array(customer_ids)
                    
                    st.info(f"Generated {len(X)} customer feature vectors of dimension {X.shape[1]}")
                
            
            
            # Step 4: Clustering Analysis
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 4: LDA Topic Modeling</h2>
                        <hr style='margin-top: -8px;margin-bottom: 10px;'>
                        """, unsafe_allow_html=True)
                
            else:
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Step 4: Advanced Clustering on Event Embeddings</h2>
                        <hr style='margin-top: -8px;margin-bottom: 10px;'>
                        """, unsafe_allow_html=True)
            # Data preprocessing summary (skip for LDA)
            preprocessing_steps = []
            
            # Skip preprocessing for LDA (uses documents, not feature vectors)
            if clustering_method != "Latent Dirichlet Allocation (LDA)" and X is not None:
                # Check for data quality issues
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                    preprocessing_steps.append("data cleaning")
            
            # Normalize if requested (skip for LDA)
            if clustering_method != "Latent Dirichlet Allocation (LDA)" and X is not None and normalize_vectors:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                preprocessing_steps.append("normalization")
                
            # Apply PCA if requested (skip for LDA)
            original_features = X.shape[1] if X is not None else 0
            pca_applied = False
            pca_info_text = ""
            
            if clustering_method != "Latent Dirichlet Allocation (LDA)" and X is not None:
                if pca_option == "Auto (Recommended)":
                    # Get optimal PCA recommendation
                    pca_info = get_optimal_pca_components(X, clustering_method)
                    recommended_components = pca_info["recommended"]
                    
                    if recommended_components > 0 and recommended_components < original_features:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=recommended_components, random_state=42)
                        X = pca.fit_transform(X)
                        pca_applied = True
                        preprocessing_steps.append("auto PCA")
                        
                        variance_explained = pca_info["variance_explained"]
                        pca_info_text = f"PCA: {original_features} → {recommended_components} dimensions ({variance_explained:.1%} variance)"
                    else:
                        pca_info_text = f"PCA: {pca_info['reason']}"
                        
                elif pca_option == "Manual" and pca_components > 0 and pca_components < original_features:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=pca_components, random_state=42)
                    X = pca.fit_transform(X)
                    pca_applied = True
                    preprocessing_steps.append("manual PCA")
            else:
                pca_info_text = "PCA: Not applicable to LDA"
            
            if pca_option == "No PCA" and clustering_method != "Latent Dirichlet Allocation (LDA)":
                pca_info_text = f"Features: {original_features} original dimensions"
            
            elif pca_option == "Manual" and pca_components >= original_features:
                pca_info_text = f"PCA: using all {original_features} features (manual components >= original features)"
            
            # Display consolidated preprocessing summary
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                st.info(f"Running {clustering_method} topic modeling on {len(sequences_df):,} customers • {feature_weighting} weighting • Document-based approach", icon=":material/settings:")
                try:
                        from sklearn.decomposition import LatentDirichletAllocation
                        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                        
                        # Use the documents prepared in the LDA preprocessing step
                        if 'lda_documents' in st.session_state and st.session_state['lda_documents'] is not None:
                            event_documents = st.session_state['lda_documents']
                            valid_customer_ids = st.session_state['lda_customer_ids']
                            st.info(f"Using prepared LDA documents: {len(event_documents):,} documents", icon=":material/check:")
                        else:
                            # Fallback: create documents from sequences_df
                            event_documents = []
                            valid_customer_ids = []
                            
                            for _, row in sequences_df.iterrows():
                                customer_id = row[customer_id_col]
                                # Convert "A → B → C" to "A B C" for LDA
                                document = row[event_seq_col].replace(' → ', ' ')
                                
                                # Filter out very short sequences
                                if len(document.split()) >= 3:
                                    event_documents.append(document)
                                    valid_customer_ids.append(customer_id)
                        
                        if len(event_documents) < 10:
                            st.error("Insufficient data for LDA. Need at least 10 customers with 3+ events.", icon=":material/chat_error:")
                            st.info("Falling back to K-Means clustering...", icon=":material/info:")
                            from sklearn.cluster import KMeans
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = clusterer.fit_predict(X)
                            customer_ids = np.array(valid_customer_ids) if valid_customer_ids else customer_ids
                        else:
                            if feature_weighting == "TF-IDF":
                                # Proper TF-IDF: Events as terms, Customers as documents
                                # Create proper document-term matrix: customers × events
                                from collections import Counter
                                
                                # Build vocabulary from filtered events
                                vocabulary = list(event_vocab)
                                vocab_to_idx = {event: idx for idx, event in enumerate(vocabulary)}
                                
                                # Create document-term matrix (customers × events)
                                doc_term_matrix = np.zeros((len(event_documents), len(vocabulary)))
                                
                                for doc_idx, doc in enumerate(event_documents):
                                    events = doc.split()
                                    event_counts = Counter(events)
                                    
                                    for event, count in event_counts.items():
                                        if event in vocab_to_idx:
                                            doc_term_matrix[doc_idx, vocab_to_idx[event]] = count
                                
                                # Calculate TF-IDF manually for better control
                                # TF: term frequency within each document
                                tf_matrix = doc_term_matrix.copy()
                                doc_lengths = np.sum(tf_matrix, axis=1, keepdims=True)
                                doc_lengths[doc_lengths == 0] = 1  # Avoid division by zero
                                tf_matrix = tf_matrix / doc_lengths
                                
                                # IDF: inverse document frequency
                                document_frequency = np.sum(doc_term_matrix > 0, axis=0)  # How many docs contain each term
                                document_frequency[document_frequency == 0] = 1  # Avoid division by zero
                                idf_vector = np.log(len(event_documents) / document_frequency)
                                
                                # TF-IDF = TF × IDF
                                tfidf_matrix = tf_matrix * idf_vector
                                
                                # IMPORTANT: LDA needs raw counts, not TF-IDF weights!
                                # Use TF-IDF for feature selection, but feed raw counts to LDA
                                avg_tfidf = np.mean(tfidf_matrix, axis=0)
                                
                                # Select top features based on TF-IDF scores
                                n_top_features = min(50, len(vocabulary))  # Keep top 50 features
                                top_features_idx = np.argsort(avg_tfidf)[-n_top_features:]
                                
                                # Create filtered vocabulary and get raw counts for selected features
                                selected_vocabulary = [vocabulary[i] for i in top_features_idx]
                                
                                # Rebuild raw count matrix for selected features only
                                raw_count_matrix = np.zeros((len(event_documents), len(selected_vocabulary)))
                                selected_vocab_to_idx = {event: idx for idx, event in enumerate(selected_vocabulary)}
                                
                                for doc_idx, doc in enumerate(event_documents):
                                    events = doc.split()
                                    event_counts = Counter(events)
                                    for event, count in event_counts.items():
                                        if event in selected_vocab_to_idx:
                                            raw_count_matrix[doc_idx, selected_vocab_to_idx[event]] = count
                                
                                # Use raw counts for LDA
                                doc_term_matrix = raw_count_matrix
                                feature_names = np.array(selected_vocabulary)
                                
                                # Store TF-IDF matrix for statistics display (only selected features)
                                tfidf_matrix_for_stats = tfidf_matrix[:, top_features_idx]
                                
                                # Critical validation: Check if we have enough features for the requested number of topics
                                n_features = doc_term_matrix.shape[1]
                                if n_features < n_topics:
                                    st.error(f"**Insufficient features for LDA**: Found only {n_features} events in TF-IDF matrix, but trying to create {n_topics} topics", icon=":material/chat_error:")
                                    st.warning(f"**Rule**: You need at least as many unique events as topics. Currently: {n_features} events < {n_topics} topics", icon=":material/warning:")
                                    
                                    with st.expander("Solutions", expanded=True,icon=":material/troubleshoot:"):
                                        st.markdown(f"""
                                        **Choose one of these options:**
                                        
                                        **Option 1: Reduce Number of Topics**
                                        - Change "Number of Topics" to **{max(2, n_features)}** or less
                                        
                                        **Option 2: Relax Data Filters**
                                        - Reduce "Min Event Frequency" (currently {min_count})
                                        - Reduce "Min Events per Customer" (currently {min_sequence_length})
                                        - This will include more events in the vocabulary
                                        
                                        **Current Analysis:**
                                        - Total events in vocabulary: {len(event_vocab)}
                                        - After TF-IDF processing: {n_features} events
                                        - Requested topics: {n_topics}
                                        - **Maximum possible topics**: {n_features}
                                        """)
                                    st.stop()
                                
                                st.success(f"Created proper TF-IDF matrix: {doc_term_matrix.shape[0]} customers × {doc_term_matrix.shape[1]} events", icon=":material/check:")
                                
                                # Show TF-IDF statistics
                                with st.expander("TF-IDF Statistics", icon=":material/query_stats:"):
                                    avg_tfidf_selected = np.mean(tfidf_matrix_for_stats, axis=0)
                                    top_tfidf_idx = np.argsort(avg_tfidf_selected)[-10:][::-1]
                                    st.markdown(f"**Top 10 Events by Average TF-IDF Score** *(among {len(event_documents):,} LDA-eligible customers)*:")
                                    for idx in top_tfidf_idx:
                                        event_name = feature_names[idx]
                                        score = avg_tfidf_selected[idx]
                                        # Calculate document frequency for selected features
                                        doc_freq = np.sum(raw_count_matrix[:, idx] > 0)
                                        pct = (doc_freq / len(event_documents)) * 100
                                        st.markdown(f"• **{event_name}**: {score:.4f} (in {doc_freq:,} customers - {pct:.1f}%)")
                                    
                                    st.markdown(f"**Bottom 5 Events by TF-IDF** *(among selected features)*:")
                                    bottom_tfidf_idx = np.argsort(avg_tfidf_selected)[:5]
                                    for idx in bottom_tfidf_idx:
                                        event_name = feature_names[idx]
                                        score = avg_tfidf_selected[idx]
                                        doc_freq = np.sum(raw_count_matrix[:, idx] > 0)
                                        pct = (doc_freq / len(event_documents)) * 100
                                        st.markdown(f"• **{event_name}**: {score:.4f} (in {doc_freq:,} customers - {pct:.1f}%)")
                                    
                                    st.markdown("**Note**: Statistics show frequency among LDA-eligible customers only (those with sufficient event sequences). Low TF-IDF = common events, High TF-IDF = distinctive events.")
                            
                            elif feature_weighting == "Smart Filtering":
                                # Calculate event document frequencies for smart filtering
                                all_events_in_docs = []
                                for doc in event_documents:
                                    all_events_in_docs.extend(doc.split())
                                
                                event_doc_freq = pd.Series(all_events_in_docs).value_counts()
                                total_docs = len(event_documents)
                                
                                # More aggressive filtering to handle "browse" dominance
                                max_df_threshold = 0.7  # More restrictive than default 0.8
                                min_df_threshold = max(3, int(total_docs * 0.02))  # At least 3 docs or 2% of docs
                                
                                st.info(f"Smart Filtering: Removing events in >{max_df_threshold*100:.0f}% of customers (too common) or <{min_df_threshold} customers (too rare)", icon=":material/info:")
                                
                                # Create document-term matrix with aggressive filtering
                                vectorizer = CountVectorizer(
                                    max_features=min(len(event_vocab), 50),  # Focus on most distinctive events
                                    min_df=min_df_threshold,  # More restrictive minimum
                                    max_df=max_df_threshold,  # More restrictive maximum  
                                    token_pattern=r'\b\w+(?:\s+\w+)*\b'  # Handle multi-word events
                                )
                                
                                doc_term_matrix = vectorizer.fit_transform(event_documents)
                                feature_names = vectorizer.get_feature_names_out()
                                
                                # Critical validation: Check if we have enough features for the requested number of topics
                                n_features = doc_term_matrix.shape[1]
                                if n_features < n_topics:
                                    st.error(f"**Insufficient features for LDA**: Found only {n_features} events after filtering, but trying to create {n_topics} topics", icon=":material/chat_error:")
                                    st.warning(f"**Rule**: You need at least as many unique events as topics. Currently: {n_features} events < {n_topics} topics", icon=":material/warning:")
                                    
                                    with st.expander("🔧 Solutions", expanded=True):
                                        st.markdown(f"""
                                        **Choose one of these options:**
                                        
                                        **Option 1: Reduce Number of Topics**
                                        - Change "Number of Topics" to **{max(2, n_features)}** or less
                                        
                                        **Option 2: Use Less Aggressive Filtering**
                                        - Switch Feature Weighting from "Smart Filtering" to "Raw Frequency", "Log Frequency", or "Binary Presence"
                                        - These methods include more events ({len(event_vocab)} available vs {n_features} after Smart Filtering)
                                        
                                        **Option 3: Relax Data Filters**
                                        - Reduce "Min Event Frequency" (currently {min_count})
                                        - Reduce "Min Events per Customer" (currently {min_sequence_length})
                                        - This will include more events in the vocabulary
                                        
                                        **Current Analysis:**
                                        - Total events in vocabulary: {len(event_vocab)}
                                        - After Smart Filtering: {n_features} events
                                        - Requested topics: {n_topics}
                                        - **Maximum possible topics**: {n_features}
                                        """)
                                    st.stop()
                                
                                st.success(f"Created filtered document-term matrix: {doc_term_matrix.shape[0]} customers × {doc_term_matrix.shape[1]} events", icon=":material/check:")
                                
                                # Show filtering results
                                with st.expander("Event Filtering Results", icon=":material/leaderboard:"):
                                    st.markdown("**Events included in LDA model:**")
                                    for event_name in feature_names:
                                        doc_freq = event_doc_freq.get(event_name, 0)
                                        doc_pct = (doc_freq / total_docs) * 100
                                        st.markdown(f"• **{event_name}**: {doc_freq} customers ({doc_pct:.1f}%)")
                                    
                                    excluded_common = event_doc_freq[event_doc_freq > max_df_threshold * total_docs]
                                    if len(excluded_common) > 0:
                                        st.markdown("**Excluded (too common):**")
                                        for event, freq in excluded_common.head().items():
                                            pct = (freq / total_docs) * 100
                                            st.markdown(f"• ~~{event}~~: {freq} customers ({pct:.1f}%)")
                                    
                                    excluded_rare = event_doc_freq[event_doc_freq < min_df_threshold]
                                    if len(excluded_rare) > 0:
                                        st.markdown(f"**Excluded (too rare)**: {len(excluded_rare)} events appearing in <{min_df_threshold} customers")
                            
                            else:  # Raw Frequency, Log Frequency, Binary Presence
                                # For these methods, apply basic filtering to ensure LDA can find meaningful patterns
                                # Calculate event document frequencies first
                                all_events_in_docs = []
                                for doc in event_documents:
                                    all_events_in_docs.extend(doc.split())
                                
                                event_doc_freq = pd.Series(all_events_in_docs).value_counts()
                                total_docs = len(event_documents)
                                
                                # Apply more lenient filtering than Smart Filtering to preserve feature weighting differences
                                max_df_threshold = 0.95  # Remove only if in >95% of customers (very common)
                                min_df_threshold = max(2, int(total_docs * 0.01))  # At least 2 customers or 1% of customers
                                
                                st.info(f"Using {feature_weighting} feature weighting with basic filtering: removing events in >{max_df_threshold*100:.0f}% of customers or <{min_df_threshold} customers", icon=":material/info:")
                                
                                # Create document-term matrix with basic filtering
                                vectorizer = CountVectorizer(
                                    min_df=min_df_threshold,  # Basic minimum threshold
                                    max_df=max_df_threshold,  # Basic maximum threshold
                                    max_features=min(len(event_vocab), 100),  # Allow more features than Smart Filtering
                                    token_pattern=r'\b\w+(?:\s+\w+)*\b'  # Handle multi-word events
                                )
                                
                                # Get raw count matrix first
                                raw_matrix = vectorizer.fit_transform(event_documents)
                                feature_names = vectorizer.get_feature_names_out()
                                
                                # Apply the specific feature weighting
                                if feature_weighting == "Raw Frequency":
                                    # Raw counts - use as is
                                    doc_term_matrix = raw_matrix
                                    
                                    # Show raw frequency statistics
                                    matrix_stats = f"min: {doc_term_matrix.data.min():.1f}, max: {doc_term_matrix.data.max():.1f}, mean: {doc_term_matrix.data.mean():.2f}"
                                    st.info(f"Using raw event frequencies for {len(feature_names)} events. Matrix stats: {matrix_stats}", icon=":material/info:")
                                    
                                elif feature_weighting == "Log Frequency":
                                    # Log(1 + count) to reduce impact of very frequent events
                                    doc_term_matrix = raw_matrix.copy()
                                    doc_term_matrix.data = np.log1p(doc_term_matrix.data)  # log(1 + x) for sparse matrix
                                    st.info(f"Using log frequency weighting: log(1 + count) for {len(feature_names)} events", icon=":material/info:")
                                    
                                elif feature_weighting == "Binary Presence":
                                    # Binary: 1 if event occurs, 0 otherwise
                                    doc_term_matrix = raw_matrix.copy()
                                    doc_term_matrix.data = (doc_term_matrix.data > 0).astype(float)  # Convert to binary
                                    
                                    # Verify binary conversion worked
                                    unique_values = np.unique(doc_term_matrix.data)
                                    st.info(f"Using binary presence weighting (0/1) for {len(feature_names)} events. Matrix values: {unique_values}", icon=":material/info:")
                                    
                                else:
                                    # Default to raw frequency
                                    doc_term_matrix = raw_matrix
                                    st.warning(f"Unknown feature weighting '{feature_weighting}', using raw frequency", icon=":material/warning:")
                                
                                # Critical validation: Check if we have enough features for the requested number of topics
                                n_features = doc_term_matrix.shape[1]
                                if n_features < n_topics:
                                    st.error(f"**Insufficient features for LDA**: Found only {n_features} events after basic filtering, but trying to create {n_topics} topics", icon=":material/chat_error:")
                                    st.warning(f"**Rule**: You need at least as many unique events as topics. Currently: {n_features} events < {n_topics} topics", icon=":material/warning:")
                                    
                                    with st.expander("🔧 Solutions", expanded=True):
                                        st.markdown(f"""
                                        **Choose one of these options:**
                                        
                                        **Option 1: Reduce Number of Topics**
                                        - Change "Number of Topics" to **{max(2, n_features)}** or less
                                        
                                        **Option 2: Relax Data Filters**
                                        - Reduce "Min Event Frequency" (currently {min_count})
                                        - Reduce "Min Events per Customer" (currently {min_sequence_length})
                                        - This will include more events in the vocabulary
                                        
                                        **Current Analysis:**
                                        - Total events in vocabulary: {len(event_vocab)}
                                        - After basic filtering: {n_features} events
                                        - Requested topics: {n_topics}
                                        - **Maximum possible topics**: {n_features}
                                        """)
                                    st.stop()
                                
                                st.success(f"Created document-term matrix: {doc_term_matrix.shape[0]} customers × {doc_term_matrix.shape[1]} events", icon=":material/check:")
                                
                                # Show filtering results to distinguish from Smart Filtering
                                with st.expander("Basic Event Filtering Results", icon=":material/filter_list:"):
                                    st.markdown("**Events included in LDA model:**")
                                    for event_name in feature_names:
                                        doc_freq = event_doc_freq.get(event_name, 0)
                                        doc_pct = (doc_freq / total_docs) * 100
                                        st.markdown(f"• **{event_name}**: {doc_freq} customers ({doc_pct:.1f}%)")
                                    
                                    excluded_common = event_doc_freq[event_doc_freq > max_df_threshold * total_docs]
                                    if len(excluded_common) > 0:
                                        st.markdown("**Excluded (too common):**")
                                        for event, freq in excluded_common.head().items():
                                            pct = (freq / total_docs) * 100
                                            st.markdown(f"• ~~{event}~~: {freq} customers ({pct:.1f}%)")
                                    
                                    excluded_rare = event_doc_freq[event_doc_freq < min_df_threshold]
                                    if len(excluded_rare) > 0:
                                        st.markdown(f"**Excluded (too rare)**: {len(excluded_rare)} events appearing in <{min_df_threshold} customers")
                                        
                                    st.markdown(f"**Note**: This basic filtering preserves more events than Smart Filtering to maintain {feature_weighting} characteristics")
                            
                            # Fit LDA model
                            lda_model = LatentDirichletAllocation(
                                n_components=n_topics,
                                doc_topic_prior=alpha,  # Alpha parameter
                                topic_word_prior=beta,   # Beta parameter
                                random_state=42,
                                max_iter=100,
                                learning_method='batch'
                            )
                            
                            # Get topic distributions for each customer
                            topic_distributions = lda_model.fit_transform(doc_term_matrix)
                            
                            # Assign each customer to their dominant topic
                            labels = np.argmax(topic_distributions, axis=1)
                            customer_ids = np.array(valid_customer_ids)
                            
                            # Update X to use topic distributions for metrics calculation
                            X = topic_distributions
                            
                            st.success(f"LDA completed! Discovered {n_topics} behavioral topics for {len(customer_ids)} customers",icon=":material/check:")
                            
                except ImportError:
                        st.error("scikit-learn LDA not available. Falling back to K-Means.", icon=":material/chat_error:")
                        from sklearn.cluster import KMeans
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = clusterer.fit_predict(X)
                        
                except Exception as e:
                        st.error(f"LDA failed: {str(e)}", icon=":material/chat_error:")
                        st.info("Falling back to K-Means clustering...")
                        from sklearn.cluster import KMeans
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = clusterer.fit_predict(X)

            elif X is not None and len(X) > 0:
                preprocessing_text = f" + {', '.join(preprocessing_steps)}" if preprocessing_steps else ""
                st.info(f"Running {clustering_method} clustering on {len(X):,} customers • {feature_weighting} weighting • {pca_info_text}{preprocessing_text}", icon=":material/settings:")
                    
                # Additional check for Gaussian Mixture
                if clustering_method == "Gaussian Mixture" and X is not None:
                    # Ensure we have enough samples per component
                    min_samples_per_component = n_clusters * 10
                    if len(X) < min_samples_per_component:
                        st.warning(f"Gaussian Mixture works best with at least {min_samples_per_component} samples for {n_clusters} clusters. Consider reducing cluster count or using K-Means.",icon=":material/warning:")
                
                # Perform clustering
                if clustering_method == "K-Means":
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(X)
                    
                elif clustering_method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = clusterer.fit_predict(X)
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_points = list(labels).count(-1)
                    
                    if n_clusters_found > 50:
                        st.warning(f"DBSCAN found {n_clusters_found} clusters with {noise_points} noise points - too fragmented! Try increasing epsilon or min_samples.", icon=":material/warning:")
                    elif n_clusters_found < 3:
                        st.warning(f"DBSCAN found only {n_clusters_found} clusters with {noise_points} noise points - try decreasing epsilon or min_samples.", icon=":material/warning:")
                    else:
                        st.success(f"DBSCAN found {n_clusters_found} clusters with {noise_points} noise points", icon=":material/check:")
                    
                elif clustering_method == "Gaussian Mixture":
                    from sklearn.mixture import GaussianMixture
                    try:
                        # Convert to float64 for better numerical stability
                        X_float64 = X.astype(np.float64)
                        
                        # Use regularization to handle ill-conditioned covariance matrices
                        clusterer = GaussianMixture(
                            n_components=n_clusters, 
                            random_state=42,
                            reg_covar=1e-4,  # Regularization to prevent singular matrices
                            covariance_type='diag',  # Diagonal covariance for stability
                            max_iter=200,
                            n_init=3
                        )
                        labels = clusterer.fit_predict(X_float64)
                        
                        # Check convergence
                        if not clusterer.converged_:
                            st.warning("Gaussian Mixture did not converge. Results may be suboptimal.", icon=":material/warning:")
                            
                    except Exception as e:
                        st.error(f"Gaussian Mixture failed: {str(e)} → Using K-Means instead", icon=":material/chat_error:")
                        
                        # Fallback to K-Means
                        from sklearn.cluster import KMeans
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = clusterer.fit_predict(X)
                    
                elif clustering_method == "Hierarchical" and X is not None:
                    from sklearn.cluster import AgglomerativeClustering
                    
                    # Check memory requirements for hierarchical clustering
                    n_samples = len(X)
                    memory_limit = 10000  # Conservative limit for hierarchical clustering
                    
                    if n_samples > memory_limit:
                        st.error(f"Hierarchical clustering: too many customers ({n_samples:,} > {memory_limit:,}) → Using K-Means instead", icon=":material/chat_error:")                    
                        # Fallback to K-Means
                        from sklearn.cluster import KMeans
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = clusterer.fit_predict(X)
                        
                    else:
                        try:
                            clusterer = AgglomerativeClustering(
                                n_clusters=n_clusters,
                                linkage='ward',  # Most memory efficient
                                metric='euclidean'  # Required for ward linkage
                            )
                            labels = clusterer.fit_predict(X)
                            
                        except Exception as e:
                            st.error(f"Hierarchical clustering failed: {str(e)} → Using K-Means instead", icon=":material/chat_error:")
                            
                            # Fallback to K-Means
                            from sklearn.cluster import KMeans
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = clusterer.fit_predict(X)
                
                
            else:
                    # Default to K-Means if method not recognized
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(X)
                    st.warning(f"Unknown clustering method '{clustering_method}', defaulting to K-Means", icon=":material/warning:")
                
                # Calculate basic metrics for session state
            valid_mask = labels != -1
            n_valid_samples = valid_mask.sum()
            n_clusters_found = len(set(labels[valid_mask])) if n_valid_samples > 0 else 0
                
                # Show success message based on clustering method
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                    st.success(f"LDA topic modeling completed! Preparing results...",icon=":material/check:")
            else:
                    st.success(f"{clustering_method} clustering completed! Preparing results...",icon=":material/check:")
                
                # Prepare PCA info for storage
            if pca_applied:
                    if pca_option == "Auto (Recommended)":
                        pca_description = f"Auto PCA: {original_features} → {X.shape[1]} dimensions ({variance_explained:.1%} variance)"
                    else:
                        pca_description = f"Manual PCA: {original_features} → {X.shape[1]} dimensions ({variance_explained:.1%} variance)"
            else:
                    pca_description = f"No PCA (using all {original_features} features)"
                
                # Store results in session state for Results tab
            session_results = {
                    'completed': True,
                    'clustering_method': clustering_method,
                    'feature_weighting': feature_weighting,
                    'X': X,
                    'labels': labels,
                    'customer_ids': customer_ids,
                    'sequences_df': sequences_df,
                    'valid_mask': valid_mask,
                    'n_valid_samples': n_valid_samples,
                    'n_clusters_found': n_clusters_found,
                    'pca_info': pca_description,
                    'original_features': original_features,
                    'pca_applied': pca_applied
                }
                
                # Add LDA-specific data if LDA was used
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                    if 'lda_model' in locals():
                        session_results['lda_model'] = lda_model
                    if 'topic_distributions' in locals():
                        session_results['topic_distributions'] = topic_distributions
                    if 'feature_names' in locals():
                        session_results['feature_names'] = feature_names
                
                # Store results in session state for persistence
            st.session_state['behavioral_results'] = {
                    'completed': True,
                    'clustering_method': clustering_method,
                    'feature_weighting': feature_weighting,
                    'X': X,
                    'labels': labels,
                    'customer_ids': customer_ids,
                    'sequences_df': sequences_df,
                    'valid_mask': valid_mask,
                    'n_valid_samples': n_valid_samples,
                    'n_clusters_found': n_clusters_found,
                    'customer_id_col': customer_id_col,
                    'event_seq_col': event_seq_col,
                    'seq_length_col': seq_length_col,
                    'unique_events_col': unique_events_col,
                    'pca_info': pca_description,
                    'original_features': original_features,
                    'pca_applied': pca_applied
                }
                
                # Add LDA-specific data if LDA was used
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                    if 'lda_model' in locals():
                        st.session_state['behavioral_results']['lda_model'] = lda_model
                    if 'topic_distributions' in locals():
                        st.session_state['behavioral_results']['topic_distributions'] = topic_distributions
                    if 'feature_names' in locals():
                        st.session_state['behavioral_results']['feature_names'] = feature_names
                
                # Also store parameters for cache invalidation
            st.session_state['last_behavioral_params'] = {
                    'database': database,
                    'schema': schema,
                    'table': table,
                    'uid': uid,
                    'evt': evt,
                    'tmstp': tmstp,
                    'period_type': period_type,
                    'clustering_method': clustering_method,
                    'feature_weighting': feature_weighting,
                    'min_sequence_length': min_sequence_length,
                    'max_sequence_length': max_sequence_length,
                    'min_count': min_count,
                    'n_clusters': n_clusters if clustering_method != "Latent Dirichlet Allocation (LDA)" else n_topics,
                    'max_customers': max_customers,
                    'pca_option': pca_option,
                    'normalize_vectors': normalize_vectors,
                    'vector_size': vector_size,
                    'window_size': window_size
                }
                
                # Add method-specific parameters
            if clustering_method == "Latent Dirichlet Allocation (LDA)":
                    st.session_state['last_behavioral_params'].update({
                        'alpha': alpha,
                        'beta': beta,
                        'n_topics': n_topics
                    })
            elif clustering_method == "DBSCAN":
                    st.session_state['last_behavioral_params'].update({
                        'eps': eps,
                        'min_samples': min_samples
                    })
                
                # Add date range parameters if applicable
            if period_type == "Last N Days":
                    st.session_state['last_behavioral_params']['days_back'] = days_back
            elif period_type == "Date Range":
                    st.session_state['last_behavioral_params'].update({
                        'start_date': str(start_date),
                        'end_date': str(end_date)
                    })
            
            # Set flag to suppress stale parameter warning on next rerun
            st.session_state['just_ran_clustering'] = True
        
        with tab2:
            # Render results from session state (persistent across reruns)
            if 'behavioral_results' in st.session_state and st.session_state['behavioral_results'].get('completed'):
                
                
                # Get results from session state
                results = st.session_state['behavioral_results']
            
            # Display results based on clustering method - simplified for LDA
                if results.get('clustering_method') == "Latent Dirichlet Allocation (LDA)":
                    # Add LDA-specific quality metrics
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Quality Metrics</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                    
                    # Check if we have LDA model results stored
                    if 'lda_model' in results and 'topic_distributions' in results:
                        lda_model = results['lda_model']
                        topic_distributions = results['topic_distributions']
                        
                        # Calculate LDA-specific metrics
                        try:
                            # Validate dimensions first to prevent index errors
                            X_shape = results['X'].shape if 'X' in results else (0, 0)
                            td_shape = topic_distributions.shape
                            comp_shape = lda_model.components_.shape
                            n_topics_actual = lda_model.n_components
                            
                            # Check for dimension mismatches that could cause index errors
                            if td_shape[1] != n_topics_actual:
                                st.warning(f"Topic distribution mismatch: got {td_shape[1]} topics, expected {n_topics_actual}", icon=":material/warning:")
                            
                            if comp_shape[0] != n_topics_actual:
                                st.warning(f"Components shape mismatch: got {comp_shape[0]} topics, expected {n_topics_actual}", icon=":material/warning:")
                            
                            # Perplexity (lower is better)
                            perplexity = lda_model.perplexity(results['X'])
                            
                            # Log-likelihood (higher is better)
                            log_likelihood = lda_model.score(results['X'])
                            
                            # Topic coherence (measure topic quality)
                            # Calculate average within-topic probability with explicit bounds checking
                            try:
                                if comp_shape[0] > 0 and comp_shape[1] > 0:
                                    # Additional safety: iterate safely through components
                                    topic_means = []
                                    for i, topic in enumerate(lda_model.components_):
                                        if i < comp_shape[0] and len(topic) == comp_shape[1]:
                                            topic_means.append(np.mean(topic))
                                    avg_topic_coherence = np.mean(topic_means) if topic_means else 0.0
                                else:
                                    avg_topic_coherence = 0.0
                            except (IndexError, ValueError) as e:
                                avg_topic_coherence = 0.0
                                st.warning(f"Topic coherence calculation failed: {str(e)}", icon=":material/warning:")
                            
                            # Topic diversity (how different topics are from each other)
                            # Add safety check for components shape
                            components_shape = lda_model.components_.shape
                            if components_shape[1] > 0 and components_shape[0] > 0:  # Ensure we have topics and features
                                topic_diversity = np.mean(np.std(lda_model.components_, axis=0))
                            else:
                                topic_diversity = 0.0
                            
                            # Customer assignment clarity (how clearly customers are assigned to topics)
                            # Ensure we have valid topic distributions
                            if topic_distributions.size > 0 and topic_distributions.shape[1] > 0:
                                max_probs = np.max(topic_distributions, axis=1)
                                assignment_clarity = np.mean(max_probs)
                            else:
                                assignment_clarity = 0.0
                            
                            # Display LDA quality metrics
                            with st.container(border=True):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                col1.metric("Perplexity", f"{perplexity:.1f}", 
                                        help="Model complexity (lower is better, typically 50-500)")
                                col2.metric("Log-Likelihood", f"{log_likelihood:.1f}", 
                                        help="Model fit quality (higher is better)")
                                col3.metric("Topic Coherence", f"{avg_topic_coherence:.3f}", 
                                        help="Topic quality (higher is better, 0-1 scale)")
                                col4.metric("Assignment Clarity", f"{assignment_clarity:.3f}", 
                                        help="Customer-topic assignment strength (higher is better, 0-1 scale)")
                            
                        except Exception as e:
                            st.error(f"Error calculating LDA metrics: {str(e)}", icon=":material/chat_error:")
                            
                        # AI Insights for LDA (main analysis)
                        def lda_ai_analysis_callback_main(selected_model):
                            with st.spinner("Analyzing LDA quality metrics with AI..."):
                                try:
                                    # Get current LDA configuration
                                    current_topics = results.get('n_clusters_found', 'Unknown')
                                    current_weighting = results.get('feature_weighting', 'Unknown')
                                    pca_info_lda = results.get('pca_info', 'No PCA applied')
                                    original_features = results.get('original_features', 'Unknown')
                                    pca_applied = results.get('pca_applied', False)
                                    
                                    prompt = f"""
                                    Analyze these LDA (Latent Dirichlet Allocation) behavioral segmentation results and provide actionable insights:

                                    CURRENT LDA QUALITY METRICS:
                                    - Perplexity: {perplexity:.1f} (lower is better, typical range 50-500)
                                    - Log-Likelihood: {log_likelihood:.1f} (higher is better)
                                    - Topic Coherence: {avg_topic_coherence:.3f} (higher is better, 0-1 scale)
                                    - Assignment Clarity: {assignment_clarity:.3f} (higher is better, 0-1 scale)

                                    AVAILABLE LDA CONTROLS IN THE APPLICATION:
                                    - Number of Topics: 3-20 (slider control)
                                    - Alpha (Topic Concentration): 0.01-2.0 (affects how many topics per customer)
                                    - Beta (Word Concentration): 0.01-2.0 (affects how specific topics are)
                                    - Feature Weighting: TF-IDF, Binary Presence, Count Frequency, Raw Frequency, Log Frequency, Smart Filtering
                                    - Alternative Methods: K-Means, DBSCAN, Gaussian Mixture, Hierarchical (use Vector Dimensions 50-300, Context Window 2-10)

                                    CURRENT UI SETTINGS USED:
                                    - Method: LDA with {current_topics} topics
                                    - Feature Weighting: {current_weighting}
                                    - Alpha: {alpha}, Beta: {beta}
                                    - PCA: {pca_info_lda}
                                    - Note: Vector Dimensions ({vector_size}) and Context Window ({window_size}) not used by LDA

                                    IMPORTANT PCA CONTEXT:
                                    PCA is not applicable to LDA since LDA uses discrete token counts, not continuous vectors. If you recommend switching to other clustering methods, PCA options include: No PCA, Auto PCA (method-specific variance targets), and Manual PCA (user-specified components).

                                    Please provide specific, actionable recommendations for improving these LDA results. Focus on parameter adjustments, alternative approaches, and interpretation guidance. Keep response concise but insightful.
                                    """
                                    
                                    ai_result = session.sql(f"""
                                        SELECT SNOWFLAKE.CORTEX.COMPLETE('{selected_model}', '{prompt}') as insights
                                    """).collect()
                                    
                                    if ai_result and len(ai_result) > 0:
                                        insights = ai_result[0]['INSIGHTS']
                                        st.markdown("**🧠 AI Quality Analysis & Recommendations**")
                                        st.markdown(insights)
                                    
                                except Exception as e:
                                    st.warning(f"AI insights not available: {str(e)}", icon=":material/warning:")
                        
                        ai_model_lda_main, ai_enabled_lda_main = display_ai_insights_section(
                            "ai_model_lda_main", 
                            "Select the LLM model for AI analysis (dynamically fetched from Snowflake)",
                            ai_content_callback=lda_ai_analysis_callback_main
                        )
                        
                    else:
                        st.warning("LDA model results not found in session state. Please re-run the analysis.")
                    
                else:
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Quality Metrics</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                    
                    # Use stored results
                    valid_mask = results.get('valid_mask')
                    n_valid_samples = results.get('n_valid_samples', 0)
                    n_clusters_found = results.get('n_clusters_found', 0)
                    
                    if valid_mask is not None and n_valid_samples > 1 and n_clusters_found > 1:
                        try:
                            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                            
                            X = results['X']
                            labels = results['labels']
                            X_valid = X[valid_mask]
                            labels_valid = labels[valid_mask]
                            
                            # Calculate metrics
                            silhouette = silhouette_score(X_valid, labels_valid)
                            calinski = calinski_harabasz_score(X_valid, labels_valid)
                            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
                            
                            # Display quality metrics
                            with st.container(border=True):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                col1.metric("Silhouette Score", f"{silhouette:.3f}", 
                                        help="Cluster separation quality (-1 to 1, higher is better)")
                                col2.metric("Calinski-Harabasz", f"{calinski:.1f}", 
                                        help="Cluster density ratio (higher is better)")
                                col3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", 
                                        help="Cluster similarity measure (lower is better)")
                                col4.metric("Valid Customers", f"{n_valid_samples:,}", 
                                        help="Customers successfully clustered")
                            
                            
                        except Exception as e:
                            st.error(f"Error calculating metrics: {str(e)}", icon=":material/chat_error:")
                    else:
                        st.warning("Insufficient data for clustering quality metrics", icon=":material/warning:")
                    
                    # AI Insights for clustering quality (non-LDA methods) - moved outside the if/else block
                    def ai_analysis_callback(selected_model):
                        with st.spinner("Analyzing clustering quality metrics with AI..."):
                            try:
                                # Prepare metrics data for AI analysis
                                if valid_mask is not None and n_valid_samples > 1 and n_clusters_found > 1:
                                    # We have calculated metrics
                                    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                    
                                    X = session_results['X']
                                    labels = session_results['labels']
                                    X_valid = X[valid_mask]
                                    labels_valid = labels[valid_mask]
                                    
                                    # Calculate metrics for AI analysis
                                    silhouette = silhouette_score(X_valid, labels_valid)
                                    calinski = calinski_harabasz_score(X_valid, labels_valid)
                                    davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
                                    
                                    metrics_info = f"""
                                    Clustering Method: {results.get('clustering_method')}
                                    Feature Weighting: {results.get('feature_weighting')}
                                    Total Customers: {len(results.get('customer_ids', []))}
                                    Number of Clusters: {results.get('n_clusters_found')}
                                    
                                    Quality Metrics:
                                    - Silhouette Score: {silhouette:.3f} (range -1 to 1, higher is better)
                                    - Calinski-Harabasz Index: {calinski:.1f} (higher is better)
                                    - Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)
                                    - Valid Customers Clustered: {n_valid_samples:,}
                                    """
                                else:
                                    metrics_info = f"""
                                    Clustering Method: {results.get('clustering_method')}
                                    Feature Weighting: {results.get('feature_weighting')}
                                    Total Customers: {len(results.get('customer_ids', []))}
                                    Number of Clusters: {results.get('n_clusters_found')}
                                    
                                    Note: Insufficient data for detailed quality metrics calculation.
                                    """
                                    
                                # Get current configuration details
                                current_method = results.get('clustering_method', 'Unknown')
                                current_weighting = results.get('feature_weighting', 'Unknown')
                                current_clusters = results.get('n_clusters_found', 'Unknown')
                                pca_info = results.get('pca_info', 'No PCA applied')
                                
                                ai_prompt = f"""
                                You are analyzing behavioral clustering results from a Streamlit application. Please provide specific, actionable recommendations based on the current configuration and available options.

                                CURRENT CONFIGURATION:
                                {metrics_info}

                                CURRENT UI SETTINGS USED:
                                - Clustering Method: {current_method}
                                - Feature Weighting: {current_weighting}
                                - Number of Clusters: {current_clusters}
                                - Vector Dimensions: {vector_size}
                                - Context Window: {window_size}
                                - Normalize Vectors: {normalize_vectors}
                                - PCA Dimensionality Reduction: {pca_info}

                                AVAILABLE OPTIONS IN THE APPLICATION:
                                - Clustering Methods: K-Means, DBSCAN, Gaussian Mixture, Hierarchical, Latent Dirichlet Allocation (LDA)
                                - Feature Weighting: TF-IDF, Binary Presence, Count Frequency, Raw Frequency, Log Frequency, Smart Filtering
                                - Number of Clusters: 3-20 (for K-Means, Gaussian Mixture, Hierarchical)
                                - Vector Dimensions: 50-300 (dimensionality of event embeddings)
                                - Context Window: 2-10 (surrounding events for context in embeddings)
                                - Normalize Vectors: Yes/No (normalize to unit length)
                                - DBSCAN Parameters: eps (0.1-2.0), min_samples (2-20)
                                - LDA Parameters: alpha (0.01-2.0), beta (0.01-2.0), number of topics (3-20)
                                - PCA Options: No PCA, Auto (Recommended), Manual (1-50 components)

                                Please provide:
                                1. **Quality Assessment**: How do these metrics rate for {current_method} clustering with the current PCA setting?
                                2. **PCA Analysis**: Is the current PCA choice ("{pca_info}") optimal for {current_method}? Should they try No PCA, Auto, or Manual with different dimensions?
                                3. **Parameter Tuning**: Specific adjustments for {current_method} parameters available in the UI (considering PCA effects)
                                4. **Alternative Methods**: Should the user try a different clustering method from the available options? Consider how PCA interacts with each method.
                                5. **Feature Engineering**: Should they switch from "{current_weighting}" to another weighting method? How does this interact with PCA?
                                6. **Data Insights**: Are there data quality issues affecting the {current_method} results? Could PCA help or hurt?
                                7. **Next Steps**: Concrete actions using the available UI controls, including specific PCA recommendations

                                IMPORTANT PCA CONTEXT:
                                - No PCA: Uses all original features, may have curse of dimensionality issues
                                - Auto PCA: System-optimized for {current_method} (K-Means/Hierarchical: ~90% variance, DBSCAN: ~95%, Gaussian Mixture: ~85%, LDA: not applicable)
                                - Manual PCA: User-controlled 1-50 components for custom dimensionality reduction
                                
                                Focus on actionable recommendations including PCA strategy for optimal {current_method} performance.
                                """
                                
                                # Use Snowflake Cortex for analysis with selected model
                                ai_sql = f"""
                                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                    '{selected_model}',
                                    $${ai_prompt}$$
                                ) as insights
                                """
                                
                                ai_result = session.sql(ai_sql).collect()
                                if ai_result:
                                    insights = ai_result[0]['INSIGHTS']
                                    
                                    st.markdown("**AI Quality Analysis & Recommendations**")
                                    st.markdown(insights)
                                    
                            except Exception as e:
                                st.warning(f"AI insights not available: {str(e)}", icon=":material/warning:")
                    
                    ai_model, ai_enabled = display_ai_insights_section(
                        "ai_model_clustering", 
                        "Select the LLM model for AI analysis (dynamically fetched from Snowflake)",
                        ai_content_callback=ai_analysis_callback
                    )
                
                # Create results DataFrame with cluster assignments using session state data
                results_df = pd.DataFrame({
                    'customer_id': results['customer_ids'],
                    'cluster': results['labels']
                })
                
                # Merge with original sequence data for analysis
                customer_id_col = results['customer_id_col']
                event_seq_col = results['event_seq_col']
                seq_length_col = results['seq_length_col']
                unique_events_col = results['unique_events_col']
                sequences_df = results['sequences_df']
                clustering_method = results['clustering_method']
                
                merge_cols = [customer_id_col, event_seq_col, seq_length_col, unique_events_col]
                results_df = results_df.merge(
                    sequences_df[merge_cols], 
                    left_on='customer_id', 
                    right_on=customer_id_col
                )
                
                # Cluster Distribution
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Cluster Distribution</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                
                # Simple cluster distribution chart - sorted by customer count descending
                cluster_counts = results_df['cluster'].value_counts().sort_values(ascending=False)  # Sort by count descending
                cluster_percentages = (cluster_counts / len(results_df) * 100).round(1)
                
                # Create bar chart with colors aligned to treemap (Blues color scale)
                # Darker colors for higher customer counts (sorted order)
                import plotly.colors
                n_colors = len(cluster_counts)
                if n_colors > 1:
                    # Use Blues color scale with darker colors for higher values
                    color_scale = plotly.colors.sequential.Blues
                    # Reverse the mapping so first (highest) gets darkest color
                    bar_colors = [color_scale[-(int(i * (len(color_scale)-1) / (n_colors-1)) + 1)] for i in range(n_colors)]
                else:
                    bar_colors = ['#29B5E8']  # Single color if only one cluster
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"Cluster {i}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {i}" for i in cluster_counts.index],
                    y=cluster_counts.values,
                    text=[f"{count:,}<br>({pct}%)" for count, pct in zip(cluster_counts.values, cluster_percentages.values)],
                    textposition='auto',
                    marker_color=bar_colors
                ))
                
                fig_bar.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Clusters" if clustering_method != "Latent Dirichlet Allocation (LDA)" else "Topics",
                    yaxis_title="Number of Customers"
                )
                
                with st.container(border=True):
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Add Treemap Visualization below the bar chart
                
                # Cluster Distribution
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Cluster & Event Distribution</h2>
                    <hr style='margin-top: -8px;margin-bottom: 10px;'>
                    """, unsafe_allow_html=True)
                # Add option to show all events vs top events

                show_all_events = st.checkbox("Show All Events", value=False, help="Uncheck to show only top 5 events per cluster")
                
                try:
                    # Create multi-level treemap using Plotly Express path approach
                    # Build a dataframe suitable for px.treemap with path parameter
                    
                    # Prepare data for multi-level treemap
                    treemap_data = []
                    
                    # Analyze events within each cluster first
                    cluster_event_analysis = {}
                    for cluster_id in cluster_counts.index:
                        cluster_customers = results_df[results_df['cluster'] == cluster_id]
                        
                        # Extract all events from customers in this cluster
                        all_events_in_cluster = []
                        for _, row in cluster_customers.iterrows():
                            events = row[event_seq_col].split(' → ')
                            all_events_in_cluster.extend(events)
                        
                        # Count event frequencies within this cluster
                        if len(all_events_in_cluster) > 0:
                            event_counts_in_cluster = pd.Series(all_events_in_cluster).value_counts()
                            cluster_event_analysis[cluster_id] = event_counts_in_cluster
                    
                    # Build hierarchical data for px.treemap using path parameter
                    for cluster_id in cluster_counts.index:
                        cluster_name = f"Cluster {cluster_id}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {cluster_id}"
                        cluster_count = cluster_counts[cluster_id]
                        cluster_percentage = cluster_percentages[cluster_id]
                        
                        # Enhanced cluster name with percentage
                        cluster_display_name = f"{cluster_name} ({cluster_count:,} customers - {cluster_percentage}%)"
                        
                        if cluster_id in cluster_event_analysis:
                            # Get events for this cluster based on user preference
                            if show_all_events:
                                events_to_show = cluster_event_analysis[cluster_id]
                            else:
                                events_to_show = cluster_event_analysis[cluster_id].head(5)
                            
                            cluster_total_events = cluster_event_analysis[cluster_id].sum()
                            
                            for event_name, event_count in events_to_show.items():
                                event_percentage = (event_count / cluster_total_events * 100)
                                
                                # Add each event as a row in our treemap data
                                treemap_data.append({
                                    'all': 'All Customers',  # Root level
                                    'cluster': cluster_display_name,  # Second level with enhanced info
                                    'event': event_name,     # Third level
                                    'count': event_count,    # Values for sizing
                                    'customers': cluster_count,  # For color coding
                                    'cluster_percentage': cluster_percentage,  # For tooltips
                                    'event_percentage': round(event_percentage, 1),  # Event % within cluster
                                    'cluster_total_events': cluster_total_events  # For context
                                })
                        else:
                            # If no events, just add cluster data
                            treemap_data.append({
                                'all': 'All Customers',
                                'cluster': cluster_display_name,
                                'event': 'No Events',
                                'count': 1,  # Minimal size
                                'customers': cluster_count,
                                'cluster_percentage': cluster_percentage,
                                'event_percentage': 0,
                                'cluster_total_events': 0
                            })
                    
                    # Convert to DataFrame
                    treemap_df = pd.DataFrame(treemap_data)
                    
                    # Create treemap using px.treemap with path parameter
                    if len(treemap_df) > 0:
                        try:
                            # Create discrete color map matching bar chart EXACTLY
                            import plotly.colors
                            
                            # Extract cluster numbers from display names  
                            treemap_df['cluster_num'] = treemap_df['cluster'].str.extract(r'Cluster (\d+)')[0].astype(int)
                            
                            # Get cluster-to-customer mapping and sort by customer count descending (SAME AS BAR CHART)
                            cluster_info = treemap_df[['cluster', 'cluster_num', 'customers']].drop_duplicates('cluster')
                            cluster_info = cluster_info.sort_values('customers', ascending=False).reset_index(drop=True)
                            
                            n_colors = len(cluster_info)
                            
                            # Safety check
                            if n_colors == 0:
                                st.warning("No clusters found for treemap visualization.", icon=":material/warning:")
                            else:
                                color_scale = plotly.colors.sequential.Blues
                                
                                # Use EXACT SAME color assignment as bar chart
                                if n_colors > 1:
                                    bar_colors = [color_scale[-(int(i * (len(color_scale)-1) / (n_colors-1)) + 1)] for i in range(n_colors)]
                                else:
                                    bar_colors = ['#29B5E8']
                                
                                # Map: cluster display name → bar chart color (in sorted order)
                                color_discrete_map = dict(zip(cluster_info['cluster'], bar_colors))
                                
                                # Add customer count to treemap_df and sort by it to match bar chart order
                                treemap_df = treemap_df.merge(cluster_info[['cluster', 'customers']], on='cluster', how='left', suffixes=('', '_merge'))
                                treemap_df = treemap_df.sort_values(['customers', 'count'], ascending=[False, False])
                                
                                fig_treemap = px.treemap(
                                    treemap_df,
                                    path=[px.Constant("All Customers"), 'cluster', 'event'],
                                    values='count',
                                    color='cluster',
                                    color_discrete_map=color_discrete_map,
                                    branchvalues='total'  # Size clusters by total of their children
                                )
                                
                                # Customize appearance and set root transparent
                                fig_treemap.update_traces(
                                    textfont_size=10,
                                    marker=dict(line=dict(width=1)),
                                    hovertemplate='<b>%{label}</b><br>Event Frequency: %{value:,}<br><extra></extra>'
                                )
                                
                                # Override just the root color to transparent
                                fig_treemap.data[0].marker.colors = [
                                    'rgba(0,0,0,0)' if label == 'All Customers' else fig_treemap.data[0].marker.colors[i]
                                    for i, label in enumerate(fig_treemap.data[0].labels)
                                ]
                                
                                fig_treemap.update_layout(
                                    height=600,
                                    margin=dict(t=50, l=0, r=0, b=0),
                                    font=dict(size=10)
                                )

                                with st.container(border=True):
                                    st.plotly_chart(fig_treemap, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate treemap visualization: {str(e)}", icon=":material/error:")
                            st.info("Treemap requires additional data processing. Bar chart above shows the same information.", icon=":material/info:")
                        
                        # Show summary of what's displayed
                        if show_all_events:
                            total_events_shown = len(treemap_df[treemap_df['event'] != 'No Events'])
                            st.info(f"Displaying all {total_events_shown} unique events across {len(cluster_counts)} clusters", icon=":material/info:")
                        else:
                            st.info("Displaying top 5 events per cluster. Check 'Show All Events' to see complete data.", icon=":material/info:")
                        
                    else:
                        # Fallback to simple treemap if data preparation fails
                        fig_treemap = go.Figure(go.Treemap(
                            labels=[f"Cluster {i}" for i in cluster_counts.index],
                            parents=["All Customers"] * len(cluster_counts),
                            values=cluster_counts.values,
                            textinfo="label",
                            textfont=dict(size=12),
                            marker=dict(colorscale='Blues'),
                            root_color="lightgrey"
                        ))
                        
                        fig_treemap.update_layout(
                            height=500,
                            margin=dict(t=50, l=0, r=0, b=0)
                        )
                        
                        with st.container(border=True):
                            st.plotly_chart(fig_treemap, use_container_width=True)
                    
                    # Add explanation for enhanced treemap
                    with st.expander("Understanding the Enhanced Treemap", icon=":material/lightbulb:"):
                        st.markdown("""
                        **How to read this enhanced treemap:**
                        - **3 Levels**: Root (All Customers) → Clusters → Events within each cluster
                        - **Size**: Rectangle area represents customer count (clusters) or event frequency (events)
                        - **Color**: Shades of blue distinguish clusters; slightly different tones show events within clusters
                        - **Hierarchy**: Click or hover to explore cluster composition and dominant events
                        
                        **Key insights:**
                        - **Cluster Level**: Larger rectangles = clusters with more customers (dominant segments)
                        - **Event Level**: Shows events within each cluster with their frequency
                        - **Behavioral Patterns**: Quickly identify which events define each behavioral segment
                        - **Event Distribution**: See how event frequency varies across different customer clusters
                        
                        **Navigation Tips:**
                        - Hover over any rectangle to see detailed information including exact percentages
                        - Compare event patterns between different clusters
                        - Use the 'Show All Events' checkbox to see complete or filtered data
                        - Identify clusters dominated by specific event types
                        """)
                        
                except Exception as e:
                    st.warning(f"Could not generate treemap visualization: {str(e)}", icon=":material/warning:")
                    st.info("Treemap requires additional data processing. Bar chart above shows the same information.", icon=":material/info:")
                
                # AI Cluster Interpretation Section
                with st.expander("AI Cluster Interpretation", expanded=False, icon=":material/network_intel_node:"):
                    st.markdown("**Analyze cluster behaviors and business implications**")
                    
                    # Model selection for cluster interpretation
                    models_data = get_available_cortex_models()
                    if models_data["status"] == "found":
                        available_models = models_data["models"]
                    else:
                        available_models = ["mixtral-8x7b", "mistral-large", "llama3-8b", "llama3-70b", "gemma-7b"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        interpretation_model = st.selectbox(
                            "Choose AI Model for Interpretation", 
                            available_models, 
                            index=0,
                            key="cluster_interpretation_model_tab1"
                        )
                        
                        interpret_clusters = st.toggle(
                            "Interpret Clusters", 
                            key="interpret_clusters_toggle_tab1"
                        )
                    # with col2 & col3: (left empty for alignment with other sections)
                    
                    if interpret_clusters:
                        with st.spinner("Analyzing cluster behaviors and business context..."):
                            try:
                                # Prepare cluster analysis data for AI
                                cluster_analysis_text = ""
                                
                                for cluster_id in cluster_counts.index:
                                    cluster_name = f"Cluster {cluster_id}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {cluster_id}"
                                    cluster_count = cluster_counts[cluster_id]
                                    cluster_percentage = cluster_percentages[cluster_id]
                                    
                                    cluster_analysis_text += f"\n**{cluster_name}:**\n"
                                    cluster_analysis_text += f"- Size: {cluster_count:,} customers ({cluster_percentage}% of total)\n"
                                    
                                    if cluster_id in cluster_event_analysis:
                                        top_events = cluster_event_analysis[cluster_id].head(10)  # Top 10 events for AI analysis
                                        cluster_analysis_text += f"- Top Events:\n"
                                        for event_name, event_count in top_events.items():
                                            event_pct = (event_count / cluster_event_analysis[cluster_id].sum() * 100)
                                            cluster_analysis_text += f"  • {event_name}: {event_count} occurrences ({event_pct:.1f}%)\n"
                                    else:
                                        cluster_analysis_text += "- No significant events identified\n"
                                
                                # Create AI prompt for cluster interpretation
                                interpretation_prompt = f"""
                                You are a business analyst specializing in customer behavioral segmentation. Analyze the following customer clusters and provide business insights.
                                
                                CLUSTERING ANALYSIS RESULTS:
                                - Method Used: {clustering_method}
                                - Feature Weighting: {feature_weighting}
                                - Total Customers Analyzed: {len(results_df):,}
                                - Number of Clusters/Topics: {len(cluster_counts)}
                                
                                CLUSTER BREAKDOWN:
                                {cluster_analysis_text}
                                
                                ANALYSIS INSTRUCTIONS:
                                Please provide a comprehensive interpretation for each cluster/topic with the following structure:
                                
                                For each cluster, provide:
                                1. **Behavioral Profile**: What type of customer behavior does this cluster represent?
                                2. **Business Significance**: What does this cluster mean from a business perspective?
                                3. **Key Characteristics**: What are the defining traits based on the top events?
                                4. **Business Implications**: How should the business approach this customer segment?
                                5. **Potential Actions**: What specific strategies or tactics would be most effective?
                                
                                CONTEXT CONSIDERATIONS:
                                - Focus on actionable business insights
                                - Consider the relative size of each cluster (larger clusters may be more strategically important)
                                - Interpret event patterns in terms of customer journey stages, preferences, or behaviors
                                - Suggest practical applications for marketing, product development, or customer experience
                                - Keep interpretations concise but insightful
                                
                                Format your response with clear headers for each cluster and bullet points for easy reading.
                                """
                                
                                # Call Snowflake Cortex for cluster interpretation
                                interpretation_query = f"""
                                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                    '{interpretation_model}',
                                    '{interpretation_prompt.replace("'", "''")}'
                                ) as interpretation
                                """
                                
                                interpretation_result = session.sql(interpretation_query).collect()
                                
                                if interpretation_result and len(interpretation_result) > 0:
                                    interpretation = interpretation_result[0]['INTERPRETATION']
                                    
                                    st.markdown("**AI Cluster Interpretation & Business Insights**")
                                    st.markdown(interpretation)
                                else:
                                    st.error("Failed to generate cluster interpretation", icon=":material/chat_error:")
                                    
                            except Exception as e:
                                st.error(f"Error generating cluster interpretation: {str(e)}", icon=":material/chat_error:")
                
                
                # Call the improved writeback function
                add_writeback_functionality_segments_improved(results_df, clustering_method, customer_id_col, event_seq_col, seq_length_col, unique_events_col)
            
            else:
                st.info("No results yet. Click the 'Run Clustering Analysis' button above to start.", icon=":material/info:")

# Show cached results when not running new clustering
if not run_button and has_cached_results and all([database, schema, table, uid, evt, tmstp]):
    # Show tabs with cached results
    tab1, tab2 = st.tabs(["Analysis Process", "Results"])
    
    with tab1:
        # Check if parameters have changed since last analysis
        try:
            params_changed, changed_param_list = parameters_changed()
            
            if params_changed:
                # Merged message: parameters changed warning with call to action
                st.warning("**Parameters Changed** - Click 'Run Clustering Analysis' button above to update results with current parameters", icon=":material/warning:")
                
                # Show what changed
                with st.expander("See What Changed", expanded=False,icon=":material/visibility:"):
                    st.markdown("Changed parameters since last analysis:")
                    for param in changed_param_list:
                        # Make parameter names more user-friendly
                        friendly_names = {
                            'clustering_method': 'Clustering Algorithm',
                            'feature_weighting': 'Feature Weighting Strategy', 
                            'n_clusters': 'Number of Clusters',
                            'pca_option': 'PCA Dimensionality Reduction',
                            'normalize_vectors': 'Normalize Customer Vectors',
                            'max_customers': 'Max Customers to Analyze',
                            'min_sequence_length': 'Min Events per Customer',
                            'max_sequence_length': 'Max Events per Customer',
                            'min_count': 'Min Event Frequency',
                            'vector_size': 'Vector Dimensions',
                            'window_size': 'Context Window',
                            'alpha': 'Alpha (Topic Concentration)',
                            'beta': 'Beta (Word Concentration)',
                            'n_topics': 'Number of Topics',
                            'eps': 'DBSCAN Epsilon',
                            'min_samples': 'DBSCAN Min Samples',
                            'days_back': 'Days Back',
                            'start_date': 'Start Date',
                            'end_date': 'End Date'
                        }
                        friendly_name = friendly_names.get(param, param.replace('_', ' ').title())
                        st.markdown(f"• {friendly_name}")
                
            else:
                # Parameters haven't changed - show normal success
                st.success("Analysis Up-to-Date - Results match current parameters", icon=":material/check:")
                st.info("Change any parameters above and click 'Run Clustering Analysis' button to re-run with new settings", icon=":material/lightbulb:")
                
        except Exception as e:
            # Fallback to original message if comparison fails
            st.success("Analysis completed! Results are cached and available in the Results tab.", icon=":material/check:")
            st.info("Tip: Change any parameters above to re-run the analysis with new settings.", icon=":material/lightbulb:")
        
    
    with tab2:
        # Display cached results (same code as in the main analysis block)
        if 'behavioral_results' in st.session_state and st.session_state['behavioral_results'].get('completed'):
            
            
            # Get results from session state
            results = st.session_state['behavioral_results']
            
            # Display results based on clustering method - simplified for LDA
            if results.get('clustering_method') == "Latent Dirichlet Allocation (LDA)":
                # Add LDA-specific quality metrics
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Quality Metrics</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
                
                # Check if we have LDA model results stored
                if 'lda_model' in results and 'topic_distributions' in results:
                    lda_model = results['lda_model']
                    topic_distributions = results['topic_distributions']
                    
                    # Calculate LDA-specific metrics
                    try:
                        # Validate dimensions first to prevent index errors
                        X_shape = results['X'].shape if 'X' in results else (0, 0)
                        td_shape = topic_distributions.shape
                        comp_shape = lda_model.components_.shape
                        n_topics_actual = lda_model.n_components
                        
                        # Check for dimension mismatches that could cause index errors
                        if td_shape[1] != n_topics_actual:
                            st.warning(f"Topic distribution mismatch: got {td_shape[1]} topics, expected {n_topics_actual}", icon=":material/warning:")
                        
                        if comp_shape[0] != n_topics_actual:
                            st.warning(f"Components shape mismatch: got {comp_shape[0]} topics, expected {n_topics_actual}", icon=":material/warning:")
                        
                        # Perplexity (lower is better)
                        perplexity = lda_model.perplexity(results['X'])
                        
                        # Log-likelihood (higher is better)
                        log_likelihood = lda_model.score(results['X'])
                        
                        # Topic coherence (measure topic quality)
                        # Calculate average within-topic probability with explicit bounds checking
                        try:
                            if comp_shape[0] > 0 and comp_shape[1] > 0:
                                # Additional safety: iterate safely through components
                                topic_means = []
                                for i, topic in enumerate(lda_model.components_):
                                    if i < comp_shape[0] and len(topic) == comp_shape[1]:
                                        topic_means.append(np.mean(topic))
                                avg_topic_coherence = np.mean(topic_means) if topic_means else 0.0
                            else:
                                avg_topic_coherence = 0.0
                        except (IndexError, ValueError) as e:
                            avg_topic_coherence = 0.0
                            st.warning(f"Topic coherence calculation failed: {str(e)}", icon=":material/warning:")
                        
                        # Topic diversity (how different topics are from each other)
                        # Add safety check for components shape
                        components_shape = lda_model.components_.shape
                        if components_shape[1] > 0 and components_shape[0] > 0:  # Ensure we have topics and features
                            topic_diversity = np.mean(np.std(lda_model.components_, axis=0))
                        else:
                            topic_diversity = 0.0
                        
                        # Customer assignment clarity (how clearly customers are assigned to topics)
                        # Ensure we have valid topic distributions
                        if topic_distributions.size > 0 and topic_distributions.shape[1] > 0:
                            max_probs = np.max(topic_distributions, axis=1)
                            assignment_clarity = np.mean(max_probs)
                        else:
                            assignment_clarity = 0.0
                        
                        # Display LDA quality metrics
                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Perplexity", f"{perplexity:.1f}", 
                                    help="Model complexity (lower is better, typically 50-500)")
                            col2.metric("Log-Likelihood", f"{log_likelihood:.1f}", 
                                    help="Model fit quality (higher is better)")
                            col3.metric("Topic Coherence", f"{avg_topic_coherence:.3f}", 
                                    help="Topic quality (higher is better, 0-1 scale)")
                            col4.metric("Assignment Clarity", f"{assignment_clarity:.3f}", 
                                    help="Customer-topic assignment strength (higher is better, 0-1 scale)")
                        
                    except Exception as e:
                        st.error(f"Error calculating LDA metrics: {str(e)}", icon=":material/chat_error:")
                        
                    # AI Insights for LDA
                    def lda_ai_analysis_callback(selected_model):
                        with st.spinner("Analyzing LDA quality metrics with AI..."):
                            try:
                                # Get current LDA configuration
                                current_topics = results.get('n_clusters_found', 'Unknown')
                                current_weighting = results.get('feature_weighting', 'Unknown')
                                pca_info_lda = results.get('pca_info', 'No PCA applied')
                                
                                ai_prompt = f"""
                                You are analyzing LDA (Latent Dirichlet Allocation) topic modeling results from a Streamlit behavioral segmentation application. Provide specific recommendations based on current settings and available controls.

                                CURRENT LDA CONFIGURATION:
                                - Clustering Method: {results.get('clustering_method')}
                                - Feature Weighting: {current_weighting}
                                - Total Customers: {len(results.get('customer_ids', []))}
                                - Number of Topics: {current_topics}
                                - PCA Setting: {pca_info_lda}
                                
                                QUALITY METRICS:
                                - Perplexity: {perplexity:.1f} (lower is better, typical range 50-500)
                                - Log-Likelihood: {log_likelihood:.1f} (higher is better)
                                - Topic Coherence: {avg_topic_coherence:.3f} (higher is better, 0-1 scale)
                                - Assignment Clarity: {assignment_clarity:.3f} (higher is better, 0-1 scale)

                                AVAILABLE LDA CONTROLS IN THE APPLICATION:
                                - Number of Topics: 3-20 (slider control)
                                - Alpha (Topic Concentration): 0.01-2.0 (affects how many topics per customer)
                                - Beta (Word Concentration): 0.01-2.0 (affects how specific topics are)
                                - Feature Weighting: TF-IDF, Binary Presence, Count Frequency, Raw Frequency, Log Frequency, Smart Filtering
                                - Alternative Methods: K-Means, DBSCAN, Gaussian Mixture, Hierarchical (use Vector Dimensions 50-300, Context Window 2-10)

                                Please provide:
                                1. **LDA Quality Assessment**: How do these metrics rate for topic modeling with {current_topics} topics?
                                2. **PCA Consideration**: Note that LDA uses discrete token counts and PCA is not applicable ({pca_info_lda}). If PCA was applied, explain why it should be "No PCA" for LDA.
                                3. **LDA Parameter Tuning**: Specific alpha/beta/topic count adjustments using the available sliders
                                4. **Feature Engineering**: Should the user switch from "{current_weighting}" for better topic discovery?
                                5. **Alternative Approaches**: Should they try non-LDA clustering methods for this data? If so, what PCA strategy would be optimal for those methods?
                                6. **Business Application**: How can these {current_topics} topics be used for customer segmentation?
                                7. **Next Actions**: Specific UI control adjustments to improve topic quality, including correct PCA setting

                                IMPORTANT LDA-PCA CONTEXT:
                                - LDA should always use "No PCA" as it works on discrete document-term matrices
                                - If considering alternative clustering methods, recommend appropriate PCA strategies:
                                  * K-Means/Hierarchical: Auto PCA (~90% variance) or Manual 15-30 components
                                  * DBSCAN: Auto PCA (~95% variance) or Manual 20-40 components  
                                  * Gaussian Mixture: Auto PCA (~85% variance) or Manual 10-25 components

                                Focus on LDA optimization and appropriate PCA guidance if suggesting method switches.
                                """
                                
                                # Use Snowflake Cortex for analysis with selected model
                                ai_sql = f"""
                                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                    '{selected_model}',
                                    $${ai_prompt}$$
                                ) as insights
                                """
                                
                                ai_result = session.sql(ai_sql).collect()
                                if ai_result:
                                    insights = ai_result[0]['INSIGHTS']
                                    
                                    st.markdown("**AI Quality Analysis & Recommendations**")
                                    st.markdown(insights)
                                
                            except Exception as e:
                                st.warning(f"AI insights not available: {str(e)}", icon=":material/warning:")
                    
                    ai_model_lda, ai_enabled_lda = display_ai_insights_section(
                        "ai_model_lda", 
                        "Select the LLM model for AI analysis (dynamically fetched from Snowflake)",
                        ai_content_callback=lda_ai_analysis_callback
                    )
                else:
                    st.warning("LDA model results not found in session state. Please re-run the analysis.", icon=":material/warning:")
                
            else:
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Quality Metrics</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
                
                # Use stored results
                valid_mask = results.get('valid_mask')
                n_valid_samples = results.get('n_valid_samples', 0)
                n_clusters_found = results.get('n_clusters_found', 0)
                
                if valid_mask is not None and n_valid_samples > 1 and n_clusters_found > 1:
                    try:
                        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                        
                        X = results['X']
                        labels = results['labels']
                        X_valid = X[valid_mask]
                        labels_valid = labels[valid_mask]
                        
                        # Calculate metrics
                        silhouette = silhouette_score(X_valid, labels_valid)
                        calinski = calinski_harabasz_score(X_valid, labels_valid)
                        davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
                        
                        # Display quality metrics
                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with st.container(border=True):
                                col1.metric("Silhouette Score", f"{silhouette:.3f}", 
                                        help="Cluster separation quality (-1 to 1, higher is better)")
                                col2.metric("Calinski-Harabasz", f"{calinski:.1f}", 
                                        help="Cluster density ratio (higher is better)")
                                col3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", 
                                        help="Cluster similarity measure (lower is better)")
                                col4.metric("Valid Customers", f"{n_valid_samples:,}", 
                                        help="Customers successfully clustered")
                            
                        
                    except Exception as e:
                        st.error(f"Error calculating metrics: {str(e)}", icon=":material/chat_error:")
                else:
                    st.warning("Insufficient data for clustering quality metrics", icon=":material/warning:")
                
                # AI Insights for clustering quality (non-LDA methods) - cached results
                def cached_ai_analysis_callback(selected_model):
                    with st.spinner("Analyzing clustering quality metrics with AI..."):
                        try:
                            # Prepare metrics data for AI analysis
                            if valid_mask is not None and n_valid_samples > 1 and n_clusters_found > 1:
                                # We have calculated metrics
                                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                
                                X = results['X']
                                labels = results['labels']
                                X_valid = X[valid_mask]
                                labels_valid = labels[valid_mask]
                                
                                # Calculate metrics for AI analysis
                                silhouette = silhouette_score(X_valid, labels_valid)
                                calinski = calinski_harabasz_score(X_valid, labels_valid)
                                davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
                                
                                metrics_info = f"""
                                Clustering Method: {results.get('clustering_method')}
                                Feature Weighting: {results.get('feature_weighting')}
                                Total Customers: {len(results.get('customer_ids', []))}
                                Number of Clusters: {results.get('n_clusters_found')}
                                
                                Quality Metrics:
                                - Silhouette Score: {silhouette:.3f} (range -1 to 1, higher is better)
                                - Calinski-Harabasz Index: {calinski:.1f} (higher is better)
                                - Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)
                                - Valid Customers Clustered: {n_valid_samples:,}
                                """
                            else:
                                metrics_info = f"""
                                Clustering Method: {results.get('clustering_method')}
                                Feature Weighting: {results.get('feature_weighting')}
                                Total Customers: {len(results.get('customer_ids', []))}
                                Number of Clusters: {results.get('n_clusters_found')}
                                
                                Note: Insufficient data for detailed quality metrics calculation.
                                """
                            
                            # Get current configuration for cached results
                            current_method = results.get('clustering_method', 'Unknown')
                            current_weighting = results.get('feature_weighting', 'Unknown')
                            current_clusters = results.get('n_clusters_found', 'Unknown')
                            pca_info_cached = results.get('pca_info', 'No PCA applied')
                            
                            ai_prompt = f"""
                            You are analyzing cached behavioral clustering results from a Streamlit application. Provide specific recommendations based on the previously run configuration and available options for re-running the analysis.

                            PREVIOUS CONFIGURATION RESULTS:
                            {metrics_info}

                            SETTINGS USED IN PREVIOUS RUN:
                            - Clustering Method: {current_method}
                            - Feature Weighting: {current_weighting}
                            - Number of Clusters: {current_clusters}
                            - Vector Dimensions: {vector_size}
                            - Context Window: {window_size}
                            - Normalize Vectors: {normalize_vectors}
                            - PCA Configuration: {pca_info_cached}

                            AVAILABLE OPTIONS FOR NEW ANALYSIS:
                            - Clustering Methods: K-Means, DBSCAN, Gaussian Mixture, Hierarchical, Latent Dirichlet Allocation (LDA)
                            - Feature Weighting: TF-IDF, Binary Presence, Count Frequency, Raw Frequency, Log Frequency, Smart Filtering
                            - Number of Clusters: 3-20 (for K-Means, Gaussian Mixture, Hierarchical)
                            - Vector Dimensions: 50-300 (dimensionality of event embeddings)
                            - Context Window: 2-10 (surrounding events for context in embeddings)
                            - Normalize Vectors: Yes/No (normalize to unit length)
                            - DBSCAN Parameters: eps (0.1-2.0), min_samples (2-20)
                            - LDA Parameters: alpha (0.01-2.0), beta (0.01-2.0), number of topics (3-20)
                            - PCA Options: No PCA, Auto (Recommended), Manual (1-50 components)

                            Please provide:
                            1. **Previous Results Assessment**: How did {current_method} with "{current_weighting}" and "{pca_info_cached}" perform?
                            2. **PCA Strategy Review**: Was the previous PCA choice optimal for {current_method}? Recommend specific PCA improvements for re-run.
                            3. **Re-run Recommendations**: Should the user run a new analysis with different settings including PCA adjustments?
                            4. **Method Comparison**: Would a different clustering method work better than {current_method}? What PCA strategy for each alternative?
                            5. **Parameter Optimization**: What specific parameter changes would improve results (including PCA settings)?
                            6. **Data Insights**: Are there data characteristics affecting {current_method} performance? How would PCA help?
                            7. **Next Analysis Steps**: Specific UI settings to try in the Analysis Process tab, including optimal PCA choice

                            CACHED RESULTS PCA GUIDANCE:
                            - Evaluate if previous PCA choice was appropriate for the clustering method used
                            - For method switches, recommend optimal PCA strategies:
                              * K-Means/Hierarchical: Auto PCA or Manual 15-30 components  
                              * DBSCAN: Auto PCA or Manual 20-40 components
                              * Gaussian Mixture: Auto PCA or Manual 10-25 components
                              * LDA: Always "No PCA"
                            - Consider dimensionality issues if "No PCA" was used with high-dimensional data

                            Focus on actionable recommendations including specific PCA strategies for improved clustering.
                            """
                            
                            # Use Snowflake Cortex for analysis with selected model
                            ai_sql = f"""
                            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                '{selected_model}',
                                $${ai_prompt}$$
                            ) as insights
                            """
                            
                            ai_result = session.sql(ai_sql).collect()
                            if ai_result:
                                insights = ai_result[0]['INSIGHTS']
                                
                                st.markdown("**AI Quality Analysis & Recommendations**")
                                st.markdown(insights)
                                
                        except Exception as e:
                            st.warning(f"AI insights not available: {str(e)}", icon=":material/warning:")
                
                ai_model_cached, ai_enabled_cached = display_ai_insights_section(
                    "ai_model_cached", 
                    "Select the LLM model for AI analysis (dynamically fetched from Snowflake)",
                    ai_content_callback=cached_ai_analysis_callback
                )
            
            # Create results DataFrame with cluster assignments using session state data
            results_df = pd.DataFrame({
                'customer_id': results['customer_ids'],
                'cluster': results['labels']
            })
            
            # Merge with original sequence data for analysis
            customer_id_col = results['customer_id_col']
            event_seq_col = results['event_seq_col']
            seq_length_col = results['seq_length_col']
            unique_events_col = results['unique_events_col']
            sequences_df = results['sequences_df']
            clustering_method = results['clustering_method']
            
            merge_cols = [customer_id_col, event_seq_col, seq_length_col, unique_events_col]
            results_df = results_df.merge(
                sequences_df[merge_cols], 
                left_on='customer_id', 
                right_on=customer_id_col
            )
            
            # Cluster Distribution
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Clusters Distribution</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            
            with st.container(border=True):
                    # Simple cluster distribution chart - sorted by customer count descending
                    cluster_counts = results_df['cluster'].value_counts().sort_values(ascending=False)  # Sort by count descending
                    cluster_percentages = (cluster_counts / len(results_df) * 100).round(1)
                    
                    # Create bar chart with colors aligned to treemap (Blues color scale)
                    # Darker colors for higher customer counts (sorted order)
                    import plotly.colors
                    n_colors = len(cluster_counts)
                    if n_colors > 1:
                        # Use Blues color scale with darker colors for higher values
                        color_scale = plotly.colors.sequential.Blues
                        # Reverse the mapping so first (highest) gets darkest color
                        bar_colors = [color_scale[-(int(i * (len(color_scale)-1) / (n_colors-1)) + 1)] for i in range(n_colors)]
                    else:
                        bar_colors = ['#29B5E8']  # Single color if only one cluster
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=[f"Cluster {i}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {i}" for i in cluster_counts.index],
                        y=cluster_counts.values,
                        text=[f"{count:,}<br>({pct}%)" for count, pct in zip(cluster_counts.values, cluster_percentages.values)],
                        textposition='auto',
                        marker_color=bar_colors
                    ))
                    
                    fig_bar.update_layout(
                        height=400,
                        showlegend=False,
                        xaxis_title="Clusters" if clustering_method != "Latent Dirichlet Allocation (LDA)" else "Topics",
                        yaxis_title="Number of Customers"
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Add Treemap Visualization below the bar chart (cached results)
            
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Clusters & Events Hierarchy Distribution</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            # Add option to show all events vs top events (cached results)
            
            show_all_events_cached = st.checkbox("Show All Events", value=False, key="show_all_events_cached", help="Uncheck to show only top 5 events per cluster")
            
            try:
                # Create multi-level treemap using Plotly Express path approach (cached results)
                # Build a dataframe suitable for px.treemap with path parameter
                
                # Prepare data for multi-level treemap
                treemap_data = []
                
                # Analyze events within each cluster first
                cluster_event_analysis = {}
                for cluster_id in cluster_counts.index:
                    cluster_customers = results_df[results_df['cluster'] == cluster_id]
                    
                    # Extract all events from customers in this cluster
                    all_events_in_cluster = []
                    for _, row in cluster_customers.iterrows():
                        events = row[event_seq_col].split(' → ')
                        all_events_in_cluster.extend(events)
                    
                    # Count event frequencies within this cluster
                    if len(all_events_in_cluster) > 0:
                        event_counts_in_cluster = pd.Series(all_events_in_cluster).value_counts()
                        cluster_event_analysis[cluster_id] = event_counts_in_cluster
                
                # Build hierarchical data for px.treemap using path parameter
                for cluster_id in cluster_counts.index:
                    cluster_name = f"Cluster {cluster_id}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {cluster_id}"
                    cluster_count = cluster_counts[cluster_id]
                    cluster_percentage = cluster_percentages[cluster_id]
                    
                    # Enhanced cluster name with percentage
                    cluster_display_name = f"{cluster_name} ({cluster_count:,} customers - {cluster_percentage}%)"
                    
                    if cluster_id in cluster_event_analysis:
                        # Get events for this cluster based on user preference
                        if show_all_events_cached:
                            events_to_show = cluster_event_analysis[cluster_id]
                        else:
                            events_to_show = cluster_event_analysis[cluster_id].head(5)
                        
                        cluster_total_events = cluster_event_analysis[cluster_id].sum()
                        
                        for event_name, event_count in events_to_show.items():
                            event_percentage = (event_count / cluster_total_events * 100)
                            
                            # Add each event as a row in our treemap data
                            treemap_data.append({
                                'all': 'All Customers',  # Root level
                                'cluster': cluster_display_name,  # Second level with enhanced info
                                'event': event_name,     # Third level
                                'count': event_count,    # Values for sizing
                                'customers': cluster_count,  # For color coding
                                'cluster_percentage': cluster_percentage,  # For tooltips
                                'event_percentage': round(event_percentage, 1),  # Event % within cluster
                                'cluster_total_events': cluster_total_events  # For context
                            })
                    else:
                        # If no events, just add cluster data
                        treemap_data.append({
                            'all': 'All Customers',
                            'cluster': cluster_display_name,
                            'event': 'No Events',
                            'count': 1,  # Minimal size
                            'customers': cluster_count,
                            'cluster_percentage': cluster_percentage,
                            'event_percentage': 0,
                            'cluster_total_events': 0
                        })
                
                # Convert to DataFrame
                treemap_df = pd.DataFrame(treemap_data)
                
                # Create treemap using px.treemap with path parameter
                if len(treemap_df) > 0:
                    try:
                        # Create discrete color map matching bar chart EXACTLY
                        import plotly.colors
                        
                        # Extract cluster numbers from display names  
                        treemap_df['cluster_num'] = treemap_df['cluster'].str.extract(r'Cluster (\d+)')[0].astype(int)
                        
                        # Get cluster-to-customer mapping and sort by customer count descending (SAME AS BAR CHART)
                        cluster_info = treemap_df[['cluster', 'cluster_num', 'customers']].drop_duplicates('cluster')
                        cluster_info = cluster_info.sort_values('customers', ascending=False).reset_index(drop=True)
                        
                        n_colors = len(cluster_info)
                        
                        # Safety check
                        if n_colors == 0:
                            st.warning("No clusters found for treemap visualization.", icon=":material/warning:")
                        else:
                            color_scale = plotly.colors.sequential.Blues
                            
                            # Use EXACT SAME color assignment as bar chart
                            if n_colors > 1:
                                bar_colors = [color_scale[-(int(i * (len(color_scale)-1) / (n_colors-1)) + 1)] for i in range(n_colors)]
                            else:
                                bar_colors = ['#29B5E8']
                            
                            # Map: cluster display name → bar chart color (in sorted order)
                            color_discrete_map = dict(zip(cluster_info['cluster'], bar_colors))
                            
                            # Add customer count to treemap_df and sort by it to match bar chart order
                            treemap_df = treemap_df.merge(cluster_info[['cluster', 'customers']], on='cluster', how='left', suffixes=('', '_merge'))
                            treemap_df = treemap_df.sort_values(['customers', 'count'], ascending=[False, False])
                            
                            fig_treemap = px.treemap(
                                treemap_df,
                                path=[px.Constant("All Customers"), 'cluster', 'event'],
                                values='count',
                                color='cluster',
                                color_discrete_map=color_discrete_map,
                                branchvalues='total'  # Size clusters by total of their children
                            )
                            
                            # Customize appearance and set root transparent
                            fig_treemap.update_traces(
                                textfont_size=10,
                                marker=dict(line=dict(width=1)),
                                hovertemplate='<b>%{label}</b><br>Event Frequency: %{value:,}<br><extra></extra>'
                            )
                            
                            # Override just the root color to transparent
                            fig_treemap.data[0].marker.colors = [
                                'rgba(0,0,0,0)' if label == 'All Customers' else fig_treemap.data[0].marker.colors[i]
                                for i, label in enumerate(fig_treemap.data[0].labels)
                            ]
                            
                            fig_treemap.update_layout(
                                height=600,
                                margin=dict(t=50, l=0, r=0, b=0),
                                font=dict(size=10)
                            )
                            
                            with st.container(border=True):
                                st.plotly_chart(fig_treemap, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate treemap visualization: {str(e)}", icon=":material/error:")
                        st.info("Treemap requires additional data processing. Bar chart above shows the same information.", icon=":material/info:")
                    
                    # Show summary of what's displayed
                    if show_all_events_cached:
                        total_events_shown = len(treemap_df[treemap_df['event'] != 'No Events'])
                        st.info(f"Displaying all {total_events_shown} unique events across {len(cluster_counts)} clusters", icon=":material/info:")
                    else:
                        st.info("Displaying top 5 events per cluster. Check 'Show All Events' to see complete data.", icon=":material/info:")
                    
                else:
                    # Fallback to simple treemap if data preparation fails
                    fig_treemap = go.Figure(go.Treemap(
                        labels=[f"Cluster {i}" for i in cluster_counts.index],
                        parents=["All Customers"] * len(cluster_counts),
                        values=cluster_counts.values,
                        textinfo="label",
                        textfont=dict(size=12),
                        marker=dict(colorscale='Blues'),
                        root_color="lightgrey"
                    ))
                    
                    fig_treemap.update_layout(
                        height=500,
                        margin=dict(t=50, l=0, r=0, b=0)
                    )
                    
                    st.plotly_chart(fig_treemap, use_container_width=True)
                
                # Add explanation for enhanced treemap (cached results)
                with st.expander("Understanding the Enhanced Treemap"):
                    st.markdown("""
                    **How to read this enhanced treemap:**
                    - **3 Levels**: Root (All Customers) → Clusters → Events within each cluster
                    - **Size**: Rectangle area represents customer count (clusters) or event frequency (events)
                    - **Color**: Shades of blue distinguish clusters; slightly different tones show events within clusters
                    - **Hierarchy**: Click or hover to explore cluster composition and dominant events
                    
                    **Key insights:**
                    - **Cluster Level**: Larger rectangles = clusters with more customers (dominant segments)
                    - **Event Level**: Shows events within each cluster with their frequency
                    - **Behavioral Patterns**: Quickly identify which events define each behavioral segment
                    - **Event Distribution**: See how event frequency varies across different customer clusters
                    
                    **Navigation Tips:**
                    - Hover over any rectangle to see detailed information including exact percentages
                    - Compare event patterns between different clusters
                    - Use the 'Show All Events' checkbox to see complete or filtered data
                    - Identify clusters dominated by specific event types
                    """)
                    
            except Exception as e:
                st.warning(f"Could not generate treemap visualization: {str(e)}", icon=":material/warning:")
                st.info("Treemap requires additional data processing. Bar chart above shows the same information.", icon=":material/info:")
            
            # AI Cluster Interpretation Section
            with st.expander("AI Cluster Interpretation", expanded=False, icon=":material/network_intel_node:"):
                st.markdown("**Analyze cluster behaviors and business implications**")
                
                # Model selection for cluster interpretation
                models_data = get_available_cortex_models()
                if models_data["status"] == "found":
                    available_models = models_data["models"]
                else:
                    available_models = ["mixtral-8x7b", "mistral-large", "llama3-8b", "llama3-70b", "gemma-7b"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    interpretation_model = st.selectbox(
                        "Choose AI Model for Interpretation", 
                        available_models, 
                        index=0,
                        key="cluster_interpretation_model_tab1"
                    )
                    
                    interpret_clusters = st.toggle(
                        "Interpret Clusters", 
                        key="interpret_clusters_toggle_tab1"
                    )
                # with col2 & col3: (left empty for alignment with other sections)
                
                if interpret_clusters:
                    with st.spinner("Analyzing cluster behaviors and business context..."):
                        try:
                            # Prepare cluster analysis data for AI
                            cluster_analysis_text = ""
                            
                            for cluster_id in cluster_counts.index:
                                cluster_name = f"Cluster {cluster_id}" if clustering_method != "Latent Dirichlet Allocation (LDA)" else f"Topic {cluster_id}"
                                cluster_count = cluster_counts[cluster_id]
                                cluster_percentage = cluster_percentages[cluster_id]
                                
                                cluster_analysis_text += f"\n**{cluster_name}:**\n"
                                cluster_analysis_text += f"- Size: {cluster_count:,} customers ({cluster_percentage}% of total)\n"
                                
                                if cluster_id in cluster_event_analysis:
                                    top_events = cluster_event_analysis[cluster_id].head(10)  # Top 10 events for AI analysis
                                    cluster_analysis_text += f"- Top Events:\n"
                                    for event_name, event_count in top_events.items():
                                        event_pct = (event_count / cluster_event_analysis[cluster_id].sum() * 100)
                                        cluster_analysis_text += f"  • {event_name}: {event_count} occurrences ({event_pct:.1f}%)\n"
                                else:
                                    cluster_analysis_text += "- No significant events identified\n"
                            
                            # Create AI prompt for cluster interpretation
                            interpretation_prompt = f"""
                            You are a business analyst specializing in customer behavioral segmentation. Analyze the following customer clusters and provide business insights.
                            
                            CLUSTERING ANALYSIS RESULTS:
                            - Method Used: {clustering_method}
                            - Feature Weighting: {feature_weighting}
                            - Total Customers Analyzed: {len(results_df):,}
                            - Number of Clusters/Topics: {len(cluster_counts)}
                            
                            CLUSTER BREAKDOWN:
                            {cluster_analysis_text}
                            
                            ANALYSIS INSTRUCTIONS:
                            Please provide a comprehensive interpretation for each cluster/topic with the following structure:
                            
                            For each cluster, provide:
                            1. **Behavioral Profile**: What type of customer behavior does this cluster represent?
                            2. **Business Significance**: What does this cluster mean from a business perspective?
                            3. **Key Characteristics**: What are the defining traits based on the top events?
                            4. **Business Implications**: How should the business approach this customer segment?
                            5. **Potential Actions**: What specific strategies or tactics would be most effective?
                            
                            CONTEXT CONSIDERATIONS:
                            - Focus on actionable business insights
                            - Consider the relative size of each cluster (larger clusters may be more strategically important)
                            - Interpret event patterns in terms of customer journey stages, preferences, or behaviors
                            - Suggest practical applications for marketing, product development, or customer experience
                            - Keep interpretations concise but insightful
                            
                            Format your response with clear headers for each cluster and bullet points for easy reading.
                            """
                            
                            # Call Snowflake Cortex for cluster interpretation
                            interpretation_query = f"""
                            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                '{interpretation_model}',
                                '{interpretation_prompt.replace("'", "''")}'
                            ) as interpretation
                            """
                            
                            interpretation_result = session.sql(interpretation_query).collect()
                            
                            if interpretation_result and len(interpretation_result) > 0:
                                interpretation = interpretation_result[0]['INTERPRETATION']
                                
                                st.markdown("**AI Cluster Interpretation & Business Insights**")
                                st.markdown(interpretation)
                            else:
                                st.error("Failed to generate cluster interpretation", icon=":material/chat_error:")
                                
                        except Exception as e:
                            st.error(f"Error generating cluster interpretation: {str(e)}", icon=":material/chat_error:")
                
            
            # Call the improved writeback function
            add_writeback_functionality_segments_improved(results_df, clustering_method, customer_id_col, event_seq_col, seq_length_col, unique_events_col)
            
        else:
            st.info("No cached results found. Use the toggle above to run the analysis.", icon=":material/info:")

else:
    # Show parameter requirements message when not all parameters are set
    if not all([database, schema, table, uid, evt, tmstp]):
        pass  # The main parameter message is already shown above

 