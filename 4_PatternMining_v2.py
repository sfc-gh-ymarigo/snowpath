# Import python packages
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import datetime 
import uuid
import altair as alt
import re
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast


# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

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

#===================================================================================
# Cached Parameter Query Functions (to avoid repeated queries)
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

# Function to display AI insights UI with model selection and toggle
def display_ai_insights_section(key_prefix, help_text="Select the LLM model for AI analysis", ai_content_callback=None):
    """Display AI insights section with model selection UI and toggle in an expander"""
    
    with st.expander("AI Insights & Recommendations", expanded=False, icon=":material/network_intel_node:"):
        models_result = get_available_cortex_models()
        available_models = models_result["models"]
        status = models_result["status"]
        
        # Show status message if needed
        if status == "not_found":
            st.warning("No Cortex models found in your Snowflake account. Using default models.", icon=":material/warning:")
            
            # Add refresh option in expander to not clutter main UI
            with st.expander("Refresh Model List", icon=":material/refresh:"):
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
            ai_enabled = st.toggle("Exp**AI**n Me!", key=f"{key_prefix}_toggle", help="Generate AI insights and recommendations for your pattern mining results")
        
        # with col2 & col3: (left empty for alignment with other sections)
        
        # If AI is enabled and callback provided, execute the AI content within the expander
        if ai_enabled and ai_content_callback:
            ai_content_callback(selected_model)
        
        return selected_model, ai_enabled

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
                
                .custom-container-1 {
                    background-color: #f0f2f6 !important;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }

                .custom-container-1 h5 {
                    color: #0f0f0f !important;
                }
                
                /* Rounded corners for all message types */
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
        </style>
        """, unsafe_allow_html=True)

st.markdown("""
    <style>
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
            border: 1px solid #29B5E8 !important;
            background-color: transparent !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

#--------------------------------------
# PATTERN MINING PAGE
#--------------------------------------

# Page Header
st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
        PATTERN MINING
    </h5>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'pattern_results' not in st.session_state:
    st.session_state['pattern_results'] = None

# Global variables
fromevt = None
toevt = None
minnbbevt = 0
maxnbbevt = 5
overlap = 'TO NEXT ROW'  # Default as requested
uid = None
evt = None
tmstp = None
database = None
schema = None
tbl = None

#--------------------------------------
# HELPER FUNCTIONS
#--------------------------------------

def generate_pattern_sql(pattern_type, sequence_config, selected_events, uid, evt, tmstp, database, schema, tbl, 
                         start_date, end_date, sql_where_clause, max_gap_days=None, overlap='TO NEXT ROW', time_gap_config=None):
    """Generate SQL for different pattern mining configurations with optional time gap filtering"""
    
    # Base partition clause
    partitionby = f"partition by {uid}" if max_gap_days else f"partition by {uid}"
    
    # Base time filter with optional gap constraint
    time_constraint = f"and {tmstp} between DATE('{start_date}') and DATE('{end_date}')"
    
    # Initialize time gap configuration
    if time_gap_config is None:
        time_gap_config = {'enabled': False, 'show_gaps': False}
    
    # Max gap constraint for time windows (only needed if max_gap_days is set)
    gap_constraint = ""
    if max_gap_days:
        gap_constraint = f"""
        WITH events_with_gap AS (
            SELECT {uid}, {evt}, {tmstp},
                   LAG({tmstp}) OVER (PARTITION BY {uid} ORDER BY {tmstp}) as prev_timestamp,
                   DATEDIFF(day, LAG({tmstp}) OVER (PARTITION BY {uid} ORDER BY {tmstp}), {tmstp}) as day_gap
            FROM {database}.{schema}.{tbl}
            WHERE 1=1 {time_constraint} {sql_where_clause}
        ),
        filtered_events AS (
            SELECT {uid}, {evt}, {tmstp}
            FROM events_with_gap
            WHERE day_gap IS NULL OR day_gap <= {max_gap_days}
        )"""
        base_table = "filtered_events"
    else:
        gap_constraint = ""
        base_table = f"{database}.{schema}.{tbl}"
    
    # Build pattern and define clauses based on pattern type
    having_clause = ""  # Initialize having_clause
    
    # Build time gap constraints for DEFINE clause using LAG()
    time_gap_constraint = ""
    if time_gap_config.get('enabled', False):
        constraints = []
        if time_gap_config.get('filter_type') in ["Maximum gap", "Both"]:
            max_op = time_gap_config.get('max_gap_operator', '<=')
            max_val = time_gap_config.get('max_gap_value', 60)
            max_unit = time_gap_config.get('max_gap_unit', 'MINUTE')
            constraints.append(f"TIMESTAMPDIFF({max_unit}, LAG({tmstp}), {tmstp}) {max_op} {max_val}")
        
        if time_gap_config.get('filter_type') in ["Minimum gap", "Both"]:
            min_op = time_gap_config.get('min_gap_operator', '>=')
            min_val = time_gap_config.get('min_gap_value', 1)
            min_unit = time_gap_config.get('min_gap_unit', 'MINUTE')
            constraints.append(f"TIMESTAMPDIFF({min_unit}, LAG({tmstp}), {tmstp}) {min_op} {min_val}")
        
        if constraints:
            time_gap_constraint = " AND " + " AND ".join(constraints)
    
    if pattern_type == "All":
        # All combinations - generate all possible sequences
        pattern_clause = f"A{{{sequence_config['min']},{sequence_config['max']}}}"
        if time_gap_constraint:
            define_clause = f"define A as true{time_gap_constraint}"
        else:
            define_clause = "define A as true"
        
    elif pattern_type == "Contains":
        # Must contain specific events - use A{min,max} pattern with post-processing filter
        if sequence_config['size_type'] == 'exact':
            pattern_clause = f"A{{{sequence_config['exact']}}}"
        else:
            pattern_clause = f"A{{{sequence_config['min']},{sequence_config['max']}}}"
        
        if time_gap_constraint:
            define_clause = f"define A as true{time_gap_constraint}"
        else:
            define_clause = "define A as true"
        
        # Create having clause to filter sequences that contain all required events
        containment_checks = []
        for event in selected_events:
            containment_checks.append(f"path LIKE '%{event}%'")
        
        having_clause = f"HAVING {' AND '.join(containment_checks)}"
    
    elif pattern_type == "Does not contain":
        # Must NOT contain specific events - use A{min,max} pattern with exclusion filter
        if sequence_config['size_type'] == 'exact':
            pattern_clause = f"A{{{sequence_config['exact']}}}"
        else:
            pattern_clause = f"A{{{sequence_config['min']},{sequence_config['max']}}}"
        
        if time_gap_constraint:
            define_clause = f"define A as true{time_gap_constraint}"
        else:
            define_clause = "define A as true"
        
        # Create having clause to filter sequences that do NOT contain any of the selected events
        exclusion_checks = []
        for event in selected_events:
            exclusion_checks.append(f"path NOT LIKE '%{event}%'")
        
        having_clause = f"HAVING {' AND '.join(exclusion_checks)}"
            
    elif pattern_type == "Starts with":
        # Must start with specific event
        start_event = selected_events[0] if selected_events else None
        if sequence_config['size_type'] == 'exact':
            pattern_clause = f"A B{{{sequence_config['exact']-1}}}"
        else:
            min_following = max(0, sequence_config['min'] - 1)
            max_following = sequence_config['max'] - 1
            pattern_clause = f"A B{{{min_following},{max_following}}}"
        
        if time_gap_constraint:
            define_clause = f"define A as {evt} = '{start_event}', B as true{time_gap_constraint}"
        else:
            define_clause = f"define A as {evt} = '{start_event}', B as true"
        
    elif pattern_type == "Ends with":
        # Must end with specific event
        end_event = selected_events[0] if selected_events else None
        if sequence_config['size_type'] == 'exact':
            pattern_clause = f"A{{{sequence_config['exact']-1}}} B"
        else:
            min_preceding = max(0, sequence_config['min'] - 1)
            max_preceding = sequence_config['max'] - 1
            pattern_clause = f"A{{{min_preceding},{max_preceding}}} B"
        
        if time_gap_constraint:
            define_clause = f"define A as true{time_gap_constraint}, B as {evt} = '{end_event}'"
        else:
            define_clause = f"define A as true, B as {evt} = '{end_event}'"
    
    # Determine MEASURES and SELECT clauses based on time gap configuration
    if time_gap_config.get('enabled') and time_gap_config.get('show_gaps'):
        # Determine which unit to use for display (prefer max_gap_unit, fallback to min_gap_unit)
        display_unit = time_gap_config.get('max_gap_unit', time_gap_config.get('min_gap_unit', 'SECOND'))
        
        # Add time gap measurements using the selected unit
        measures_clause = f"""match_number() as MATCH_NUMBER,
                    match_sequence_number() as msq,
                    classifier() as cl,
                    {tmstp} as event_time,
                    TIMESTAMPDIFF({display_unit}, PREV({tmstp}), {tmstp}) as gap_value"""
        
        time_gap_select = """,
            AVG(avg_gap_value) as avg_gap_value,
            MIN(min_gap_value) as min_gap_value,
            MAX(max_gap_value) as max_gap_value"""
        
        inner_select_gap = """,
                AVG(gap_value) as avg_gap_value,
                MIN(gap_value) as min_gap_value,
                MAX(gap_value) as max_gap_value"""
    else:
        measures_clause = """match_number() as MATCH_NUMBER,
                    match_sequence_number() as msq,
                    classifier() as cl"""
        time_gap_select = ""
        inner_select_gap = ""
    
    # Build the complete SQL
    if max_gap_days:
        # When max gap is enabled, use the WITH clause structure
        base_query = f"""
        {gap_constraint}
        SELECT 
            path,
            COUNT(*) as frequency,
            COUNT(DISTINCT {uid}) as unique_users,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
            AVG(sequence_length) as avg_sequence_length,
            MIN(sequence_length) as min_sequence_length,
            MAX(sequence_length) as max_sequence_length{time_gap_select}
        FROM (
            SELECT 
                {uid},
                LISTAGG({evt}, ' → ') WITHIN GROUP (ORDER BY msq) as path,
                COUNT(*) as sequence_length{inner_select_gap}
            FROM {base_table}
            MATCH_RECOGNIZE(
                {partitionby}
                ORDER BY {tmstp}
                MEASURES 
                    {measures_clause}
                ALL ROWS PER MATCH
                AFTER MATCH SKIP {overlap}
                PATTERN({pattern_clause})
                {define_clause}
            )
            GROUP BY {uid}, MATCH_NUMBER
        )
        GROUP BY path
        {having_clause if 'having_clause' in locals() and having_clause else ''}
        ORDER BY FREQUENCY DESC
        """
    else:
        # When no max gap, use the simple structure
        base_query = f"""
        SELECT 
            path,
            COUNT(*) as frequency,
            COUNT(DISTINCT {uid}) as unique_users,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
            AVG(sequence_length) as avg_sequence_length,
            MIN(sequence_length) as min_sequence_length,
            MAX(sequence_length) as max_sequence_length{time_gap_select}
        FROM (
            SELECT 
                {uid},
                LISTAGG({evt}, ' → ') WITHIN GROUP (ORDER BY msq) as path,
                COUNT(*) as sequence_length{inner_select_gap}
            FROM (SELECT * FROM {base_table} WHERE 1=1 {time_constraint} {sql_where_clause})
            MATCH_RECOGNIZE(
                {partitionby}
                ORDER BY {tmstp}
                MEASURES 
                    {measures_clause}
                ALL ROWS PER MATCH
                AFTER MATCH SKIP {overlap}
                PATTERN({pattern_clause})
                {define_clause}
            )
            GROUP BY {uid}, MATCH_NUMBER
        )
        GROUP BY path
        {having_clause if 'having_clause' in locals() and having_clause else ''}
        ORDER BY FREQUENCY DESC
        """
    
    return base_query

def pattern_sankey_chart(results_df, pattern_type="All", direction="from", topN_percentage=100):
    """Create Sankey chart for pattern mining results"""
    dataDict = defaultdict(lambda: {"count": 0, "uids": []})
    eventDict = defaultdict(int)
    indexed_paths = []
    
    if results_df.empty:
        return False
    
    # Convert pattern mining results to sankey format
    df_sankey = results_df.copy()
    
    # Create UID_LIST from UNIQUE_USERS (simplified for visualization)
    df_sankey['UID_LIST'] = df_sankey.apply(lambda x: list(range(int(x['UNIQUE_USERS']))), axis=1)
    df_sankey['COUNT'] = df_sankey['FREQUENCY']
    
    # Select top N patterns
    max_patterns = len(df_sankey)
    topN = int(max_patterns * (topN_percentage / 100))
    if topN > 0:
        df_sankey = df_sankey.sort_values(by='COUNT', ascending=False).head(topN)
    
    # Determine direction based on pattern type
    if pattern_type == "Starts with":
        direction = "from"  # Left-aligned for patterns starting with specific event
    elif pattern_type == "Ends with":  
        direction = "to"    # Right-aligned for patterns ending with specific event
    elif pattern_type == "Contains":
        direction = "from"  # Left-aligned to show progression
    elif pattern_type == "Does not contain":
        direction = "from"  # Left-aligned to show progression
    else:  # "All"
        direction = "from"  # Default left-aligned
    
    if direction == "to":
        # Right-aligned (for "Ends with" patterns)
        maxCount = df_sankey['COUNT'].max()
        for _, row in df_sankey.iterrows():
            rowList = row['PATH'].split(' → ')
            pathCnt = row['COUNT']
            uid_list = row['UID_LIST']
            pathLen = len(rowList)
            indexedRowList = [f"{150 + i + maxCount - pathLen}_{rowList[i].strip()}" for i in range(len(rowList))]
            indexed_paths.append(",".join(indexedRowList))
            for i in range(len(indexedRowList) - 1):
                leftValue = indexedRowList[i]
                rightValue = indexedRowList[i + 1]
                valuePair = leftValue + '|||' + rightValue
                dataDict[valuePair]["count"] += pathCnt
                dataDict[valuePair]["uids"].extend(uid_list)
                eventDict[leftValue] += pathCnt
                eventDict[rightValue] += pathCnt
        
        # Create tooltips for right-aligned
        for key, val in dataDict.items():
            source_node, target_node = key.split('|||')
            tooltip_text = f"""
                Pattern Flow: {source_node.split('_', 1)[1]} → {target_node.split('_', 1)[1]}<br>
                Frequency: {val["count"]}<br>
                Pattern Type: {pattern_type}
            """
            val["tooltip"] = tooltip_text
            
    else:
        # Left-aligned (for "Starts with", "Contains", and "All" patterns)
        for _, row in df_sankey.iterrows():
            rowList = row['PATH'].split(' → ')
            pathCnt = row['COUNT']
            uid_list = row['UID_LIST']
            indexedRowList = [f"{i}_{rowList[i].strip()}" for i in range(len(rowList))]
            indexed_paths.append(",".join(indexedRowList))
            for i in range(len(indexedRowList) - 1):
                leftValue = indexedRowList[i]
                rightValue = indexedRowList[i + 1]
                valuePair = leftValue + '|||' + rightValue
                dataDict[valuePair]["count"] += pathCnt
                dataDict[valuePair]["uids"].extend(uid_list)
                eventDict[rightValue] += pathCnt
        
        # Calculate flow percentages for left-aligned
        dropoffDict = defaultdict(int)
        for node in eventDict:
            total_at_node = eventDict[node]
            outgoing = sum(dataDict[f"{node}|||{target}"]["count"] for target in eventDict if f"{node}|||{target}" in dataDict)
            dropoff = total_at_node - outgoing
            dropoffDict[node] = dropoff
        
        # Create tooltips for left-aligned with flow percentages
        for key, val in dataDict.items():
            source_node, target_node = key.split('|||')
            total_at_source = eventDict[source_node]
            forward_percentage = (val["count"] / total_at_source * 100) if total_at_source > 0 else 0
            dropoff_percentage = (dropoffDict[source_node] / total_at_source * 100) if total_at_source > 0 else 0
            val["tooltip"] = f"""
                Pattern Flow: {source_node.split('_', 1)[1]} → {target_node.split('_', 1)[1]}<br>
                Frequency: {val["count"]}<br>
                Forward %: {forward_percentage:.2f}%<br>
                Drop-off %: {dropoff_percentage:.2f}%<br>
                Pattern Type: {pattern_type}
            """
    
    if not dataDict:
        return False  # Return False instead of None to indicate failure
    
    # Build the sankey chart data
    sortedEventList = sorted(eventDict.keys())
    sankeyLabel = [event.split('_', 1)[1] for event in sortedEventList]
    
    sankeyLinks = []
    for key, val in dataDict.items():
        source_node, target_node = key.split('|||')
        sankeyLinks.append({
            "source": sortedEventList.index(source_node),
            "target": sortedEventList.index(target_node),
            "value": val["count"],
            "tooltip": {"formatter": val["tooltip"]},
            "uids": val["uids"],
            "source_node": source_node,
            "target_node": target_node
        })
    
    # Create the sankey chart options
    options = {
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "sankey",
            "layout": "none",
            "top": "4%",
            "left": "4%",
            "bottom": "4%",
            "right": "4%",
            "data": [{"label": {"show": True, "formatter": label}} for node, label in zip(sortedEventList, sankeyLabel)],
            "links": sankeyLinks,
            "lineStyle": {"color": "source", "curveness": 0.5},
            "label": {"color": "#888888"},
            "emphasis": {"focus": "adjacency"}
        }]
    }
    
    # Display the chart and return True to indicate success
    st_echarts(options=options, height="600px", key=f"pattern_sankey_{pattern_type}_{direction}")
    return True  # Return True to indicate successful creation

def create_pattern_visualization(results_df, title="Pattern Mining Results"):
    """Create interactive visualizations for pattern mining results"""
    
    if results_df.empty:
        st.warning("No patterns found with the current configuration.", icon=":material/warning:")
        return
    
    # Ensure numeric types for visualization
    results_df = results_df.copy()
    results_df['FREQUENCY'] = pd.to_numeric(results_df['FREQUENCY'], errors='coerce').fillna(0)
    results_df['UNIQUE_USERS'] = pd.to_numeric(results_df['UNIQUE_USERS'], errors='coerce').fillna(0)
    results_df['PERCENTAGE'] = pd.to_numeric(results_df['PERCENTAGE'], errors='coerce').fillna(0)
    results_df['AVG_SEQUENCE_LENGTH'] = pd.to_numeric(results_df['AVG_SEQUENCE_LENGTH'], errors='coerce').fillna(0)
    
    # Take top 20 patterns for visualization
    top_patterns = results_df.head(20)
    
    # 1. Frequency Bar Chart
    fig1 = go.Figure(data=[
        go.Bar(
            x=top_patterns['FREQUENCY'],
            y=top_patterns['PATH'],
            orientation='h',
            marker_color='#29B5E8',
            text=top_patterns['FREQUENCY'],
            textposition='outside'
        )
    ])
    
    fig1.update_layout(
        title="",
        xaxis_title="Frequency",
        yaxis_title="Patterns",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Frequency Analysis</h3>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Sankey Chart for Pattern Flow Visualization
    # Get pattern type from session state if available
    pattern_type = st.session_state.get('pattern_config', {}).get('pattern_type', 'All')
    
    # Create help text for the legend
    legend_text = f"Shows pattern flow progression. Node size = event frequency, link width = transition frequency. {'Left-to-right progression' if pattern_type in ['Starts with', 'Contains', 'Does not contain', 'All'] else 'Right-aligned convergence'}. Hover for detailed statistics."
    
    # Add controls for sankey visualization
    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Pattern Flow (Sankey Chart)</h3>""", unsafe_allow_html=True, help=legend_text)

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])  
        with col1:
             # Adjust slider bounds to handle small result sets
             max_patterns = min(100, len(results_df))
             min_patterns = min(10, max_patterns) if max_patterns > 1 else 1
             
             if min_patterns >= max_patterns:
                 # If we have very few results, just show them all
                 top_n_sankey = len(results_df)
                 st.caption(f"Showing all {len(results_df)} patterns")
             else:
                 top_n_sankey = st.slider("Top N Patterns", min_value=min_patterns, max_value=max_patterns, 
                                        value=min(20, max_patterns), key='sankey_top_n')
        
        # Calculate percentage for top N
        sankey_percentage = (top_n_sankey / len(results_df)) * 100
       
        with col2:
            st.write("")
        
        # Create nd display sankey chart
        sankey_success = pattern_sankey_chart(results_df, pattern_type=pattern_type, 
                                            topN_percentage=sankey_percentage)
        
    if not sankey_success:
        st.warning("Cannot create Sankey chart - insufficient data or no pattern flows detected.", icon=":material/warning:")
    
    # Note: Pattern Reach vs Frequency chart is now handled in the conditional section below
    
    # Get pattern configuration from session state to check if it's range or exact
    pattern_config = st.session_state.get('pattern_config', {})
    size_type = pattern_config.get('size_type', 'range')
    
    # Create two columns for Pattern Reach and Distribution of Pattern Lengths (if range)
    if size_type == 'range':
        col1, col2 = st.columns(2)
        
        # Move Pattern Reach vs Frequency to first column
        with col1:
            # Calculate exact sequence length for each pattern
            results_df['EXACT_LENGTH'] = results_df['PATH'].apply(lambda x: len(x.split(' → ')))
            
            # Add jitter to avoid overlapping points
            import numpy as np
            np.random.seed(42)  # For consistent jitter
            jitter_x = np.random.normal(0, results_df['UNIQUE_USERS'].std() * 0.02, len(results_df))
            jitter_y = np.random.normal(0, results_df['FREQUENCY'].std() * 0.02, len(results_df))
            
            # Update Pattern Reach vs Frequency with inverted gradient colors
            fig2 = go.Figure(data=[
                go.Scatter(
                    x=results_df['UNIQUE_USERS'] + jitter_x,
                    y=results_df['FREQUENCY'] + jitter_y,
                    mode='markers',
                    marker=dict(
                        size=results_df['PERCENTAGE'] * 2 + 5,  # Reduced size for better balance
                        color=results_df['EXACT_LENGTH'],
                        colorscale=[[0, '#B8E6F0'], [0.5, '#75CDD7'], [1, '#29B5E8']],  # Inverted: longer = darker
                        showscale=True,
                        colorbar=dict(
                            title="Sequence Length",
                            dtick=1,  # Force integer ticks
                            tick0=results_df['EXACT_LENGTH'].min(),  # Start from minimum value
                            tickmode='linear'
                        ),
                        sizemin=6,  # Reduced minimum size
                        line=dict(width=1, color='white')  # Add white border for better visibility
                    ),
                    text=results_df['PATH'],
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Unique Users: %{customdata[0]}<br>' +
                                 'Frequency: %{customdata[1]}<br>' +
                                 'Sequence Length: %{customdata[2]}<br>' +
                                 'Percentage: %{customdata[3]:.1f}%<extra></extra>',
                    customdata=np.column_stack((
                        results_df['UNIQUE_USERS'].astype(int),  # Original unique users (integer)
                        results_df['FREQUENCY'].astype(int),     # Original frequency (integer)
                        results_df['EXACT_LENGTH'].astype(int),  # Sequence length (integer)
                        results_df['PERCENTAGE']                 # Percentage (float)
                    ))
                )
            ])
            
            fig2.update_layout(
                title="",  # Empty title to avoid 'undefined' text
                xaxis_title="Unique Users",
                yaxis_title="Frequency",
                height=500,
                xaxis=dict(
                    tickformat='d',  # Format x-axis as integers
                    tickmode='linear',
                    dtick=max(1, int((results_df['UNIQUE_USERS'].max() - results_df['UNIQUE_USERS'].min()) / 10)) if len(results_df) > 0 else 1
                ),
                yaxis=dict(
                    tickformat='d',  # Format y-axis as integers
                    tickmode='linear',
                    dtick=max(1, int((results_df['FREQUENCY'].max() - results_df['FREQUENCY'].min()) / 10)) if len(results_df) > 0 else 1
                )
            )
            
            # Add title with tooltip
            st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Pattern Reach vs Frequency</h3>""", unsafe_allow_html=True, help="Shows how many unique users follow each pattern (reach) versus how often the pattern occurs (frequency). Patterns in the top-right are both popular and frequent.")
            with st.container(border=True):
                st.plotly_chart(fig2, use_container_width=True)
        
        # 4. Pattern Length Distribution (only show for range patterns, not exact)
        with col2:
            # Calculate count of patterns for each sequence length value
            length_counts = results_df['AVG_SEQUENCE_LENGTH'].round().value_counts().sort_index()
            
            fig3 = go.Figure(data=[
                go.Bar(
                    x=length_counts.index,
                    y=length_counts.values,
                    marker_color='#29B5E8',  # Same blue as All Mining Results bar chart
                    text=length_counts.values,
                    textposition='outside'
                )
            ])
            
            fig3.update_layout(
                title="",  # Empty title to avoid 'undefined' text
                xaxis_title="Sequence Length",
                yaxis_title="Count of Patterns",
                height=500,
                xaxis=dict(dtick=1),  # Force integer ticks on x-axis
                bargap=0.3  # Add space between bars
            )
            
            # Add title to match the style
            st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Distribution of Pattern Lengths</h3>""", unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(fig3, use_container_width=True)
    else:
        # If exact size, show only Pattern Reach vs Frequency (full width)
        # Calculate exact sequence length for each pattern
        results_df['EXACT_LENGTH'] = results_df['PATH'].apply(lambda x: len(x.split(' → ')))
        
        # Add jitter to avoid overlapping points
        import numpy as np
        np.random.seed(42)  # For consistent jitter
        jitter_x = np.random.normal(0, results_df['UNIQUE_USERS'].std() * 0.02, len(results_df))
        jitter_y = np.random.normal(0, results_df['FREQUENCY'].std() * 0.02, len(results_df))
        
        # Update Pattern Reach vs Frequency with inverted gradient colors
        fig2 = go.Figure(data=[
            go.Scatter(
                x=results_df['UNIQUE_USERS'] + jitter_x,
                y=results_df['FREQUENCY'] + jitter_y,
                mode='markers',
                marker=dict(
                    size=results_df['PERCENTAGE'] * 2 + 5,  # Reduced size for better balance
                    color=results_df['EXACT_LENGTH'],
                    colorscale=[[0, '#B8E6F0'], [0.5, '#75CDD7'], [1, '#29B5E8']],  # Inverted: longer = darker
                    showscale=True,
                    colorbar=dict(
                        title="Sequence Length",
                        dtick=1,  # Force integer ticks
                        tick0=results_df['EXACT_LENGTH'].min(),  # Start from minimum value
                        tickmode='linear'
                    ),
                    sizemin=6,  # Reduced minimum size
                    line=dict(width=1, color='white')  # Add white border for better visibility
                ),
                text=results_df['PATH'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Unique Users: %{customdata[0]}<br>' +
                             'Frequency: %{customdata[1]}<br>' +
                             'Sequence Length: %{customdata[2]}<br>' +
                             'Percentage: %{customdata[3]:.1f}%<extra></extra>',
                customdata=np.column_stack((
                    results_df['UNIQUE_USERS'].astype(int),  # Original unique users (integer)
                    results_df['FREQUENCY'].astype(int),     # Original frequency (integer)
                    results_df['EXACT_LENGTH'].astype(int),  # Sequence length (integer)
                    results_df['PERCENTAGE']                 # Percentage (float)
                ))
            )
        ])
        
        fig2.update_layout(
            title="",  # Empty title to avoid 'undefined' text
            xaxis_title="Unique Users",
            yaxis_title="Frequency",
            height=500,
            xaxis=dict(
                tickformat='d',  # Format x-axis as integers
                tickmode='linear',
                dtick=max(1, int((results_df['UNIQUE_USERS'].max() - results_df['UNIQUE_USERS'].min()) / 10)) if len(results_df) > 0 else 1
            ),
            yaxis=dict(
                tickformat='d',  # Format y-axis as integers
                tickmode='linear',
                dtick=max(1, int((results_df['FREQUENCY'].max() - results_df['FREQUENCY'].min()) / 10)) if len(results_df) > 0 else 1
            )
        )
        
        # Add title with tooltip
        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Pattern Reach vs Frequency</h3>""", unsafe_allow_html=True, help="Shows how many unique users follow each pattern (reach) versus how often the pattern occurs (frequency). Patterns in the top-right are both popular and frequent.")
        with st.container(border=True):
            st.plotly_chart(fig2, use_container_width=True)

#--------------------------------------
# DATA INPUT SECTION
#--------------------------------------

with st.expander("Input Parameters", icon=":material/settings:"):
    
    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
    <hr style='margin-top: -8px;margin-bottom: 10px;'>
    """, unsafe_allow_html=True)

    # Database, Schema, Table Selection (cached)
    db0 = fetch_databases(session)

    col1, col2, col3 = st.columns(3)

    # Database Selection
    with col1:
        database = st.selectbox('Select Database', key='pattern_db', index=None, 
                               placeholder="Choose from list...", options=db0['name'].unique())

    # Schema Selection
    if database:
        schema0 = fetch_schemas(session, database)
        
        with col2:
            schema = st.selectbox('Select Schema', key='pattern_schema', index=None, 
                                 placeholder="Choose from list...", options=schema0['name'].unique())
    else:
        schema = None

    # Table Selection
    if database and schema:
        try:
            sqltables = f"SHOW TABLES IN SCHEMA {database}.{schema}"
            tables = session.sql(sqltables).collect()
            table0 = pd.DataFrame(tables)
            
            # Debug: Check what columns are returned
            if table0.empty:
                st.warning(f"No tables found in schema {database}.{schema}")
                tbl = None
            else:
                # Handle different possible column names from SHOW TABLES
                table_column = None
                for col_name in ['TABLE_NAME', 'name', 'Name', 'table_name']:
                    if col_name in table0.columns:
                        table_column = col_name
                        break
                
                if table_column is None:
                    st.error(f"Cannot find table name column. Available columns: {list(table0.columns)}", icon=":material/chat_error:")
                    tbl = None
                else:
                    with col3:
                        tbl = st.selectbox('Select Event Table or View', key='pattern_table', index=None, 
                                          placeholder="Choose from list...", options=table0[table_column].unique(),
                                          help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp.")
        except Exception as e:
            st.error(f"Error loading tables: {str(e)}", icon=":material/chat_error:")
            tbl = None
    else:
        tbl = None

    # Column Selection (cached)
    if database and schema and tbl:
        colsdf = fetch_columns(session, database, schema, tbl)
        
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Column Mapping</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;'>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...", 
                              key='pattern_uid', help="The identifier column for partitioning (user id, customer id, etc.)")
        with col2: 
            evt = st.selectbox('Select event column', colsdf, index=None, placeholder="Choose from list...",
                              key='pattern_evt', help="The event column containing the actual events for pattern analysis.")
        with col3:
            tmstp = st.selectbox('Select timestamp column', colsdf, index=None, placeholder="Choose from list...",
                                key='pattern_timestamp', help="The timestamp column for sequential ordering.")
        
        # Get distinct events for pattern configuration
        if uid and evt and tmstp:
            events_sql = f"SELECT DISTINCT {evt} FROM {database}.{schema}.{tbl} ORDER BY {evt}"
            distinct_events = session.sql(events_sql).collect()
            events_df = pd.DataFrame(distinct_events)
            event_list = events_df[evt].tolist()

    #--------------------------------------
    # PATTERN CONFIGURATION SECTION
    #--------------------------------------

    if uid and evt and tmstp and database and schema and tbl:
        
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Pattern Configuration</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
    
        # Pattern Definition Type
        with col1:
            pattern_type = st.selectbox(
                "Pattern Definition",
                ["All", "Contains", "Does not contain", "Starts with", "Ends with"],
                index=0,
                key='pattern_type',
                help="Define how patterns should be matched: All combinations, must contain/exclude specific events, start with event, or end with event"
            )
        
        # Event Selection based on pattern type
        with col2:
            selected_events = []
            if pattern_type == "Contains":
                selected_events = st.multiselect(
                    "Select events that must be present",
                    event_list,
                    key='pattern_contains_events',
                    help="Patterns must contain all selected events"
                )
            elif pattern_type == "Does not contain":
                selected_events = st.multiselect(
                    "Select events that must be excluded",
                    event_list,
                    key='pattern_excludes_events',
                    help="Patterns must NOT contain any of the selected events"
                )
            elif pattern_type == "Starts with":
                start_event = st.selectbox(
                    "Select starting event",
                    event_list,
                    index=None,
                    key='pattern_start_event',
                    help="Patterns must start with this event"
                )
                selected_events = [start_event] if start_event else []
            elif pattern_type == "Ends with":
                end_event = st.selectbox(
                    "Select ending event", 
                    event_list,
                    index=None,
                    key='pattern_end_event',
                    help="Patterns must end with this event"
                )
                selected_events = [end_event] if end_event else []
        
        # Sequence Size Configuration (under Pattern Configuration)
        st.markdown('<p style="font-size: 14px; margin-bottom: 5px;">Sequence Size Configuration</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            size_type = st.radio(
                "Size Type",
                ["range", "exact"],
                index=0,
                key='pattern_size_type',
                help="Choose between a range of sequence lengths or an exact length"
            )
        
        sequence_config = {}
        if size_type == "range":
            with col2:
                min_size = st.number_input("Min Size", min_value=1, value=2, key='pattern_min_size')
            with col3:
                max_size = st.number_input("Max Size", min_value=min_size, value=5, key='pattern_max_size')
            sequence_config = {'size_type': 'range', 'min': min_size, 'max': max_size}
        else:
            with col2:
                exact_size = st.number_input("Exact Size", min_value=1, value=3, key='pattern_exact_size')
            sequence_config = {'size_type': 'exact', 'exact': exact_size, 'min': exact_size, 'max': exact_size}

        # Pattern Matching Mode (under Pattern Configuration)
        st.markdown('<p style="font-size: 14px; margin-bottom: 5px;">Pattern Matching Mode</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            checkoverlap = st.toggle("Allow overlap", value=True, key="pattern_checkoverlap",
                                    help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern. Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
            overlap = 'TO NEXT ROW'  # Default to overlapping mode
        if not checkoverlap:
            overlap = 'PAST LAST ROW'
        with col2:
            st.write("")

        # Date Range Selection (following PathAnalysis approach)
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Analysis Time Range</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;'>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([4,4,4])
        
        # SQL query to get the min start date
        minstartdt = f"SELECT TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {database}.{schema}.{tbl}"
        # Get min start date :
        defstartdt = session.sql(minstartdt).collect()
        defstartdt_str = defstartdt[0][0] 
        defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
        
        with col1:
            start_date = st.date_input('Start date', value=defstartdt1, key='pattern_start_date')
        with col2:
            end_date = st.date_input('End date', key='pattern_end_date', 
                                    help="Apply a time window to the data set to find patterns on a specific date range or over the entire lifespan of your data (default values)")
        with col3:
            st.write("")
        
        # Lookback Parameter (under Analysis Time Range)
        st.markdown('<p style="font-size: 14px; margin-bottom: 5px;">Lookback Configuration</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            enable_gap = st.checkbox("Enable Lookback", key='pattern_enable_gap',
                                    help="Limit patterns to events occurring within a time window")
            
            max_gap_days = None
            if enable_gap:
                # Put the Lookback input under the checkbox, constrained to column width
                max_gap_days = st.number_input("Lookback (days)", min_value=1, value=30, key='pattern_max_gap',
                                              help="Maximum days between consecutive events in a pattern")
        with col2:
            st.write("")
        with col3:
            st.write("")
    
        # Time Gap Filtering Section
        st.markdown('<p style="font-size: 14px; margin-bottom: 5px;">Time Gap Between Events</p>', unsafe_allow_html=True)
        
        # Row 1: Enable checkbox
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            enable_time_gap = st.checkbox("Enable Time Gap Filtering", key='pattern_enable_time_gap',
                                        help="Filter patterns based on time elapsed between consecutive events")
        # col2 and col3 are empty
            
        time_gap_config = {'enabled': enable_time_gap, 'show_gaps': False}
        
        if enable_time_gap:
            # Row 2: Show gaps checkbox
            col1, col2, col3 = st.columns([4, 4, 4])
            
            with col1:
                show_gaps = st.checkbox("Show time gaps in results", value=True, key='pattern_show_gaps',
                                       help="Display average, min, and max time gaps between events in the results")
            # col2 and col3 are empty
            
            time_gap_config['show_gaps'] = show_gaps
            
            # Row 3: Filter Type, Operator, Max Time Gap, Unit all on same level
            col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 2])
            
            with col1:
                gap_filter_type = st.selectbox("Filter Type", 
                                              ["Maximum gap", "Minimum gap", "Both"],
                                              key='pattern_gap_filter_type',
                                              help="Apply maximum, minimum, or both constraints on time gaps")
            
            time_gap_config['filter_type'] = gap_filter_type
            
            if gap_filter_type in ["Maximum gap", "Both"]:
                with col2:
                    max_gap_operator = st.selectbox("Operator", 
                                                   ["<=", "<"],
                                                   index=0,
                                                   key='pattern_max_gap_operator')
                with col3:
                    max_gap_value = st.number_input("Max Time Gap", min_value=1, value=60, key='pattern_max_gap_value',
                                                   help="Maximum time allowed between consecutive events")
                with col4:
                    max_gap_unit = st.selectbox("Unit", 
                                               ["Seconds", "Minutes", "Hours", "Days"],
                                               index=1,
                                               key='pattern_max_gap_unit')
                
                time_gap_config['max_gap_value'] = max_gap_value
                time_gap_config['max_gap_unit'] = max_gap_unit.upper()[:-1]  # Convert "Seconds" to "SECOND"
                time_gap_config['max_gap_operator'] = max_gap_operator
            
            # Row 4: If Both, show Min gap controls aligned under the Max gap controls
            if gap_filter_type == "Both":
                col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 2])
                
                with col2:
                    min_gap_operator = st.selectbox("Operator", 
                                                   [">=", ">"],
                                                   index=0,
                                                   key='pattern_min_gap_operator')
                with col3:
                    min_gap_value = st.number_input("Min Time Gap", min_value=0, value=1, key='pattern_min_gap_value',
                                                   help="Minimum time required between consecutive events")
                with col4:
                    min_gap_unit = st.selectbox("Unit", 
                                               ["Seconds", "Minutes", "Hours", "Days"],
                                               index=1,
                                               key='pattern_min_gap_unit')
                
                time_gap_config['min_gap_value'] = min_gap_value
                time_gap_config['min_gap_unit'] = min_gap_unit.upper()[:-1]  # Convert "Minutes" to "MINUTE"
                time_gap_config['min_gap_operator'] = min_gap_operator
            
            elif gap_filter_type == "Minimum gap":
                # If only Minimum gap is selected, show it on the same row as Filter Type
                col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 2])
                
                with col2:
                    min_gap_operator = st.selectbox("Operator", 
                                                   [">=", ">"],
                                                   index=0,
                                                   key='pattern_min_gap_operator')
                with col3:
                    min_gap_value = st.number_input("Min Time Gap", min_value=0, value=1, key='pattern_min_gap_value',
                                                   help="Minimum time required between consecutive events")
                with col4:
                    min_gap_unit = st.selectbox("Unit", 
                                               ["Seconds", "Minutes", "Hours", "Days"],
                                               index=1,
                                               key='pattern_min_gap_unit')
                
                time_gap_config['min_gap_value'] = min_gap_value
                time_gap_config['min_gap_unit'] = min_gap_unit.upper()[:-1]  # Convert "Minutes" to "MINUTE"
                time_gap_config['min_gap_operator'] = min_gap_operator
        
        # Filters Section
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Additional filters</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;'>
        """, unsafe_allow_html=True)
        
        # Advanced Filters - exclude event and timestamp (they have dedicated sections)
        # Check if there are any filterable columns (excluding event and timestamp which have dedicated sections)
        filterable_columns = colsdf[~colsdf['COLUMN_NAME'].isin([evt, tmstp])]['COLUMN_NAME']
        
        # Initialize default filter state
        sql_where_clause = ""
        
        if not filterable_columns.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                if len(filterable_columns) == 1:  # Only ID column available
                    checkfilters = st.toggle("Filter on ID column", key='pattern_id_filters',
                                           help="Apply conditional filters to the identifier column. Event and timestamp filtering are handled in dedicated sections above.")
                else:  # ID + additional columns available
                    checkfilters = st.toggle("Additional filters", key='pattern_additional_filters',
                                           help="Apply conditional filters to identifier and other available columns. Event and timestamp filtering are handled in dedicated sections above.")
            with col2:
                st.write("")
        else:
            st.info("No additional columns available for filtering. Event and timestamp filtering are handled in dedicated sections above.", icon=":material/chat_info:")
            checkfilters = False
        
        # Only execute filter logic if there are available columns AND the toggle is enabled
        if checkfilters and not filterable_columns.empty:
            
            with st.container():
                # Helper function to fetch distinct values from a column
                def fetch_distinct_values(column):
                    """Query the distinct values for a column, except for dates"""
                    query = f"SELECT DISTINCT {column} FROM {database}.{schema}.{tbl}"
                    result = session.sql(query).collect()
                    distinct_values = [row[column] for row in result]
                    return distinct_values
        
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
                        distinct_values = fetch_distinct_values(col_name)
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
                        distinct_values = fetch_distinct_values(col_name)
                        value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key, accept_new_options=True)
                        return value
                    
                    # Handle other data types (strings, etc.)
                    else:
                        distinct_values = fetch_distinct_values(col_name)
                        value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key)
                        return value
        
                # Initialize variables to store filters and logical conditions
                filters = []
                logic_operator = None
                filter_index = 0
        
                while True:
                    available_columns = filterable_columns  # Columns available for filtering (excluding event and timestamp)
        
                    if available_columns.empty:
                        st.write("No columns available for filtering.")
                        break  # Stop the loop if no columns remain
        
                    # Create 3 columns for column selection, operator, and value input
                    col1, col2, col3 = st.columns([2, 1, 2])
        
                    with col1:
                        selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns, key=f"pattern_filter_col_{filter_index}")
        
                    # Determine column data type
                    # Get column data type (cached)
                    col_data_type = fetch_column_type(session, database, schema, tbl, selected_column)
        
                    with col2:
                        operator = get_operator_input(selected_column, col_data_type, filter_index)
        
                    with col3:
                        value = get_value_input(selected_column, col_data_type, operator, filter_index)
        
                    # Append filter as a tuple (column, operator, value)
                    # For NULL operators, value is None and that's expected
                    # For other operators, we need a valid value (except for empty lists in IN/NOT IN)
                    if operator:
                        if operator in ['IS NULL', 'IS NOT NULL']:
                            filters.append((selected_column, operator, None))
                        elif operator in ['IN', 'NOT IN'] and value:  # Must have at least one value for IN/NOT IN
                            filters.append((selected_column, operator, value))
                        elif operator not in ['IS NULL', 'IS NOT NULL', 'IN', 'NOT IN'] and (value is not None and value != ''):
                            filters.append((selected_column, operator, value))
        
                    # Ask user if they want another filter
                    add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"pattern_add_filter_{filter_index}")
        
                    if add_filter == 'Yes':
                        col1, col2 = st.columns([2, 13])
                        with col1: 
                            logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"pattern_logic_operator_{filter_index}")
                            filter_index += 1
                        with col2:
                            st.write("")
                    else:
                        break
                
                # Generate SQL WHERE clause based on selected filters and logic
                if filters:
                    sql_where_clause = " AND "
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
                                # Handle mixed types by converting all to strings for SQL safety
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

#--------------------------------------
# PATTERN MINING EXECUTION
#--------------------------------------

# Only show mining toggle if essential parameters are configured
if (uid and evt and tmstp and database and schema and tbl):
    # Create columns for layout similar to Path Analysis
    col1, col2 = st.columns([1, 4])
    
    with col1:
        show_patterns = st.toggle("Show me!", help="Mine and display pattern analysis results", key='mine_patterns_toggle')
    
    with col2:
        st.write("")  # Empty space for alignment
    
    # Execute pattern mining when toggle is turned on
    if show_patterns:
        # Validation
        if pattern_type in ["Contains", "Does not contain", "Starts with", "Ends with"] and not selected_events:
            st.error("Please select at least one event for the chosen pattern type.")
            st.stop()
        
        # Check if we need to run the query (if parameters changed or no results exist)
        current_params = {
            'database': database,
            'schema': schema,
            'tbl': tbl,
            'uid': uid,
            'evt': evt,
            'tmstp': tmstp,
            'pattern_type': pattern_type,
            'sequence_config': sequence_config,
            'selected_events': selected_events,
            'max_gap_days': max_gap_days,
            'overlap': overlap,
            'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None,
            'sql_where_clause': sql_where_clause,
            'time_gap_config': time_gap_config
        }
        
        # Check if we need to rerun the query
        stored_params = st.session_state.get('pattern_last_params', {})
        need_refresh = (current_params != stored_params or 
                       st.session_state.get('pattern_results') is None)
        
        if need_refresh:
            with st.spinner("Mining patterns... This may take a few moments."):
                try:
                    # Generate and execute SQL
                    pattern_sql = generate_pattern_sql(
                        pattern_type=pattern_type,
                        sequence_config=sequence_config,
                        selected_events=selected_events,
                        uid=uid,
                        evt=evt,
                        tmstp=tmstp,
                        database=database,
                        schema=schema,
                        tbl=tbl,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        sql_where_clause=sql_where_clause,
                        max_gap_days=max_gap_days,
                        overlap=overlap,
                        time_gap_config=time_gap_config
                    )
                    
                    # Execute the query
                    results = session.sql(pattern_sql).collect()
                    results_df = pd.DataFrame(results)
                    
                    # Store results in session state
                    st.session_state['pattern_results'] = results_df
                    st.session_state['pattern_config'] = {
                        'pattern_type': pattern_type,
                        'sequence_config': sequence_config,
                        'selected_events': selected_events,
                        'max_gap_days': max_gap_days,
                        'time_gap_config': time_gap_config
                    }
                    st.session_state['pattern_last_params'] = current_params
                    
                except Exception as e:
                    st.error(f"Error mining patterns: {str(e)}")
                    st.code(pattern_sql)  # Show SQL for debugging
    else:
        # Toggle is off, define show_patterns for scope
        show_patterns = False
else:
    # Essential parameters not configured, define show_patterns for scope
    show_patterns = False
    # Show info message when essential parameters are not configured
    if not (uid and evt and tmstp and database and schema and tbl):
        st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 14px; font-weight: 200 ; margin-top: 0px; margin-bottom: -15px;">
            Please ensure all required inputs parameters to enable pattern mining.
        </h5>
    </div>
    """, unsafe_allow_html=True)
        #st.info("Please configure all required parameters (Database, Schema, Table, and Column Mapping) to enable pattern mining.")

# Parameter change detection is now handled within the toggle logic above

#--------------------------------------
# RESULTS DISPLAY
#--------------------------------------

# Only show results if toggle is on and results exist
if (show_patterns and 
    st.session_state.get('pattern_results') is not None and 
    'database' in locals() and database and 
    'schema' in locals() and schema and
    'tbl' in locals() and tbl and
    'uid' in locals() and uid and
    'evt' in locals() and evt and
    'tmstp' in locals() and tmstp):
    results_df = st.session_state['pattern_results']
    config = st.session_state.get('pattern_config', {})
    
    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Pattern Mining Results</h2>
    <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
    """, unsafe_allow_html=True)
    
    if not results_df.empty:
        
        # Summary Statistics
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            # Ensure numeric types for metrics calculations
            results_df['FREQUENCY'] = pd.to_numeric(results_df['FREQUENCY'], errors='coerce').fillna(0)
            results_df['AVG_SEQUENCE_LENGTH'] = pd.to_numeric(results_df['AVG_SEQUENCE_LENGTH'], errors='coerce').fillna(0)
            results_df['UNIQUE_USERS'] = pd.to_numeric(results_df['UNIQUE_USERS'], errors='coerce').fillna(0)
            results_df['PERCENTAGE'] = pd.to_numeric(results_df['PERCENTAGE'], errors='coerce').fillna(0)
            
            with col1:
                st.metric("Total Patterns", len(results_df))
            with col2:
                st.metric("Total Frequency", int(results_df['FREQUENCY'].sum()))
            with col3:
                st.metric("Avg Pattern Length", f"{results_df['AVG_SEQUENCE_LENGTH'].mean():.1f}")
            with col4:
                st.metric("Max Frequency", int(results_df['FREQUENCY'].max()))
        
        # Pattern Results Table
        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>All Pattern Mining Results</h3>""", unsafe_allow_html=True)
        
        # Add rank column
        display_df = results_df.copy()
        display_df['rank'] = range(1, len(display_df) + 1)
        
        # Build column list - start with base columns
        display_columns = ['rank', 'PATH', 'FREQUENCY', 'UNIQUE_USERS', 'PERCENTAGE', 'AVG_SEQUENCE_LENGTH']
        
        # Add time gap columns if they exist in results
        if 'AVG_GAP_VALUE' in display_df.columns:
            display_columns.extend(['AVG_GAP_VALUE', 'MIN_GAP_VALUE', 'MAX_GAP_VALUE'])
        
        display_df = display_df[display_columns]
        
        # Format columns - ensure numeric types first
        display_df['PERCENTAGE'] = pd.to_numeric(display_df['PERCENTAGE'], errors='coerce').fillna(0).round(2).astype(str) + '%'
        display_df['AVG_SEQUENCE_LENGTH'] = pd.to_numeric(display_df['AVG_SEQUENCE_LENGTH'], errors='coerce').fillna(0).round(1)
        display_df['FREQUENCY'] = pd.to_numeric(display_df['FREQUENCY'], errors='coerce').fillna(0)
        display_df['UNIQUE_USERS'] = pd.to_numeric(display_df['UNIQUE_USERS'], errors='coerce').fillna(0)
        
        # Format time gap columns if present
        if 'AVG_GAP_VALUE' in display_df.columns:
            display_df['AVG_GAP_VALUE'] = pd.to_numeric(display_df['AVG_GAP_VALUE'], errors='coerce').fillna(0).round(1)
            display_df['MIN_GAP_VALUE'] = pd.to_numeric(display_df['MIN_GAP_VALUE'], errors='coerce').fillna(0).round(1)
            display_df['MAX_GAP_VALUE'] = pd.to_numeric(display_df['MAX_GAP_VALUE'], errors='coerce').fillna(0).round(1)
        
        # Determine unit label for time gaps
        time_gap_cfg = config.get('time_gap_config', {})
        display_unit = time_gap_cfg.get('max_gap_unit', time_gap_cfg.get('min_gap_unit', 'SECOND'))
        
        # Convert unit to display format (SECOND -> sec, MINUTE -> min, HOUR -> hr, DAY -> day)
        unit_map = {'SECOND': 'sec', 'MINUTE': 'min', 'HOUR': 'hr', 'DAY': 'day'}
        unit_label = unit_map.get(display_unit, 'sec')
        
        # Rename columns for better display
        rename_dict = {
            'rank': 'Rank',
            'PATH': 'Pattern Path',
            'FREQUENCY': 'Frequency',
            'UNIQUE_USERS': 'Unique Users',
            'PERCENTAGE': 'Percentage',
            'AVG_SEQUENCE_LENGTH': 'Length'
        }
        
        # Add time gap column renames if present with correct unit
        if 'AVG_GAP_VALUE' in display_df.columns:
            rename_dict.update({
                'AVG_GAP_VALUE': f'Avg Gap ({unit_label})',
                'MIN_GAP_VALUE': f'Min Gap ({unit_label})',
                'MAX_GAP_VALUE': f'Max Gap ({unit_label})'
            })
        
        display_df = display_df.rename(columns=rename_dict)

        with st.container(border=True):
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
        
        # Add spacing before visualizations
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations with aligned title formatting
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Pattern Visualizations</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
        """, unsafe_allow_html=True)
        create_pattern_visualization(results_df, f"{config.get('pattern_type', 'Pattern')} Mining Results")
        
        # AI-Powered Insights with model selection
        def pattern_ai_analysis_callback(selected_model):
            """Callback function for pattern mining AI analysis"""
            with st.spinner("Generating AI insights..."):
                try:
                    # Get top 10 patterns for analysis
                    top_patterns = results_df.head(10)
                    patterns_text = "\n".join([f"{row['PATH']} (frequency: {row['FREQUENCY']}, users: {row['UNIQUE_USERS']})" 
                                             for _, row in top_patterns.iterrows()])
                    
                    ai_prompt = f"""
                    Analyze these top sequential patterns from user behavior data:
                    
                    {patterns_text}
                    
                    Pattern Type: {config.get('pattern_type', 'Unknown')}
                    Total Patterns Found: {len(results_df)}
                    
                    Please provide insights on:
                    1. Most significant behavioral patterns
                    2. User journey implications 
                    3. Potential optimization opportunities
                    4. Anomalies or interesting findings
                    
                    Keep your analysis concise and actionable.
                    """
                    
                    # Use selected model for Snowflake Cortex analysis
                    ai_sql = f"""
                    SELECT SNOWFLAKE.CORTEX.COMPLETE(
                        '{selected_model}',
                        '{ai_prompt.replace("'", "''")}'
                    ) as insights
                    """
                    
                    ai_result = session.sql(ai_sql).collect()
                    if ai_result:
                        insights = ai_result[0]['INSIGHTS']
                        
                        st.markdown("**AI-Generated Insights**")
                        st.markdown(insights)
                    
                except Exception as e:
                    st.warning(f"AI insights not available: {str(e)}", icon=":material/warning:")
        
        # Display AI insights section with model selection
        ai_model, ai_enabled = display_ai_insights_section(
            "pattern_ai", 
            "Select the LLM model for AI analysis of pattern mining results",
            ai_content_callback=pattern_ai_analysis_callback
        )
        
        # Export Options
        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Export Results</h2>
        <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
        """, unsafe_allow_html=True)
        
        # Download CSV as toggle
        if st.toggle("Download Results as CSV", key="pattern_download_csv"):
            # Fix CSV encoding issue by using UTF-8 encoding without BOM
            csv = results_df.to_csv(index=False, sep=',', encoding='utf-8-sig')
            st.download_button(
                label="Download CSV File",
                data=csv,
                file_name=f"pattern_mining_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Writeback to Snowflake (reuse the function from AttributionAnalysis)
        if st.toggle("Writeback results to Snowflake", key="pattern_writeback"):
            
            # Fetch DBs (cached)
            db0 = fetch_databases(session)
            
            # Row with DB / Schema / Table Name
            db_col, schema_col, tbl_col = st.columns(3)
            
            with db_col:
                wb_database = st.selectbox("Select Database", db0["name"].unique(), index=None, 
                                         key="pattern_wb_db", placeholder="Choose...")
            
            wb_schema = None
            if wb_database:
                schema0 = fetch_schemas(session, wb_database)
                
                with schema_col:
                    wb_schema = st.selectbox("Select Schema", schema0["name"].unique(), index=None, 
                                           key="pattern_wb_schema", placeholder="Choose...")
            
            wb_table_name = None
            if wb_database and wb_schema:
                with tbl_col:
                    wb_table_name = st.text_input("Enter Table Name", key="pattern_wb_tbl", 
                                                placeholder="e.g. pattern_mining_results")
            
            # Write button and success message - left aligned
            success = False
            if wb_database and wb_schema and wb_table_name:
                st.markdown("---")
                
                if st.button("Write Table", key="pattern_wb_btn"):
                        try:
                            session.write_pandas(
                                results_df,
                                wb_table_name,
                                database=wb_database,
                                schema=wb_schema,
                                auto_create_table=True,
                                overwrite=True
                            )
                            success = True
                        except Exception as e:
                            st.error(f"Write failed: {e}", icon=":material/chat_error:")

                # Show success message immediately after button click
                if success:
                    st.success(f"Successfully wrote to `{wb_database}.{wb_schema}.{wb_table_name}`", icon=":material/check:")
        
        # AI-Powered Insights moved to Pattern Mining Results section
    
    else:
        st.warning("No patterns found with the current configuration. Try adjusting your parameters.", icon=":material/warning:")

else:
    pass
    #st.info("Configure your parameters above and click 'Show me!' to start the analysis.")
