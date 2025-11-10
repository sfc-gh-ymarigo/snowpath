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
from snowflake.snowpark import Session#
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast



# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

# Dark mode compatible styling
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

/* Block container styling */
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
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

.custom-container-1 h5 {
    color: #0f0f0f !important;
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
    
    /* Custom styling for all message types in dark mode */
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

#===================================================================================
# STORED PROCEDURE CREATION FUNCTIONS (Server-Side Optimization)
#===================================================================================

def check_procedure_exists(session, procedure_name):
    """Check if a stored procedure exists in the current database/schema"""
    try:
        result = session.sql(f"SHOW PROCEDURES LIKE '{procedure_name}'").collect()
        return len(result) > 0
    except:
        return False

def create_markov_stored_procedure(session):
    """
    Create Snowflake stored procedure for Markov Chain attribution.
    This procedure handles all computation server-side using NumPy in Snowflake.
    
    Note: If procedure already exists (pre-created via SETUP_STORED_PROCEDURES.sql),
    it will be reused instead of recreated to save 90+ seconds.
    """
    
    sp_name = "MARKOV_ATTRIBUTION_SP"
    
    # Check if procedure already exists (pre-created)
    if check_procedure_exists(session, sp_name):
        return sp_name
    
    try:
        # Only create if doesn't exist (fallback for users who haven't run setup script)
        st.info(f"Creating {sp_name} stored procedure (first-time setup, ~60 seconds)...", icon=":material/engineering:")
        create_sp_sql = f"""
        CREATE OR REPLACE PROCEDURE {sp_name}(
            paths_table STRING,
            path_column STRING,
            frequency_column STRING
        )
        RETURNS TABLE(channel STRING, attribution_pct FLOAT, removal_effect FLOAT, conversions FLOAT)
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.10'
        PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
        HANDLER = 'run_markov_attribution'
        AS
        $$
import numpy as np
import pandas as pd
from collections import defaultdict
import re

def run_markov_attribution(session, paths_table, path_column, frequency_column):
    # Load paths from table (using SQL to handle fully qualified names)
    query = f"SELECT {{path_column}}, {{frequency_column}} FROM {{paths_table}}"
    df = session.sql(query).to_pandas()
    df.columns = ['path', 'frequency']
    
    # DEBUG: Log first few paths
    import sys
    print(f"DEBUG SP: Received {{len(df)}} paths", file=sys.stderr)
    print(f"DEBUG SP: First 3 paths: {{df.head(3).to_dict('records')}}", file=sys.stderr)
    
    # Clean paths
    regex = re.compile('[^a-zA-Z0-9>_ -]')
    df['path'] = df['path'].apply(lambda x: regex.sub('', str(x)))
    
    # Parse paths
    all_paths = []
    all_frequencies = []
    all_touchpoints = set()
    
    for _, row in df.iterrows():
        path_list = [tp.strip() for tp in row['path'].split(' > ')]
        freq = int(row['frequency'])
        all_paths.append(path_list)
        all_frequencies.append(freq)
        all_touchpoints.update(path_list)
    
    print(f"DEBUG SP: Parsed {{len(all_paths)}} paths, {{len(all_touchpoints)}} unique touchpoints", file=sys.stderr)
    print(f"DEBUG SP: Touchpoints: {{sorted(all_touchpoints)}}", file=sys.stderr)
    print(f"DEBUG SP: First path example: {{all_paths[0] if all_paths else 'No paths'}}", file=sys.stderr)
    
    channels = [tp for tp in all_touchpoints if tp not in ['start', 'conv', 'null']]
    
    if not channels:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        empty_result = pd.DataFrame([{{'CHANNEL': 'No channels', 'ATTRIBUTION_PCT': 0.0, 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0}}])
        return session.create_dataframe(empty_result, schema=schema)
    
    # Match the exact logic from local optimized version
    def build_transition_matrix_local_style(paths, frequencies, exclude_channel=None):
        # Get unique touch list from paths (already includes start, conv, null)
        unique_touch_list = set()
        for path in paths:
            unique_touch_list.update(path)
        
        # Filter out excluded channel from paths and touch list if specified
        filtered_paths = []
        for path in paths:
            if exclude_channel:
                filtered_path = [tp for tp in path if tp != exclude_channel]
                filtered_paths.append(filtered_path)
            else:
                filtered_paths.append(path)
        
        if exclude_channel and exclude_channel in unique_touch_list:
            unique_touch_list.remove(exclude_channel)
        
        # Count transitions (matching lines 1625-1640 in local version)
        transitionStates = {{}}
        for x in unique_touch_list:
            for y in unique_touch_list:
                transitionStates[x + ">" + y] = 0
        
        # Only count transitions for non-null, non-conv states (matching line 1631-1640)
        for possible_state in unique_touch_list:
            if possible_state != "null" and possible_state != "conv":
                for i, user_path in enumerate(filtered_paths):
                    freq = frequencies[i]
                    if possible_state in user_path:
                        indices = [j for j, s in enumerate(user_path) if possible_state == s]
                        for col in indices:
                            if col + 1 < len(user_path):  # Check bounds
                                transitionStates[user_path[col] + ">" + user_path[col + 1]] += freq
        
        # Build transition matrix (matching lines 1642-1656)
        actual_paths = []
        for state in unique_touch_list:
            if state != "null" and state != "conv":
                counter = 0
                # Get all transitions from this state
                state_transitions = {{k: v for k, v in transitionStates.items() if k.startswith(state + '>')}}
                counter = sum(state_transitions.values())
                
                if counter > 0:
                    for trans, count in state_transitions.items():
                        if count > 0:
                            state_prob = float(count) / float(counter)
                            actual_paths.append({{trans: state_prob}})
        
        # Build DataFrame (matching lines 1667-1686)
        transState = []
        transMatrix = []
        for item in actual_paths:
            for key in item:
                transState.append(key)
                transMatrix.append(item[key])
        
        # Create transition dataframe
        if not transState:
            # No transitions found
            return None, None, None
            
        tmatrix_df = pd.DataFrame({{'paths': transState, 'prob': transMatrix}})
        tmatrix_split = tmatrix_df['paths'].str.split('>', expand=True)
        tmatrix_df['channel0'] = tmatrix_split[0]
        tmatrix_df['channel1'] = tmatrix_split[1]
        
        # Create full state matrix (matching lines 1678-1686)
        test_df = pd.DataFrame(0.0, index=list(unique_touch_list), columns=list(unique_touch_list))
        
        for _, v in tmatrix_df.iterrows():
            x = v['channel0']
            y = v['channel1']
            val = v['prob']
            test_df.loc[x, y] = val
        
        # Set absorbing states (matching lines 1687-1688)
        test_df.loc['conv', 'conv'] = 1.0
        test_df.loc['null', 'null'] = 1.0
        
        return test_df, unique_touch_list, None
    
    # Calculate base conversion rate (matching lines 1689-1701)
    def calculate_conversion_rate_from_df(test_df):
        # Extract R and Q matrices (matching lines 1689-1692)
        R = test_df[['null', 'conv']]
        R = R.drop(['null', 'conv'], axis=0)
        Q = test_df.drop(['null', 'conv'], axis=1)
        Q = Q.drop(['null', 'conv'], axis=0)
        
        t = len(Q.columns)
        if t == 0:
            return 0.0
        
        try:
            # Calculate absorption matrix (matching lines 1699-1701)
            N = np.linalg.inv(np.identity(t) - np.asarray(Q))
            M = np.dot(N, np.asarray(R))
            base_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
            return base_cvr
        except:
            return 0.0
    
    # Calculate removal effects (matching lines 1565-1593)
    def calculate_removals(df, base_cvr):
        removal_effect_list = dict()
        channels_to_remove = [col for col in df.columns if col not in ['conv', 'null', 'start']]
        
        for channel in channels_to_remove:
            removal_df = df.drop(channel, axis=1)
            removal_df = removal_df.drop(channel, axis=0)
            
            # Renormalize rows (matching lines 1573-1580)
            for col in removal_df.columns:
                if col not in ['null', 'conv']:
                    one = float(1)
                    row_sum = np.sum(list(removal_df.loc[col]))
                    null_percent = one - row_sum
                    if null_percent != 0:
                        removal_df.loc[col, 'null'] = null_percent
            
            removal_df.loc['null', 'null'] = 1.0
            
            # Calculate removal CVR (matching lines 1582-1591)
            R = removal_df[['null', 'conv']]
            R = R.drop(['null', 'conv'], axis=0)
            Q = removal_df.drop(['null', 'conv'], axis=1)
            Q = Q.drop(['null', 'conv'], axis=0)
            t = len(Q.columns)
            
            try:
                N = np.linalg.inv(np.identity(t) - np.asarray(Q))
                M = np.dot(N, np.asarray(R))
                removal_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
                removal_effect = 1 - removal_cvr / base_cvr
                removal_effect_list[channel] = removal_effect
            except:
                removal_effect_list[channel] = 0.0
        
        return removal_effect_list
    
    # Build base transition matrix
    test_df, unique_touch_list, _ = build_transition_matrix_local_style(all_paths, all_frequencies)
    
    if test_df is None:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        equal_share = 100.0 / len(channels)
        equal_result = pd.DataFrame([{{'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(equal_share), 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0}} for ch in channels])
        return session.create_dataframe(equal_result, schema=schema)
    
    # Calculate base CVR (matching line 1701)
    base_cvr = calculate_conversion_rate_from_df(test_df)
    
    if base_cvr == 0:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        equal_share = 100.0 / len(channels)
        equal_result = pd.DataFrame([{{'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(equal_share), 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0}} for ch in channels])
        return session.create_dataframe(equal_result, schema=schema)
    
    # Calculate removal effects (matching line 1702)
    removal_effects = calculate_removals(test_df, base_cvr)
    
    # Calculate attributions (matching lines 1703-1721)
    denominator = np.sum(list(removal_effects.values()))
    total_conversions = sum(all_frequencies)
    
    if denominator > 0:
        attribution_pcts = {{ch: (removal_effects[ch] / denominator) * 100 for ch in channels}}
        conversions = {{ch: (removal_effects[ch] / denominator) * total_conversions for ch in channels}}
    else:
        equal_share = 100.0 / len(channels)
        attribution_pcts = {{ch: equal_share for ch in channels}}
        conversions = {{ch: (equal_share / 100) * total_conversions for ch in channels}}
    
    result = pd.DataFrame([
        {{'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(attribution_pcts[ch]), 'REMOVAL_EFFECT': float(removal_effects.get(ch, 0)), 'CONVERSIONS': float(conversions[ch])}}
        for ch in channels
    ])
    result_sorted = result.sort_values('ATTRIBUTION_PCT', ascending=False)
    # Convert pandas DataFrame to Snowpark DataFrame with explicit types
    from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
    schema = StructType([
        StructField("CHANNEL", StringType()),
        StructField("ATTRIBUTION_PCT", FloatType()),
        StructField("REMOVAL_EFFECT", FloatType()),
        StructField("CONVERSIONS", FloatType())
    ])
    return session.create_dataframe(result_sorted, schema=schema)
$$;
        """
        
        session.sql(create_sp_sql).collect()
        return sp_name
    except Exception as e:
        st.warning(f"Note: Could not create stored procedure (will use fallback): {str(e)}")
        return None


def create_shapley_stored_procedure(session):
    """
    Create Snowflake stored procedure for Shapley Value attribution.
    This procedure handles all computation server-side with frequency-weighted sampling.
    
    Note: If procedure already exists (pre-created via SETUP_STORED_PROCEDURES.sql),
    it will be reused instead of recreated to save 90+ seconds.
    """
    
    sp_name = "SHAPLEY_ATTRIBUTION_SP"
    
    # Check if procedure already exists (pre-created)
    if check_procedure_exists(session, sp_name):
        return sp_name
    
    try:
        # Only create if doesn't exist (fallback for users who haven't run setup script)
        st.info(f"Creating {sp_name} stored procedure (first-time setup, ~60 seconds)...", icon=":material/engineering:")
        create_sp_sql = f"""
        CREATE OR REPLACE PROCEDURE {sp_name}(
            paths_table STRING,
            path_column STRING,
            frequency_column STRING,
            n_samples INTEGER
        )
        RETURNS TABLE(channel STRING, shapley_value FLOAT, attribution_pct FLOAT, conversions FLOAT)
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.10'
        PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
        HANDLER = 'run_shapley_attribution'
        AS
        $$
import numpy as np
import pandas as pd

def run_shapley_attribution(session, paths_table, path_column, frequency_column, n_samples):
    # Load paths from table (using SQL to handle fully qualified names)
    query = f"SELECT {{path_column}}, {{frequency_column}} FROM {{paths_table}}"
    df = session.sql(query).to_pandas()
    df.columns = ['path', 'frequency']
    
    # Parse paths
    all_touchpoints = set()
    path_data = []
    for _, row in df.iterrows():
        touchpoints = [tp.strip() for tp in str(row['path']).split(',')]
        freq = int(row['frequency'])
        path_data.append((touchpoints, freq))
        all_touchpoints.update(touchpoints)
    
    channels = sorted(list(all_touchpoints))
    if not channels:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("SHAPLEY_VALUE", FloatType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        empty_result = pd.DataFrame([{{'CHANNEL': 'No channels', 'SHAPLEY_VALUE': 0.0, 'ATTRIBUTION_PCT': 0.0, 'CONVERSIONS': 0.0}}])
        return session.create_dataframe(empty_result, schema=schema)
    
    # Calculate coalition values (frequency-weighted)
    def coalition_value(coalition, path_data):
        if not coalition:
            return 0.0
        coalition_set = set(coalition)
        matched_conversions = 0
        total_freq = 0
        for touchpoints, freq in path_data:
            if any(tp in coalition_set for tp in touchpoints):
                matched_conversions += freq
            total_freq += freq
        return matched_conversions / total_freq if total_freq > 0 else 0.0
    
    # Monte Carlo sampling
    shapley_values = {{ch: 0.0 for ch in channels}}
    for _ in range(n_samples):
        permutation = np.random.permutation(channels)
        for i, channel in enumerate(permutation):
            coalition_without = set(permutation[:i])
            coalition_with = coalition_without | {{channel}}
            value_without = coalition_value(coalition_without, path_data)
            value_with = coalition_value(coalition_with, path_data)
            marginal = value_with - value_without
            shapley_values[channel] += marginal
    
    # Average across samples
    for ch in shapley_values:
        shapley_values[ch] /= n_samples
    
    # Normalize to percentages
    abs_values = {{ch: abs(val) for ch, val in shapley_values.items()}}
    total_abs = sum(abs_values.values())
    if total_abs > 0:
        attribution_pcts = {{ch: (abs_values[ch] / total_abs) * 100 for ch in channels}}
    else:
        equal_share = 100.0 / len(channels)
        attribution_pcts = {{ch: equal_share for ch in channels}}
    
    # Calculate conversions
    total_conversions = sum(freq for _, freq in path_data)
    conversions = {{ch: (pct / 100) * total_conversions for ch, pct in attribution_pcts.items()}}
    
    # Create result DataFrame
    result = pd.DataFrame([
        {{'CHANNEL': str(ch), 'SHAPLEY_VALUE': float(shapley_values[ch]), 'ATTRIBUTION_PCT': float(attribution_pcts[ch]), 'CONVERSIONS': float(conversions[ch])}}
        for ch in channels
    ])
    result_sorted = result.sort_values('ATTRIBUTION_PCT', ascending=False)
    # Convert pandas DataFrame to Snowpark DataFrame with explicit types
    from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
    schema = StructType([
        StructField("CHANNEL", StringType()),
        StructField("SHAPLEY_VALUE", FloatType()),
        StructField("ATTRIBUTION_PCT", FloatType()),
        StructField("CONVERSIONS", FloatType())
    ])
    return session.create_dataframe(result_sorted, schema=schema)
$$;
        """
        
        session.sql(create_sp_sql).collect()
        return sp_name
    except Exception as e:
        st.warning(f"Note: Could not create stored procedure (will use fallback): {str(e)}")
        return None

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
# AI Helper Functions (from Path Analysis)
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_available_cortex_models():
    """Fetch available Cortex models from Snowflake"""
    try:
        models_query = "SHOW MODELS IN SNOWFLAKE.MODELS"
        models_result = session.sql(models_query).collect()
        
        if models_result:
            # Extract model names and filter out Arctic (text-to-SQL model)
            models = [row['name'] for row in models_result if 'arctic' not in row['name'].lower()]
            return {"models": models, "status": "found"}
        else:
            return {"models": [], "status": "not_found"}
    except Exception as e:
        return {"models": [], "status": "error", "error": str(e)}

def refresh_cortex_models():
    """Refresh the Cortex models list"""
    try:
        # Clear the cache first
        get_available_cortex_models.clear()
        
        # Try to refresh the models list
        refresh_query = "CALL SNOWFLAKE.MODELS.CORTEX_BASE_MODELS_REFRESH()"
        session.sql(refresh_query).collect()
        
        # Get updated models
        return get_available_cortex_models()
    except Exception as e:
        return {"models": [], "status": "refresh_error", "error": str(e)}

# Function to display AI insights UI with model selection and toggle
def display_ai_insights_section(key_prefix, help_text="Select the LLM model for AI analysis", ai_content_callback=None):
    
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
                            st.success(f"Found {len(refresh_result['models'])} Cortex models after refresh!")
                            st.rerun()  # Refresh the UI to show new models
                        elif refresh_result["status"] == "refresh_error":
                            st.error(f"Failed to refresh models: {refresh_result.get('error', 'Unknown error')}", icon=":material/chat_error:")
                        else:
                            st.error("No Cortex models found even after refresh.", icon=":material/chat_error:")
        
        elif status == "error":
            st.warning(f"Could not fetch model list: {models_result.get('error', 'Unknown error')}. Using default models.", icon=":material/warning:")
        
        # Always show model selection (with fallback models if needed)
        if not available_models:
            available_models = ["mixtral-8x7b", "mistral-large", "llama2-70b-chat"]
        
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
        
        # Create two columns for toggle and pills layout (like Show Me pattern)
        col1, col2 = st.columns([2, 7])
        
        # Place the toggle in the first column
        with col1:
            ai_enabled = st.toggle("ExpAIn Me!", key=f"{key_prefix}_toggle", help="Generate AI insights and recommendations for your attribution analysis results. **Auto**: Pre-built analysis with structured insights. **Custom**: Enter your own questions and prompts for personalized analysis.")
        
        # Place the pills in the second column, but only if the toggle is on
        with col2:
            if ai_enabled:
                prompt_type = st.pills(
                    "Choose prompt type:",
                    ["Auto", "Custom"],
                    key=f"{key_prefix}_prompt_pills",
                    label_visibility="collapsed"
                )
            else:
                prompt_type = None
        
        # If AI is enabled, pills are selected, and callback provided, execute the AI content within the expander
        if ai_enabled and prompt_type and ai_content_callback:
            ai_content_callback(selected_model, prompt_type)
        
        return selected_model, ai_enabled, prompt_type

#st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

#--------------------------------------
#ATTRIBUTION ANALYSIS - SERVER-SIDE OPTIMIZED
#--------------------------------------


# Get the current credentials
session = get_active_session()
#Initialize variables
toevt = None
maxnbbevt = 5
uid = None
evt = None
tmstp = None
tbl = None
partitionby = None
startdt_input = None
enddt_input = None
sess = None
excl3 = "''"
conv= None
conv_value="''"
sess1=""
model = None
cols=''
colsdf = pd.DataFrame()

st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
        ATTRIBUTION ANALYSIS
    </h5>
</div>
""", unsafe_allow_html=True) 
with st.expander("Input Parameters", icon=":material/settings:"):
        
        # DATA SOURCE 
        st.markdown("""
    <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
    <hr style='margin-top: -8px;margin-bottom: 5px;'>
    """, unsafe_allow_html=True)
        
                        # Get list of databases (cached)
        db0 = fetch_databases(session)
            
        col1, col2, col3 = st.columns(3)
            
            # **Database Selection**
        with col1:
            database = st.selectbox('Select Database', key='attribdb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
        if database:
            schema0 = fetch_schemas(session, database)
            
            with col2:
                schema = st.selectbox('Select Schema', key='attribsch', index=None, 
                                          placeholder="Choose from list...", options=schema0['name'].unique())
        else:
            schema = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected)**
        if database and schema:
            table0 = fetch_tables(session, database, schema)
            
            with col3:
                tbl = st.selectbox('Select Event Table or View', key='attribtbl', index=None, 
                                       placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
        else:
            tbl = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected)**
        if database and schema and tbl:
            colsdf = fetch_columns(session, database, schema, tbl)
        
        # Selecting identifier (uid), event (evt), and timestamp (tmstp) columns
        col1, col2, col3 = st.columns([4,4,4])
        with col1:
            uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
        with col2: 
            evt = st.selectbox('Select event column', colsdf, index=None, placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
        with col3:
            tmstp = st.selectbox('Select timestamp column', colsdf, index=None, placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
            
        if (uid != None and evt != None and tmstp != None):
            # Get Distinct Events Of Interest from Event Table
            EOI = f"SELECT DISTINCT {evt} FROM {database}.{schema}.{tbl} ORDER BY {evt}"
    
            # Get excluded EOI :
            excl =session.sql(EOI).collect()
    
            # Write query output in a pandas dataframe
            excl0 =pd.DataFrame(excl)   
            # Get remaining columns except uid, evt, tmstp
            remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
            
            #--------------------------------------
            # SESSION
            #-------------------------------------
            with st.container():
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session (optional)</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help= "A session, in the context of customer journey analysis, refers to a defined period of activity during which a customer interacts or events occur. It includes all events, actions, or transactions performed by the user within this timeframe. If a session field is present in the event table, select it from the 'Column' tab below. If no session field is available, use the 'Sessionize' tab to create unique session identifiers by grouping events based on time gaps (e.g., a gap of more than 30 minutes starts a new session). Once a session is selected or created, it will be used alongside the unique identifier to partition the input rows before applying pattern matching.")
            tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
            with tab11:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sess = st.selectbox('Select session column ', colsdf, index=None, placeholder="Choose from list...", help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")  # Empty column for spacing
                    
                    with col3:
                        st.write("")  # Empty column for spacing
                
            with tab22:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        unitoftime =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout=  st.number_input( "Insert a timeout value",value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
            if sess == None and unitoftime==None and timeout==None: 
                        partitionby = f"partition by {uid}"
                        groupby = f'group by {uid}'
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                
            elif sess != None and unitoftime==None and timeout==None:
                        partitionby=f"partition by {uid},{sess} "
                        sess1=f"{sess},"
                        groupby = f'group by {uid}, {sess}'
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
            elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        sess1=f"SESSION,"
                        groupby = f'group by {uid}, SESSION'
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
            else :
                    st.write("")
            
            #--------------------------------------
            # CONVERSION EVENT (CONV)
            #--------------------------------------
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Conversion Event</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help="While 'conversion' typically refers to a positive outcome, in this analysis, it is used more broadly to represent any outcome—whether a positive (e.g., a purchase) or a negative (e.g., churn) event")
            
            col1, col2,col3 = st.columns([5,5,5])
            
            # Select the conversion column (conv), allowing evt and conv to be the same
            with col1:
                # Add a placeholder at the top of the options
                conv_options = ['Select...'] + colsdf['COLUMN_NAME'].tolist()  # Add placeholder
                conv = st.selectbox('Select conversion column:', conv_options, index=0, help="Usually the event column")
            
            # Once conv is selected and not 'Select...', constrain the options based on distinct values from that column
            if conv != 'Select...':
                # Fetch distinct values from the selected conversion column
                conv_query = f"SELECT DISTINCT {conv} FROM {database}.{schema}.{tbl} ORDER BY {conv}"
                conv_values = session.sql(conv_query).collect()
                conv_values_df = pd.DataFrame(conv_values)
            
                # Add a placeholder at the top of the options for conv_value
                conv_value_options = ['Select...'] + conv_values_df[conv].unique().tolist()
                
                # Select a specific conversion value from the conversion column
                with col2:
                    conv_value = st.selectbox(f"Select conversion value:", conv_value_options, index=0)

                #--------------------------------------
                #CONVERSION CONDITION (Optional additional filter)
                #--------------------------------------
                # Initialize outside the container to ensure scope
                conv_condition_clause = ""
                
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px; margin-top: 15px;'>Conversion Condition (Optional)</h2>
                    <hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True, help="Add an additional condition that must be met for an event to be considered a conversion")
                    
                    col_toggle, col_empty = st.columns([2, 3])
                    with col_toggle:
                        enable_conv_condition = st.toggle("Add condition to conversion", key="enable_conversion_condition")
                    with col_empty:
                        st.write("")
                    
                    if enable_conv_condition:
                        # Get filterable columns (exclude conversion column itself to avoid confusion)
                        filterable_conv_columns = colsdf[~colsdf['COLUMN_NAME'].isin([evt, tmstp])]['COLUMN_NAME'].tolist()
                        
                        if filterable_conv_columns:
                            col1, col2, col3 = st.columns([2, 1, 2])
                            
                            with col1:
                                conv_cond_column = st.selectbox("Column for condition", filterable_conv_columns, key="conv_cond_column")
                            
                            # Get column data type
                            conv_cond_col_type = fetch_column_type(session, database, schema, tbl, conv_cond_column)
                            
                            with col2:
                                # Operator selection based on column type
                                if conv_cond_col_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                    conv_cond_operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'], key="conv_cond_operator")
                                elif conv_cond_col_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                    conv_cond_operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IS NULL', 'IS NOT NULL'], key="conv_cond_operator")
                                else:  # String/categorical
                                    conv_cond_operator = st.selectbox("Operator", ['=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'IS NULL', 'IS NOT NULL'], key="conv_cond_operator")
                            
                            with col3:
                                # Value input based on column type and operator
                                if conv_cond_operator in ['IS NULL', 'IS NOT NULL']:
                                    conv_cond_value = None
                                    st.write("(No value needed)")
                                elif conv_cond_operator in ['IN', 'NOT IN']:
                                    conv_cond_distinct_vals = fetch_distinct_values(session, database, schema, tbl, conv_cond_column).iloc[:, 0].tolist()
                                    conv_cond_value = st.multiselect(f"Values for {conv_cond_column}", conv_cond_distinct_vals, key="conv_cond_value")
                                elif conv_cond_operator in ['LIKE', 'NOT LIKE']:
                                    conv_cond_value = st.text_input("Pattern (use % for wildcards)", key="conv_cond_value", placeholder="e.g., %text%")
                                elif conv_cond_col_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                    conv_cond_value = st.date_input(f"Value for {conv_cond_column}", key="conv_cond_value")
                                elif conv_cond_col_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                    conv_cond_distinct_vals = fetch_distinct_values(session, database, schema, tbl, conv_cond_column).iloc[:, 0].tolist()
                                    conv_cond_value = st.selectbox(f"Value for {conv_cond_column}", conv_cond_distinct_vals, key="conv_cond_value", accept_new_options=True, placeholder="Select or enter value...")
                                else:
                                    conv_cond_distinct_vals = fetch_distinct_values(session, database, schema, tbl, conv_cond_column).iloc[:, 0].tolist()
                                    conv_cond_value = st.selectbox(f"Value for {conv_cond_column}", conv_cond_distinct_vals, key="conv_cond_value", accept_new_options=True, placeholder="Select or enter value...")
                            
                            # Build SQL condition
                            if conv_cond_operator in ['IS NULL', 'IS NOT NULL']:
                                conv_condition_clause = f" AND {conv_cond_column} {conv_cond_operator}"
                            elif conv_cond_operator in ['IN', 'NOT IN']:
                                if conv_cond_value:
                                    if len(conv_cond_value) == 1:
                                        single_op = '=' if conv_cond_operator == 'IN' else '!='
                                        if isinstance(conv_cond_value[0], (int, float)):
                                            conv_condition_clause = f" AND {conv_cond_column} {single_op} {conv_cond_value[0]}"
                                        else:
                                            conv_condition_clause = f" AND {conv_cond_column} {single_op} '{conv_cond_value[0]}'"
                                    else:
                                        formatted_values = []
                                        for v in conv_cond_value:
                                            if isinstance(v, (int, float)):
                                                formatted_values.append(str(v))
                                            else:
                                                formatted_values.append(f"'{v}'")
                                        conv_condition_clause = f" AND {conv_cond_column} {conv_cond_operator} ({', '.join(formatted_values)})"
                            elif conv_cond_operator in ['LIKE', 'NOT LIKE']:
                                if conv_cond_value:
                                    conv_condition_clause = f" AND {conv_cond_column} {conv_cond_operator} '{conv_cond_value}'"
                            else:  # =, !=, <, <=, >, >=
                                if conv_cond_value is not None and conv_cond_value != '':
                                    if isinstance(conv_cond_value, (int, float)):
                                        conv_condition_clause = f" AND {conv_cond_column} {conv_cond_operator} {conv_cond_value}"
                                    else:
                                        conv_condition_clause = f" AND {conv_cond_column} {conv_cond_operator} '{conv_cond_value}'"
                            
                            # Display the generated condition for verification
                            if conv_condition_clause:
                                st.caption(f"✓ Conversion condition: `{conv}='{conv_value}'{conv_condition_clause}`")
                        else:
                            st.info("No additional columns available for conversion condition.", icon=":material/info:")

        #--------------------------------------
        #WINDOW
        #--------------------------------------
                with st.container():
                    st.markdown("""
        <h2 style='font-size: 14px; margin-bottom: 0px;'>Window Size</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                # SQL query to get the min start date
                minstartdt = f"SELECT TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {database}.{schema}.{tbl}"
                # Get min start date :
                defstartdt = session.sql(minstartdt).collect()
                defstartdt_str = defstartdt[0][0] 
                defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
                
                # Attribution Window Strategy
   
                window_strategy = st.pills(
                     "Choose the window size type",
                     ["Most Events Window", "Lookback Time Window"],
                     default="Most Events Window",
                    help="Specify how to determine the maximum window size for the attribution calculation. Most Events Window assigns attributions to at most the selected events before conversion event, excluding events of types specified. Lookback Time Window assigns attributions to events within a specified maximum gap time window before the conversion event, excluding events of types specified.",
                    key="window_strategy"
                )
                
                col1, col2, col3 = st.columns(3)
                
                if window_strategy == "Most Events Window":
                    with col1:
                        maxnbbevt = st.number_input("Max events preceding", value=5, min_value=1, placeholder="Type a number...",help="Select the maximum number of events preceding the 'conversion'.")
                        max_gap_days = None
                        pattern_type = "limited"
                else:  # Lookback Time Window
                    with col1:
                        max_gap_days = st.number_input("Max gap (days)", value=7, min_value=1, placeholder="Type a number...",help="Maximum days for the entire A* sequence duration (from first A to last A)")
                        maxnbbevt = None  # Not used in A* pattern
                        pattern_type = "gap_window"
                
                with col2:
                    startdt_input = st.date_input('Start date', value=defstartdt1)
                with col3:
                    enddt_input = st.date_input('End date',help="Apply a time window to the data on a specific date range or over the entire lifespan of your data (default values)")
           
                
    #--------------------------------------
    #PATTERN GENERATION LOGIC
    #--------------------------------------
                
                # Generate pattern and define clauses based on strategy
                if pattern_type == "limited":
                    # Traditional approach: A{0,N} B
                    attribution_pattern = f"A{{0,{maxnbbevt}}} B"
                    attribution_define = f"A as true, B AS {conv}='{conv_value}'{conv_condition_clause}"
                    markov_pattern = f"A{{0,{maxnbbevt}}}"
                    markov_define = f"A AS {conv} !='{conv_value}'"
                else:  # gap_window
                    # New approach: A* B with time constraint
                    attribution_pattern = "A* B"
                    attribution_define = f"A as true, B AS {conv}='{conv_value}'{conv_condition_clause}"
                    markov_pattern = "A*"
                    markov_define = f"A AS {conv} != '{conv_value}'"

    #--------------------------------------
    #FILTERS
    #--------------------------------------

                #initialize sql_where_clause
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 10px;'>""", unsafe_allow_html=True)

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0, placeholder="Select event(s)...", help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                            excl3 = "''"
                        else:
                            excl3 = ', '.join([f"'{excl2}'" for excl2 in excl1])
                    with col2:
                        st.write("")  # Empty column for spacing
                
                # ADDITIONAL FILTERS
                # Check if there are any filterable columns (excluding event and timestamp which have dedicated sections)
                filterable_columns = colsdf[~colsdf['COLUMN_NAME'].isin([evt, tmstp])]['COLUMN_NAME']
                
                if not filterable_columns.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if len(filterable_columns) == 1:  # Only ID column available
                            checkfilters = st.toggle("Filter on ID column", key="additional_filters_attribution", help="Apply conditional filters to the identifier column. Event and timestamp filtering are handled in dedicated sections above.")
                        else:  # ID + additional columns available
                            checkfilters = st.toggle("Additional filters", key="additional_filters_attribution", help="Apply conditional filters to identifier and other available columns. Event and timestamp filtering are handled in dedicated sections above.")
                    with col2:
                        st.write("")
                else:
                    st.info("No additional columns available for filtering. Event and timestamp filtering are handled in dedicated sections above.", icon=":material/info:")
                    checkfilters = False
                    
                # Initialize sql_where_clause - retrieve from session state if it exists
                sql_where_clause = st.session_state.get('attribution_sql_where_clause', "")
                
                if checkfilters:

                    with st.container():
                        # Helper function to get distinct values as a list (uses cached global function)
                        def get_distinct_values_list(column):
                            df_vals = fetch_distinct_values(session, database, schema, tbl, column)
                            return df_vals.iloc[:, 0].tolist() if not df_vals.empty else []
                        
                        # Helper function to display operator selection based on column data type
                        def get_operator_input(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_{filter_index}"  # Ensure unique key
                            
                            # Get the actual column data type from the table data
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                operator = st.selectbox(f"Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME','TIMESTAMP_NTZ']:
                                operator = st.selectbox(f"Operator", ['=', '<', '<=', '>', '>=', '!=', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox(f"Operator", ['=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'IS NULL', 'IS NOT NULL'], key=operator_key)
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
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME','TIMESTAMP_NTZ']:
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
                        filter_index = 0  # A unique index to assign unique keys to each filter
                        
                        # Dynamic filter loop with one selectbox at a time
                        while True:
                            # Get the columns to select from - exclude event and timestamp (they have dedicated sections)
                            # Include ID column and any additional columns, but exclude event and timestamp
                            filterable_columns = colsdf[~colsdf['COLUMN_NAME'].isin([evt, tmstp])]['COLUMN_NAME']
                            available_columns = filterable_columns
                                
                            if available_columns.empty:
                                st.write("No columns available for filtering.")
                                break
                            
                            # Create 3 columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])  # Adjust width ratios for better layout
                            
                            with col1:
                                # Select a column to filter on
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns)
                            
                            # Determine column data type (cached)
                            col_data_type = fetch_column_type(session, database, schema, tbl, selected_column)
                            
                            with col2:
                                # Display operator selection based on column data type
                                operator = get_operator_input(selected_column, col_data_type, filter_index)
                            
                            with col3:
                                # Display value input based on column data type and selected operator
                                value = get_value_input(selected_column, col_data_type, operator, filter_index)
                                #st.write(value)
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
                            
                            # Ask user if they want to add another filter
                            add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"add_filter_{filter_index}")
                            
                            if add_filter == 'Yes':
                                # If another filter is to be added, ask for AND/OR logic
                                col1, col2 = st.columns([2, 13])
                                with col1: 
                                    logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"logic_operator_{filter_index}")
                                    filter_index += 1  # Increment the index for the next filter
                                with col2:
                                    st.write("")
                            else:
                                break
                        
                        # Generate SQL WHERE clause based on selected filters and logic
                        # Only rebuild if filters list is not empty (has actual filters)
                        if filters:
                            sql_where_clause = " AND "
                            #st.write(filters)
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
                                    # Get the column data type to determine if we need quotes
                                    col_type = fetch_column_type(session, database, schema, tbl, col)
                                    if col_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                        # Numeric column - no quotes needed
                                        sql_where_clause += f"{col} {operator} {value}"
                                    else:
                                        # For non-numeric values (strings, dates), enclose the value in quotes
                                        sql_where_clause += f"{col} {operator} '{value}'"        
                            
                            # Store sql_where_clause in session state so it persists across reruns
                            st.session_state['attribution_sql_where_clause'] = sql_where_clause
                        # If filters is empty, keep the existing sql_where_clause from session state
                else:
                    # If filters toggle is off, reset the clause
                    sql_where_clause = ""
                    st.session_state['attribution_sql_where_clause'] = ""
                    
                if all([uid, evt, tmstp]) and conv!= None and conv_value != 'Select...' and window_strategy:
                 with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Modeling Technique</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True, help="""
**Rule Based Models:** Simple, predefined credit allocation rules:
• First Touch: All credit to the first event
• Last Touch: All credit to the final event before conversion
• Linear: Equal credit distributed across all events
• U-shaped: More credit to first and last events
• Time Decay: More credit to recent events

**Markov Chain Model:** Data-driven probabilistic approach that analyzes transition patterns between events to calculate each touchpoint's contribution to conversion likelihood.

**Shapley Value Model:** Game theory approach that fairly distributes credit by calculating each event's marginal contribution across all possible combinations of touchpoints in the customer journey.

**Performance Tip:** For optimal performance when running Markov Chain or Shapley Value models, use a Snowpark-optimized warehouse to leverage enhanced computational capabilities for complex probabilistic calculations.""")
                    st.write("")  # Add small space
                    model = st.pills(
                         "Select Model",
                         ["Rule Based", "Markov Chain", "Shapley Value", "Rule Based & Markov Chain", "Rule Based & Shapley Value", "All Models"],
                         key='modeltech',
                         label_visibility="collapsed"
                     )

                    col1, col2 = st.columns([2, 1])
                    with col1:                     
                     # Add Shapley Value configuration options
                     if model and 'Shapley' in model:
                         with st.expander("⚙️ Shapley Value Configuration"):
                             shapley_samples = st.number_input(
                                 "Monte Carlo Samples",
                                 min_value=100,
                                 max_value=5000,
                                 value=1000,
                                 step=100,
                                 help="Number of random permutations for Shapley value approximation. Higher values = more accurate but slower. 💡 **Shapley Values** provide fair attribution by considering all possible coalitions of touchpoints. Each touchpoint gets credit for its marginal contribution across all scenarios."
                             )
                             st.session_state['shapley_samples'] = shapley_samples
# SQL LOGIC
        
#partitionby = f"partition by {uid}"

if all([uid, evt, tmstp]) and conv!= None and conv_value !="''":
    # Rule based models
    def add_writeback_functionality(dfattrib, unique_key_suffix=""):
        """Reusable writeback functionality for any dataframe"""
        if st.toggle("Writeback the detailed scores table to Snowflake", key=f"writeback_toggle_{unique_key_suffix}"):
        
            # Fetch DBs (cached)
            db0 = fetch_databases(session)
        
            # Row with DB / Schema / Table Name
            db_col, schema_col, tbl_col = st.columns(3)
        
            with db_col:
                database = st.selectbox("Select Database", db0["name"].unique(), index=None, key=f"wb_db_{unique_key_suffix}", placeholder="Choose...")
        
            schema = None
            if database:
                schema0 = fetch_schemas(session, database)
        
                with schema_col:
                    schema = st.selectbox("Select Schema", schema0["name"].unique(), index=None, key=f"wb_schema_{unique_key_suffix}", placeholder="Choose...")
        
            table_name = None
            if database and schema:
                with tbl_col:
                    table_name = st.text_input("Enter Table Name", key=f"wb_tbl_{unique_key_suffix}", placeholder="e.g. scores_table")
        
            # Write button and success message - left aligned
            success = False
            if database and schema and table_name:
                st.markdown("---")
                
                if st.button("Write Table", key=f"wb_btn_{unique_key_suffix}", type="primary"):
                        cleaned_df = dfattrib.copy()
                        if "color" in cleaned_df.columns:
                            cleaned_df = cleaned_df.drop(columns=["color"])
                        if "COLOR" in cleaned_df.columns:
                            cleaned_df = cleaned_df.drop(columns=["COLOR"])
                        try:
                            session.write_pandas(
                                cleaned_df,
                                table_name,
                                database=database,
                                schema=schema,
                                auto_create_table=True,
                                overwrite=True
                            )
                            success = True
                        except Exception as e:
                            st.error(f"Write failed: {e}", icon=":material/chat_error:")

                # Show success message immediately after button click
                if success:
                    st.success(f"Successfully wrote to `{database}.{schema}.{table_name}`", icon=":material/check:")

    def rulebased ():
            if unitoftime ==None and timeout ==None:
                    
                # Optimize SQL structure based on gap filtering requirement
                if max_gap_days is None:
                    # Most Events Window - no gap filtering needed, use simple structure
                    source_table_alias = "MATCHED_SEQUENCES"
                    gap_filtering_ctes = ""
                    comma_separator = ""
                else:
                    # Lookback Time Window - gap filtering needed, use full structure
                    source_table_alias = "FILTERED_SEQUENCES"
                    gap_filtering_ctes = f"""
                GAP_FILTERED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl,
                    MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                    MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                    from MATCHED_SEQUENCES
                ),
                FILTERED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl
                    from GAP_FILTERED_SEQUENCES
                    WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
                )"""
                    comma_separator = ","

                attributionsql= f"""
                WITH MATCHED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl
                    from (select * from {database}.{schema}.{tbl} where {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                    match_recognize(
                        {partitionby}
                        order by {tmstp}
                        measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl
                        all rows per match
                        pattern({attribution_pattern})
                        define {attribution_define}
                    )
                ){comma_separator}{gap_filtering_ctes}
                ,ALLOTHERS AS (
                    select {uid}, msq, {tmstp}, {evt},
                    CASE WHEN msq = (max(msq) OVER ({partitionby}))-1 THEN '1' 
                         WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE '0' END AS LASTCLICK,
                    CASE WHEN msq = 1 THEN '1' 
                         WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE '0' END AS FIRSTCLICK,
                    CASE WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE 1/((max(msq) OVER ({partitionby})-1)) END AS UNIFORM,        
                    CASE 
                        WHEN MAX(msq) OVER ({partitionby}) = 1 THEN 1.0
                        WHEN MAX(msq) OVER ({partitionby}) = 2 THEN 0.5
                        WHEN msq = 1 THEN 0.4
                        WHEN msq = (MAX(msq) OVER ({partitionby}) - 1) THEN 0.4
                        ELSE 0.2 / NULLIF((MAX(msq) OVER ({partitionby}) - 2), 0)
                        END AS USHAPE,
                    TIMESTAMPDIFF (second, LAST_value({tmstp}) OVER ({partitionby} ORDER BY {tmstp}), {tmstp}) as TIMETOCONVERSION
                    from {source_table_alias}
                    order by 1,2,3
                ),
                EXPDECAY AS (
                    select {uid}, msq,
                    CASE WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL
                         ELSE POWER(0.5, max_msq - MSQ) / 
                              (SUM( CASE 
                     WHEN msq < max_msq THEN POWER(0.5, max_msq - MSQ) 
                     ELSE 0 
                 END) OVER ({partitionby})) END as expdecay
                    from (
                        SELECT {uid}, msq, MAX(MSQ) OVER ({partitionby}) AS max_msq
                        FROM {source_table_alias}
                    )
                )
                SELECT {evt}, count(1) as "Count", ROUND((SUM(lastclick) * 100.0) / NULLIF(SUM(SUM(lastclick)) OVER (), 0),2) as "Last Click", ROUND((SUM(firstclick) * 100.0) / NULLIF(SUM(SUM(firstclick)) OVER (), 0),2) AS "First Click", 
                       ROUND((SUM(uniform) * 100.0) / NULLIF(SUM(SUM(uniform)) OVER (), 0),2) AS "Uniform", ROUND((SUM(expdecay) * 100.0) / NULLIF(SUM(SUM(expdecay)) OVER (), 0),2) AS "Exponential Decay",  ROUND((SUM(ushape) * 100.0) / NULLIF(SUM(SUM(ushape)) OVER (), 0),2) AS "U Shape",
                       avg(-timetoconversion) AS "Time To Conversion (Sec)",  avg(-timetoconversion)/60 AS "Time To Conversion (Min)",avg(-timetoconversion)/3600 AS "Time To Conversion (Hour)",avg(-timetoconversion)/86400 AS "Time To Conversion (Day)"
                FROM ALLOTHERS 
                LEFT OUTER JOIN EXPDECAY ON ALLOTHERS.{uid} = EXPDECAY.{uid} AND ALLOTHERS.msq = EXPDECAY.msq
                WHERE lastclick is not null
                GROUP BY 1 """
                
            elif sess == None and unitoftime !=None and timeout !=None:
                
                # Optimize SQL structure based on gap filtering requirement
                if max_gap_days is None:
                    # Most Events Window - no gap filtering needed, use simple structure
                    source_table_alias = "MATCHED_SEQUENCES"
                    gap_filtering_ctes = ""
                    comma_separator = ""
                else:
                    # Lookback Time Window - gap filtering needed, use full structure
                    source_table_alias = "FILTERED_SEQUENCES"
                    gap_filtering_ctes = f"""
                GAP_FILTERED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl,
                    MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                    MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                    from MATCHED_SEQUENCES
                ),
                FILTERED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl
                    from GAP_FILTERED_SEQUENCES
                    WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
                )"""
                    comma_separator = ","

                attributionsql= f"""
                WITH MATCHED_SEQUENCES AS (
                    select {uid}, msq, {tmstp}, {evt}, cl
                    from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                                {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                                SELECT *FROM sessions) 
                    match_recognize(
                        {partitionby}
                        order by {tmstp}
                        measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl
                        all rows per match
                        pattern({attribution_pattern})
                        define {attribution_define}
                    )
                ){comma_separator}{gap_filtering_ctes}
                ,ALLOTHERS AS (
                    select {uid}, msq, {tmstp}, {evt},
                    CASE WHEN msq = (max(msq) OVER ({partitionby}))-1 THEN '1' 
                         WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE '0' END AS LASTCLICK,
                    CASE WHEN msq = 1 THEN '1' 
                         WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE '0' END AS FIRSTCLICK,
                    CASE WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL 
                         ELSE 1/((max(msq) OVER ({partitionby})-1)) END AS UNIFORM,        
                    CASE 
                        WHEN MAX(msq) OVER ({partitionby}) = 1 THEN 1.0
                        WHEN MAX(msq) OVER ({partitionby}) = 2 THEN 0.5
                        WHEN msq = 1 THEN 0.4
                        WHEN msq = (MAX(msq) OVER ({partitionby}) - 1) THEN 0.4
                        ELSE 0.2 / NULLIF((MAX(msq) OVER ({partitionby}) - 2), 0)
                        END AS USHAPE,
                    TIMESTAMPDIFF (second, LAST_value({tmstp}) OVER ({partitionby} ORDER BY {tmstp}), {tmstp}) as TIMETOCONVERSION
                    from {source_table_alias}
                    order by 1,2,3
                ),
                EXPDECAY AS (
                    select {uid}, msq,
                    CASE WHEN msq = (max(msq) OVER ({partitionby})) THEN NULL
                         ELSE POWER(0.5, max_msq - MSQ) / 
                              (SUM(  CASE 
                     WHEN msq < max_msq THEN POWER(0.5, max_msq - MSQ) 
                     ELSE 0 
                 END) OVER ({partitionby})) END as expdecay
                    from (
                        SELECT {uid}, msq, MAX(MSQ) OVER ({partitionby}) AS max_msq
                        FROM {source_table_alias}
                    )
                )
               SELECT {evt}, count(1) as "Count", ROUND((SUM(lastclick) * 100.0) / NULLIF(SUM(SUM(lastclick)) OVER (), 0),2) as "Last Click", ROUND((SUM(firstclick) * 100.0) / NULLIF(SUM(SUM(firstclick)) OVER (), 0),2) AS "First Click", 
                       ROUND((SUM(uniform) * 100.0) / NULLIF(SUM(SUM(uniform)) OVER (), 0),2) AS "Uniform", ROUND((SUM(expdecay) * 100.0) / NULLIF(SUM(SUM(expdecay)) OVER (), 0),2) AS "Exponential Decay",  ROUND((SUM(ushape) * 100.0) / NULLIF(SUM(SUM(ushape)) OVER (), 0),2) AS "U Shape",
                       avg(-timetoconversion) AS "Time To Conversion (Sec)",  avg(-timetoconversion)/60 AS "Time To Conversion (Min)",avg(-timetoconversion)/3600 AS "Time To Conversion (Hour)",avg(-timetoconversion)/86400 AS "Time To Conversion (Day)"
                FROM ALLOTHERS
                LEFT OUTER JOIN EXPDECAY ON ALLOTHERS.{uid} = EXPDECAY.{uid} AND ALLOTHERS.msq = EXPDECAY.msq
                WHERE lastclick is not null
                GROUP BY 1 """
                
            #st.write (attributionsql)
            #st.code(attributionsql, language='sql')
            attribution = session.sql(attributionsql).collect()
            return (attribution)        
    # Shapley Value Attribution model
    def shapley_attribution():
        """
        Calculate Shapley value attribution using sampling approximation
        for computational efficiency with large datasets
        """
        # Create the same temporary tables as Markov chain for consistency
        crttblrawsmceventsrefsql= None
        crttblrawsmceventsref = None
        crttblrawsmceventscompsql = None
        crttblrawsmceventscomp = None
        
        def generate_unique_reftable_name(base_name="RAWMCEVENTSREF"):
            unique_refid = uuid.uuid4().hex
            return f"{base_name}_{unique_refid}"
        unique_reftable_name = generate_unique_reftable_name()
        
        # CREATE TABLE individual reference Paths (same as Markov)
        if unitoftime==None and timeout ==None :
            # Optimize SQL structure based on gap filtering requirement
            if max_gap_days is None:
                # Most Events Window - no gap filtering needed, use simple structure
                source_table_alias = "MATCHED_SEQUENCES"
                gap_filtering_ctes = ""
                comma_separator = ""
            else:
                # Max Gap Time Window - gap filtering needed, use full structure
                source_table_alias = "FILTERED_SEQUENCES"
                gap_filtering_ctes = f"""
            GAP_FILTERED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl,
                MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                from MATCHED_SEQUENCES
            ),
            FILTERED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl
                from GAP_FILTERED_SEQUENCES
                WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
            )"""
                comma_separator = ","

            crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
            WITH MATCHED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl
                from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                    match_recognize(
                    {partitionby} 
                    order by {tmstp} 
                    measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                    all rows per match
                    pattern({attribution_pattern})
                    define {attribution_define}
                )
            ){comma_separator}{gap_filtering_ctes}
            select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
            from {source_table_alias}
            WHERE cl != 'B' {groupby}) """
            
        elif unitoftime != None and timeout !=None :
            # Optimize SQL structure based on gap filtering requirement
            if max_gap_days is None:
                # Most Events Window - no gap filtering needed, use simple structure
                source_table_alias = "MATCHED_SEQUENCES"
                gap_filtering_ctes = ""
                comma_separator = ""
            else:
                # Max Gap Time Window - gap filtering needed, use full structure
                source_table_alias = "FILTERED_SEQUENCES"
                gap_filtering_ctes = f"""
            GAP_FILTERED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl,
                MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                from MATCHED_SEQUENCES
            ),
            FILTERED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl
                from GAP_FILTERED_SEQUENCES
                WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
            )"""
                comma_separator = ","

            crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
            WITH MATCHED_SEQUENCES AS (
                select {uid}, msq, {tmstp}, {evt}, cl
                from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) match_recognize(
                    {partitionby} 
                    order by {tmstp} 
                    measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                    all rows per match
                    pattern({attribution_pattern})
                    define {attribution_define}
                )
            ){comma_separator}{gap_filtering_ctes}
            select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
            from {source_table_alias}
            WHERE cl != 'B' {groupby}) """
            
        # Run the SQL
        crttblrawsmceventsref = session.sql(crttblrawsmceventsrefsql).collect()
        
        # OPTIMIZATION: Aggregate by unique paths with frequency counts to reduce data volume
        # Create aggregated table for stored procedure (TRANSIENT for SP compatibility)
        unique_shapley_agg_name = f"{unique_reftable_name}_AGG"
        shapley_agg_sql = f"""
        CREATE OR REPLACE TRANSIENT TABLE {unique_shapley_agg_name} AS
        SELECT 
            path AS PATH,
            COUNT(*) as PATH_FREQUENCY
        FROM {unique_reftable_name}
        GROUP BY path
        """
        session.sql(shapley_agg_sql).collect()
        
        # For fallback: also get in pandas
        shapley_input = session.sql(f"SELECT * FROM {unique_shapley_agg_name}").collect()
        shapley_df = pd.DataFrame(shapley_input)
        
        # Show optimization info
        total_customers_sql = f"SELECT COUNT(*) as total_customers FROM {unique_reftable_name}"
        total_customers = session.sql(total_customers_sql).collect()[0]['TOTAL_CUSTOMERS']
        unique_paths = len(shapley_df)
        if unique_paths < total_customers:
            reduction_pct = ((total_customers - unique_paths) / total_customers) * 100
            #st.info(f"Shapley Values: Processing {unique_paths:,} unique paths instead of {total_customers:,} individual customer journeys ({reduction_pct:.1f}% reduction)", icon=":material/speed:")
        
        # SERVER-SIDE: Check/create Shapley stored procedure (cached per session)
        # If pre-created via SETUP_STORED_PROCEDURES.sql, reuses immediately (instant)
        # Otherwise creates dynamically (one-time 60-90s overhead)
        if 'shapley_sp_created' not in st.session_state:
            with st.spinner("⚙️ Checking for Shapley stored procedure..."):
                shapley_sp = create_shapley_stored_procedure(session)
                if shapley_sp:
                    st.session_state['shapley_sp_created'] = shapley_sp
                    st.session_state['shapley_sp_name'] = shapley_sp
        else:
            shapley_sp = st.session_state['shapley_sp_name']
        
        n_samples = st.session_state.get('shapley_samples', 1000)
        
        if shapley_sp:
            try:
                # Get current database and schema for fully qualified table name
                current_db = session.sql("SELECT CURRENT_DATABASE()").collect()[0][0]
                current_schema = session.sql("SELECT CURRENT_SCHEMA()").collect()[0][0]
                fully_qualified_table = f"{current_db}.{current_schema}.{unique_shapley_agg_name}"
                
                # Call stored procedure (all computation happens in Snowflake)
                shapley_call_sql = f"""
                CALL {shapley_sp}(
                    '{fully_qualified_table}',
                    'PATH',
                    'PATH_FREQUENCY',
                    {n_samples}
                )
                """
                # Use .collect() then convert to DataFrame (same as Markov)
                shapley_results = session.sql(shapley_call_sql).collect()
                shapley_results_df = pd.DataFrame([row.asDict() for row in shapley_results])
                
                # Normalize column names to uppercase for consistency
                shapley_results_df.columns = [col.upper() for col in shapley_results_df.columns]
                
                # Convert to expected format (dict with percentages)
                shapley_results = dict(zip(
                    shapley_results_df['CHANNEL'],
                    shapley_results_df['ATTRIBUTION_PCT']
                ))
                
                # Clean up temporary tables (server-side success path)
                session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
                session.sql(f"DROP TABLE IF EXISTS {unique_shapley_agg_name}").collect()
                
                # st.success("✅ Shapley computation completed server-side")
            except Exception as e:
                st.warning(f"⚠️ Server-side Shapley failed, using local fallback: {str(e)}")
                # Clean up temporary tables (server-side failure path)
                try:
                    session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
                    session.sql(f"DROP TABLE IF EXISTS {unique_shapley_agg_name}").collect()
                except:
                    pass  # Ignore cleanup errors in fallback path
                # Fallback to local computation
                def calculate_shapley_values(paths_df, n_samples=1000):
                    if paths_df.empty:
                        return {}
                    all_touchpoints = set()
                    journey_data = []
                    for _, row in paths_df.iterrows():
                        path_events = [event.strip() for event in str(row['PATH']).split(',')]
                        frequency = int(row['PATH_FREQUENCY'])
                        for _ in range(frequency):
                            journey_data.append(path_events)
                        all_touchpoints.update(path_events)
                    touchpoints = list(all_touchpoints)
                    if not touchpoints:
                        return {}
                    shapley_values = {tp: 0.0 for tp in touchpoints}
                    
                    def calculate_coalition_value(coalition, journey_data):
                        if not coalition:
                            return 0.0
                        conversions = sum(1 for journey in journey_data if coalition.issubset(set(journey)))
                        return conversions / len(journey_data) if journey_data else 0.0
                    
                    for _ in range(n_samples):
                        permutation = np.random.permutation(touchpoints)
                        for i, touchpoint in enumerate(permutation):
                            coalition_without = set(permutation[:i])
                            coalition_with = coalition_without | {touchpoint}
                            value_without = calculate_coalition_value(coalition_without, journey_data)
                            value_with = calculate_coalition_value(coalition_with, journey_data)
                            shapley_values[touchpoint] += value_with - value_without
                    
                    for tp in shapley_values:
                        shapley_values[tp] /= n_samples
                    
                    abs_values = {tp: abs(value) for tp, value in shapley_values.items()}
                    total_abs_value = sum(abs_values.values())
                    if total_abs_value > 0:
                        shapley_values = {tp: (abs_values[tp] / total_abs_value) * 100 for tp in touchpoints}
                    else:
                        equal_share = 100.0 / len(touchpoints)
                        shapley_values = {tp: equal_share for tp in touchpoints}
                    return shapley_values
                
                shapley_results = calculate_shapley_values(shapley_df, n_samples)
        else:
            st.warning("⚠️ Stored procedure not available, using local computation")
            # Fallback to local computation (same as above)
            def calculate_shapley_values(paths_df, n_samples=1000):
                if paths_df.empty:
                    return {}
                all_touchpoints = set()
                journey_data = []
                for _, row in paths_df.iterrows():
                    path_events = [event.strip() for event in str(row['PATH']).split(',')]
                    frequency = int(row['PATH_FREQUENCY'])
                    for _ in range(frequency):
                        journey_data.append(path_events)
                    all_touchpoints.update(path_events)
                touchpoints = list(all_touchpoints)
                if not touchpoints:
                    return {}
                shapley_values = {tp: 0.0 for tp in touchpoints}
                
                def calculate_coalition_value(coalition, journey_data):
                    if not coalition:
                        return 0.0
                    conversions = sum(1 for journey in journey_data if coalition.issubset(set(journey)))
                    return conversions / len(journey_data) if journey_data else 0.0
                
                for _ in range(n_samples):
                    permutation = np.random.permutation(touchpoints)
                    for i, touchpoint in enumerate(permutation):
                        coalition_without = set(permutation[:i])
                        coalition_with = coalition_without | {touchpoint}
                        value_without = calculate_coalition_value(coalition_without, journey_data)
                        value_with = calculate_coalition_value(coalition_with, journey_data)
                        shapley_values[touchpoint] += value_with - value_without
                
                for tp in shapley_values:
                    shapley_values[tp] /= n_samples
                
                abs_values = {tp: abs(value) for tp, value in shapley_values.items()}
                total_abs_value = sum(abs_values.values())
                if total_abs_value > 0:
                    shapley_values = {tp: (abs_values[tp] / total_abs_value) * 100 for tp in touchpoints}
                else:
                    equal_share = 100.0 / len(touchpoints)
                    shapley_values = {tp: equal_share for tp in touchpoints}
                return shapley_values
            
            shapley_results = calculate_shapley_values(shapley_df, n_samples)
        
        # Clean up temporary tables
        session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
        session.sql(f"DROP TABLE IF EXISTS {unique_shapley_agg_name}").collect()
        
        return shapley_results
    
    # Markov Chain model
    def markovchain ():

         crttblrawsmceventsrefsql= None
         crttblrawsmceventsref = None
         crttblrawsmceventscompsql = None
         crttblrawsmceventscomp = None
         
         def generate_unique_reftable_name(base_name="RAWMCEVENTSREF"):
             unique_refid = uuid.uuid4().hex  # Generate a random UUID
             return f"{base_name}_{unique_refid}"
         unique_reftable_name = generate_unique_reftable_name()
         
             # CREATE TABLE individiual reference Paths 
         if unitoftime==None and timeout ==None :
             
             # Optimize SQL structure based on gap filtering requirement
             if max_gap_days is None:
                 # Most Events Window - no gap filtering needed, use simple structure
                 source_table_alias = "MATCHED_SEQUENCES"
                 gap_filtering_ctes = ""
                 comma_separator = ""
             else:
                 # Max Gap Time Window - gap filtering needed, use full structure
                 source_table_alias = "FILTERED_SEQUENCES"
                 gap_filtering_ctes = """
             GAP_FILTERED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl,
                 MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                 MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                 from MATCHED_SEQUENCES
             ),
             FILTERED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl
                 from GAP_FILTERED_SEQUENCES
                 WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
             )""".format(uid=uid, msq="msq", tmstp=tmstp, evt=evt, partitionby=partitionby, max_gap_days=max_gap_days)
                 comma_separator = ","

             crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
             WITH MATCHED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl
                 from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                     match_recognize(
                     {partitionby} 
                     order by {tmstp} 
                     measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                     all rows per match
                     pattern({attribution_pattern})
                     define {attribution_define}
                 )
             ){comma_separator}{gap_filtering_ctes}
             select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
             from {source_table_alias}
             WHERE cl != 'B' {groupby}) """
             
         elif unitoftime != None and timeout !=None :
             # Optimize SQL structure based on gap filtering requirement
             if max_gap_days is None:
                 # Most Events Window - no gap filtering needed, use simple structure
                 source_table_alias = "MATCHED_SEQUENCES"
                 gap_filtering_ctes = ""
                 comma_separator = ""
             else:
                 # Max Gap Time Window - gap filtering needed, use full structure
                 source_table_alias = "FILTERED_SEQUENCES"
                 gap_filtering_ctes = f"""
             GAP_FILTERED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl,
                 MIN(CASE WHEN cl = 'A' THEN {tmstp} END) OVER ({partitionby}) as first_a_timestamp,
                 MAX(CASE WHEN cl = 'B' THEN {tmstp} END) OVER ({partitionby}) as last_a_timestamp
                 from MATCHED_SEQUENCES
             ),
             FILTERED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl
                 from GAP_FILTERED_SEQUENCES
                 WHERE TIMESTAMPDIFF(day, first_a_timestamp, last_a_timestamp) <= {max_gap_days}
             )"""
                 comma_separator = ","

             crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
             WITH MATCHED_SEQUENCES AS (
                 select {uid}, msq, {tmstp}, {evt}, cl
                 from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                 {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                 ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                 SELECT *FROM sessions) match_recognize(
                     {partitionby} 
                     order by {tmstp} 
                     measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                     all rows per match
                     pattern({attribution_pattern})
                     define {attribution_define}
                 )
             ){comma_separator}{gap_filtering_ctes}
             select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
             from {source_table_alias}
             WHERE cl != 'B' {groupby}) """
             
         # Run the SQL
         crttblrawsmceventsref = session.sql(crttblrawsmceventsrefsql).collect()
        
         # Generate a unique comp table name
         def generate_unique_comptable_name(base_namec="RAWMCEVENTSCOMP"):
             unique_compid = uuid.uuid4().hex  # Generate a random UUID
             return f"{base_namec}_{unique_compid}"
         # Generate a unique table name
         unique_comptable_name = generate_unique_comptable_name()  
        
         # CREATE TABLE individiual compared (complement set) Paths 
         if unitoftime==None and timeout ==None :
             crttblrawsmceventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
             select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
             from  (select * from {tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
             {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {database}.{schema}.{tbl} where {conv}='{conv_value}' )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                 match_recognize(
                 {partitionby} 
                 order by {tmstp} 
                 measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                 all rows per match
                 pattern({markov_pattern})
                 define {markov_define}
             )  {groupby}) """

             
         elif unitoftime != None and timeout !=None :
             
             crttblrawsmceventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
             select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
             from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
             {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
             {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {database}.{schema}.{tbl} where {conv}='{conv_value}' )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause})
             ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
             OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
             SELECT *FROM sessions)
                 match_recognize(
                 {partitionby} 
                 order by {tmstp} 
                 measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl 
                 all rows per match
                 pattern({markov_pattern})
                 define {markov_define}
             )  {groupby}) """
         # Run the SQL
         crttblrawsmceventscomp = session.sql(crttblrawsmceventscompsql).collect()
         
         # OPTIMIZATION: Aggregate by unique paths with frequency counts to reduce data volume
         # Create aggregated table for stored procedure (TRANSIENT for SP compatibility)
         unique_markov_agg_name = f"RAWMCEVENTSAGG_{uuid.uuid4().hex}"
         markov_agg_sql = f"""
         CREATE OR REPLACE TRANSIENT TABLE {unique_markov_agg_name} AS
         SELECT 
             'start > ' || REPLACE(path, ',', ' > ') || ' > conv' AS PATHS,
             COUNT(*) as PATH_FREQUENCY
         FROM {unique_reftable_name}
         GROUP BY path
         UNION ALL
         SELECT 
             'start > ' || REPLACE(path, ',', ' > ') || ' > null' AS PATHS,
             COUNT(*) as PATH_FREQUENCY
         FROM {unique_comptable_name}
         GROUP BY path
         """
         session.sql(markov_agg_sql).collect()
         
         # For fallback: also get in pandas
         markovinput = session.sql(f"SELECT * FROM {unique_markov_agg_name}").collect()
         markov=pd.DataFrame(markovinput)
         
         # Show optimization info
         total_customers_sql = f"""
         SELECT 
             (SELECT COUNT(*) FROM {unique_reftable_name}) + 
             (SELECT COUNT(*) FROM {unique_comptable_name}) as total_customers
         """
         total_customers = session.sql(total_customers_sql).collect()[0]['TOTAL_CUSTOMERS']
         unique_paths = len(markov)
         if unique_paths < total_customers:
             reduction_pct = ((total_customers - unique_paths) / total_customers) * 100
             #st.info(f"Markov Chain Attribution: Processing {unique_paths:,} unique paths instead of {total_customers:,} individual customer journeys ({reduction_pct:.1f}% reduction)", icon=":material/speed:")
        
         # SERVER-SIDE: Check/create Markov stored procedure (cached per session)
         # If pre-created via SETUP_STORED_PROCEDURES.sql, reuses immediately (instant)
         # Otherwise creates dynamically (one-time 60-90s overhead)
         if 'markov_sp_created' not in st.session_state:
             with st.spinner("⚙️ Checking for Markov Chain stored procedure..."):
                 markov_sp = create_markov_stored_procedure(session)
                 if markov_sp:
                     st.session_state['markov_sp_created'] = markov_sp
                     st.session_state['markov_sp_name'] = markov_sp
         else:
             markov_sp = st.session_state['markov_sp_name']
         
         if markov_sp:
             try:
                 # Get current database and schema for fully qualified table name
                 current_db = session.sql("SELECT CURRENT_DATABASE()").collect()[0][0]
                 current_schema = session.sql("SELECT CURRENT_SCHEMA()").collect()[0][0]
                 fully_qualified_table = f"{current_db}.{current_schema}.{unique_markov_agg_name}"
                 
                 # Call stored procedure (all computation happens in Snowflake)
                 markov_call_sql = f"""
                 CALL {markov_sp}(
                     '{fully_qualified_table}',
                     'PATHS',
                     'PATH_FREQUENCY'
                 )
                 """
                 markov_results = session.sql(markov_call_sql).collect()
                 markov_results_df = pd.DataFrame([row.asDict() for row in markov_results])
                 
                 # Normalize column names to uppercase for consistency
                 markov_results_df.columns = [col.upper() for col in markov_results_df.columns]
                 
                 # Convert to expected format
                 markov_conversions_pct = dict(zip(
                     markov_results_df['CHANNEL'],
                     markov_results_df['ATTRIBUTION_PCT']
                 ))
                 
                 # For compatibility with existing code, create full result dict
                 mcmodel = {
                     'markov_conversions': markov_conversions_pct,
                     'last_touch_conversions': {},  # Not calculated server-side
                     'removal_effects': dict(zip(markov_results_df['CHANNEL'], markov_results_df['REMOVAL_EFFECT'])),
                     'base_cvr': 1.0,  # Placeholder
                     'transition_matrix': None,  # Not needed for display
                     'absorption_matrix': None  # Not needed for display
                 }
                 
                 # Clean up temporary tables (server-side success path)
                 session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
                 session.sql(f"DROP TABLE IF EXISTS {unique_comptable_name}").collect()
                 session.sql(f"DROP TABLE IF EXISTS {unique_markov_agg_name}").collect()
                 
                 # st.success("✅ Markov computation completed server-side")
                 return mcmodel
                 
             except Exception as e:
                 st.warning(f"⚠️ Server-side Markov failed, using local fallback: {str(e)}")
                 # Clean up temporary tables (server-side failure path)
                 try:
                     session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
                     session.sql(f"DROP TABLE IF EXISTS {unique_comptable_name}").collect()
                     session.sql(f"DROP TABLE IF EXISTS {unique_markov_agg_name}").collect()
                 except:
                     pass  # Ignore cleanup errors in fallback path
                 # Fall through to local computation
         else:
             st.warning("⚠️ Stored procedure not available, using local computation")
         
         # Fallback: Local computation
         def run_model(paths):
            #regex = re.compile('[^a-zA-Z>_ -]')
            regex = re.compile('[^a-zA-Z0-9>_ -]')
            paths.rename(columns={paths.columns[0]: "Paths", paths.columns[1]: "Frequency"}, inplace=True)
            paths['Paths'] = paths['Paths'].apply(lambda x: regex.sub('', x))
            markov_conversions = first_order(paths)
            return markov_conversions
        
            
         def calculate_removals(df, base_cvr):
            removal_effect_list = dict()
            channels_to_remove = df.drop(['conv', 'null', 'start'], axis=1).columns
            for channel in channels_to_remove:
                removal_cvr_array = list()
                removal_channel = channel
                removal_df = df.drop(removal_channel, axis=1)
                removal_df = removal_df.drop(removal_channel, axis=0)
                for col in removal_df.columns:
                    one = float(1)
                    row_sum = np.sum(list(removal_df.loc[col]))
                    null_percent = one - row_sum
                    if null_percent == 0:
                        continue
                    else:
                        removal_df.loc[col,'null'] = null_percent
                removal_df.loc['null','null'] = 1.0
                R = removal_df[['null', 'conv']]
                R = R.drop(['null', 'conv'], axis=0)
                Q = removal_df.drop(['null', 'conv'], axis=1)
                Q = Q.drop(['null', 'conv'], axis=0)
                t = len(Q.columns)
        
                N = np.linalg.inv(np.identity(t) - np.asarray(Q))
                M = np.dot(N, np.asarray(R))
                removal_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
                removal_effect = 1 - removal_cvr / base_cvr
                removal_effect_list[channel] = removal_effect
            return removal_effect_list
        
        
         def first_order(paths):
            # OPTIMIZATION: Process paths with frequency weights
            paths_with_freq = []
            frequencies = []
            total_paths = 0
            
            # Extract paths and their frequencies
            for idx, row in paths.iterrows():
                path_str = row['Paths']
                freq = int(row['Frequency'])
                userpath = path_str.split(' > ')
                paths_with_freq.append(userpath)
                frequencies.append(freq)
                total_paths += freq  # Account for frequency in total count
            
            paths = paths_with_freq
        
            unique_touch_list = set(x for element in paths for x in element)
            # get total last touch conversion counts
            conv_dict = {}
            total_conversions = 0
            for item in unique_touch_list:
                conv_dict[item] = 0
            for i, path in enumerate(paths):
                if 'conv' in path:
                    freq = frequencies[i]
                    total_conversions += freq  # Weight by frequency
                    conv_dict[path[-2]] += freq  # Weight by frequency
        
            transitionStates = {}
            base_cvr = total_conversions / total_paths
            for x in unique_touch_list:
                for y in unique_touch_list:
                    transitionStates[x + ">" + y] = 0
        
            for possible_state in unique_touch_list:
                if possible_state != "null" and possible_state != "conv":
                    # print(possible_state)
                    for i, user_path in enumerate(paths):
                        freq = frequencies[i]  # Get frequency for this path
        
                        if possible_state in user_path:
                            indices = [j for j, s in enumerate(user_path) if possible_state == s]
                            for col in indices:
                                transitionStates[user_path[col] + ">" + user_path[col + 1]] += freq  # Weight by frequency
        
            transitionMatrix = []
            actual_paths = []
            for state in unique_touch_list:
        
                if state != "null" and state != "conv":
                    counter = 0
                    index = [i for i, s in enumerate(transitionStates) if s.startswith(state + '>')]
                    for col in index:
                        if transitionStates[list(transitionStates)[col]] > 0:
                            counter += transitionStates[list(transitionStates)[col]]
                    for col in index:
                        if transitionStates[list(transitionStates)[col]] > 0:
                            state_prob = float((transitionStates[list(transitionStates)[col]])) / float(counter)
                            actual_paths.append({list(transitionStates)[col]: state_prob})
            transitionMatrix.append(actual_paths)
        
            flattened_matrix = [item for sublist in transitionMatrix for item in sublist]
            transState = []
            transMatrix = []
            for item in flattened_matrix:
                for key in item:
                    transState.append(key)
                for key in item:
                    transMatrix.append(item[key])
        
            tmatrix = pd.DataFrame({'paths': transState,
                                    'prob': transMatrix})
            # unique_touch_list = model['unique_touch_list']
            tmatrix = tmatrix.join(tmatrix['paths'].str.split('>', expand=True).add_prefix('channel'))[
                ['channel0', 'channel1', 'prob']]
            column = list()
            for k, v in tmatrix.iterrows():
                if v['channel0'] in column:
                    continue
                else:
                    column.append(v['channel0'])
            test_df = pd.DataFrame()
            for col in unique_touch_list:
                test_df[col] = 0.00
                test_df.loc[col] = 0.00
            for k, v in tmatrix.iterrows():
                x = v['channel0']
                y = v['channel1']
                val = v['prob']
                test_df.loc[x,y] = val
            test_df.loc['conv','conv'] = 1.0
            test_df.loc['null','null'] = 1.0
            R = test_df[['null', 'conv']]
            R = R.drop(['null', 'conv'], axis=0)
            Q = test_df.drop(['null', 'conv'], axis=1)
            Q = Q.drop(['null', 'conv'], axis=0)
            O = pd.DataFrame()
            t = len(Q.columns)
            for col in range(0, t):
                O[col] = 0.00
            for col in range(0, len(R.columns)):
                O.loc[col] = 0.00
            N = np.linalg.inv(np.identity(t) - np.asarray(Q))
            M = np.dot(N, np.asarray(R))
            base_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
            removal_effects = calculate_removals(test_df, base_cvr)
            denominator = np.sum(list(removal_effects.values()))
            allocation_amount = list()
            for i in removal_effects.values():
                allocation_amount.append((i / denominator) * total_conversions)
            # print(allocation_amount)
            markov_conversions = dict()
            i = 0
            for channel in removal_effects.keys():
                markov_conversions[channel] = allocation_amount[i]
                i += 1
            conv_dict.pop('conv', None)
            conv_dict.pop('null', None)
            conv_dict.pop('start', None)
        
           # After calculating markov_conversions
            total_score = sum(markov_conversions.values())
        
        # Normalize the attribution scores
            normalized_markov_conversions = {k: (v / total_score) * 100 for k, v in markov_conversions.items()}
        
        # Return normalized results
            return {
            'markov_conversions': normalized_markov_conversions,
            'last_touch_conversions': conv_dict,
            'removal_effects': removal_effects,
            'base_cvr': base_cvr,
            'transition_matrix': test_df,
            'absorption_matrix': M
        }
             
        
         mc_paths_df = markov[['PATHS', 'PATH_FREQUENCY']].copy()
         mcmodel = run_model(paths=mc_paths_df)
         
         # Clean up temporary tables
         session.sql(f"DROP TABLE IF EXISTS {unique_reftable_name}").collect()
         session.sql(f"DROP TABLE IF EXISTS {unique_comptable_name}").collect()
         session.sql(f"DROP TABLE IF EXISTS {unique_markov_agg_name}").collect()
         
         return mcmodel
        
    if model== 'Rule Based & Markov Chain':
        # Function to compute models (only run when not cached)
        def compute_models():
            # Always recompute when called (button clicked)
            attribution = rulebased()
            mcmodel = markovchain()
            
            dfattrib = pd.DataFrame(attribution)
            markov_df = pd.DataFrame.from_dict(mcmodel['markov_conversions'], orient='index', columns=['Markov Conversions'])
            markov_df.index.name = conv
            markov_df.reset_index(inplace=True)
            
            # Merge Markov results
            dfattrib = dfattrib.merge(markov_df, on=conv, how='left')
            desired_position = 6
            column_to_move = dfattrib.pop('Markov Conversions')
            dfattrib.insert(desired_position, 'Markov Conversions', column_to_move)
            
            # Ensure all model columns are numeric
            model_columns = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape', 'Markov Conversions']
            for col in model_columns:
                if col in dfattrib.columns:
                    dfattrib[col] = pd.to_numeric(dfattrib[col], errors='coerce').fillna(0)
            
            # Cache results in session state
            st.session_state['dfattrib'] = dfattrib

                    
        # Track last-used parameters for cache invalidation
        last_tbl = st.session_state.get('last_tbl_combined')
        last_conv = st.session_state.get('last_conv_combined')
        last_conv_value = st.session_state.get('last_conv_value_combined')
        last_excl3 = st.session_state.get('last_excl3_combined')
        last_database = st.session_state.get('last_database_combined')
        last_schema = st.session_state.get('last_schema_combined')
        last_strategy = st.session_state.get('last_strategy_combined')
        last_maxnbbevt = st.session_state.get('last_maxnbbevt_combined')
        last_max_gap_days = st.session_state.get('last_max_gap_days_combined')
        last_uid = st.session_state.get('last_uid_combined')
        last_evt = st.session_state.get('last_evt_combined')
        last_tmstp = st.session_state.get('last_tmstp_combined')
        last_startdt = st.session_state.get('last_startdt_combined')
        last_enddt = st.session_state.get('last_enddt_combined')
        last_sql_where_clause = st.session_state.get('last_sql_where_clause_combined')
        
        # If any key parameter changed, invalidate cached results
        if (tbl != last_tbl or conv != last_conv or conv_value != last_conv_value or 
            excl3 != last_excl3 or database != last_database or schema != last_schema or 
            window_strategy != last_strategy or maxnbbevt != last_maxnbbevt or 
            max_gap_days != last_max_gap_days or uid != last_uid or evt != last_evt or 
            tmstp != last_tmstp or str(startdt_input) != str(last_startdt) or 
            str(enddt_input) != str(last_enddt) or sql_where_clause != last_sql_where_clause):
            if 'dfattrib' in st.session_state:
                del st.session_state['dfattrib']
        
        # Update session state with current selections
        st.session_state['last_tbl_combined'] = tbl
        st.session_state['last_conv_combined'] = conv
        st.session_state['last_conv_value_combined'] = conv_value
        st.session_state['last_excl3_combined'] = excl3
        st.session_state['last_database_combined'] = database
        st.session_state['last_schema_combined'] = schema
        st.session_state['last_strategy_combined'] = window_strategy
        st.session_state['last_maxnbbevt_combined'] = maxnbbevt
        st.session_state['last_max_gap_days_combined'] = max_gap_days
        st.session_state['last_uid_combined'] = uid
        st.session_state['last_evt_combined'] = evt
        st.session_state['last_tmstp_combined'] = tmstp
        st.session_state['last_startdt_combined'] = startdt_input
        st.session_state['last_enddt_combined'] = enddt_input
        st.session_state['last_sql_where_clause_combined'] = sql_where_clause
        
        if st.button("Run Attribution Analysis", help="Detailed Scores Table, Attribution Models Summary Bar Chart and Models Bubble Charts"):
            # Call computation when button is clicked
            with st.spinner("Computing attribution models... This may take a few moments."):
                compute_models()
                # Load results from session state
                dfattrib = st.session_state['dfattrib']
        
        # Display results if they exist in session state
        if 'dfattrib' in st.session_state:
            dfattrib = st.session_state['dfattrib']
            #st.write (dfattrib)
            
            # Check if results are empty due to gap filtering
            if len(dfattrib) == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    # Optionally show debug info
                    with st.expander("Debug Information", icon=":material/lightbulb:"):
                        st.write("This means all conversion paths in your data have gaps longer than the specified window.")
                        st.write("**Common reasons:**")
                        st.write("- Long consideration periods in your customer journey")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()

            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(dfattrib, use_container_width=True)

            # Add writeback functionality
            add_writeback_functionality(dfattrib, "rule_markov")

    
            
            # Bar chart summary
            model_colors = {
                'Last Click': '#29B5E8',  
                'First Click': '#75CDD7',  
                'Uniform': '#FF9F36',  
                'Exponential Decay': '#11567F' ,
                'U Shape':'#7d44cf',
                'Markov Conversions': '#5B5B5B'
                
            }
            # Initialize figure
            fig2 = go.Figure()
            # Define models to display
            models = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape','Markov Conversions']
            # Add bars for each model
            for model in models:
                # Ensure the column is numeric before rounding
                y_values = pd.to_numeric(dfattrib[model], errors='coerce').fillna(0)
                text_values = y_values.round(2)
                
                fig2.add_trace(go.Bar(
                    x=dfattrib[evt],  # Events on x-axis
                    y=y_values,  # Model values on y-axis
                    name=model,
                    marker_color=model_colors[model],  # Use consistent color by model
                    text=text_values,  # Display value on top of the bar
                    textposition='outside',  # Place text above the bar
                    textfont=dict(size=9),  # Consistent text size for bar labels
                ))
            # Add vertical separators between events
            for i in range(len(dfattrib[evt]) - 1):
                fig2.add_shape(
                    type="line",
                    x0=i + 0.5,
                    x1=i + 0.5,
                    y0=0,
                    y1=max(dfattrib[models].max()),  # Set height based on data range
                    line=dict(color="lightgray", width=1, dash="dash")
                )
            # Update layout
            fig2.update_layout(
                title={
                    'text': "<span style='font-size:12px;'>Event</span>",
                    'font': {
                        'size': 16,
                        'color': 'black',
                        'family': 'Arial'
                    },
                    'x': 0.5  # Center the title
                },
                xaxis=dict(
                    title="",  # Remove x-axis title since it's now in the main title
                    tickangle=45,  # Rotate x-axis labels
                    tickfont=dict(size=10),
                    tickmode="array",
                    tickvals=list(range(len(dfattrib[dfattrib.columns[0]]))),
                    ticktext=dfattrib[dfattrib.columns[0]]
                ),
                yaxis=dict(
                    title="Attribution Score",
                    showgrid=True,
                    zeroline=True,
                    tickfont=dict(size=10)
                ),
                barmode='group',  # Grouped bars
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                height=600,  # Increased height
                margin=dict(l=50, r=50, t=80, b=150),  # Better margins
                legend=dict(
                    title="Model",
                    orientation="h",
                    yanchor="top",
                    y=-0.25,  # Moved legend further down
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )
            # Display chart below the table
            with st.container():
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Models Summary</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
                st.write("")
                with st.container(border=True):
                    st.plotly_chart(fig2, use_container_width=True)

                    # User selection for Time To Conversion Unit
                st.write("")
            timetoconversionunit = st.radio(
                "Time To Conversion Unit",
                ["Sec", "Min", "Hour", "Day"],
                horizontal=True,
                key='timetoconv',
                help="Select the most appropriate time unit to display the Time To Conversion value in the bubble charts"
            )
            # Function to generate random color (can be reused)
            def random_color():
                return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            
            # Debug: Check if evt column exists
            if evt not in dfattrib.columns:
                st.error(f"Column '{evt}' not found in dataframe. Available columns: {list(dfattrib.columns)}", icon=":material/chat_error:")
                st.stop()
            
            unique_events = dfattrib[evt].unique()
            
            # Reset color map if table, conversion column, or unique events change
            if (
                'events_color_map' not in st.session_state
                or st.session_state.get('events_color_map_evt') != evt
                or st.session_state.get('events_color_map_tbl') != tbl
                or st.session_state.get('events_color_map_conv') != conv
            ):
                st.session_state['events_color_map'] = {}
                st.session_state['events_color_map_evt'] = evt
                st.session_state['events_color_map_tbl'] = tbl
                st.session_state['events_color_map_conv'] = conv
            
            # Ensure all current events have colors
            for event in unique_events:
                if event not in st.session_state['events_color_map']:
                    st.session_state['events_color_map'][event] = random_color()
            
            dfattrib['color'] = dfattrib[evt].map(st.session_state['events_color_map'])
            # Define y-axis column dynamically based on selected time unit
            y_col = f"Time To Conversion ({timetoconversionunit})"
            # ✅ Function to create bubble chart (DRY principle)
            def create_bubble_chart(x_col, y_col, title):
                # Define model colors matching the bar chart
                model_title_colors = {
                    'Last Click': '#29B5E8',
                    'First Click': '#75CDD7', 
                    'Uniform': '#FF9F36',  # Orange like in bar chart
                    'Exponential Decay': '#11567F',
                    'U Shape': '#7d44cf',
                    'Markov Conversions': '#5B5B5B'  # Gray like in bar chart
                }
                
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[evt] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],  # Use event-based colors for bubbles
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                # Extract model name for title color
                model_name = x_col.replace(' Conversions', '').replace(' Values', '') if 'Conversions' in x_col or 'Values' in x_col else x_col
                title_color = model_title_colors.get(model_name, model_title_colors.get(x_col, '#000000'))
                
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': f"<span style='color:{title_color}'>{title}</span>",
                        'font': {'size': 12, 'color': title_color, 'family': 'Arial'}
                    }
                )
                return fig
            
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)

            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)  
                    
            with col2:
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)      
                with st.container(border=True):
                    st.plotly_chart(create_bubble_chart('Markov Conversions', y_col, "Markov Conversions"), use_container_width=True)
        
            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def attribution_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for attribution analysis AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_combined_custom_prompt",
                        help="Enter your custom analysis prompt. The attribution data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare attribution data for analysis
                        def attrib_row_to_text(row):
                            return (
                                f"{row[conv]}: Count={row['Count']}, "
                                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                                f"Markov Conversions={row.get('Markov Conversions', 'N/A')}, "
                                f"Time To Conversion (Day)={row['Time To Conversion (Day)']:.2f}"
                            )
                        
                        # Get all attribution results for comprehensive analysis
                        attribution_text = "\n".join([attrib_row_to_text(row) 
                                                     for _, row in dfattrib.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive attribution analysis results:
                            
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
                            
                            Please provide insights on:
                            1. Most significant attribution patterns across different models
                            2. Comparison between rule-based and Markov chain models
                            3. Time to conversion patterns and implications
                            4. Recommendations for marketing optimization
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 attribution analysis results:
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_combined_ai", 
                "Select the LLM model for AI analysis of attribution results",
                ai_content_callback=attribution_ai_analysis_callback
            )

    elif model== 'Markov Chain':
        if st.button("Run Attribution Analysis", key="run_markov", help="Detailed Scores Table and Attribution Model Bar Chart"):
            # Only compute when button is clicked
            with st.spinner("Computing Markov Chain attribution model... This may take a few moments."):
                mcmodel = markovchain()
                markov_df = pd.DataFrame.from_dict(mcmodel['markov_conversions'], orient='index', columns=['Markov Conversions'])
                markov_df.index.name = conv  # Set index name to match the key in dfattrib
                markov_df.reset_index(inplace=True)  # Reset index to make conversion column a column
                st.session_state['markov_df'] = markov_df
        
        # Display results if they exist in session state
        if 'markov_df' in st.session_state:
            markov_df = st.session_state['markov_df']
            
            # Sort by Markov Conversions highest to lowest
            markov_df = markov_df.sort_values('Markov Conversions', ascending=False)
            
            # Check if results are empty or all zeros due to gap filtering
            if len(markov_df) == 0 or markov_df['Markov Conversions'].sum() == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    with st.expander("**Debugging Information**", icon=":material/lightbulb:"):
                        st.write("**Possible reasons for no results:**")
                        st.write("- Gap window too restrictive for your customer journey patterns")
                        st.write("- Most conversions take longer than the specified gap period")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()
            
            # Display Scores Table
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                        <hr style='margin-top: -8px;margin-bottom: 5px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(markov_df, use_container_width=True)
            
            # Bar Chart
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Model Summary</h2>
                        <hr style='margin-top: -8px;margin-bottom: 5px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
            st.write("")
            fig = go.Figure([go.Bar(
                x=markov_df[conv],
                y=markov_df['Markov Conversions'],
                marker_color='#29B5E8',
                text=markov_df['Markov Conversions'],
                texttemplate='%{text:.2f}',
                textposition='outside'
            )])
            
            fig.update_layout(
                xaxis_title=conv,
                yaxis_title="Markov Conversions",
                showlegend=False,
                height=500,
                title="",
                title_font_size=14
            )
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)
            
            # Add writeback functionality
            add_writeback_functionality(markov_df, "markov")
            
            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def markov_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for Markov Chain AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_markov_custom_prompt",
                        help="Enter your custom analysis prompt. The Markov Chain data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare Markov data for analysis
                        def markov_row_to_text(row):
                            return (
                                f"{row[conv]}: Markov Conversions={row['Markov Conversions']:.2f}. "
                                f"This represents the probabilistic contribution based on transition patterns."
                            )
                        
                        # Get all Markov results for comprehensive analysis
                        markov_text = "\n".join([markov_row_to_text(row) 
                                               for _, row in markov_df.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive Markov Chain attribution results:
                            
                            {markov_text}
                            
                            Total Markov Results: {len(markov_df)}
                            
                            Please provide insights on:
                            1. Most significant touchpoints based on Markov Chain analysis
                            2. Probabilistic attribution patterns and transition behaviors
                            3. Comparison with traditional rule-based attribution models
                            4. Strategic recommendations for customer journey optimization
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 Markov Chain attribution results:
                            {markov_text}
                            
                            Total Markov Results: {len(markov_df)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_markov_ai", 
                "Select the LLM model for AI analysis of Markov Chain results",
                ai_content_callback=markov_ai_analysis_callback
            )
        
        
    elif model== 'Rule Based':
        # Function to compute rule-based model and store in session state
        def compute_rulebased():
            # Always recompute when called (button clicked)
            attribution = rulebased()
            dfattrib = pd.DataFrame(attribution)
            
            # Ensure all model columns are numeric
            model_columns = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape']
            for col in model_columns:
                if col in dfattrib.columns:
                    dfattrib[col] = pd.to_numeric(dfattrib[col], errors='coerce').fillna(0)
            
            st.session_state['dfattrib_rulebased'] = dfattrib
        # Track last-used parameters for cache invalidation
        last_tbl = st.session_state.get('last_tbl_rulebased')
        last_conv = st.session_state.get('last_conv_rulebased')
        last_conv_value = st.session_state.get('last_conv_value_rulebased')
        last_database = st.session_state.get('last_database_rulebased')
        last_schema = st.session_state.get('last_schema_rulebased')
        last_strategy = st.session_state.get('last_strategy_rulebased')
        last_maxnbbevt = st.session_state.get('last_maxnbbevt_rulebased')
        last_max_gap_days = st.session_state.get('last_max_gap_days_rulebased')
        last_uid = st.session_state.get('last_uid_rulebased')
        last_evt = st.session_state.get('last_evt_rulebased')
        last_tmstp = st.session_state.get('last_tmstp_rulebased')
        last_startdt = st.session_state.get('last_startdt_rulebased')
        last_enddt = st.session_state.get('last_enddt_rulebased')
        last_sql_where_clause = st.session_state.get('last_sql_where_clause_rulebased')
        
        # If any key parameter changed, invalidate cached results
        if (tbl != last_tbl or conv != last_conv or conv_value != last_conv_value or 
            database != last_database or schema != last_schema or window_strategy != last_strategy or
            maxnbbevt != last_maxnbbevt or max_gap_days != last_max_gap_days or
            uid != last_uid or evt != last_evt or tmstp != last_tmstp or
            str(startdt_input) != str(last_startdt) or str(enddt_input) != str(last_enddt) or
            sql_where_clause != last_sql_where_clause):
            if 'dfattrib_rulebased' in st.session_state:
                del st.session_state['dfattrib_rulebased']
        
        # Update session state with current selections
        st.session_state['last_tbl_rulebased'] = tbl
        st.session_state['last_conv_rulebased'] = conv
        st.session_state['last_conv_value_rulebased'] = conv_value
        st.session_state['last_database_rulebased'] = database
        st.session_state['last_schema_rulebased'] = schema
        st.session_state['last_strategy_rulebased'] = window_strategy
        st.session_state['last_maxnbbevt_rulebased'] = maxnbbevt
        st.session_state['last_max_gap_days_rulebased'] = max_gap_days
        st.session_state['last_uid_rulebased'] = uid
        st.session_state['last_evt_rulebased'] = evt
        st.session_state['last_tmstp_rulebased'] = tmstp
        st.session_state['last_startdt_rulebased'] = startdt_input
        st.session_state['last_enddt_rulebased'] = enddt_input
        st.session_state['last_sql_where_clause_rulebased'] = sql_where_clause
        
        if st.button("Run Attribution Analysis", key="run_rulebased", help="Detailed Scores Table, Attribution Models Summary Bar Chart and Models Bubble Charts"):
            # Call computation when button is clicked
            with st.spinner("Computing rule-based attribution models... This may take a few moments."):
                compute_rulebased()
                # Load cached results
                dfattrib = st.session_state['dfattrib_rulebased']
        
        # Display results if they exist in session state
        if 'dfattrib_rulebased' in st.session_state:
            dfattrib = st.session_state['dfattrib_rulebased']
            #dfattrib = dfattrib.drop(columns=['color'])
            
            # Check if results are empty due to gap filtering
            if len(dfattrib) == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    # Optionally show debug info
                    with st.expander("Debug Information", icon=":material/lightbulb:"):
                        st.write("This means all conversion paths in your data have gaps longer than the specified window.")
                        st.write("**Common reasons:**")
                        st.write("- Long consideration periods in your customer journey")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()
            # Display Scores Table
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(dfattrib, use_container_width=True)
            
            # Add writeback functionality
            add_writeback_functionality(dfattrib, "rule_based")
            
            # User selection for Time To Conversion Unit (affects only the display)
            timetoconversionunit = st.radio(
                "Time To Conversion Unit",
                ["Sec", "Min", "Hour", "Day"],
                horizontal=True,
                key='timetoconv',
                help="Select the most appropriate time unit to display the Time To Conversion value in the bubble charts"
            )
            # Function to generate random color (if not already cached)
            def random_color():
                return "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            # Generate colors for events if not already cached
            if 'events_color_map_rulebased' not in st.session_state:
                st.session_state['events_color_map_rulebased'] = {}
            
            # Ensure all current events have colors
            # Use the actual event column from the dataframe (first column)
            actual_evt_col = dfattrib.columns[0]  # First column is always the event column from the SQL query
            
            unique_events = dfattrib[actual_evt_col].unique()
            for event in unique_events:
                if event not in st.session_state['events_color_map_rulebased']:
                    st.session_state['events_color_map_rulebased'][event] = random_color()
            
            # Map colors using the actual column name
            dfattrib['color'] = dfattrib[actual_evt_col].map(st.session_state['events_color_map_rulebased']).fillna('#808080')
            # Define y-axis column dynamically based on selected time unit
            y_col = f"Time To Conversion ({timetoconversionunit})"
            # ✅ Function to create bubble chart (DRY principle)
            def create_bubble_chart(x_col, y_col, title):
                # Define model colors matching the bar chart (for TITLES only)
                model_title_colors = {
                    'Last Click': '#29B5E8',
                    'First Click': '#75CDD7', 
                    'Uniform': '#FF9F36',  # Orange like in bar chart
                    'Exponential Decay': '#11567F',
                    'U Shape': '#7d44cf',
                    'Markov Conversions': '#5B5B5B'  # Gray like in bar chart
                }
                
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[actual_evt_col] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],  # Use event-based colors for bubbles
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                # Extract model name for title color
                model_name = x_col.replace(' Conversions', '').replace(' Values', '') if 'Conversions' in x_col or 'Values' in x_col else x_col
                title_color = model_title_colors.get(model_name, model_title_colors.get(x_col, '#000000'))
                
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': f"<span style='color:{title_color}'>{title}</span>",
                        'font': {'size': 12, 'color': title_color, 'family': 'Arial'}
                    }
                )
                return fig
            
            # Use the actual event column from the dataframe (same as above)
            unique_events = dfattrib[actual_evt_col].unique()
            events_color_map = {event: f"#{random.randint(0, 0xFFFFFF):06x}" for event in unique_events}
            # Bar chart summary
            model_colors = {
                'Last Click': '#29B5E8',  
                'First Click': '#75CDD7',  
                'Uniform': '#FF9F36',  
                'U Shape':'#7d44cf',
                'Exponential Decay': '#11567F'  
                
            }
            # Initialize figure
            fig = go.Figure()
            # Define models to display
            models = ['Last Click', 'First Click', 'U Shape','Uniform', 'Exponential Decay']
            # Add bars for each model
            for model in models:
                # Ensure the column is numeric before rounding
                y_values = pd.to_numeric(dfattrib[model], errors='coerce').fillna(0)
                text_values = y_values.round(2)
                
                fig.add_trace(go.Bar(
                    x=dfattrib[actual_evt_col],  # Events on x-axis
                    y=y_values,  # Model values on y-axis
                    name=model,
                    marker_color=model_colors[model],  # Use consistent color by model
                    text=text_values,  # Display value on top of the bar
                    textposition='outside',  # Place text above the bar
                    textfont=dict(size=9),  # Consistent text size for bar labels
                ))
            # Add vertical separators between events
            for i in range(len(dfattrib[actual_evt_col]) - 1):
                fig.add_shape(
                    type="line",
                    x0=i + 0.5,
                    x1=i + 0.5,
                    y0=0,
                    y1=max(dfattrib[models].max()),  # Set height based on data range
                    line=dict(color="lightgray", width=1, dash="dash")
                )
            # Update layout
            fig.update_layout(
                title={
                    'text': "",
                    'font': {
                        'size': 16,
                        'color': 'black',
                        'family': 'Arial'
                    },
                    'x': 0.5  # Center the title
                },
                xaxis=dict(
                    title="",
                    tickangle=45,  # Rotate x-axis labels
                    tickfont=dict(size=10),
                    tickmode="array",
                    tickvals=list(range(len(dfattrib[dfattrib.columns[0]]))),
                    ticktext=dfattrib[dfattrib.columns[0]]
                ),
                yaxis=dict(
                    title="Attribution Score",
                    showgrid=True,
                    zeroline=True,
                    tickfont=dict(size=10)
                ),
                barmode='group',  # Grouped bars
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                height=600,  # Increased height
                margin=dict(l=50, r=50, t=80, b=150),  # Better margins
                legend=dict(
                    title="Model",
                    orientation="h",
                    yanchor="top",
                    y=-0.25,  # Moved legend further down
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )
            # Display chart below the table
            with st.container():
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Models Summary</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
                st.write("")
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                with col2:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)
        
            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def attribution_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for attribution analysis AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_rulebased_custom_prompt",
                        help="Enter your custom analysis prompt. The attribution data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare attribution data for analysis
                        def attrib_row_to_text(row):
                            return (
                                f"{row[conv]}: Count={row['Count']}, "
                                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                                f"Time To Conversion (Day)={row['Time To Conversion (Day)']:.2f}"
                            )
                        
                        # Get all attribution results for comprehensive analysis
                        attribution_text = "\n".join([attrib_row_to_text(row) 
                                                     for _, row in dfattrib.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive rule-based attribution analysis results:
                            
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
                            
                            Please provide insights on:
                            1. Most significant attribution patterns across rule-based models
                            2. Comparison between different rule-based approaches (First Click, Last Click, Uniform, etc.)
                            3. Time to conversion patterns and implications
                            4. Recommendations for marketing optimization
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 rule-based attribution analysis results:
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_rulebased_ai", 
                "Select the LLM model for AI analysis of rule-based attribution results",
                ai_content_callback=attribution_ai_analysis_callback
            )

    elif model == 'Shapley Value':
        if st.button("Run Attribution Analysis", key="run_shapley", help="Detailed Scores Table and Shapley Values Chart"):
            # Only compute when button is clicked
            with st.spinner("Computing Shapley value attribution... This may take a few moments."):
                shapley_results = shapley_attribution()
                shapley_df = pd.DataFrame.from_dict(shapley_results, orient='index', columns=['Shapley Values'])
                shapley_df.index.name = conv
                shapley_df.reset_index(inplace=True)
                st.session_state['shapley_df'] = shapley_df
        
        # Display results if they exist in session state
        if 'shapley_df' in st.session_state:
            shapley_df = st.session_state['shapley_df']
            
            # Ensure Shapley Values column is numeric
            shapley_df['Shapley Values'] = pd.to_numeric(shapley_df['Shapley Values'], errors='coerce').fillna(0)
            
            # Sort by Shapley Values highest to lowest
            shapley_df = shapley_df.sort_values('Shapley Values', ascending=False)
            
            # Check if results are empty or all zeros due to gap filtering
            if len(shapley_df) == 0 or shapley_df['Shapley Values'].sum() == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    with st.expander("**Debugging Information**", icon=":material/lightbulb:"):
                        st.write("**Possible reasons for no results:**")
                        st.write("- Gap window too restrictive for your customer journey patterns")
                        st.write("- Most conversions take longer than the specified gap period")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()
            
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 5px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(shapley_df, use_container_width=True)
            
            # Add writeback functionality
            add_writeback_functionality(shapley_df, "shapley")
            
            # Bar Chart
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Model Summary</h2>
                        <hr style='margin-top: -8px;margin-bottom: 5px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
            st.write("")
            # Create Shapley Values bar chart
            # Ensure the column is numeric before rounding
            y_values = pd.to_numeric(shapley_df['Shapley Values'], errors='coerce').fillna(0)
            text_values = y_values.round(2)
            
            fig_shapley = go.Figure(data=[
                go.Bar(
                    x=shapley_df[conv],
                    y=y_values,
                    marker_color='#29B5E8',
                    text=text_values,
                    textposition='outside',
                    textfont=dict(size=9)  # Consistent text size for bar labels
                )
            ])
            
            fig_shapley.update_layout(
                xaxis=dict(
                    title="",
                    tickangle=45,  # Rotate x-axis labels for better readability
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Attribution Score (%)",
                    tickfont=dict(size=10)
                ),
                showlegend=False,
                height=600,  # Increased height
                margin=dict(l=50, r=50, t=80, b=150),  # Better margins
                font=dict(size=11)
            )
            with st.container(border=True):
                st.plotly_chart(fig_shapley, use_container_width=True)

            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def shapley_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for Shapley values AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_shapley_custom_prompt",
                        help="Enter your custom analysis prompt. The Shapley values data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare Shapley data for analysis
                        def shapley_row_to_text(row):
                            shapley_val = pd.to_numeric(row['Shapley Values'], errors='coerce')
                            if pd.isna(shapley_val):
                                shapley_val = 0.0
                            return (
                                f"{row[conv]}: Shapley Value={shapley_val:.2f}%. "
                                f"This represents the fair marginal contribution of this touchpoint to conversions."
                            )
                        
                        # Get all Shapley results for comprehensive analysis
                        shapley_text = "\n".join([shapley_row_to_text(row) 
                                                 for _, row in shapley_df.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive Shapley value attribution results:
                            
                            {shapley_text}
                            
                            Total Shapley Results: {len(shapley_df)}
                            
                            Please provide insights on:
                            1. Most significant touchpoints based on Shapley values
                            2. Fair attribution distribution across the customer journey
                            3. Comparison with traditional attribution models
                            4. Strategic recommendations for touchpoint optimization
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 Shapley value attribution results:
                            {shapley_text}
                            
                            Total Shapley Results: {len(shapley_df)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_shapley_ai", 
                "Select the LLM model for AI analysis of Shapley value results",
                ai_content_callback=shapley_ai_analysis_callback
            )

    elif model == 'Rule Based & Shapley Value':
        # Function to compute both models and store in session state
        def compute_rulebased_shapley():
            # Always recompute when called (button clicked)
            attribution = rulebased()
            shapley_results = shapley_attribution()
            
            dfattrib = pd.DataFrame(attribution)
            shapley_df = pd.DataFrame.from_dict(shapley_results, orient='index', columns=['Shapley Values'])
            shapley_df.index.name = conv
            shapley_df.reset_index(inplace=True)
            
            # Ensure Shapley Values column is numeric
            shapley_df['Shapley Values'] = pd.to_numeric(shapley_df['Shapley Values'], errors='coerce').fillna(0)
            
            # Merge Shapley results
            dfattrib = dfattrib.merge(shapley_df, on=conv, how='left')
            desired_position = 6
            column_to_move = dfattrib.pop('Shapley Values')
            dfattrib.insert(desired_position, 'Shapley Values', column_to_move)
            
            # Ensure all model columns are numeric
            model_columns = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape', 'Shapley Values']
            for col in model_columns:
                if col in dfattrib.columns:
                    dfattrib[col] = pd.to_numeric(dfattrib[col], errors='coerce').fillna(0)
            
            st.session_state['dfattrib_rulebased_shapley'] = dfattrib
        
        # Track last-used parameters for cache invalidation
        last_tbl = st.session_state.get('last_tbl_shapley')
        last_conv = st.session_state.get('last_conv_shapley')
        last_conv_value = st.session_state.get('last_conv_value_shapley')
        last_excl3 = st.session_state.get('last_excl3_shapley')
        last_database = st.session_state.get('last_database_shapley')
        last_schema = st.session_state.get('last_schema_shapley')
        last_strategy = st.session_state.get('last_strategy_shapley')
        last_maxnbbevt = st.session_state.get('last_maxnbbevt_shapley')
        last_max_gap_days = st.session_state.get('last_max_gap_days_shapley')
        last_uid = st.session_state.get('last_uid_shapley')
        last_evt = st.session_state.get('last_evt_shapley')
        last_tmstp = st.session_state.get('last_tmstp_shapley')
        last_startdt = st.session_state.get('last_startdt_shapley')
        last_enddt = st.session_state.get('last_enddt_shapley')
        last_shapley_samples = st.session_state.get('last_shapley_samples')
        last_sql_where_clause = st.session_state.get('last_sql_where_clause_shapley')
        
        # If any key parameter changed, invalidate cached results
        if (tbl != last_tbl or conv != last_conv or conv_value != last_conv_value or 
            excl3 != last_excl3 or database != last_database or schema != last_schema or 
            window_strategy != last_strategy or maxnbbevt != last_maxnbbevt or 
            max_gap_days != last_max_gap_days or uid != last_uid or evt != last_evt or 
            tmstp != last_tmstp or str(startdt_input) != str(last_startdt) or 
            str(enddt_input) != str(last_enddt) or 
            st.session_state.get('shapley_samples') != last_shapley_samples or
            sql_where_clause != last_sql_where_clause):
            if 'dfattrib_rulebased_shapley' in st.session_state:
                del st.session_state['dfattrib_rulebased_shapley']
        
        # Update session state with current selections
        st.session_state['last_tbl_shapley'] = tbl
        st.session_state['last_conv_shapley'] = conv
        st.session_state['last_conv_value_shapley'] = conv_value
        st.session_state['last_excl3_shapley'] = excl3
        st.session_state['last_database_shapley'] = database
        st.session_state['last_schema_shapley'] = schema
        st.session_state['last_strategy_shapley'] = window_strategy
        st.session_state['last_maxnbbevt_shapley'] = maxnbbevt
        st.session_state['last_max_gap_days_shapley'] = max_gap_days
        st.session_state['last_uid_shapley'] = uid
        st.session_state['last_evt_shapley'] = evt
        st.session_state['last_tmstp_shapley'] = tmstp
        st.session_state['last_startdt_shapley'] = startdt_input
        st.session_state['last_enddt_shapley'] = enddt_input
        st.session_state['last_shapley_samples'] = st.session_state.get('shapley_samples')
        st.session_state['last_sql_where_clause_shapley'] = sql_where_clause
        
        if st.button("Run Attribution Analysis", key="run_rulebased_shapley", help="Detailed Scores Table, Attribution Models Summary Bar Chart and Models Charts"):
            # Call computation when button is clicked
            with st.spinner("Computing rule-based and Shapley value attribution models... This may take a few moments."):
                compute_rulebased_shapley()
                dfattrib = st.session_state['dfattrib_rulebased_shapley']
        
        # Display results if they exist in session state
        if 'dfattrib_rulebased_shapley' in st.session_state:
            dfattrib = st.session_state['dfattrib_rulebased_shapley']
            
            # Check if results are empty due to gap filtering
            if len(dfattrib) == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    # Optionally show debug info
                    with st.expander("Debug Information", icon=":material/lightbulb:"):
                        st.write("This means all conversion paths in your data have gaps longer than the specified window.")
                        st.write("**Common reasons:**")
                        st.write("- Long consideration periods in your customer journey")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(dfattrib, use_container_width=True)
            
            # Add writeback functionality
            add_writeback_functionality(dfattrib, "rule_shapley")
            
            # Attribution Models Comparison title with consistent formatting
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Models Comparison (including Shapley Values)</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            
            # Enhanced bar chart with Shapley Values
            model_colors = {
                'Last Click': '#29B5E8',
                'First Click': '#75CDD7',
                'Uniform': '#FF9F36',
                'Exponential Decay': '#11567F',
                'U Shape': '#7d44cf',
                'Shapley Values': '#FF6B6B'
            }
            
            fig = go.Figure()
            models = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape', 'Shapley Values']
            
            for model_name in models:
                if model_name in dfattrib.columns:
                    # Ensure the column is numeric before rounding
                    y_values = pd.to_numeric(dfattrib[model_name], errors='coerce').fillna(0)
                    text_values = y_values.round(2)
                    
                    fig.add_trace(go.Bar(
                        x=dfattrib[evt],
                        y=y_values,
                        name=model_name,
                        marker_color=model_colors[model_name],
                        text=text_values,
                        textposition='outside',
                        textfont=dict(size=9),  # Consistent text size for bar labels
                    ))
            
            fig.update_layout(
                xaxis=dict(
                    title="",
                    tickangle=45,  # Rotate x-axis labels for better readability
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Attribution Score",
                    tickfont=dict(size=10)
                ),
                barmode='group',
                height=600,  # Increased height
                margin=dict(l=50, r=50, t=80, b=150),  # Better margins
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,  # Moved legend further down
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

            # Bubble Charts section
            # Set up color mapping for events (needed for bubble chart)
            # Ensure events_color_map exists in session state
            if 'events_color_map' not in st.session_state:
                st.session_state['events_color_map'] = {}
                unique_events = dfattrib[evt].unique()
                for event in unique_events:
                    st.session_state['events_color_map'][event] = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            
            dfattrib['color'] = dfattrib[evt].map(st.session_state['events_color_map'])
            y_col = "Time To Conversion (Day)"  # Default time column
            def create_bubble_chart(x_col, y_col, title, color_map=None):
                # Define model colors matching the bar chart (for TITLES only)
                model_title_colors = {
                    'Last Click': '#29B5E8',
                    'First Click': '#75CDD7',
                    'Uniform': '#FF9F36',  # Orange like in bar chart
                    'Exponential Decay': '#11567F',
                    'U Shape': '#7d44cf',
                    'Shapley Values': '#FF6B6B'  # Red like in bar chart
                }
                
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[evt] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],  # Use event-based colors for bubbles
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                # Extract model name for title color (handle "Shapley Values" case)
                model_name = x_col.replace(' Values', '') if 'Values' in x_col else x_col
                title_color = model_title_colors.get(model_name, model_title_colors.get(x_col, '#000000'))
                
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': f"<span style='color:{title_color}'>{title}</span>",
                        'font': {'size': 12, 'color': title_color, 'family': 'Arial'}
                    }
                )
                return fig

            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)
                    
                with col2:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)      
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Shapley Values', y_col, "Shapley Values Model"), use_container_width=True)

            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def rulebased_shapley_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for Rule-Based & Shapley attribution AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_rulebased_shapley_custom_prompt",
                        help="Enter your custom analysis prompt. The rule-based and Shapley values data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare combined attribution data for analysis
                        def attrib_shapley_row_to_text(row):
                            shapley_val = pd.to_numeric(row.get('Shapley Values', 0), errors='coerce')
                            if pd.isna(shapley_val):
                                shapley_val = 0.0
                            return (
                                f"{row[conv]}: Count={row['Count']}, "
                                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                                f"U Shape={row['U Shape']}, Shapley Values={shapley_val:.2f}"
                            )
                        
                        # Get all combined attribution results for comprehensive analysis
                        attribution_text = "\n".join([attrib_shapley_row_to_text(row) 
                                                     for _, row in dfattrib.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive rule-based and Shapley value attribution results:
                            
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
                            
                            Please provide insights on:
                            1. Comparison between rule-based models and Shapley values
                            2. Most significant touchpoints across all attribution methods
                            3. Fair attribution vs traditional models - key differences
                            4. Strategic recommendations for comprehensive attribution strategy
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 rule-based and Shapley value attribution results:
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_rulebased_shapley_ai", 
                "Select the LLM model for AI analysis of rule-based and Shapley value results",
                ai_content_callback=rulebased_shapley_ai_analysis_callback
            )

    elif model == 'All Models':
        # Function to compute all models
        def compute_all_models():
            # Always recompute when called (button clicked)
            attribution = rulebased()
            mcmodel = markovchain()
            shapley_results = shapley_attribution()
            
            dfattrib = pd.DataFrame(attribution)
            
            # Add Markov results
            markov_df = pd.DataFrame.from_dict(mcmodel['markov_conversions'], orient='index', columns=['Markov Conversions'])
            markov_df.index.name = conv
            markov_df.reset_index(inplace=True)
            dfattrib = dfattrib.merge(markov_df, on=conv, how='left')
            
            # Add Shapley results
            shapley_df = pd.DataFrame.from_dict(shapley_results, orient='index', columns=['Shapley Values'])
            shapley_df.index.name = conv
            shapley_df.reset_index(inplace=True)
            
            # Ensure Shapley Values column is numeric
            shapley_df['Shapley Values'] = pd.to_numeric(shapley_df['Shapley Values'], errors='coerce').fillna(0)
            
            dfattrib = dfattrib.merge(shapley_df, on=conv, how='left')
            
            # Reorder columns
            desired_position = 6
            markov_column = dfattrib.pop('Markov Conversions')
            shapley_column = dfattrib.pop('Shapley Values')
            dfattrib.insert(desired_position, 'Markov Conversions', markov_column)
            dfattrib.insert(desired_position + 1, 'Shapley Values', shapley_column)
            
            # Ensure all model columns are numeric
            model_columns = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape', 'Markov Conversions', 'Shapley Values']
            for col in model_columns:
                if col in dfattrib.columns:
                    dfattrib[col] = pd.to_numeric(dfattrib[col], errors='coerce').fillna(0)
            
            st.session_state['dfattrib_all'] = dfattrib
        
        # Track last-used parameters for cache invalidation
        last_tbl = st.session_state.get('last_tbl_all')
        last_conv = st.session_state.get('last_conv_all')
        last_conv_value = st.session_state.get('last_conv_value_all')
        last_excl3 = st.session_state.get('last_excl3_all')
        last_database = st.session_state.get('last_database_all')
        last_schema = st.session_state.get('last_schema_all')
        last_strategy = st.session_state.get('last_strategy_all')
        last_maxnbbevt = st.session_state.get('last_maxnbbevt_all')
        last_max_gap_days = st.session_state.get('last_max_gap_days_all')
        last_uid = st.session_state.get('last_uid_all')
        last_evt = st.session_state.get('last_evt_all')
        last_tmstp = st.session_state.get('last_tmstp_all')
        last_startdt = st.session_state.get('last_startdt_all')
        last_enddt = st.session_state.get('last_enddt_all')
        last_shapley_samples = st.session_state.get('last_shapley_samples_all')
        last_sql_where_clause = st.session_state.get('last_sql_where_clause_all')
        
        # If any key parameter changed, invalidate cached results
        if (tbl != last_tbl or conv != last_conv or conv_value != last_conv_value or 
            excl3 != last_excl3 or database != last_database or schema != last_schema or 
            window_strategy != last_strategy or maxnbbevt != last_maxnbbevt or 
            max_gap_days != last_max_gap_days or uid != last_uid or evt != last_evt or 
            tmstp != last_tmstp or str(startdt_input) != str(last_startdt) or 
            str(enddt_input) != str(last_enddt) or 
            st.session_state.get('shapley_samples') != last_shapley_samples or
            sql_where_clause != last_sql_where_clause):
            if 'dfattrib_all' in st.session_state:
                del st.session_state['dfattrib_all']
        
        # Update session state with current selections
        st.session_state['last_tbl_all'] = tbl
        st.session_state['last_conv_all'] = conv
        st.session_state['last_conv_value_all'] = conv_value  
        st.session_state['last_excl3_all'] = excl3
        st.session_state['last_database_all'] = database
        st.session_state['last_schema_all'] = schema
        st.session_state['last_strategy_all'] = window_strategy
        st.session_state['last_maxnbbevt_all'] = maxnbbevt
        st.session_state['last_max_gap_days_all'] = max_gap_days
        st.session_state['last_uid_all'] = uid
        st.session_state['last_evt_all'] = evt
        st.session_state['last_tmstp_all'] = tmstp
        st.session_state['last_startdt_all'] = startdt_input
        st.session_state['last_enddt_all'] = enddt_input
        st.session_state['last_shapley_samples_all'] = st.session_state.get('shapley_samples')
        st.session_state['last_sql_where_clause_all'] = sql_where_clause
        
        if st.button("Run Attribution Analysis", key="run_all_models", help="Comprehensive Attribution Analysis with All Models"):
            # Call computation when button is clicked
            with st.spinner("Computing comprehensive attribution analysis with all models... This may take a few moments."):
                compute_all_models()
                dfattrib = st.session_state['dfattrib_all']
        
        # Display results if they exist in session state
        if 'dfattrib_all' in st.session_state:
            dfattrib = st.session_state['dfattrib_all']
            
            # Check if results are empty due to gap filtering
            if len(dfattrib) == 0:
                if max_gap_days is not None:
                    st.warning(f"No conversion paths found within the {max_gap_days}-day gap window.",icon=":material/warning:")
                    st.info(f"**Suggestions:**\n- Try increasing the gap window (e.g., 14, 30, or 60 days)\n- Switch to 'Most Events Window' strategy\n- Check if your conversion cycles are typically longer than {max_gap_days} days",icon=":material/lightbulb:")
                    
                    # Optionally show debug info
                    with st.expander("Debug Information", icon=":material/lightbulb:"):
                        st.write("This means all conversion paths in your data have gaps longer than the specified window.")
                        st.write("**Common reasons:**")
                        st.write("- Long consideration periods in your customer journey")
                        st.write("- Seasonal patterns or purchase cycles")
                        st.write("- Data sparsity (few touchpoints per customer)")
                else:
                    st.warning("No conversion paths found. Please check your data filters and date range.",icon=":material/warning:")
                st.stop()
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Comprehensive Attribution Analysis</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.dataframe(dfattrib, use_container_width=True)
            
            # Add writeback functionality
            add_writeback_functionality(dfattrib, "all_models")
            
            # Attribution Models Comparison title with consistent formatting
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Complete Attribution Models Comparison</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            # Comprehensive model comparison chart
            model_colors = {
                'Last Click': '#29B5E8',
                'First Click': '#75CDD7', 
                'Uniform': '#FF9F36',
                'Exponential Decay': '#11567F',
                'U Shape': '#7d44cf',
                'Markov Conversions': '#5B5B5B',
                'Shapley Values': '#FF6B6B'
            }
            
            fig = go.Figure()
            models = ['Last Click', 'First Click', 'Uniform', 'Exponential Decay', 'U Shape', 'Markov Conversions', 'Shapley Values']
            
            for model_name in models:
                if model_name in dfattrib.columns:
                    # Ensure the column is numeric before rounding
                    y_values = pd.to_numeric(dfattrib[model_name], errors='coerce').fillna(0)
                    text_values = y_values.round(2)
                    
                    fig.add_trace(go.Bar(
                        x=dfattrib[evt],
                        y=y_values,
                        name=model_name,
                        marker_color=model_colors[model_name],
                        text=text_values,
                        textposition='outside',
                        textfont=dict(size=9),  # Consistent text size for bar labels
                    ))
            
            fig.update_layout(
                xaxis=dict(
                    title="",
                    tickangle=45,  # Rotate x-axis labels for better readability
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Attribution Score",
                    tickfont=dict(size=10)
                ),
                barmode='group',
                height=700,  # Increased height to accommodate legend
                margin=dict(l=50, r=50, t=80, b=150),  # Increased bottom margin for rotated labels and legend
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,  # Moved legend further down
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                # Improve text label appearance
                font=dict(size=11),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            with st.container(border=True):
             st.plotly_chart(fig, use_container_width=True)

            # Bubble Charts section
            # Set up color mapping for events (needed for bubble chart)
            # Ensure events_color_map exists in session state
            if 'events_color_map' not in st.session_state:
                st.session_state['events_color_map'] = {}
                unique_events = dfattrib[evt].unique()
                for event in unique_events:
                    st.session_state['events_color_map'][event] = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            
            dfattrib['color'] = dfattrib[evt].map(st.session_state['events_color_map'])
            y_col = "Time To Conversion (Day)"  # Default time column
            def create_bubble_chart(x_col, y_col, title, color_map=None):
                # Define model colors matching the bar chart (for TITLES only)
                model_title_colors = {
                    'Last Click': '#29B5E8',
                    'First Click': '#75CDD7',
                    'Uniform': '#FF9F36',  # Orange like in bar chart
                    'Exponential Decay': '#11567F',
                    'U Shape': '#7d44cf',
                    'Markov Conversions': '#5B5B5B',  # Gray like in bar chart
                    'Shapley Values': '#FF6B6B'  # Red like in bar chart
                }
                
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[evt] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],  # Use event-based colors for bubbles
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                # Extract model name for title color (handle "Markov Conversions" and "Shapley Values" cases)
                model_name = x_col.replace(' Conversions', '').replace(' Values', '') if 'Conversions' in x_col or 'Values' in x_col else x_col
                title_color = model_title_colors.get(model_name, model_title_colors.get(x_col, '#000000'))
                
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': f"<span style='color:{title_color}'>{title}</span>",
                        'font': {'size': 12, 'color': title_color, 'family': 'Arial'}
                    }
                )
                return fig

            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                """, unsafe_allow_html=True)
            st.write("")
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Shapley Values', y_col, "Shapley Values Model"), use_container_width=True)
                    
                with col2:
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)      
                    with st.container(border=True):
                        st.plotly_chart(create_bubble_chart('Markov Conversions', y_col, "Markov Conversions Model"), use_container_width=True)

            # AI-Powered Insights with model selection (only show if Show me! toggle is on)
            def all_models_ai_analysis_callback(selected_model, prompt_type):
                """Callback function for All Models attribution AI insights"""
                
                # Show custom prompt input if Custom is selected
                if prompt_type == "Custom":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        value="",
                        key="attribution_all_models_custom_prompt",
                        help="Enter your custom analysis prompt. The comprehensive attribution data will be automatically included.",
                        placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                    )
                    
                    # Only proceed if custom prompt is not empty
                    if not custom_prompt or custom_prompt.strip() == "":
                        st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                        return
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Prepare comprehensive attribution data for analysis
                        def attrib_all_row_to_text(row):
                            markov_val = pd.to_numeric(row.get('Markov Conversions', 0), errors='coerce')
                            shapley_val = pd.to_numeric(row.get('Shapley Values', 0), errors='coerce')
                            if pd.isna(markov_val):
                                markov_val = 0.0
                            if pd.isna(shapley_val):
                                shapley_val = 0.0
                            return (
                                f"{row[conv]}: Count={row['Count']}, "
                                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                                f"U Shape={row['U Shape']}, Markov={markov_val:.2f}, "
                                f"Shapley Values={shapley_val:.2f}"
                            )
                        
                        # Get all comprehensive attribution results for exhaustive analysis
                        attribution_text = "\n".join([attrib_all_row_to_text(row) 
                                                     for _, row in dfattrib.iterrows()])
                        
                        if prompt_type == "Auto":
                            ai_prompt = f"""
                            Analyze these comprehensive attribution results across all models:
                            
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
                            
                            Please provide insights on:
                            1. Comprehensive comparison across all attribution models (Rule-based, Markov, Shapley)
                            2. Most significant touchpoints and their performance across different methodologies
                            3. Strategic insights from combining traditional, probabilistic, and game-theory approaches
                            4. Recommendations for optimal attribution strategy based on all model results
                            
                            Keep your analysis concise and actionable.
                            """
                        else:  # Custom
                            ai_prompt = f"""
                            {custom_prompt}
                            
                            Data to analyze - Top 10 comprehensive attribution results across all models:
                            {attribution_text}
                            
                            Total Attribution Results: {len(dfattrib)}
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
            ai_model, ai_enabled, prompt_type = display_ai_insights_section(
                "attribution_all_models_ai", 
                "Select the LLM model for AI analysis of comprehensive attribution results",
                ai_content_callback=all_models_ai_analysis_callback
            )
                    
    else:
        st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200 ; margin-top: 0px; margin-bottom: -15px;">
                    Please select one or more modeling technique
                </h5>
            </div>
            """, unsafe_allow_html=True)
        #st.warning('Please select one or more modeling technique')
        
else:   
        st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200 ; margin-top: 0px; margin-bottom: -15px;">
                    Please ensure all required inputs are selected before running the app. Performance Tip: For optimal performance when running Markov Chain or Shapley Value models, use a Snowpark-optimized warehouse to leverage enhanced computational capabilities for complex probabilistic calculations.
                </h5>
            </div>
            """, unsafe_allow_html=True)
        #st.warning("Please ensure all required inputs are selected before running the app.")

        
 

