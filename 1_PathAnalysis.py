
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

# AI Helper Functions (from Pattern Mining)
@st.cache_data(ttl=3600)  # Cache for 1 hour
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
            ai_enabled = st.toggle("Expl**AI**n Me!", key=f"{key_prefix}_toggle", help="Generate AI insights and recommendations for your path analysis results. **Auto**: Pre-built analysis with structured insights. **Custom**: Enter your own questions and prompts for personalized analysis.")
        
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

#st.set_page_config(layout="wide")

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
 #--------------------------------------
 #SANKEY VIZ
 #--------------------------------------
 # Function for Sankey plot
def sankeyPlot(res, direction, title_text="Sankey nPath", topN=15):
    npath_pandas = res.copy()
    if topN:
        npath_pandas = npath_pandas.sort_values(by='COUNT', ascending=False).head(topN)

    if direction == "from":
        dataDict = defaultdict(int)
        for index, row in npath_pandas.iterrows():
            pathCnt = row['COUNT']
            rowList = [item.strip() for item in row['PATH'].split(',')]
            for i in range(len(rowList) - 1):
                leftValue = rowList[i] + str(i)
                rightValue = rowList[i + 1] + str(i + 1)
                valuePair = leftValue + '+' + rightValue
                dataDict[valuePair] += pathCnt
        
        eventList = []
        for key in dataDict.keys():
            leftValue, rightValue = key.split('+')
            if leftValue not in eventList:
                eventList.append(leftValue)
            if rightValue not in eventList:
                eventList.append(rightValue)
        
        sankeyLabel = [s[:-1] for s in eventList]
        sankeySource = []
        sankeyTarget = []
        sankeyValue = []
         
        for key, val in dataDict.items():
             sankeySource.append(eventList.index(key.split('+')[0]))
             sankeyTarget.append(eventList.index(key.split('+')[1]))
             sankeyValue.append(val)
         
        sankeyColor = []
        for i in sankeyLabel:
             sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
         
         # Updated Sankey configuration with font properties
        data = go.Sankey(
         arrangement = "snap",
         node = dict(
             pad = 15,
             thickness = 20,
             line = dict(color = "black", width = 0.5),
             label = sankeyLabel,
             color = sankeyColor
         ),
         link = dict(
             source = sankeySource,
             target = sankeyTarget,
             value = sankeyValue,
             color = 'light grey'
         ),
         textfont = dict(
             family = "Arial",
             size = 14,
             color = "black"
         )
     )
         
        fig = go.Figure(data)
        fig.update_layout(
             hovermode='closest',
             title=dict(
                 text=title_text,
                 font=dict(size=20, family="Arial", color="black")
             ),
             plot_bgcolor='white',
             paper_bgcolor='white',
             font=dict(
                 family="Arial",
                 size=14,
                 color="black"
             )
        )
        st.plotly_chart(fig)

    elif direction == "to":
        dataDict = defaultdict(int)
        eventDict = defaultdict(int)
        maxPath = npath_pandas['COUNT'].max()
        
        for index, row in npath_pandas.iterrows():
            rowList = row['PATH'].split(',')
            pathCnt = row['COUNT']
            pathLen = len(rowList)
            for i in range(len(rowList) - 1):
                leftValue = str(150 + i + maxPath - pathLen) + rowList[i].strip()
                rightValue = str(150 + i + 1 + maxPath - pathLen) + rowList[i + 1].strip()
                valuePair = leftValue + '+' + rightValue
                dataDict[valuePair] += pathCnt
                eventDict[leftValue] += 1
                eventDict[rightValue] += 1

        eventList = []
        for key, val in eventDict.items():
            eventList.append(key)
        sortedEventList = sorted(eventList)

        sankeyLabel = []
        for event in sortedEventList:
            sankeyLabel.append(event[3:])
 
        sankeySource = []
        sankeyTarget = []
        sankeyValue = []
        for key, val in dataDict.items():
             sankeySource.append(sortedEventList.index(key.split('+')[0]))
             sankeyTarget.append(sortedEventList.index(key.split('+')[1]))
             sankeyValue.append(val)
 
        sankeyColor = []
        for i in sankeyLabel:
             sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
 
         # Updated Sankey configuration with font properties
             data = go.Sankey(
              arrangement = "snap",
         node = dict(
             align="right",
             pad = 15,
             thickness = 20,
             line = dict(color = "black", width = 0.5),
             label = sankeyLabel,
             color = sankeyColor
         ),
         link = dict(
             source = sankeySource,
             target = sankeyTarget,
             value = sankeyValue,
             color = 'light grey'
         ),
         textfont = dict(
             family = "Arial",
             size = 14,
             color = "black"
         )
     )
         
        fig = go.Figure(data)
        fig.update_layout(
             hovermode='closest',
             title=dict(
                 text=title_text,
                 font=dict(size=20, family="Arial", color="black")
             ),
             plot_bgcolor='white',
             paper_bgcolor='white',
             font=dict(
                 family="Arial",
                 size=14,
                 color="black"
             )
         )
        st.plotly_chart(fig)
 
    elif direction == "no_direction":
        dataDict = defaultdict(int)
        for index, row in npath_pandas.iterrows():
            pathCnt = row['COUNT']
            rowList = [item.strip() for item in row['PATH'].split(',')]
            for i in range(len(rowList) - 1):
                leftValue = rowList[i] + str(i)
                rightValue = rowList[i + 1] + str(i + 1)
                valuePair = leftValue + '+' + rightValue
                dataDict[valuePair] += pathCnt

        eventList = []
        for key in dataDict.keys():
            leftValue, rightValue = key.split('+')
            if leftValue not in eventList:
                eventList.append(leftValue)
            if rightValue not in eventList:
                eventList.append(rightValue)

        sankeyLabel = [s[:-1] for s in eventList]
        sankeySource = []
        sankeyTarget = []
        sankeyValue = []
        for key, val in dataDict.items():
            sankeySource.append(eventList.index(key.split('+')[0]))
            sankeyTarget.append(eventList.index(key.split('+')[1]))
            sankeyValue.append(val)

        sankeyColor = []
        for i in sankeyLabel:
            sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))

        # Updated Sankey configuration with font properties
        data = go.Sankey(
            arrangement = "snap",
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = sankeyLabel,
                color = sankeyColor
            ),
            link = dict(
                source = sankeySource,
                target = sankeyTarget,
                value = sankeyValue,
                color = 'light grey'
            ),
            textfont = dict(
                family = "Arial",
                size = 14,
                color = "black"
            )
        )
        
        fig = go.Figure(data)
        fig.update_layout(
            hovermode='closest',
            title=dict(
                text=title_text,
                font=dict(size=20, family="Arial", color="black")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(
                family="Arial",
                size=14,
                color="black"
            )
        )
        st.plotly_chart(fig)

    else:
        st.write("Invalid direction.")
 
 #--------------------------------------
 #INTERACTIVE CLICKABLE SANKEY VIZ
 #--------------------------------------
def sankey_chart(df, direction="from",topN_percentage=100, middle_events=None):
     dataDict = defaultdict(lambda: {"count": 0, "uids": []})  # Store counts + UIDs
     eventDict = defaultdict(int)
     indexed_paths = []
     dropoffDict = defaultdict(int)  # To store drop-off at each node
     df = df.copy()
 
 
     max_paths = df['PATH'].nunique()
     topN = int(max_paths * (topN_percentage / 100))
 
     if topN:
      df = df.sort_values(by='COUNT', ascending=False).head(topN)
     
     if direction == "before_after" and middle_events:
        # Special handling for BEFORE/AFTER pattern - center on middle event
        # Parse middle_events (could be comma-separated string like "'Event1', 'Event2'")
        if isinstance(middle_events, str):
            middle_event_list = [e.strip().strip("'\"") for e in middle_events.split(',')]
        else:
            middle_event_list = [middle_events]
        
        # Find max events before middle event across all paths
        max_before = 0
        for _, row in df.iterrows():
            rowList = [e.strip() for e in row['PATH'].split(',')]
            for middle_event in middle_event_list:
                if middle_event in rowList:
                    pos = rowList.index(middle_event)
                    max_before = max(max_before, pos)
                    break
        
        # Now index paths with middle events aligned at max_before position
        for _, row in df.iterrows():
            rowList = [e.strip() for e in row['PATH'].split(',')]
            pathCnt = row['COUNT']
            uid_list = row['UID_LIST']
            
            # Find middle event position in this path
            middle_pos = -1
            for middle_event in middle_event_list:
                if middle_event in rowList:
                    middle_pos = rowList.index(middle_event)
                    break
            
            if middle_pos == -1:
                continue  # Skip paths without middle event
            
            # Calculate offset to align this path's middle event with max_before position
            offset = max_before - middle_pos
            indexedRowList = [f"{i + offset}_{rowList[i]}" for i in range(len(rowList))]
            indexed_paths.append(",".join(indexedRowList))
            
            for i in range(len(indexedRowList) - 1):
                leftValue = indexedRowList[i]
                rightValue = indexedRowList[i + 1]
                valuePair = leftValue + '|||' + rightValue
                dataDict[valuePair]["count"] += pathCnt
                dataDict[valuePair]["uids"].extend(uid_list)
                eventDict[leftValue] += pathCnt
                eventDict[rightValue] += pathCnt
        
        # Compute tooltips
        for key, val in dataDict.items():
            source_node, target_node = key.split('|||')
            total_at_source = eventDict[source_node]
            forward_percentage = (val["count"] / total_at_source * 100) if total_at_source > 0 else 0
            source_parts = source_node.split('_', 1)
            target_parts = target_node.split('_', 1)
            source_display = source_parts[1] if len(source_parts) > 1 else source_node
            target_display = target_parts[1] if len(target_parts) > 1 else target_node
            val["tooltip"] = f"""
                Path: {source_display} → {target_display}<br>
                Count: {val["count"]}<br>
                Forward %: {forward_percentage:.2f}%
            """
     
     elif direction == "to":
         maxPath = df['COUNT'].max()
         for _, row in df.iterrows():
             rowList = row['PATH'].split(',')
             pathCnt = row['COUNT']
             uid_list = row['UID_LIST']
             pathLen = len(rowList)
             indexedRowList = [f"{150 + i + maxPath - pathLen}_{rowList[i].strip()}" for i in range(len(rowList))]
             indexed_paths.append(",".join(indexedRowList))
             for i in range(len(indexedRowList) - 1):
                 leftValue = indexedRowList[i]
                 rightValue = indexedRowList[i + 1]
                 valuePair = leftValue + '|||' + rightValue
                 dataDict[valuePair]["count"] += pathCnt
                 dataDict[valuePair]["uids"].extend(uid_list)
                 eventDict[leftValue] += pathCnt
                 eventDict[rightValue] += pathCnt
         for key, val in dataDict.items():
             # Split on delimiter (handles event names containing special chars)
             source_node, target_node = key.split('|||')
             # Safe extraction - handle cases where underscore might be missing
             source_parts = source_node.split('_', 1)
             target_parts = target_node.split('_', 1)
             source_display = source_parts[1] if len(source_parts) > 1 else source_node
             target_display = target_parts[1] if len(target_parts) > 1 else target_node
             tooltip_text = f"""
                 Path: {source_display} → {target_display}<br>
                 Count: {val["count"]}
             """
             val["tooltip"] = tooltip_text
     elif direction == "from":
        for _, row in df.iterrows():
            rowList = row['PATH'].split(',')
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
                eventDict[leftValue] += pathCnt
                eventDict[rightValue] += pathCnt
        # Step 1: Compute drop-offs and forward percentage
        for node in eventDict:
            total_at_node = eventDict[node]
            outgoing = sum(dataDict[f"{node}|||{target}"]["count"] for target in eventDict if f"{node}|||{target}" in dataDict)
            dropoff = total_at_node - outgoing
            dropoffDict[node] = dropoff
        
        # Step 2: Display drop-offs and forward percentages in the tooltip
        for key, val in dataDict.items():
            # Split on delimiter (handles event names containing special chars)
            source_node, target_node = key.split('|||')
            total_at_source = eventDict[source_node]
            forward_percentage = (val["count"] / total_at_source * 100) if total_at_source > 0 else 0
            dropoff_percentage = (dropoffDict[source_node] / total_at_source * 100) if total_at_source > 0 else 0
            # Safe extraction - handle cases where underscore might be missing
            source_parts = source_node.split('_', 1)
            target_parts = target_node.split('_', 1)
            source_display = source_parts[1] if len(source_parts) > 1 else source_node
            target_display = target_parts[1] if len(target_parts) > 1 else target_node
            val["tooltip"] = f"""
                Path: {source_display} → {target_display}<br>
                Count: {val["count"]}<br>
                Forward %: {forward_percentage:.2f}%<br>
                Drop-off %: {dropoff_percentage:.2f}%
            """
     # Ensure all nodes from links are also in the event list
     all_nodes = set(eventDict.keys())
     for key in dataDict.keys():
         source_node, target_node = key.split('|||')
         all_nodes.add(source_node)
         all_nodes.add(target_node)
     
     sortedEventList = sorted(all_nodes)
     # Safe extraction - handle cases where underscore might be missing
     sankeyLabel = []
     for event in sortedEventList:
         parts = event.split('_', 1)
         sankeyLabel.append(parts[1] if len(parts) > 1 else event)
     
     st.session_state["sankey_labels"] = sankeyLabel
     st.session_state["sortedEventList"] = sortedEventList
     st.session_state["sankey_links"] = dataDict
     sankeySource = []
     sankeyTarget = []
     sankeyValue = []
     sankeyLinks = []
     for key, val in dataDict.items():
         # Split on delimiter (handles event names containing special chars)
         source_node, target_node = key.split('|||')
         sankeySource.append(sortedEventList.index(source_node))
         sankeyTarget.append(sortedEventList.index(target_node))
         sankeyValue.append(val["count"])
         sankeyLinks.append({
             "source": sortedEventList.index(source_node),
             "target": sortedEventList.index(target_node),
             "value": val["count"],
             "tooltip": {"formatter": val["tooltip"]},
             "uids": val["uids"],
             "source_node": source_node,
             "target_node": target_node
         })
     options = {
         "tooltip": {"trigger": "item"},
         "series": [{
             "type": "sankey",
             "left": 30.0,
             "top": 20.0,
             "right": 120.0,
             "bottom": 20.0,
             "layout": "none",
             "data": [{"label": {"show": True, "formatter": label, "color": "#888888"}} for node, label in zip(sortedEventList, sankeyLabel)],
             "links": sankeyLinks,
             "lineStyle": {"color": "source", "curveness": 0.5},
             "label": {"color": "#888888"},
             "emphasis": {"focus": "adjacency"}
         }]
     }
     return st_echarts(options=options, height="600px", key=f"sankey_chart_{st.session_state['last_df_hash']}", events={"click": "function(params) { return params.data; }"})
 #----------------------
 # TREE VISUALIZATION
 #----------------------    
def build_tree_hierarchy(df, target, direction="to"):
     """
     Builds a hierarchical structure from path data.
     - If direction = "to", treats target as the final event and merges paths leading to it.
     - If direction = "from", treats target as the starting event and builds diverging branches.
     """
     hierarchy = {}
 
     for _, row in df.iterrows():
         path, count, uids = row["PATH"], row["COUNT"], row["UID_LIST"]
         events = path.split(", ")
 
         if target in events:
             if direction == "to":
                 # Target is the final event (reverse the path)
                 target_index = len(events) - 1  
                 processed_events = events[:target_index + 1][::-1]  
             else:
                 # Target is the first event (preserve sequence)
                 target_index = 0  
                 processed_events = events[target_index:]  
 
             # Build hierarchy structure
             current_level = hierarchy
             for event in processed_events:
                 if event not in current_level:
                     current_level[event] = {"children": {}, "count": 0, "uids": set(), "selected": False}  
                 current_level[event]["count"] += count  
                 current_level[event]["uids"].update(uids)  
                 current_level = current_level[event]["children"]  
 
     return hierarchy
 
def convert_to_echarts_format_t(hierarchy, parent_path="", selected_node=None):
     """
     Converts the hierarchy dictionary into ECharts-compatible format.
     Highlights selected node path.
     """
     def recurse(node, path_prefix):
         return [
             {
                 "name": key,
                 "tooltip": {"formatter": f"{key}: {value['count']}"},
                 "lineStyle": {
                     "width": 1 + math.log(value['count'] + 1),
                     "color": "red" if selected_node and selected_node in path_prefix else "#aaa"
                 },
                 "full_path": f"{path_prefix}, {key}".strip(", "),
                 "uids": list(value.get("uids", set())),  
                 "children": recurse(value["children"], f"{path_prefix}, {key}".strip(", "))
             } if value["children"] else {
                 "name": key,
                 "tooltip": {"formatter": f"{key}: {value['count']}"},
                 "lineStyle": {
                     "width": 1 + math.log(value['count'] + 1),
                     "color": "red" if selected_node and selected_node in path_prefix else "#aaa"
                 },
                 "full_path": f"{path_prefix}, {key}".strip(", "),
                 "uids": list(value.get("uids", set()))  
             }
             for key, value in node.items()
         ]
     return recurse(hierarchy, parent_path)
     
 
def plot_tree(df, target, direction="to", selected_node=None):
    hierarchy = build_tree_hierarchy(df, target, direction)
    echarts_data = convert_to_echarts_format_t(hierarchy, selected_node=selected_node)
    
    # Set tree orientation based on direction
    orient = "RL" if direction == "to" else "LR"
    label_position = "left" if direction == "to" else "right"
    align = "right" if direction == "to" else "left"
    
    # Detect theme and set label styles accordingly
    try:
        theme_type = st.context.theme.type
    except:
        theme_type = "light"  # Default to light if detection fails
    
    if theme_type == "dark":
        # Dark mode: light gray labels with dark border (original user setting)
        label_config = {
            "position": "right",
            "verticalAlign": "middle",
            "align": "left",
            "fontSize": 11,
            "color": "#888888",
            "textBorderColor": "#1a1e24",
            "textBorderWidth": 3,
            "textShadowBlur": 2,
            "textShadowColor": "#1a1e24"
        }
    else:
        # Light mode: dark labels with white border for visibility
        label_config = {
            "position": "right",
            "verticalAlign": "middle",
            "align": "left",
            "fontSize": 11,
            "color": "#333333",
            "textBorderColor": "#ffffff",
            "textBorderWidth": 3,
            "textShadowBlur": 2,
            "textShadowColor": "#ffffff"
        }

    tree_options = {
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "tree",
                "data": echarts_data,
                "top": "1%",
                "left": "2%",
                "bottom": "1%",
                "right": "4%",
                "symbolSize": 15,
                "orient": orient,
                "label": label_config,
                "leaves": {
                    "label": label_config
                },
                "emphasis": {
                    "focus": 'descendant'
                },
                "expandAndCollapse": True,
                "initialTreeDepth": -1,
                "animationDuration": 550,
                "animationDurationUpdate": 750,
                "lineStyle": {
                    "curveness": 0.6,
                    "color": "#aaa"  
                },
                "layout": "orthogonal",
                "nodePadding": 40,
                "depth": 6
             }
         ]
     }
 
    events = {
         "click": "function(params) { return {full_path: params.data.full_path, uids: params.data.uids}; }"
     }
     
    clicked_node = st_echarts(options=tree_options, events=events, height="900px", width="100%")
 
    return clicked_node
     
 #---------------------------
 # SIGMA VIZ
 #---------------------------
def rgba_to_str(rgba):
  return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"
def sigma_graph(df):
    # Step 1: Extract event pairs and calculate weights
    event_pairs = []
    event_counts = Counter()
    for _, row in df.iterrows():
        path_str, weight = row["PATH"], row["COUNT"]
        events = path_str.split(", ")
        pairs = [(events[i], events[i + 1]) for i in range(len(events) - 1)]
        for pair in pairs:
            event_pairs.extend([pair] * weight)
        for event in events:
            event_counts[event] += weight
 
    pair_counts = Counter(event_pairs)
    total_events = sum(event_counts.values())
    total_pairs = sum(pair_counts.values())
 
    # Step 2: Create nodes with size based on event counts
    unique_events = list(event_counts.keys())
    max_count = max(event_counts.values())  # Normalize node size for visualization
 
    nodes = [
        {
            "name": event,
            "symbolSize": 10 + (event_counts[event] / max_count) * 50,  # Base size + scaled size
            "itemStyle": {
                "color": f"hsl({i * 360 / len(unique_events)}, 70%, 50%, 0.7)"  # Add transparency (0.7)
            },
            "label": {
                "show": True,
                "position": "right",
                "color": "#888888",  # Light gray - works in both light and dark mode
                "fontWeight": "normal",
            },
            "emphasis": {
                "itemStyle": {
                    "color": f"hsl({i * 360 / len(unique_events)}, 70%, 50%, 0.7)"  # Maintain same color on hover
                },
                "label": {
                    "color": "#ffffff",  # White label on hover
                    "fontWeight": "bold",
                }
            },
            "tooltip": {
                "formatter": f"{event}<br>Count: {event_counts[event]}<br>Percentage: {event_counts[event] / total_events:.2%}"
            },
        }
        for i, event in enumerate(unique_events)
    ]
 
    # Step 3: Create edges with normalized thickness and YlOrRd color palette (Yellow-Orange-Red)
     
    max_pair_count = max(pair_counts.values())
    min_pair_count = min(pair_counts.values())
    
    # Normalize color scaling based on count (heat-map style: yellow → orange → red)
    norm = Normalize(vmin=min_pair_count, vmax=max_pair_count)
    
    # Normalize thickness: always range from 1 to 5 regardless of actual values
    min_width = 1.0
    max_width = 5.0

    edges = [
        {
            "source": src,
            "target": tgt,
            "lineStyle": {
                # Normalize thickness to range [1, 4]
                "width": min_width + (count - min_pair_count) / (max_pair_count - min_pair_count) * (max_width - min_width) if max_pair_count > min_pair_count else 2.0,
                # Heat-map color: YlOrRd (Yellow-Orange-Red)
                "color": rgba_to_str(plt.cm.YlOrRd(norm(count))),
                "opacity": 0.85
            },
            "tooltip": {
                "formatter": f"{src} → {tgt}<br>Count: {count}<br>Percentage: {count / total_pairs:.2%}"
            },
        }
        for (src, tgt), count in pair_counts.items()
    ]
 
    # Step 4: ECharts options for force-directed graph
    options = {
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "symbolSize": 20,
                "roam": True,
                "label": {
                    "show": True, 
                    "fontSize": 12,
                    "color": "#888888",  # Light gray - works in both light and dark mode
                },
                "edgeSymbol": ["circle", "arrow"],
                "edgeSymbolSize": [4, 10],
                "force": {
                    "repulsion": 500,  # Increase repulsion to reduce overlap
                    "edgeLength": [50, 200],  # Set minimum and maximum edge lengths
                },
                "data": nodes,
                "links": edges,
                "lineStyle": {
                    "curveness": 0.1,  # Slight curvature for readability
                    "opacity": 0.7,    # Slight transparency for overlapping edges
                },
                "emphasis": {
                    "focus": "adjacency",
                    "label": {
                        "color": "#ffffff",  # White on hover
                        "fontWeight": "bold",
                    }
                },
            }
        ],
    }
 
    # Step 5: Render the ECharts graph in Streamlit
    st_echarts(options=options, height="800px")
 #--------------------
 #SUNBURST
 #--------------------
 # Function to generate random colors for each event
def generate_colors(events):
    # Use a consistent color palette similar to Altair's default scheme
    # This ensures colors are consistent and match what users expect
    altair_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    color_map = {}
    events_list = sorted(list(events))  # Sort for consistency
    for i, event in enumerate(events_list):
        color_map[event] = altair_colors[i % len(altair_colors)]
    return color_map
# Build a hierarchical structure based on direction
def build_hierarchy(paths, direction="from"):
    if direction == "to":
        # Reverse the order of the path for "to" visualization
        processed_paths = []
        for path, size in paths:
            # Handle both comma-space and comma-only separators
            if ", " in path:
                events = path.split(", ")
            else:
                events = path.split(",")
            events = [event.strip() for event in events]  # Strip whitespace
            processed_paths.append((events[::-1], size))
        paths = processed_paths
    else:
        processed_paths = []
        for path, size in paths:
            # Handle both comma-space and comma-only separators
            if ", " in path:
                events = path.split(", ")
            else:
                events = path.split(",")
            events = [event.strip() for event in events]  # Strip whitespace
            processed_paths.append((events, size))
        paths = processed_paths
    
    root = {"name": "", "children": [], "path": [], "value": 0}
    
    for parts, size in paths:
        current_node = root
        current_path = []
        for node_name in parts:
            current_path.append(node_name)
            children = current_node.setdefault("children", [])
            found_child = None
            
            for child in children:
                if child["name"] == node_name:
                    found_child = child
                    break
            
            if found_child:
                current_node = found_child
            else:
                new_node = {
                    "name": node_name,
                    "children": [],
                    "path": current_path.copy(),
                    "value": 0
                }
                children.append(new_node)
                current_node = new_node
            
            current_node["value"] += size
    # ✅ If direction is "to", reverse order of children to reflect "to" logic
    if direction == "to":
        root["children"] = root["children"][::-1]
    return root
# Convert hierarchy to ECharts-compatible format
def convert_to_echarts_format(hierarchy, color_map, total_count, direction="from"):
    def recurse(node):
        color = color_map.get(node["name"], "#FF5722")  # Default color if not in the map
        percentage = f"{(node.get('value', 0) / total_count * 100):.2f}%" if node.get("value") else ""
        full_path = " → ".join(node.get("path", []))
        
        # ✅ Fix tooltip direction based on the `direction` parameter
        if direction == "to":
            #full_path = " ← ".join(reversed(node.get("path", [])))
            full_path = " → ".join(reversed(node.get("path", [])))
        
        return {
            "name": node["name"],
            "value": node.get("value", None),
            "tooltip": {
                "formatter": f"Path: {full_path}<br>Count: {node.get('value', 0)}<br>Percentage: {percentage}"
            },
            "itemStyle": {
                "color": color
            },
            "children": [recurse(child) for child in node.get("children", [])] if "children" in node else None
        }
    
    return [recurse(hierarchy)]
# Extract events and counts from DataFrame
def extract_events_and_count(df):
    global total_count, events
    events = set()
    total_count = 0
    for _, row in df.iterrows():
        path = row['PATH']
        size = row['COUNT']
        # Handle both comma-space and comma-only separators
        if ", " in path:
            path_events = path.split(", ")
        else:
            path_events = path.split(",")
        for event in path_events:
            events.add(event.strip())  # Strip any extra whitespace
        total_count += size
#Main function to generate Sunburst chart
def process_and_generate_sunburst(df, direction="to"):
    extract_events_and_count(df)
    
    # Generate consistent colors for each event
    color_map = generate_colors(events)
    
    #Build hierarchy with the correct path order
    paths = [(row['PATH'], row['COUNT']) for _, row in df.iterrows()]
    hierarchy = build_hierarchy(paths, direction)
    
    #Convert hierarchy to ECharts format
    echarts_data = convert_to_echarts_format(hierarchy, color_map, total_count, direction)[0]["children"]
    
    #Sunburst chart configuration
    sunburst_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {c}", 
            "position": "function(point, params, dom, rect, size) { var x = point[0]; var y = point[1]; var viewWidth = size.viewSize[0]; var viewHeight = size.viewSize[1]; var boxWidth = size.contentSize[0]; var boxHeight = size.contentSize[1]; var posX = x + 20; var posY = y - boxHeight - 20; if (posX + boxWidth > viewWidth) { posX = x - boxWidth - 20; } if (posX < 0) { posX = 10; } if (posY < 0) { posY = y + 20; } return [posX, posY]; }",
            "backgroundColor": "rgba(240, 242, 246, 0.5)",
            "textStyle": {"color": "#0f0f0f", "fontSize": 12},
            "borderColor": "#3b82f6",
            "borderWidth": 1,
            "padding": [8, 12],
            "extraCssText": "max-width: 300px; white-space: normal; word-wrap: break-word; z-index: 9999;"
        },
        "series": [
            {
                "type": "sunburst",
                "data": echarts_data,
                "radius": ["20%", "80%"],
                "center": ["50%", "60%"],
                "label": {
                    "show": False
                },
                "itemStyle": {
                    "borderColor": "#fff",
                    "borderWidth": 1
                },
                "highlightPolicy": "ancestor",
            }
        ]
    }
    
    # Display the Sunburst chart
    st_echarts(options=sunburst_options, height="600px")
    
    # Legend configuration in a bordered container with expander
    with st.expander("Event legend", expanded=False, icon=":material/palette:"):
        legend_columns = st.columns(3)  
        col_idx = 0
        for event, color in color_map.items():
            with legend_columns[col_idx]:
                st.markdown(f"<span style='color:{color};font-weight:normal'>■</span> {event}", unsafe_allow_html=True)
        
        col_idx = (col_idx + 1) % 3 

# Modified function to generate Sunburst chart with consistent colors
def process_and_generate_sunburst_with_colors(df, direction="to", color_map=None):
    # Use the EXACT same logic as the working analyze tab version
    extract_events_and_count(df)
    
    # Use the provided color_map instead of generating new colors
    if color_map is None:
        color_map = generate_colors(events)
    
    #Build hierarchy with the correct path order
    paths = [(row['PATH'], row['COUNT']) for _, row in df.iterrows()]
    hierarchy = build_hierarchy(paths, direction)
    
    #Convert hierarchy to ECharts format
    echarts_data = convert_to_echarts_format(hierarchy, color_map, total_count, direction)[0]["children"]
    
    #Sunburst chart configuration with improved tooltip
    sunburst_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "function(params) { return params.name + ': ' + params.value + '<br/>Percentage: ' + params.percent + '%'; }",
            "position": "function(point, params, dom, rect, size) { var x = point[0]; var y = point[1]; var viewWidth = size.viewSize[0]; var viewHeight = size.viewSize[1]; var boxWidth = size.contentSize[0]; var boxHeight = size.contentSize[1]; var posX = x + 20; var posY = y - boxHeight - 20; if (posX + boxWidth > viewWidth) { posX = x - boxWidth - 20; } if (posX < 0) { posX = 10; } if (posY < 0) { posY = y + 20; } return [posX, posY]; }",
            "backgroundColor": "rgba(240, 242, 246, 0.5)",
            "textStyle": {"color": "#0f0f0f", "fontSize": 12},
            "borderColor": "#3b82f6",
            "borderWidth": 1,
            "padding": [8, 12],
            "extraCssText": "max-width: 300px; white-space: normal; word-wrap: break-word; z-index: 9999;"
        },
        "series": [
            {
                "type": "sunburst",
                "data": echarts_data,
                "radius": ["20%", "80%"],
                "center": ["50%", "60%"],
                "label": {
                    "show": False
                },
                "itemStyle": {
                    "borderColor": "#fff",
                    "borderWidth": 1
                },
                "highlightPolicy": "ancestor",
            }
        ]
    }
    
    # Display the Sunburst chart
    st_echarts(options=sunburst_options, height="600px")

        # Get the current credentials
session = get_active_session()
#--------------------------------------
#VAR INIT
#--------------------------------------
#Initialize variables
fromevt = None
fromevt1=None
toevt = None
toevt1=None
minnbbevt = 0
minnbbevt1 = 0
maxnbbevt = 5
maxnbbevt1 = 5
overlap = 'PAST LAST ROW'
overlap1 = 'PAST LAST ROW'
uid = None
evt = None
display = None
tmstp = None
tbl = None
partitionby = None
groupby = None
startdt_input = None
enddt_input = None
sess = None
uid1 = None
evt1 = None
tmstp1 = None
tbl1 = None
partitionby1 = None
groupby1 = None
startdt_input1 = None
enddt_input1 = None
sess1 = None
excl3_instance = "''"
timeout = None
unitoftime= None
timeout1 = None
unitoftime1= None
cols = ''
cols1=''
colsdf = pd.DataFrame()
colsdf1 = pd.DataFrame()
#--------------------------------------
#PAGE TITLE
#--------------------------------------
st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
            PATH ANALYSIS
        </h5>
    </div>
    """, unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Analyze", "Compare"])


#--------------------------------------
#ANALYZE TAB
#--------------------------------------
with tab1:
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
                database = st.selectbox('Select Database', key='analyzedb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected - cached)**
            if database:
                schema0 = fetch_schemas(session, database)
            
                with col2:
                    schema = st.selectbox('Select Schema', key='analyzesch', index=None, 
                                          placeholder="Choose from list...", options=schema0['name'].unique())
            else:
                schema = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected - cached)**
            if database and schema:
                table0 = fetch_tables(session, database, schema)
            
                with col3:
                    tbl = st.selectbox('Select Event Table or View', key='analyzetbl', index=None, 
                                       placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
            else:
                tbl = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected - cached)**
            if database and schema and tbl:
                colsdf = fetch_columns(session, database, schema, tbl)

            col1, col2, col3 = st.columns([4,4,4])
            with col1:
                uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col2: 
                evt = st.selectbox('Select event column', colsdf, index=None, placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col3:
                tmstp = st.selectbox('Select timestamp column', colsdf, index=None, placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
                        
                
            checkdisplay = st.toggle ("Display Column", key="checkdisplaycolumn",help="When toggled on, you can select an alternative column (such as channel, campaign, etc.) to be displayed in the visualizations and used by the AI-powered explanations. This enables you to view and interpret user paths at a higher level of abstraction, while the event column remains the basis for the underlying path analysis. When toggled off, the event column is used consistently for pattern definition, visual display, and AI-powered explanations.")
            display=evt
            if checkdisplay:

                col1, col2 = st.columns([4,8])
                with col1 :
                    selecteddisplay = st.selectbox('Select display column', colsdf, index=None, placeholder="Choose from list...",help="The column that will be used in the path diagrams.")
                    if selecteddisplay is not None:
                        display = selecteddisplay
                with col2:
                    st.write("")
            else: display = evt
                
            #Get Distinct Events Of Interest from Event Table (cached)
            if (uid != None and evt != None and tmstp != None):
                distinct_evt_df = fetch_distinct_values(session, database, schema, tbl, evt)
                # Write query output in a pandas dataframe
                startdf0 = distinct_evt_df.copy()
                enddf0 = distinct_evt_df.copy()
                excl0 = distinct_evt_df.copy()


            #Add "any" to the distinct events list
                any_row = {evt: 'Any'}

        # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df = pd.DataFrame([any_row])
        # Add any to list of FROM 
                startdf1 = pd.concat([ any_row_df, startdf0], ignore_index=True)
        # Add any to list of TO
                enddf1 = pd.concat([ any_row_df, enddf0], ignore_index=True)

        #EVENTS PATTERN
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)
                    
                    # Pattern Mode Selection
                    col1_mode, col2_mode, col3_mode = st.columns([1, 1, 1])
                    with col1_mode:
                        pattern_mode = st.radio("Pattern Mode", ["FROM/TO", "BEFORE&AFTER"], horizontal=True, 
                                               help="FROM/TO: Analyze paths leading to or from or between selected events. BEFORE&AFTER: Analyze what happens before and after a middle event.")
                    with col2_mode:
                        st.write("")
                    with col3_mode:
                        st.write("")
                
                # Conditional UI based on pattern mode
                if pattern_mode == "FROM/TO":
                    # Original FROM/TO logic
                    # Add a None placeholder to force user to select an event
                    options_with_placeholder_from = ["🔍"] + startdf1[evt].unique().tolist()
                    options_with_placeholder_to = ["🔎"] + enddf1[evt].unique().tolist()
                        
                    col1, col2, col3 = st.columns([1,1,1])

                    with col1:
                        frm = st.multiselect('Select events FROM:', options=options_with_placeholder_from[1:], default=[],help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                        #filtered_frmevt = startdf1[(startdf1[evt] == frm)]
                        #fromevt = filtered_frmevt.iloc[0, 0]
                        if frm != "🔍":
                            fromevt= ", ".join([f"'{value}'" for value in frm])
                        else:
                            fromevt = None  # Set to None if the placeholder is still selected
                            
                    with col2:
                        to = st.multiselect('Select events TO:', options=options_with_placeholder_to[1:], default=[],help="Select one or more events of interest to visualize paths TO the selected point(s). 'Any' matches all values.")
                        #filtered_toevt = enddf1[(enddf1[evt] == to)]
                        #toevt =filtered_toevt.iloc[0, 0]
                        if to != "🔎":
                            toevt = ", ".join([f"'{value}'" for value in to])
                        else:
                            toevt = None  # Set to None if the placeholder is still selected
                            
                    # Pattern approach pills - placed after both FROM and TO selections so we know the path type
                    with col1:
                        # Determine path type and set appropriate labels and tooltips
                        both_any = (fromevt and fromevt.strip("'") == 'Any') and (toevt and toevt.strip("'") == 'Any') if 'toevt' in locals() else False
                        is_path_to = (not fromevt or fromevt.strip("'") == 'Any') and toevt and toevt.strip("'") != 'Any'
                        is_path_from = fromevt and fromevt.strip("'") != 'Any' and (not toevt or toevt.strip("'") == 'Any')
                        is_path_between = fromevt and fromevt.strip("'") != 'Any' and toevt and toevt.strip("'") != 'Any'
                        is_before_after = False  # Not in BEFORE/AFTER mode
                        middle_event = None  # Initialize for FROM/TO mode
                        
                        if both_any:
                            # Force Min/Max Events for Any→Any pattern
                            st.info("ℹ️ Time Window not available for Any → Any patterns. Using Min/Max Events.")
                            pattern_approach = "Min/Max Events"
                            time_window_label = "Time Window"
                            time_window_help = "Choose between event count limits or time-based window filtering"
                        elif is_path_to:
                            # PATH TO: Use "Lookback Window"
                            time_window_label = "Lookback Window"
                            time_window_help = "Choose between event count limits or lookback time window (captures events BEFORE the target event)"
                            pattern_approach = st.pills("Pattern approach", ["Min/Max Events", time_window_label], default="Min/Max Events", help=time_window_help)
                        elif is_path_from or is_path_between:
                            # PATH FROM or PATH BETWEEN: Use "Look-forward Window"
                            time_window_label = "Look-forward Window"
                            time_window_help = "Choose between event count limits or look-forward time window (captures events AFTER the starting event)"
                            pattern_approach = st.pills("Pattern approach", ["Min/Max Events", time_window_label], default="Min/Max Events", help=time_window_help)
                        else:
                            # Default fallback
                            time_window_label = "Time Window"
                            time_window_help = "Choose between event count limits or time-based window filtering"
                            pattern_approach = st.pills("Pattern approach", ["Min/Max Events", time_window_label], default="Min/Max Events", help=time_window_help)
                        
                        # Show input controls below pills within the same column
                        if pattern_approach == "Min/Max Events":
                            # Min/Max events inputs - aligned under FROM selectbox width
                            col_min, col_max = st.columns([1, 1])
                            with col_min:
                                minnbbevt = st.number_input("Min # events", value=0, placeholder="Type a number...", help="Select the minimum number of events either preceding or following the event(s) of interest.")
                            with col_max:
                                maxnbbevt = st.number_input("Max # events", value=5, min_value=1, placeholder="Type a number...", help="Select the maximum number of events either preceding or following the event(s) of interest.")
                            
                            # Set lookback variables to None for traditional approach
                            max_gap_value = None
                            gap_unit = None
                            use_lookback = False
                            
                        else:  # Time Window (Lookback or Look-forward)
                            # Time window inputs - aligned under FROM selectbox width
                            col_time, col_unit = st.columns([1.5, 1])
                            with col_time:
                                if is_path_to:
                                    max_gap_value = st.number_input("Lookback period", value=7, min_value=1, placeholder="Type a number...", help="Maximum lookback time window (how far back to look before the target event)")
                                else:
                                    max_gap_value = st.number_input("Look-forward period", value=7, min_value=1, placeholder="Type a number...", help="Maximum look-forward time window (how far forward to look after the starting event)")
                            with col_unit:
                                gap_unit = st.selectbox("Time unit", ["MINUTE", "HOUR", "DAY"], index=2, help="Select the time unit for the time window")
                            
                            # Set traditional variables for time window approach  
                            minnbbevt = 0
                            maxnbbevt = 999999  # Large number to effectively disable event count limits
                            use_lookback = True
                        
                    with col3:
                        # Empty column for visual balance
                        pass
                
                else:  # BEFORE/AFTER mode
                    # UI for BEFORE/AFTER pattern
                    col1, col2, col3 = st.columns([1,1,1])
                    
                    with col1:
                        # Middle event selection (no Any option)
                        options_middle_event = ["🎯"] + [e for e in startdf1[evt].unique().tolist() if e != 'Any']
                        middle_event_list = st.multiselect('Select middle event(s):', options=options_middle_event[1:], default=[],
                                                          help="Select the event(s) of interest to see what happens before and after")
                        
                        if middle_event_list:
                            middle_event = ", ".join([f"'{value}'" for value in middle_event_list])
                        else:
                            middle_event = None
                        
                        # Min/Max events inputs - aligned under middle event selectbox width
                        col_min, col_max = st.columns([1, 1])
                        with col_min:
                            minnbbevt = st.number_input("Min # events before/after", value=0, placeholder="Type a number...", 
                                                       help="Minimum number of events before and after the middle event")
                        with col_max:
                            maxnbbevt = st.number_input("Max # events before/after", value=5, min_value=1, placeholder="Type a number...",
                                                       help="Maximum number of events before and after the middle event")
                        
                        # Set variables for BEFORE/AFTER mode
                        fromevt = None
                        toevt = None
                        max_gap_value = None
                        gap_unit = None
                        use_lookback = False
                        is_path_to = False
                        is_path_from = False
                        is_path_between = False
                        is_before_after = True
                        
                    with col2:
                        st.write("")
                    with col3:
                        st.write("")
               
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapanalyze",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                if checkoverlap:
                    overlap='TO NEXT ROW'
                with col2:
                    st.write("")    
        
                
    #--------------------------------------
    #DATE RANGE
    #--------------------------------------
                with st.container():
                    st.markdown("""
                    <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
                    <hr style='margin-top: -8px;margin-bottom: 5px;'>
                    """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2.4,2.4,10])

                # SQL query to get the min start date
                minstartdt = f"SELECT   TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {database}.{schema}.{tbl}"
                # Get min start date :
                defstartdt = session.sql(minstartdt).collect()
                defstartdt_str = defstartdt[0][0] 
                defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
                
                with col1:
                    startdt_input = st.date_input('Start date', value=defstartdt1)
                with col2:
                    enddt_input = st.date_input('End date',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3:
                    st.write("")
                    
    #--------------------------------------
    #SESSION
    #--------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session (optional)</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help= "A session, in the context of customer journey analysis, refers to a defined period of activity during which a customer interacts or events occur. It includes all events, actions, or transactions performed by the user within this timeframe. If a session field is present in the event table, select it from the 'Column' tab below. If no session field is available, use the 'Sessionize' tab to create unique session identifiers by grouping events based on time gaps (e.g., a gap of more than 30 minutes starts a new session). Once a session is selected or created, it will be used alongside the unique identifier to partition the input rows before applying pattern matching.")
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess = st.selectbox('Select session column ', colsdf, index=None, placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout=  st.number_input( "Insert a timeout value",value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess == None and unitoftime==None and timeout==None: 
                        partitionby = f"partition by {uid}"
                        groupby = f"group by {uid}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                
                elif sess != None and unitoftime==None and timeout==None:
                        partitionby=f"partition by {uid},{sess} "
                        groupby = f"group by {uid}, match_number,{sess} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
                elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        groupby = f"group by {uid}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                else :
                    st.write("")
                    
                    
    #--------------------------------------
    #FILTERS
    #--------------------------------------

                # Initialize sql_where_clause - will be populated by filters if enabled
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2, col3  =st.columns([5,2,8])
                    with col1:
                # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0,placeholder="Select event(s)...",help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                          excl3 = "''"
                        else:
                         excl3= ', '.join([f"'{excl2}'" for excl2 in excl1])
                    with col2:
                        # Top n paths
                        topn = st.number_input("Top paths to show", value=100, placeholder="Type a number...",help="Shows the most frequent paths")
                    with col3:
                        st.write("")
                
                    # Additional filters toggle inside the main filters container
                # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                
                    with col1:
                        checkfilters = st.toggle("Additional filters", key="additional_filters_main", help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters = False  # Disable filters when no columns are left
                
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters and not remaining_columns.empty:
                    with st.container():
                        # Helper to get cached distinct values as a Python list
                        def get_distinct_values_list(column):
                            df_vals = fetch_distinct_values(session, database, schema, tbl, column)
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
                            col_data_type = fetch_column_type(session, database, schema, tbl, selected_column)
                
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
                                    if isinstance(value, (int, float)):
                                        sql_where_clause += f"{col} {operator} {value}"
                                    else:
                                 # For non-numeric values (strings, dates), enclose the value in quotes
                                        sql_where_clause += f"{col} {operator} '{value}'"        
                            
                        else:
                                # If no filters were created, ensure sql_where_clause is empty
                            sql_where_clause = ""
        
    # SQL LOGIC
    # Check pattern an run SQL accordingly
            

    # Initialize pattern_mode and middle_event if not already defined
    if 'pattern_mode' not in locals():
        pattern_mode = "FROM/TO"
    if 'middle_event' not in locals():
        middle_event = None

    # Validation check depends on pattern mode
    inputs_valid = False
    if pattern_mode == "BEFORE&AFTER":
        inputs_valid = all([uid, evt, tmstp, middle_event])
    else:  # FROM/TO mode
        inputs_valid = all([uid, evt, tmstp, fromevt, toevt])
    
    if inputs_valid:
        # Now we can proceed with the SQL logic and any further processing
        
        # Continue with SQL generation and execution based on inputs...

        # BEFORE/AFTER: Pattern = A{{{minnbbevt},{maxnbbevt}}} B A{{{minnbbevt},{maxnbbevt}}}
        if pattern_mode == "BEFORE&AFTER" and middle_event:
            
            # Initialize result containers
            before_after_agg=None
            before_after_agg_sql= None
            
            # Aggregate results for plot
            if unitoftime==None and timeout ==None :
                before_after_agg_sql = f"""
                select path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B A{{{minnbbevt},{maxnbbevt}}})
                            define A as true, B AS {evt} IN({middle_event}))
                    {groupby} ) 
                group by path order by count desc 
                """
            elif unitoftime != None and timeout !=None :
                before_after_agg_sql = f"""
            select path, count(*) as count,array_agg({uid}) as uid_list from (
                select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                    from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM
                {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
         ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
        FROM events_with_diff)
        SELECT *FROM sessions)
                        match_recognize(
                        {partitionby} 
                        order by {tmstp}  
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap} 
                        pattern(A{{{minnbbevt},{maxnbbevt}}} B A{{{minnbbevt},{maxnbbevt}}})
                        define A as true, B AS {evt} IN({middle_event}))
                {groupby} ) 
            group by path order by count desc 
            """
            
            before_after_agg = session.sql(before_after_agg_sql).collect()
            res = pd.DataFrame(before_after_agg)
            import ast
            
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except:
                            return []
                    elif isinstance(uid_entry, list):
                        return uid_entry
                    else:
                        return []
                
                res['UID_LIST'] = res['UID_LIST'].apply(convert_uid_list)
                
                # Save results
                st.session_state['before_after_agg'] = res
                
                # Display success message
                st.markdown(f"""
                <div class="custom-container-1">
                    <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                        Analysis complete: {len(res):,} unique paths retrieved from {res['COUNT'].sum():,} customer journeys
                    </h5>
                </div>
                """, unsafe_allow_html=True)
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 7])
            
                # Place the toggle in the first column
                with col1:
                    show_details_ba = st.toggle("Show me!", key="show_details_ba", help="Select a visualization option: Sankey or Graph.")

                # Place the pills in the second column, but only if the toggle is on
                with col2:
                    if 'show_details_ba' in locals() and show_details_ba:
                        genre_ba = st.pills(
                            "Choose a visualization:",
                            ["Sankey", "Graph"],
                            label_visibility="collapsed",
                            key="genre_ba"
                        )
                    else:
                        genre_ba = None
            
                # Place the visualization outside of the columns layout
                if show_details_ba and genre_ba:
                    # Initialize hash for session state tracking
                    current_df_hash = hash(res.to_json())
                    if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                        st.session_state["last_df_hash"] = current_df_hash
                    
                    if genre_ba == 'Sankey':
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                percentage_ba = st.slider("Display Top % of Paths", 1, 100, 100, key="percentage_ba")
                            with col2:
                                st.write("")
                            
                            clicked_sankey_ba = sankey_chart(res, direction="before_after", topN_percentage=percentage_ba, middle_events=middle_event)
                        
                        if clicked_sankey_ba:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey_ba and "target" in clicked_sankey_ba:
                                source_index = clicked_sankey_ba["source"]
                                target_index = clicked_sankey_ba["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                source_parts = clicked_source.split('_', 1)
                                target_parts = clicked_target.split('_', 1)
                                source_name = source_parts[1] if len(source_parts) >= 2 else clicked_source
                                target_name = target_parts[1] if len(target_parts) >= 2 else clicked_target
                                st.caption(f"Selected Edge: {source_name} → {target_name}")
                    
                    elif genre_ba == 'Graph':
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                percentage_graph_ba = st.slider("Display Top % of Paths", 1, 100, 100, key="percentage_graph_ba")
                            with col2:
                                st.write("")
                            
                            topN_graph_ba = int(len(res) * percentage_graph_ba / 100)
                            sigma_graph(res.head(topN_graph_ba))
                
                # View SQL toggle
                if st.toggle("View SQL for Aggregated Paths", key="view_sql_ba", help="View the SQL query used for this analysis"):
                    st.code(before_after_agg_sql, language="sql")
                
                # Writeback toggle
                if st.toggle("Writeback Segments to Snowflake", key="writeback_before_after",
                            help="Export selected path patterns with their counts and associated user IDs to a Snowflake table for targeted segmentation."):
                    with st.expander("Writeback Segments to Snowflake", expanded=True, icon=":material/upload:"):
                        # Path selection
                        st.markdown("**Select Paths to Export**")
                        
                        # Select All checkbox
                        select_all_before_after = st.checkbox("Select All Paths", value=False, key="select_all_before_after")
                        
                        # Multiselect for paths
                        if select_all_before_after:
                            default_paths_before_after = res['PATH'].tolist()
                        else:
                            default_paths_before_after = []
                        
                        selected_paths_before_after = st.multiselect(
                            "Choose path(s):",
                            options=res['PATH'].tolist(),
                            default=default_paths_before_after,
                            key="selected_paths_before_after"
                        )
                        
                        if selected_paths_before_after:
                            # Filter dataframe
                            filtered_df_before_after = res[res['PATH'].isin(selected_paths_before_after)]
                            
                            # Show export preview
                            st.info(f"📊 Export Preview: {len(selected_paths_before_after)} path(s) selected, "
                                   f"{filtered_df_before_after['COUNT'].sum():,} total occurrences, "
                                   f"{filtered_df_before_after['UID_LIST'].apply(len).sum():,} total user IDs", 
                                   icon=":material/info:")
                            
                            # Database/Schema/Table selection
                            col1, col2 = st.columns(2)
                            with col1:
                                wb_database_before_after = st.selectbox("Database", fetch_databases(session), key="wb_db_before_after")
                            with col2:
                                if wb_database_before_after:
                                    wb_schema_before_after = st.selectbox("Schema", fetch_schemas(session, wb_database_before_after), key="wb_schema_before_after")
                            
                            wb_table_before_after = st.text_input("Table Name", value="PATH_SEGMENTS_BEFORE_AFTER", key="wb_table_before_after")
                            
                            wb_mode_before_after = st.radio("Write Mode", ["Create or Replace", "Append to Existing"], key="wb_mode_before_after")
                            
                            if st.button("Write Table", key="write_btn_before_after"):
                                if wb_database_before_after and wb_schema_before_after and wb_table_before_after:
                                    try:
                                        with st.spinner("Writing to Snowflake..."):
                                            # Create Snowpark DataFrame
                                            export_df_before_after = session.create_dataframe(filtered_df_before_after[['PATH', 'COUNT', 'UID_LIST']])
                                            
                                            # Write to table
                                            if wb_mode_before_after == "Create or Replace":
                                                export_df_before_after.write.mode("overwrite").save_as_table(f"{wb_database_before_after}.{wb_schema_before_after}.{wb_table_before_after}")
                                            else:
                                                export_df_before_after.write.mode("append").save_as_table(f"{wb_database_before_after}.{wb_schema_before_after}.{wb_table_before_after}")
                                            
                                            st.success(f"✅ Successfully wrote {len(filtered_df_before_after)} rows to {wb_database_before_after}.{wb_schema_before_after}.{wb_table_before_after}", 
                                                      icon=":material/check:")
                                    except Exception as e:
                                        st.error(f"Error writing to Snowflake: {str(e)}", icon=":material/error:")
                                else:
                                    st.warning("Please select database, schema, and table name", icon=":material/warning:")
            else:
                st.info("No patterns found matching the criteria.", icon=":material/info:")
        
        # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
        elif pattern_mode == "FROM/TO" and fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any':
            
            path_to_agg=None
            path_to_det_df=None
            path_to_det_sql=None
            path_to_agg_sql= None
            # Aggregate results for plot
            if unitoftime==None and timeout ==None :
                
                if use_lookback:
                    # Lookback window approach for PATH TO (captures events BEFORE the target event)
                    path_to_agg_sql = f"""
                    select path, count(*) as count,array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (
                            WITH TO_EVENT_TIMES AS (
                                SELECT {uid}, {tmstp}, {evt},
                                       MIN(CASE WHEN {evt} IN ({toevt}) THEN {tmstp} END) OVER (PARTITION BY {uid}) as to_event_time
                                FROM {database}.{schema}.{tbl} 
                                WHERE {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}
                            ),
                            TIME_WINDOWED AS (
                                SELECT {uid}, {tmstp}, {evt}
                                FROM TO_EVENT_TIMES
                                WHERE to_event_time IS NOT NULL 
                                  AND TIMESTAMPDIFF({gap_unit}, {tmstp}, to_event_time) BETWEEN 0 AND {max_gap_value}
                            )
                            SELECT * FROM TIME_WINDOWED
                        ) match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A* B)
                            define A as true, B AS {evt} IN({toevt}))
                        {groupby} ) 
                    group by path order by count desc 
                    """
                else:
                    # Traditional min/max events approach
                    path_to_agg_sql = f"""
                    select path, count(*) as count,array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                            from (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                                match_recognize(
                                {partitionby} 
                                order by {tmstp}  
                                measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                                all rows per match
                                AFTER MATCH SKIP {overlap} 
                                pattern(A{{{minnbbevt},{maxnbbevt}}} B)
                                define A as true, B AS {evt} IN({toevt}))
                        {groupby} ) 
                    group by path order by count desc 
                    """
    
                path_to_agg = session.sql(path_to_agg_sql).collect()
            elif unitoftime != None and timeout !=None :
                path_to_agg_sql = f"""
            select path, count(*) as count,array_agg({uid}) as uid_list from (
                select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                    from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM
                {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
         ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
        FROM events_with_diff)
        SELECT *FROM sessions)
                        match_recognize(
                        {partitionby} 
                        order by {tmstp}  
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap} 
                        pattern(A{{{minnbbevt},{maxnbbevt}}} B)
                        define A as true, B AS {evt} IN({toevt}))
                {groupby} ) 
            group by path order by count desc 
            """
            
                path_to_agg = session.sql(path_to_agg_sql).collect()
            # If the DataFrame is not empty, show Sankey plot
            
            res = pd.DataFrame(path_to_agg)
            #st.write (res)
            import ast
        
                # Print to verify
            #st.write("✅ Processed DataFrame:", res)
                            
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                
                # Store total counts before limiting
                total_paths = len(res)
                total_customers = res['COUNT'].sum()
                
                # Apply TOP N limit at pandas level
                res = res.head(topn)
                
                # Show data summary for transparency
                if not res.empty:
                    displayed_paths = len(res)
                    if total_paths <= topn:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {displayed_paths:,} of {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys (top {topn} results)
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-container-1">
                        <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                            No paths found matching the specified criteria
                        </h5>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 7])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst.")

                # Place the pills in the second column, but only if the toggle is on
                with col2:
                    if 'show_details' in locals() and show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey","Tree", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
                    else:
                        genre = None
            
                # Place the visualization outside of the columns layout
                if show_details and genre:
                    if genre == 'Sankey':

                        with st.container(border=True):
                        
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                percentage = st.slider("Display Top % of Paths", 1, 100, 100)
                            with col2:
                                st.write("")
                        
                            # STEP 2: Reset Session State on Query Change**
                            current_df_hash = hash(res.to_json())
                            if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                                st.session_state["last_df_hash"] = current_df_hash  # Update hash
                                # Reset state variables
                                st.session_state["clicked_sankey"] = None
                                st.session_state["clicked_source"] = None
                                st.session_state["clicked_target"] = None
                                st.session_state["selected_uids"] = set()
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                                st.session_state["sankey_chart"] = None
                                st.session_state["sankey_links"] = {}
                                st.session_state["sankey_labels"] = []
                                st.session_state["sortedEventList"] = []
                                
                                #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                            # Session state to store extracted user data
                            if "selected_paths_df" not in st.session_state:
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            
                            
                            clicked_sankey = sankey_chart(res, direction="to", topN_percentage=percentage)
                            
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}|||{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No UIDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                    elif genre == 'Tree':
                        target_event = toevt.strip("'")
                        with st.container(border=True):
                            clicked_node= plot_tree(res, target_event, "to")
                        if clicked_node:
                            selected_path = clicked_node["full_path"]  
                            selected_uids = clicked_node["uids"]  # Directly use cleaned UIDs
                        
                            direction = "to"
                            if direction == "to":
                                selected_path = ", ".join(reversed(selected_path.split(", ")))  # Reverse only in "to" mode
                        
                            st.write(f"Selected path: **{selected_path}**")
                            
                            with st.expander("CREATE SEGMENT"):  # Keep the form visible
                                # **Fetch available databases**
                                sqldb = "SHOW DATABASES"
                                databases = session.sql(sqldb).collect()
                                db0 = pd.DataFrame(databases)
                        
                                # **Row Layout: Database, Schema, Table Name, Create & Append Buttons**
                                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        
                                # **Database Selection**
                                with col1:
                                    database = st.selectbox("Select Database", key="segmentdb", index=None, 
                                                            placeholder="Choose from list...", options=db0["name"].unique())
                        
                                # **Schema Selection**
                                schema = None
                                if database:
                                    sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                                    schemas = session.sql(sqlschemas).collect()
                                    schema0 = pd.DataFrame(schemas)
                        
                                    with col2:
                                        schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                                                              placeholder="Choose from list...", options=schema0["name"].unique())
                        
                                # **Table Name Input**
                                table_name = None
                                if database and schema:
                                    with col3:
                                        table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        
                                # **Ensure all fields are filled before proceeding**
                                if database and schema and table_name and selected_uids:
                                    # ✅ **STEP 1: GENERATE SQL STATEMENTS**
                                    create_table_sql = f"""
                                    CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                                        ID STRING
                                    )
                                    """
                        
                                    values = ", ".join([f"('{uid}')" for uid in selected_uids])  # Format UIDs
                                    insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};"
                        
                                    # ✅ **STEP 2: SHOW SQL PREVIEW**
                                    #st.write("#### SQL Preview:")
                                    #st.code(create_table_sql, language="sql")
                                    #st.code(insert_sql, language="sql")
                        
                                    # ✅ **STEP 3: Align Buttons with Select Boxes**
                                    with col4:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Create Segment", use_container_width=True, help="Creates a new segment table and inserts selected identifiers"):
                                            try:
                                                session.sql(create_table_sql).collect()  # Create table
                                                session.sql(insert_sql).collect()  # Insert UIDs
                                                st.success(f"Segment `{database}.{schema}.{table_name}` created successfully!")
                                            except Exception as e:
                                                st.error(f"Error executing SQL: {e}", icon=":material/chat_error:")
                        
                                    with col5:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Append to Segment", use_container_width=True, help="Appends selected identifiers into an existing table"):
                                            try:
                                                session.sql(insert_sql).collect()  # Insert UIDs only
                                                st.success(f"UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                                            except Exception as e:
                                                st.error(f"Error executing SQL: {e}", icon=":material/chat_error:")
                                                
                    elif genre == 'Graph':
                        with st.container(border=True):
                            sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        with st.container(border=True):
                            process_and_generate_sunburst(res, direction="to")

                # AI-Powered Insights with model selection (only show if toggle is on)
                if 'show_details' in locals() and show_details:
                    def path_ai_analysis_callback(selected_model, prompt_type):
                        """Callback function for path analysis AI insights"""
                        
                        # Show custom prompt input if Custom is selected
                        if prompt_type == "Custom":
                            custom_prompt = st.text_area(
                                "Enter your custom prompt:",
                                value="",
                                key="path_to_custom_prompt",
                                help="Enter your custom analysis prompt. The path data will be automatically included.",
                                placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                            )
                            
                            # Only proceed if custom prompt is not empty
                            if not custom_prompt or custom_prompt.strip() == "":
                                st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                                return
                        
                        with st.spinner("Generating AI insights..."):
                            try:
                                # Use the same number of paths as selected by user (topn)
                                top_paths = res.head(topn)
                                paths_text = "\n".join([f"{row['PATH']} (count: {row['COUNT']})" 
                                                for _, row in top_paths.iterrows()])
                                
                                if prompt_type == "Auto":
                                    ai_prompt = f"""
                                    Analyze these top {topn} customer journey paths:
                                    
                                    {paths_text}
                                    
                                    Total Paths Found: {len(res)}
                                    
                                    Please provide insights on:
                                    1. Most significant customer journey patterns
                                    2. User behavior implications 
                                    3. Potential optimization opportunities
                                    4. Anomalies or interesting findings
                                    
                                    Keep your analysis concise and actionable.
                                    """
                                else:  # Custom
                                    ai_prompt = f"""
                                    {custom_prompt}
                                    
                                    Data to analyze - Top {topn} customer journey paths:
                                    {paths_text}
                                    
                                    Total Paths Found: {len(res)}
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
                        "path_to_ai", 
                        "Select the LLM model for AI analysis of path results",
                        ai_content_callback=path_ai_analysis_callback
                    )
                
                # View Aggregated Paths SQL
                if st.toggle("View SQL for Aggregated Paths"):    
                    st.code(path_to_agg_sql, language='sql')

            if st.toggle("Writeback Segments to Snowflake", key="writeback_toggle_path_to", help="Export selected path patterns with their counts and associated user IDs to a Snowflake table for targeted segmentation."):
                with st.expander("Writeback Segments to Snowflake", icon=":material/upload:", expanded=True):
                    # Use aggregated paths data already computed for visualization
                    if path_to_agg is not None and len(path_to_agg) > 0:
                        # Convert to DataFrame for easier manipulation
                        paths_df = pd.DataFrame(path_to_agg)
                        
                        # Path selector
                        available_paths = paths_df['PATH'].tolist()
                        
                        # Select All checkbox
                        select_all = st.checkbox("Select All Paths", value=False, key="wb_select_all_to")
                        
                        # Determine default selection based on checkbox
                        if select_all:
                            default_selection = available_paths
                        else:
                            default_selection = available_paths[:min(10, len(available_paths))]  # Default to top 10
                        
                        selected_paths = st.multiselect(
                            "Choose one or more paths",
                            options=available_paths,
                            default=default_selection,
                            key="wb_path_select_to",
                            help="Select which paths to export. Each path will include COUNT and UID_LIST"
                        )
                        
                        if selected_paths:
                            # Show preview
                            filtered_df = paths_df[paths_df['PATH'].isin(selected_paths)]
                            total_users = sum([len(uid_list) for uid_list in filtered_df['UID_LIST']])
                            st.info(f"Export Preview: {len(selected_paths)} paths | {filtered_df['COUNT'].sum():,} occurrences | {total_users:,} unique users", icon=":material/info:")
                            
                            # Fetch DBs using cached method
                            db0 = fetch_databases(session)
                            
                            # Database, Schema, Table selection
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                wb_database = st.selectbox("Database", db0['name'].unique(), index=None, key="wb_db_path_to", placeholder="Choose...")
                            
                            wb_schema = None
                            if wb_database:
                                schema0 = fetch_schemas(session, wb_database)
                                
                                with col2:
                                    wb_schema = st.selectbox("Schema", schema0['name'].unique(), index=None, key="wb_schema_path_to", placeholder="Choose...")
                            
                            with col3:
                                if wb_database and wb_schema:
                                    wb_table_name = st.text_input("Table Name", key="wb_tbl_path_to", placeholder="e.g. path_to_aggregated")
                                else:
                                    wb_table_name = None
                            
                            # Write mode and button
                            if wb_database and wb_schema and wb_table_name:
                                write_mode = st.radio("Write Mode", ["Create or Replace", "Append to Existing"], key="wb_mode_path_to", horizontal=True)
                                
                                if st.button("Write Table", key="wb_btn_path_to"):
                                    try:
                                        with st.spinner("Writing aggregated paths to Snowflake..."):
                                            # Filter and convert to Snowpark DataFrame
                                            export_df = filtered_df[['PATH', 'COUNT', 'UID_LIST']]
                                            snowpark_df = session.create_dataframe(export_df)
                                            
                                            # Write to table based on mode
                                            if write_mode == "Create or Replace":
                                                snowpark_df.write.mode("overwrite").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            else:
                                                snowpark_df.write.mode("append").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            
                                            # Get actual row count from written table
                                            written_count = session.sql(f"SELECT COUNT(*) as count FROM {wb_database}.{wb_schema}.{wb_table_name}").collect()[0]['COUNT']
                                            
                                            # Success message
                                            st.success(f"Table {wb_database}.{wb_schema}.{wb_table_name} {'created' if write_mode == 'Create or Replace' else 'updated'} successfully with {written_count:,} paths", icon=":material/check:")
                                            
                                    except Exception as e:
                                        st.error(f"Error writing to Snowflake: {str(e)}", icon=":material/chat_error:")
                        else:
                            st.warning("Please select at least one path to export", icon=":material/warning:")
                    else:
                        st.info("No paths available. Please run the analysis first.", icon=":material/info:")
                
            else:
                    st.write("") 
                
        

        # Separate block for PATH FROM 
        elif pattern_mode == "FROM/TO" and fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any':

            path_frm_agg=None
            path_frm_agg_sql=None
            path_frm_det_df=None
            path_frm_det_sql = None
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                
                if use_lookback:
                    # Look-forward window approach for PATH FROM (captures events AFTER the starting event)
                    path_frm_agg_sql = f"""
                    select path, count(*) as count,array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (
                            WITH FROM_EVENT_TIMES AS (
                                SELECT {uid}, {tmstp}, {evt},
                                       MIN(CASE WHEN {evt} IN ({fromevt}) THEN {tmstp} END) OVER (PARTITION BY {uid}) as from_event_time
                                FROM {database}.{schema}.{tbl} 
                                WHERE {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}
                            ),
                            TIME_WINDOWED AS (
                                SELECT {uid}, {tmstp}, {evt}
                                FROM FROM_EVENT_TIMES
                                WHERE from_event_time IS NOT NULL 
                                  AND TIMESTAMPDIFF({gap_unit}, from_event_time, {tmstp}) BETWEEN 0 AND {max_gap_value}
                            )
                            SELECT * FROM TIME_WINDOWED
                        ) match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A B*)
                            define A AS {evt} IN ({fromevt}), B as true)
                        {groupby} ) 
                    group by path order by count desc 
                    """
                else:
                    # Traditional min/max events approach for PATH FROM
                    path_frm_agg_sql = f"""
                    select path, count(*) as count,array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                            from (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                                match_recognize(
                                {partitionby} 
                                order by {tmstp}  
                                measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                                all rows per match
                                AFTER MATCH SKIP {overlap} 
                                pattern(A B{{{minnbbevt},{maxnbbevt}}})
                                define A AS {evt} IN ({fromevt}), B as true)
                        {groupby} ) 
                    group by path order by count desc 
                    """
    
                path_frm_agg = session.sql(path_frm_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                 
                path_frm_agg_sql = f"""
                select path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
                FROM events_with_diff)
                 SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A B{{{minnbbevt},{maxnbbevt}}})
                            define A AS {evt} IN ({fromevt}), B as true)
                    {groupby} ) 
                group by path order by count desc 
                """
    
                path_frm_agg = session.sql(path_frm_agg_sql).collect()
            
            # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_frm_agg)
            import ast
            
            
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                
                # Store total counts before limiting
                total_paths = len(res)
                total_customers = res['COUNT'].sum()
                
                # Apply TOP N limit at pandas level
                res = res.head(topn)
                
                # Show data summary for transparency
                if not res.empty:
                    displayed_paths = len(res)
                    if total_paths <= topn:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {displayed_paths:,} of {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys (top {topn} results)
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-container-1">
                        <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                            No paths found matching the specified criteria
                        </h5>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 7])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst.")

                # Place the pills in the second column, but only if the toggle is on
                with col2:
                    if 'show_details' in locals() and show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey", "Tree", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
                    else:
                        genre = None
            
                # Place the visualization outside of the columns layout
                if show_details and genre:
                    if genre == 'Tree':
                        target_event = fromevt.strip("'")
                        with st.container(border=True):
                            clicked_node= plot_tree(res, target_event, "from")
                        if clicked_node:
                            selected_path = clicked_node["full_path"]  
                            selected_uids = clicked_node["uids"]  # Directly use cleaned UIDs
                        
                            st.write(f"🔀 Selected path: **{selected_path}**")
                            
                            with st.expander("CREATE SEGMENT"):  # Keep the form visible
                                # **Fetch available databases**
                                sqldb = "SHOW DATABASES"
                                databases = session.sql(sqldb).collect()
                                db0 = pd.DataFrame(databases)
                        
                                # **Row Layout: Database, Schema, Table Name, Create & Append Buttons**
                                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        
                                # **Database Selection**
                                with col1:
                                    database = st.selectbox("Select Database", key="segmentdb", index=None, 
                                                            placeholder="Choose from list...", options=db0["name"].unique())
                        
                                # **Schema Selection**
                                schema = None
                                if database:
                                    sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                                    schemas = session.sql(sqlschemas).collect()
                                    schema0 = pd.DataFrame(schemas)
                        
                                    with col2:
                                        schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                                                              placeholder="Choose from list...", options=schema0["name"].unique())
                        
                                # **Table Name Input**
                                table_name = None
                                if database and schema:
                                    with col3:
                                        table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        
                                # **Ensure all fields are filled before proceeding**
                                if database and schema and table_name and selected_uids:
                                    # ✅ **STEP 1: GENERATE SQL STATEMENTS**
                                    create_table_sql = f"""
                                    CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                                        ID STRING
                                    )
                                    """
                        
                                    values = ", ".join([f"('{uid}')" for uid in selected_uids])  # Format UIDs
                                    insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};"
                        
                                    # ✅ **STEP 2: SHOW SQL PREVIEW**
                                    #st.write("#### SQL Preview:")
                                    #st.code(create_table_sql, language="sql")
                                    #st.code(insert_sql, language="sql")
                        
                                    # ✅ **STEP 3: Align Buttons with Select Boxes**
                                    with col4:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Create Segment", use_container_width=True, help="Creates a new segment table and inserts selected identifiers"):
                                            try:
                                                session.sql(create_table_sql).collect()  # Create table
                                                session.sql(insert_sql).collect()  # Insert UIDs
                                                st.success(f"Segment `{database}.{schema}.{table_name}` created successfully!")
                                            except Exception as e:
                                                st.error(f"Error executing SQL: {e}",icon=":material/error:")
                        
                                    with col5:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Append to Segment", use_container_width=True, help="Appends selected identifiers into an existing table"):
                                            try:
                                                session.sql(insert_sql).collect()  # Insert UIDs only
                                                st.success(f"UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                                            except Exception as e:
                                                st.error(f"Error executing SQL: {e}", icon=":material/chat_error:")
                    
                    elif genre == 'Sankey':
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                percentage = st.slider("Display Top % of Paths", 1, 100, 100)
                            with col2:
                                st.write("")                        
                            # 🔹 **STEP 2: Reset Session State on Query Change**
                            current_df_hash = hash(res.to_json())
                            if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                                st.session_state["last_df_hash"] = current_df_hash  # Update hash
                                # Reset state variables
                                st.session_state["clicked_sankey"] = None
                                st.session_state["clicked_source"] = None
                                st.session_state["clicked_target"] = None
                                st.session_state["selected_uids"] = set()
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                                st.session_state["sankey_chart"] = None
                                st.session_state["sankey_links"] = {}
                                st.session_state["sankey_labels"] = []
                                st.session_state["sortedEventList"] = []
                                
                                #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                            # Session state to store extracted user data
                            if "selected_paths_df" not in st.session_state:
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            
                            clicked_sankey = sankey_chart(res, direction="from", topN_percentage=percentage)
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}|||{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No UIDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                    elif genre == 'Graph':
                        with st.container(border=True):
                            sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        with st.container(border=True):
                            process_and_generate_sunburst(res, direction="from")
                        
            # AI-Powered Insights with model selection (only show if toggle is on)
            if 'show_details' in locals() and show_details:
                def path_ai_analysis_callback(selected_model, prompt_type):
                    """Callback function for path analysis AI insights"""
                    
                    # Show custom prompt input if Custom is selected
                    if prompt_type == "Custom":
                        custom_prompt = st.text_area(
                            "Enter your custom prompt:",
                            value="",
                            key="path_from_custom_prompt",
                            help="Enter your custom analysis prompt. The path data will be automatically included.",
                            placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                        )
                        
                        # Only proceed if custom prompt is not empty
                        if not custom_prompt or custom_prompt.strip() == "":
                            st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                            return
                    
                    with st.spinner("Generating AI insights..."):
                        try:
                            # Use the same number of paths as selected by user (topn)
                            top_paths = res.head(topn)
                            paths_text = "\n".join([f"{row['PATH']} (count: {row['COUNT']})" 
                                                    for _, row in top_paths.iterrows()])
                            
                            if prompt_type == "Auto":
                                ai_prompt = f"""
                                Analyze these top {topn} customer journey paths:
                                
                                {paths_text}
                                
                                Total Paths Found: {len(res)}
                                
                                Please provide insights on:
                                1. Most significant customer journey patterns
                                2. User behavior implications 
                                3. Potential optimization opportunities
                                4. Anomalies or interesting findings
                                
                                Keep your analysis concise and actionable.
                                """
                            else:  # Custom
                                ai_prompt = f"""
                                {custom_prompt}
                                
                                Data to analyze - Top {topn} customer journey paths:
                                {paths_text}
                                
                                Total Paths Found: {len(res)}
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
                    "path_from_ai", 
                    "Select the LLM model for AI analysis of path results",
                    ai_content_callback=path_ai_analysis_callback
                )  
                            
                # View Aggregated Paths SQL
            if st.toggle("View SQL for Aggregated Paths"):     
                    st.code(path_frm_agg_sql, language='sql')

            if st.toggle("Writeback Segments to Snowflake", key="writeback_toggle_path_from", help="Export selected path patterns with their counts and associated user IDs to a Snowflake table for targeted segmentation."):
                with st.expander("Writeback Segments to Snowflake", icon=":material/upload:", expanded=True):
                    # Use aggregated paths data already computed for visualization
                    if path_frm_agg is not None and len(path_frm_agg) > 0:
                        # Convert to DataFrame for easier manipulation
                        paths_df = pd.DataFrame(path_frm_agg)
                        
                        # Path selector
                        available_paths = paths_df['PATH'].tolist()
                        
                        # Select All checkbox
                        select_all = st.checkbox("Select All Paths", value=False, key="wb_select_all_from")
                        
                        # Determine default selection based on checkbox
                        if select_all:
                            default_selection = available_paths
                        else:
                            default_selection = available_paths[:min(10, len(available_paths))]  # Default to top 10
                        
                        selected_paths = st.multiselect(
                            "Choose one or more paths",
                            options=available_paths,
                            default=default_selection,
                            key="wb_path_select_from",
                            help="Select which paths to export. Each path will include COUNT and UID_LIST"
                        )
                        
                        if selected_paths:
                            # Show preview
                            filtered_df = paths_df[paths_df['PATH'].isin(selected_paths)]
                            total_users = sum([len(uid_list) for uid_list in filtered_df['UID_LIST']])
                            st.info(f"📊 Export Preview: {len(selected_paths)} paths | {filtered_df['COUNT'].sum():,} occurrences | {total_users:,} unique users", icon=":material/info:")
                            
                            # Fetch DBs using cached method
                            db0 = fetch_databases(session)
                            
                            # Database, Schema, Table selection
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                wb_database = st.selectbox("Database", db0['name'].unique(), index=None, key="wb_db_path_from", placeholder="Choose...")
                            
                            wb_schema = None
                            if wb_database:
                                schema0 = fetch_schemas(session, wb_database)
                                
                                with col2:
                                    wb_schema = st.selectbox("Schema", schema0['name'].unique(), index=None, key="wb_schema_path_from", placeholder="Choose...")
                            
                            with col3:
                                if wb_database and wb_schema:
                                    wb_table_name = st.text_input("Table Name", key="wb_tbl_path_from", placeholder="e.g. path_from_aggregated")
                                else:
                                    wb_table_name = None
                            
                            # Write mode and button
                            if wb_database and wb_schema and wb_table_name:
                                write_mode = st.radio("Write Mode", ["Create or Replace", "Append to Existing"], key="wb_mode_path_from", horizontal=True)
                                
                                if st.button("Write Table", key="wb_btn_path_from"):
                                    try:
                                        with st.spinner("Writing aggregated paths to Snowflake..."):
                                            # Filter and convert to Snowpark DataFrame
                                            export_df = filtered_df[['PATH', 'COUNT', 'UID_LIST']]
                                            snowpark_df = session.create_dataframe(export_df)
                                            
                                            # Write to table based on mode
                                            if write_mode == "Create or Replace":
                                                snowpark_df.write.mode("overwrite").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            else:
                                                snowpark_df.write.mode("append").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            
                                            # Get actual row count from written table
                                            written_count = session.sql(f"SELECT COUNT(*) as count FROM {wb_database}.{wb_schema}.{wb_table_name}").collect()[0]['COUNT']
                                            
                                            # Success message
                                            st.success(f"Table {wb_database}.{wb_schema}.{wb_table_name} {'created' if write_mode == 'Create or Replace' else 'updated'} successfully with {written_count:,} paths", icon=":material/check:")
                                            
                                    except Exception as e:
                                        st.error(f"Error writing to Snowflake: {str(e)}", icon=":material/chat_error:")
                        else:
                            st.warning("Please select at least one path to export", icon=":material/warning:")
                    else:
                        st.info("No paths available. Please run the analysis first.", icon=":material/info:")
                
            else:
                st.write("")
                
        # Separate block for PATH BETWEEN 
        elif pattern_mode == "FROM/TO" and fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any':
            
            path_betw_agg=None
            path_betw_agg_sql=None
            path_betw_det_df=None
            path_betw_det_sql = None
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                
                if use_lookback:
                    # Look-forward window approach for PATH BETWEEN (captures events AFTER starting event, up to target event)
                    path_betw_agg_sql = f"""
                    select path, count(*) as count, array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (
                            WITH FROM_EVENT_TIMES AS (
                                SELECT {uid}, {tmstp}, {evt},
                                       MIN(CASE WHEN {evt} IN ({fromevt}) THEN {tmstp} END) OVER (PARTITION BY {uid}) as from_event_time
                                FROM {database}.{schema}.{tbl} 
                                WHERE {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}
                            ),
                            TIME_WINDOWED AS (
                                SELECT {uid}, {tmstp}, {evt}
                                FROM FROM_EVENT_TIMES
                                WHERE from_event_time IS NOT NULL 
                                  AND TIMESTAMPDIFF({gap_unit}, from_event_time, {tmstp}) BETWEEN 0 AND {max_gap_value}
                            )
                            SELECT * FROM TIME_WINDOWED
                        ) match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A X* B) 
                            define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt}))
                        {groupby} ) 
                    group by path order by count desc 
                    """
                else:
                    # Traditional min/max events approach for PATH BETWEEN
                    path_betw_agg_sql = f"""
                    select path, count(*) as count, array_agg({uid}) as uid_list from (
                        select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                            from (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                                match_recognize(
                                {partitionby} 
                                order by {tmstp}  
                                measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                                all rows per match
                                AFTER MATCH SKIP {overlap} 
                                pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                                define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt}))
                        {groupby} ) 
                    group by path order by count desc 
                    """
              
                path_betw_agg = session.sql(path_betw_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                 
                path_betw_agg_sql = f"""
                select path, count(*) as count, array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
                FROM events_with_diff)
                 SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                            define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt}))
                    {groupby} ) 
                group by path order by count desc 
                """
               
                path_betw_agg = session.sql(path_betw_agg_sql).collect()
            
            # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_betw_agg)
            import ast
           
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                
                # Store total counts before limiting
                total_paths = len(res)
                total_customers = res['COUNT'].sum()
                
                # Apply TOP N limit at pandas level
                res = res.head(topn)
                
                # Show data summary for transparency
                if not res.empty:
                    displayed_paths = len(res)
                    if total_paths <= topn:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="custom-container-1">
                            <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                                Analysis complete: {displayed_paths:,} of {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys (top {topn} results)
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-container-1">
                        <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                            No paths found matching the specified criteria
                        </h5>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 7])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Forced Layout Graph or Sunburst")
            
                # Place the pills in the second column, but only if the toggle is on
                with col2:
                    if 'show_details' in locals() and show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
                    else:
                        genre = None
                # Place the visualization outside of the columns layout
                if show_details and genre:
            
                    if genre == 'Graph':
                        with st.container(border=True):
                            sigma_graph(res)
                    
                    elif genre == 'Sankey':

                        with st.container(border=True):
                        
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                percentage = st.slider("Display Top % of Paths", 1, 100, 100)
                            with col2:
                                st.write("")
                        
                            # STEP 2: Reset Session State on Query Change**
                            current_df_hash = hash(res.to_json())
                            if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                                st.session_state["last_df_hash"] = current_df_hash  # Update hash
                                # Reset state variables
                                st.session_state["clicked_sankey"] = None
                                st.session_state["clicked_source"] = None
                                st.session_state["clicked_target"] = None
                                st.session_state["selected_uids"] = set()
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                                st.session_state["sankey_chart"] = None
                                st.session_state["sankey_links"] = {}
                                st.session_state["sankey_labels"] = []
                                st.session_state["sortedEventList"] = []
                                
                                #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                            # Session state to store extracted user data
                            if "selected_paths_df" not in st.session_state:
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            
                            
                            clicked_sankey = sankey_chart(res, direction="to", topN_percentage=percentage)
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}|||{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ IDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No IDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
            
                    elif genre == 'Sunburst':
                        with st.container(border=True):
                            process_and_generate_sunburst(res, direction="to")
                        
            # AI-Powered Insights with model selection (only show if toggle is on)
            if 'show_details' in locals() and show_details:
                def path_ai_analysis_callback(selected_model, prompt_type):
                    """Callback function for path analysis AI insights"""
                    
                    # Show custom prompt input if Custom is selected
                    if prompt_type == "Custom":
                        custom_prompt = st.text_area(
                            "Enter your custom prompt:",
                            value="",
                            key="path_between_custom_prompt",
                            help="Enter your custom analysis prompt. The path data will be automatically included.",
                            placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                        )
                        
                        # Only proceed if custom prompt is not empty
                        if not custom_prompt or custom_prompt.strip() == "":
                            st.info("Please enter your custom prompt above to generate AI insights.", icon=":material/info:")
                            return
                    
                    with st.spinner("Generating AI insights..."):
                        try:
                            # Use the same number of paths as selected by user (topn)
                            top_paths = res.head(topn)
                            paths_text = "\n".join([f"{row['PATH']} (count: {row['COUNT']})" 
                                                    for _, row in top_paths.iterrows()])
                            
                            if prompt_type == "Auto":
                                ai_prompt = f"""
                                Analyze these top {topn} customer journey paths:
                                
                                {paths_text}
                                
                                Total Paths Found: {len(res)}
                                
                                Please provide insights on:
                                1. Most significant customer journey patterns
                                2. User behavior implications 
                                3. Potential optimization opportunities
                                4. Anomalies or interesting findings
                                
                                Keep your analysis concise and actionable.
                                """
                            else:  # Custom
                                ai_prompt = f"""
                                {custom_prompt}
                                
                                Data to analyze - Top {topn} customer journey paths:
                                {paths_text}
                                
                                Total Paths Found: {len(res)}
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
                    "path_between_ai", 
                    "Select the LLM model for AI analysis of path results",
                    ai_content_callback=path_ai_analysis_callback
                )                  
                # View Aggregated Paths SQL
            if st.toggle("View SQL for Aggregated Paths"):      
                st.code(path_betw_agg_sql, language='sql')

            if st.toggle("Writeback Segments to Snowflake", key="writeback_toggle_path_between", help="Export selected path patterns with their counts and associated user IDs to a Snowflake table for targeted segmentation."):
                with st.expander("Writeback Segments to Snowflake", icon=":material/upload:", expanded=True):
                    # Use aggregated paths data already computed for visualization
                    if path_betw_agg is not None and len(path_betw_agg) > 0:
                        # Convert to DataFrame for easier manipulation
                        paths_df = pd.DataFrame(path_betw_agg)
                        
                        # Path selector
                        available_paths = paths_df['PATH'].tolist()
                        
                        # Select All checkbox
                        select_all = st.checkbox("Select All Paths", value=False, key="wb_select_all_between")
                        
                        # Determine default selection based on checkbox
                        if select_all:
                            default_selection = available_paths
                        else:
                            default_selection = available_paths[:min(10, len(available_paths))]  # Default to top 10
                        
                        selected_paths = st.multiselect(
                            "Choose one or more paths",
                            options=available_paths,
                            default=default_selection,
                            key="wb_path_select_between",
                            help="Select which paths to export. Each path will include COUNT and UID_LIST"
                        )
                        
                        if selected_paths:
                            # Show preview
                            filtered_df = paths_df[paths_df['PATH'].isin(selected_paths)]
                            total_users = sum([len(uid_list) for uid_list in filtered_df['UID_LIST']])
                            st.info(f"📊 Export Preview: {len(selected_paths)} paths | {filtered_df['COUNT'].sum():,} occurrences | {total_users:,} unique users", icon=":material/info:")
                            
                            # Fetch DBs using cached method
                            db0 = fetch_databases(session)
                            
                            # Database, Schema, Table selection
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                wb_database = st.selectbox("Database", db0['name'].unique(), index=None, key="wb_db_path_between", placeholder="Choose...")
                            
                            wb_schema = None
                            if wb_database:
                                schema0 = fetch_schemas(session, wb_database)
                                
                                with col2:
                                    wb_schema = st.selectbox("Schema", schema0['name'].unique(), index=None, key="wb_schema_path_between", placeholder="Choose...")
                            
                            with col3:
                                if wb_database and wb_schema:
                                    wb_table_name = st.text_input("Table Name", key="wb_tbl_path_between", placeholder="e.g. path_between_aggregated")
                                else:
                                    wb_table_name = None
                            
                            # Write mode and button
                            if wb_database and wb_schema and wb_table_name:
                                write_mode = st.radio("Write Mode", ["Create or Replace", "Append to Existing"], key="wb_mode_path_between", horizontal=True)
                                
                                if st.button("Write Table", key="wb_btn_path_between"):
                                    try:
                                        with st.spinner("Writing aggregated paths to Snowflake..."):
                                            # Filter and convert to Snowpark DataFrame
                                            export_df = filtered_df[['PATH', 'COUNT', 'UID_LIST']]
                                            snowpark_df = session.create_dataframe(export_df)
                                            
                                            # Write to table based on mode
                                            if write_mode == "Create or Replace":
                                                snowpark_df.write.mode("overwrite").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            else:
                                                snowpark_df.write.mode("append").save_as_table(f"{wb_database}.{wb_schema}.{wb_table_name}")
                                            
                                            # Get actual row count from written table
                                            written_count = session.sql(f"SELECT COUNT(*) as count FROM {wb_database}.{wb_schema}.{wb_table_name}").collect()[0]['COUNT']
                                            
                                            # Success message
                                            st.success(f"Table {wb_database}.{wb_schema}.{wb_table_name} {'created' if write_mode == 'Create or Replace' else 'updated'} successfully with {written_count:,} paths", icon=":material/check:")
                                            
                                    except Exception as e:
                                        st.error(f"Error writing to Snowflake: {str(e)}", icon=":material/chat_error:")
                        else:
                            st.warning("Please select at least one path to export", icon=":material/warning:")
                    else:
                        st.info("No paths available. Please run the analysis first.", icon=":material/info:")
                
            else:
                st.write("")
            

        elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any':
            st.warning("This is tuple generator",icon=":material/warning:")
            
            path_tupl_agg=None
            path_tupl_agg_sql=None
            
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                path_tupl_agg_sql = f"""
                select path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ) as path
                        from (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define  A as true)
                    {groupby} ) 
                group by path order by count desc 
                """
                path_tupl_agg = session.sql(path_tupl_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                path_tupl_agg_sql = f"""
                select path, count(*) as count ,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define  A as true)
                    {groupby} ) 
                group by path order by count desc 
                """
                path_tupl_agg = session.sql(path_tupl_agg_sql).collect()
                    # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_tupl_agg)
            if not res.empty:
                import ast
                
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                
                # Store total counts before limiting
                total_paths = len(res)
                total_customers = res['COUNT'].sum()
                
                # Apply TOP N limit at pandas level
                res = res.head(topn)
                
                # Show data summary for transparency
                displayed_paths = len(res)
                if total_paths <= topn:
                    st.info(f"Analysis complete: {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys", icon=":material/info:")
                else:
                    st.info(f"Analysis complete: {displayed_paths:,} of {total_paths:,} unique paths retrieved from {total_customers:,} customer journeys (top {topn} results)", icon=":material/info:")
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 7])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Forced Layout Graph or Sunburst")
            
                # Place the pills in the second column, but only if the toggle is on
                with col2:
                    if 'show_details' in locals() and show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
                    else:
                        genre = None
            
                # Place the visualization outside of the columns layout
                if show_details and genre:
                    if genre == 'Sankey':
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                sli = st.slider("Path Count Filter", 0, topn, topn)
                            with col2:
                                st.write("")
                            #sankeyPlot(res, "to", "", sli)
                            current_df_hash = hash(res.to_json())
                            if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                                st.session_state["last_df_hash"] = current_df_hash  # Update hash
                                # Reset state variables
                                st.session_state["clicked_sankey"] = None
                                st.session_state["clicked_source"] = None
                                st.session_state["clicked_target"] = None
                                st.session_state["selected_uids"] = set()
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                                st.session_state["sankey_chart"] = None
                                st.session_state["sankey_links"] = {}
                                st.session_state["sankey_labels"] = []
                                st.session_state["sortedEventList"] = []
                                
                                #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                            # Session state to store extracted user data
                            if "selected_paths_df" not in st.session_state:
                                st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            
                            clicked_sankey = sankey_chart(res, direction="to")
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}|||{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ IDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No IDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
            
                    elif genre == 'Graph':
                        with st.container(border=True):
                            sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        with st.container(border=True):
                            process_and_generate_sunburst(res, direction="from")
     
                
        else:
            st.write("Please select appropriate options for 'from' and 'to'")
    else:
        st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
            Please ensure all required inputs are selected before running the app.
        </h5>
    </div>
    """, unsafe_allow_html=True)
        #st.warning("Please ensure all required inputs are selected before running the app.")
        
#--------------------------------------
#COMPARE TAB
#--------------------------------------
# PATH COMPARISON
# DEFINE ONE OR TWO TARGET SETS USING THE CONTROL PANE
# 2 MODES AVAILABLE
# COMPLEMENT : Define one reference target set and compare importance of events from events from everything but this target set
# UNION : Define two target sets and compare importance of events for each set

with tab2:
    with st.expander("Comparison Mode", icon=":material/compare:"):
        st.caption("**Complement Mode**(default mode): This mode focuses on ensemble-based analysis by comparing a population defined by a (**Reference**) set of paths against everything outside of it.")
        st.caption("**Union Mode**: This mode performs a comparative analysis between two defined (**Reference**) and (**Compared**) sets of paths (i.e., populations), accounting for any potential overlap.")
        mode=st.radio(
    "Mode:",
    ["Complement", "Union"],
    key="horizontal", horizontal=True,
           label_visibility="collapsed"
)
        
    with st.expander("Input Parameters (Reference)", icon=":material/settings:"):
            
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
                database = st.selectbox('Select Database', key='comparerefdb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected - cached)**
            if database:
                schema0 = fetch_schemas(session, database)
            
                with col2:
                    schema = st.selectbox('Select Schema', key='comparerefsch', index=None, 
                                          placeholder="Choose from list...", options=schema0['name'].unique())
            else:
                schema = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected - cached)**
            if database and schema:
                table0 = fetch_tables(session, database, schema)
            
                with col3:
                    tbl = st.selectbox('Select Event Table or View', key='comparereftbl', index=None, 
                                       placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
            else:
                tbl = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected - cached)**
            if database and schema and tbl:
                colsdf = fetch_columns(session, database, schema, tbl)

            col1, col2, col3 = st.columns([4,4,4])
            with col1:
                uid = st.selectbox('Select identifier column', colsdf, index=None,  key='uidcompareref',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col2: 
                evt = st.selectbox('Select event column', colsdf, index=None, key='evtcompareref', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col3:
                tmstp = st.selectbox('Select timestamp column', colsdf, index=None, key='tsmtpcompareref',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
            
            #Get Distinct Events Of Interest from Event Table (cached)
            if (uid != None and evt != None and tmstp != None):
                distinct_evt_df = fetch_distinct_values(session, database, schema, tbl, evt)
                # Write query output in a pandas dataframe
                startdf0 = distinct_evt_df.copy()
                enddf0 = distinct_evt_df.copy()
                excl0 = distinct_evt_df.copy()


            #Add "any" to the distinct events list
                any_row = {evt: 'Any'}

        # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df = pd.DataFrame([any_row])
        # Add any to list of FROM 
                startdf1 = pd.concat([ any_row_df, startdf0], ignore_index=True)
        # Add any to list of TO
                enddf1 = pd.concat([ any_row_df, enddf0], ignore_index=True)

        #EVENTS PATTERN
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)
                                
                # Add a None placeholder to force user to select an event
                options_with_placeholder_from = ["🔍"] + startdf1[evt].unique().tolist()
                options_with_placeholder_to = ["🔎"] + enddf1[evt].unique().tolist()
                    
                col1, col2, col3,col4 = st.columns([4,4,2,2])

                with col1:
                    frm = st.multiselect('Select events FROM:', options=options_with_placeholder_from[1:], key='evtfrmref',default=[], help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                    #filtered_frmevt = startdf1[(startdf1[evt] == frm)]
                    #fromevt = filtered_frmevt.iloc[0, 0]
                    if frm != "🔍":
                        fromevt= ", ".join([f"'{value}'" for value in frm])
 
                    else:
                        fromevt = None  # Set to None if the placeholder is still selected
                    
                    
                with col2:
                    to = st.multiselect('Select events TO:', options=options_with_placeholder_to[1:],  key='evttoref',default=[],help="Select one or more events of interest to visualize paths TO the selected point(s). 'Any' matches all values.")
                    #filtered_toevt = enddf1[(enddf1[evt] == to)]
                    #toevt =filtered_toevt.iloc[0, 0]
                    if to != "🔎":
                        toevt = ", ".join([f"'{value}'" for value in to])
    
                    else:
                        toevt = None  # Set to None if the placeholder is still selected

                with col3:
                    minnbbevt = st.number_input("Min # events", value=0,  key='minnbbevtref',placeholder="Type a number...",help="Select the minimum number of events either preceding or following the event(s) of interest.")

                with col4:
                    maxnbbevt = st.number_input("Max # events", value=5, min_value=1, key='maxnbbevtref' ,placeholder="Type a number...",help="Select the maximum number of events either preceding or following the event(s) of interest.")
                
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapcompare1",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                    if checkoverlap:
                        overlap='TO NEXT ROW'
                with col2:
                    st.write("")


                #--------------------------------------
                #DATE RANGE
                #--------------------------------------
                with st.container():
                    st.markdown("""
        <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2.4,2.4,10])

                # SQL query to get the min start date
                minstartdt = f"SELECT TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {database}.{schema}.{tbl}"
                # Get min start date :
                defstartdt = session.sql(minstartdt).collect()
                defstartdt_str = defstartdt[0][0] 
                defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
                
                with col1:
                    startdt_input = st.date_input('Start date', key='startdtref', value=defstartdt1)
                with col2:
                    enddt_input = st.date_input('End date', key='enddtref',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3: 
                    st.write("")
               
                #--------------------------------------
                #SESSION
                #--------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session (optional)</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help= "A session, in the context of customer journey analysis, refers to a defined period of activity during which a customer interacts or events occur. It includes all events, actions, or transactions performed by the user within this timeframe. If a session field is present in the event table, select it from the 'Column' tab below. If no session field is available, use the 'Sessionize' tab to create unique session identifiers by grouping events based on time gaps (e.g., a gap of more than 30 minutes starts a new session). Once a session is selected or created, it will be used alongside the unique identifier to partition the input rows before applying pattern matching.")
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess = st.selectbox('Select session column', colsdf, index=None, key='sessref', placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, key='unitoftimeref', placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout=  st.number_input( "Insert a timeout value",key='timeoutref',value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess == None and unitoftime==None and timeout==None: 
                        partitionby = f"partition by {uid}"
                        groupby = f"group by {uid}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                
                elif sess != None and unitoftime==None and timeout==None:
                        partitionby=f"partition by {uid},{sess} "
                        groupby = f"group by {uid}, match_number,{sess} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
                elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        groupby = f"group by {uid}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                else :
                    st.write("")
                #--------------------------------------
                #FILTERS
                #--------------------------------------

                #initialize sql_where_clause
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2  =st.columns([5,10])
                    with col1:
                # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0,placeholder="Select event(s)...", key='reference',help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                          excl3 = "''"
                        else:
                         excl3= ', '.join([f"'{excl2}'" for excl2 in excl1])
                    
                                            
                                    # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                
                    with col1:
                        checkfilters = st.toggle("Additional filters", key='filterscompareref',help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters = False  # Disable filters when no columns are left
                
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters and not remaining_columns.empty:
                
                    with st.container():
                        # Helper to get cached distinct values
                        def get_distinct_values_list(column):
                            df_vals = fetch_distinct_values(session, database, schema, tbl, column)
                            return df_vals.iloc[:, 0].tolist() if not df_vals.empty else []
                
                        # Helper function to display operator selection based on column data type
                        def get_operator_input(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_ref_{filter_index}"  # Ensure unique key
                
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
                            value_key = f"{col_name}_value_ref_{filter_index}"  # Ensure unique key

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
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns, key=f"column_filter_ref_{filter_index}")
                
                            # Determine column data type (cached)
                            col_data_type = fetch_column_type(session, database, schema, tbl, selected_column)
                
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
                            add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"add_filter_ref_{filter_index}")
                
                            if add_filter == 'Yes':
                                col1, col2 = st.columns([2, 13])
                                with col1: 
                                    logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"logic_operator_ref_{filter_index}")
                                    filter_index += 1
                                with col2:
                                    st.write("")
                            else:
                                break
                        
                        # Generate SQL WHERE clause based on selected filters and logic
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
                                    if isinstance(value, (int, float)):
                                        sql_where_clause += f"{col} {operator} {value}"
                                    else:
                                 # For non-numeric values (strings, dates), enclose the value in quotes
                                        sql_where_clause += f"{col} {operator} '{value}'"        
                        else:
                            sql_where_clause = ""
    if mode == "Union":
        
        with st.expander ("Input Parameters (Compared)", icon=":material/settings:"):
            # DATA SOURCE FOR SECOND INSTANCE
            st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
            # Get list of databases (cached)
            db01 = fetch_databases(session)
            
            col1, col2, col3 = st.columns(3)
            
            # **Database Selection**
            with col1:
                database1 = st.selectbox('Select Database', key='comparecompdb', index=None, 
                                        placeholder="Choose from list...", options=db01['name'].unique())
            
            # **Schema Selection (Only if a database is selected - cached)**
            if database1:
                schema01 = fetch_schemas(session, database1)
            
                with col2:
                    schema1 = st.selectbox('Select Schema', key='comparecompsch', index=None, 
                                          placeholder="Choose from list...", options=schema01['name'].unique())
            else:
                schema1 = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected - cached)**
            if database1 and schema1:
                table01 = fetch_tables(session, database1, schema1)
            
                with col3:
                    tbl1 = st.selectbox('Select Event Table or View', key='comparecomptbl', index=None, 
                                       placeholder="Choose from list...", options=table01['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
            else:
                tbl1 = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected - cached)**
            if database1 and schema1 and tbl1:
                colsdf1 = fetch_columns(session, database1, schema1, tbl1)
            
            col4, col5, col6 = st.columns([4,4,4])
            with col4:
                uid1 = st.selectbox('Select identifier column', colsdf1, index=None,  key='uidcomparecomp',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col5:
                evt1 = st.selectbox('Select event column', colsdf1, index=None, key='evtcomparecomp', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col6:
                tmstp1 = st.selectbox('Select timestamp column', colsdf1, index=None, key='tsmtpcomparecomp',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
            
            # Get Distinct Events Of Interest from Event Table (cached)
            if (uid1 is not None and evt1 is not None and tmstp1 is not None):
                distinct_evt_df1 = fetch_distinct_values(session, database1, schema1, tbl1, evt1)
                # Write query output in a pandas dataframe
                startdf1_instance = distinct_evt_df1.copy()
                enddf1_instance = distinct_evt_df1.copy()
                excl0_instance = distinct_evt_df1.copy()
            
                # Add "any" to the distinct events list
                any_row_instance = {evt1: 'Any'}
            
                # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df_instance = pd.DataFrame([any_row_instance])
                # Add any to list of FROM
                startdf1_instance_with_any = pd.concat([any_row_df_instance, startdf1_instance], ignore_index=True)
                # Add any to list of TO
                enddf1_instance_with_any = pd.concat([any_row_df_instance, enddf1_instance], ignore_index=True)
            
                # EVENTS PATTERN (Comp)
                with st.container():
                    st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
                # Add a None placeholder to force user to select an event
                options_with_placeholder_from_instance = ["🔍"] + startdf1_instance_with_any[evt1].unique().tolist()
                options_with_placeholder_to_instance = ["🔎"] + enddf1_instance_with_any[evt1].unique().tolist()
            
                col4, col5, col6, col7 = st.columns([4,4,2,2])
            
                with col4:
                    frm1 = st.multiselect('Select events FROM:', key='evtfromcomp', options=options_with_placeholder_from_instance[1:], default=[], help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                    if frm1 != "🔍":
                        fromevt1 = ", ".join([f"'{value}'" for value in frm1])
                    else:
                        fromevt1 = None  # Set to None if the placeholder is still selected
            
                with col5:
                    to1 = st.multiselect('Select events TO:',  key='evttocomp', options=options_with_placeholder_to_instance[1:], default=[],help="The event column contains the actual events that will be used in the path analysis.")
                    if to1 != "🔎":
                        toevt1 = ", ".join([f"'{value}'" for value in to1])
                    else:
                        toevt1 = None  # Set to None if the placeholder is still selected
            
                with col6:
                    minnbbevt1 = st.number_input("Min # events", value=0,  key='minnbbevtcomp',placeholder="Type a number...",help="Select the minimum number of events either preceding or following the event(s) of interest.")
            
                with col7:
                    maxnbbevt1 = st.number_input("Max # events", value=5, min_value=1,key='maxnbbevtcomp', placeholder="Type a number...",help="Select the maximum number of events either preceding or following the event(s) of interest.")
            
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapcompare2",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                    if checkoverlap:
                        overlap='TO NEXT ROW'
                with col2:
                    st.write("")

                # --------------------------------------
                # DATE RANGE (Comp)
                # --------------------------------------
                with st.container():
                    st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
                col1, col2, col3 = st.columns([2.4,2.4,10])
            
                # SQL query to get the min start date
                minstartdt1 = f"SELECT TO_VARCHAR(MIN({tmstp1}), 'YYYY/MM/DD') FROM {tbl1}"
                # Get min start date :
                defstartdt1_instance = session.sql(minstartdt1).collect()
                defstartdt1_str_instance = defstartdt1_instance[0][0]
                defstartdt1_date_instance = datetime.datetime.strptime(defstartdt1_str_instance, '%Y/%m/%d').date()
            
                
                with col1:
                    startdt_input1 = st.date_input('Start date', key='startdtcomp',value=defstartdt1_date_instance)
                with col2:
                    enddt_input1  = st.date_input('End date', key='enddtcomp',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3: 
                    st.write("")
            
                # --------------------------------------
                # SESSION (Comp)
                # --------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
            
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess1 = st.selectbox('Select session column', colsdf1, index=None, key='sesscomp', placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime1 =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, key='unitoftimecomp', placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout1=  st.number_input( "Insert a timeout value",key='timeoutcomp',value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess1 == None and unitoftime1==None and timeout1==None: 
                        partitionby1 = f"partition by {uid1}"
                        groupby1 = f"group by {uid1}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns1 = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1])]['COLUMN_NAME']
                
                elif sess1 != None and unitoftime1==None and timeout1==None:
                        partitionby1=f"partition by {uid1},{sess1} "
                        groupby1 = f"group by {uid1}, match_number,{sess1} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1,sess1])]['COLUMN_NAME']
    
                elif sess1 == None and unitoftime1 !=None and timeout1 !=None:
                        partitionby1=f"partition by {uid1},SESSION "
                        groupby1 = f"group by {uid1}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns1 = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1])]['COLUMN_NAME']
                else :
                    st.write("")
            
                # --------------------------------------
                # FILTERS (Comp)
                # --------------------------------------
                #initialize sql_where_clause
                sql_where_clause_instance = ""
                
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
            
                col4, col5 = st.columns([5, 10])
                with col4:
                    # Exclude Events
                    excl1 = st.multiselect('Exclude event(s) - optional', excl0_instance, placeholder="Select event(s)...",key='compared',help="Event(s) to be excluded from the pattern evaluation and the ouput.")
            
                    if not excl1:
                        excl3_instance = "''"
                    else:
                        excl3_instance = ', '.join([f"'{excl2}'" for excl2 in excl1])
                # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                    with col1:
                        checkfilters1 = st.toggle("Additional filters", key='filterscomparecomp', help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters1 = False  # Disable filters when no columns are left
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters1 and not remaining_columns.empty:
                # ADDITIONAL FILTERS (Comp)
                #checkfilters1 = st.toggle("Additional filters",key='addfilterscomp',help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                
                #if checkfilters1:
                    with st.container():
                        # Helper function to get cached distinct values for compared instance
                        def get_distinct_values_list_instance(column):
                            df_vals = fetch_distinct_values(session, database1, schema1, tbl1, column)
                            return df_vals.iloc[:, 0].tolist() if not df_vals.empty else []
            
                        # Helper function to display operator selection based on column data type
                        def get_operator_input_instance(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_comp_{filter_index}"  # Ensure unique key
            
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                operator = st.selectbox(f"Operator (Comp)", ['=', '<', '<=', '>', '>=', '!=', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                operator = st.selectbox(f"Operator (Comp)", ['=', '<', '<=', '>', '>=', '!=', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox(f"Operator (Comp)", ['=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'IS NULL', 'IS NOT NULL'], key=operator_key)
                            return operator
            
                        # Helper function to display value input based on column data type
                        def get_value_input_instance(col_name, col_data_type, operator, filter_index):
                            """ Returns the value for filtering based on column type """
                            value_key = f"{col_name}_value_comp_{filter_index}"  # Ensure unique key
            
                            # Handle NULL operators - no value input needed
                            if operator in ['IS NULL', 'IS NOT NULL']:
                                return None
                            
                            # Handle IN and NOT IN operators
                            elif operator in ['IN', 'NOT IN']:
                                distinct_values = get_distinct_values_list_instance(col_name)
                                value = st.multiselect(f"Values for {col_name} (Comp)", distinct_values, key=value_key)
                                return value
                            
                            # Handle LIKE and NOT LIKE operators
                            elif operator in ['LIKE', 'NOT LIKE']:
                                value = st.text_input(f"Pattern for {col_name} (Comp) (use % for wildcards)", key=value_key, placeholder="e.g., %text% or prefix%")
                                return value
                            
                            # Handle date/timestamp columns
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                value = st.date_input(f"Value for {col_name} (Comp)", key=value_key)
                                return value
                            
                            # Handle numeric columns with accept_new_options for manual input
                            elif col_data_type in ['NUMBER', 'FLOAT', 'INT', 'DECIMAL']:
                                distinct_values = get_distinct_values_list_instance(col_name)
                                value = st.selectbox(f"Value for {col_name} (Comp)", distinct_values, key=value_key, accept_new_options=True)
                                return value
                            
                            # Handle other data types (strings, etc.)
                            else:
                                distinct_values = get_distinct_values_list_instance(col_name)
                                value = st.selectbox(f"Value for {col_name} (Comp)", distinct_values, key=value_key)
                            return value
            
                        # Initialize variables to store filters and logical conditions
                        filters_instance = []
                        logic_operator_instance = None
                        filter_index_instance = 0  # A unique index to assign unique keys to each filter
            
                        # Dynamic filter loop with one selectbox at a time
                        while True:
                            available_columns1 = remaining_columns1
            
                            if available_columns1.empty:
                                st.write("No more columns available for filtering (Comp).")
                                break
            
                            # Create columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])
            
                            with col1:
                                selected_column_instance = st.selectbox(f"Column (Filter {filter_index_instance + 1} - Comp)", available_columns1, key=f"column_filter_comp_{filter_index_instance}")
            
                            # Determine column data type (cached)
                            col_data_type_instance = fetch_column_type(session, database1, schema1, tbl1, selected_column_instance)
            
                            with col2:
                                operator_instance = get_operator_input_instance(selected_column_instance, col_data_type_instance, filter_index_instance)
            
                            with col3:
                                value_instance = get_value_input_instance(selected_column_instance, col_data_type_instance, operator_instance, filter_index_instance)
            
                            # Append filter as a tuple (column, operator, value)
                            # For NULL operators, value is None and that's expected
                            # For other operators, we need a valid value (except for empty lists in IN/NOT IN)
                            if operator_instance:
                                if operator_instance in ['IS NULL', 'IS NOT NULL']:
                                    filters_instance.append((selected_column_instance, operator_instance, None))
                                elif operator_instance in ['IN', 'NOT IN'] and value_instance:  # Must have at least one value for IN/NOT IN
                                    filters_instance.append((selected_column_instance, operator_instance, value_instance))
                                elif operator_instance not in ['IS NULL', 'IS NOT NULL', 'IN', 'NOT IN'] and (value_instance is not None and value_instance != ''):
                                    filters_instance.append((selected_column_instance, operator_instance, value_instance))
            
                            add_filter_instance = st.radio(f"Add another filter after {selected_column_instance} (Comp)?", ['No', 'Yes'], key=f"add_filter_comp_{filter_index_instance}")
            
                            if add_filter_instance == 'Yes':
                                col1, col2 = st.columns([2, 13])
                                with col1:
                                    logic_operator_instance = st.selectbox(f"Choose logical operator after filter {filter_index_instance + 1} (Comp)", ['AND', 'OR'], key=f"logic_operator_comp_{filter_index_instance}")
                                    filter_index_instance += 1
                                with col2:
                                    st.write("")
                            else:
                                break
            
                        # Generate SQL WHERE clause for the second instance
                        if filters_instance:
                            sql_where_clause_instance = " AND "
                        for i, (col_instance, operator_instance, value_instance) in enumerate(filters_instance):
                            if i > 0 and logic_operator_instance:
                                sql_where_clause_instance += f" {logic_operator_instance} "
            
                            # Handle NULL operators
                            if operator_instance in ['IS NULL', 'IS NOT NULL']:
                                sql_where_clause_instance += f"{col_instance} {operator_instance}"
                            
                            # Handle IN and NOT IN operators
                            elif operator_instance in ['IN', 'NOT IN']:
                                if len(value_instance) == 1:
                                    # Single value - convert to = or != for better performance
                                    single_op = '=' if operator_instance == 'IN' else '!='
                                    if isinstance(value_instance[0], (int, float)):
                                        sql_where_clause_instance += f"{col_instance} {single_op} {value_instance[0]}"
                                    else:
                                        sql_where_clause_instance += f"{col_instance} {single_op} '{value_instance[0]}'"
                                else:
                                    # Multiple values - use proper IN/NOT IN with tuple
                                    # Handle mixed types by converting all to strings for SQL safety
                                    formatted_values = []
                                    for v in value_instance:
                                        if isinstance(v, (int, float)):
                                            formatted_values.append(str(v))
                                        else:
                                            formatted_values.append(f"'{v}'")
                                    sql_where_clause_instance += f"{col_instance} {operator_instance} ({', '.join(formatted_values)})"
                            
                            # Handle LIKE and NOT LIKE operators
                            elif operator_instance in ['LIKE', 'NOT LIKE']:
                                sql_where_clause_instance += f"{col_instance} {operator_instance} '{value_instance}'"
                            
                            # Handle other operators (=, !=, <, <=, >, >=)
                            else:
                                if isinstance(value_instance, (int, float)):
                                    sql_where_clause_instance += f"{col_instance} {operator_instance} {value_instance}"
                                else:
                                    # For non-numeric values (strings, dates), enclose the value in quotes
                                    sql_where_clause_instance += f"{col_instance} {operator_instance} '{value_instance}'"
            
                        # Display the generated SQL WHERE clause
                        #st.write(f"Generated SQL WHERE clause (Comp): {sql_where_clause_instance}")
                        else:
                            sql_where_clause_instance = ""
            
            else:
                sql_where_clause_instance = ""
        
    # SQL LOGIC
    # Check pattern an run SQL accordingly
    if mode == 'Complement':
        
        if all([uid, evt, tmstp,fromevt, toevt]):
            with st.expander("Group Labels", icon=":material/label:"):
                    #Name Reference and Compared Group
                rename_groups= st.checkbox("Label Reference and Compared groups", key="disabled")
                
                reference_label = 'REFERENCE'
                compared_label = 'COMPARED'
                
                if rename_groups:
                    col1, col2 = st.columns(2)
                    with col1:
                            reference_label = st.text_input(
                            "Reference Group Name",
                            placeholder="Please name the reference group",
                            )     
                    with col2:
                             compared_label = st.text_input(
                             "Compared Group Name",
                            placeholder="Please name the compared group",
                            )
    
            # Continue with SQL generation and execution based on inputs...
    
            # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
            if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.toggle("Show me!", key='complementto'):
                    with st.spinner("Analyzing path comparison..."):
                        # Generate unique table names
                        def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                            unique_refid = uuid.uuid4().hex  # Generate a random UUID
                            return f"{base_name}_{unique_refid}"
                        unique_reftable_name = generate_unique_reftable_name()
                    
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                    unique_comptable_name = generate_unique_comptable_name()
                    
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                        # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                        
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                    # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime==None and timeout ==None :
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {database}.{schema}.{tbl} where {evt} = {toevt} )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {database}.{schema}.{tbl} where {evt} = {toevt} )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({toevt})
                        )  {groupby}) """
    
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                    
                    # Reshape the data for plotting
                    restfidf_pivot = (
                        restfidf
                        .pivot(index="EVENT", columns="GRP_Labeled", values="RANKING")
                        .reset_index()
                        .melt(id_vars="EVENT", value_name="RANKING", var_name="GRP_Labeled")
                    )
                    
                    # Identify duplicates within each group (GRP_Labeled) and adjust ranking for duplicates only
                    restfidf_pivot['is_duplicate'] = restfidf_pivot.duplicated(subset=['GRP_Labeled', 'RANKING'], keep=False)
                    
                    # Create the adjusted ranking column
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING']
                    restfidf_pivot.loc[restfidf_pivot['is_duplicate'], 'adjusted_ranking'] = (
                        restfidf_pivot.loc[restfidf_pivot['is_duplicate']]
                        .groupby(['GRP_Labeled', 'RANKING'])
                        .cumcount() * 0.3 + restfidf_pivot['RANKING']
                    )
                    
                    # Drop the temporary 'is_duplicate' column if no longer needed
                    restfidf_pivot.drop(columns='is_duplicate', inplace=True)
                    
                    # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["adjusted_ranking"].nunique()  # Number of unique adjusted ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                                            
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', axis=alt.Axis(grid=False, title='Ranking', labelAngle=0), sort='descending'),
                        color=alt.Color('EVENT:N', legend=None),
                        detail='EVENT:N',
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                     ).properties(
                         width=800,
                         height=dynamic_height,
                         title=""
                     )
                     
                     # Add labels for each point
                    # For the COMPARED group (right-side labels)
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the REFERENCE group (left-side labels)
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0),sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
           
                    
                     # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                     
                     # Add styled title for slope chart
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Event Ranking Comparison</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                     # Display in Streamlit
                    with st.container(border=True):
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    # Add legend in expander
                    with st.expander("Event Legend", expanded=False,icon=":material/palette:"):
                        # Get unique events and their colors from the chart
                        unique_events = sorted(restfidf_pivot['EVENT'].unique())
                        legend_columns = st.columns(4)
                        col_idx = 0
                        
                        # Use Altair's default color scheme
                        altair_colors = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
                        ]
                        
                        for i, event in enumerate(unique_events):
                            color = altair_colors[i % len(altair_colors)]
                            with legend_columns[col_idx]:
                                st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                            col_idx = (col_idx + 1) % 4
                                        
                     # Add styled title for sunburst charts
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Path Comparison - Sunburst Charts</h2>
                         <hr style='margin-top: -8px;margin-bottom:10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                    # Create aggregations for sunburst charts - use same structure as analyze tab
                    ref_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_reftable_name} GROUP BY path ORDER BY COUNT DESC"
                    comp_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_comptable_name} GROUP BY path ORDER BY COUNT DESC"
                    
                    try:
                        ref_sunburst_data = session.sql(ref_sunburst_sql).collect()
                        comp_sunburst_data = session.sql(comp_sunburst_sql).collect()
                        
                        ref_sunburst_df = pd.DataFrame(ref_sunburst_data)
                        comp_sunburst_df = pd.DataFrame(comp_sunburst_data)
                        
                        if not ref_sunburst_df.empty and not comp_sunburst_df.empty:
                            # Columns are already named correctly: PATH, COUNT, UID_LIST
                            # Convert UID_LIST from string to actual list (same as analyze tab)
                            import ast
                            def convert_uid_list(uid_entry):
                                if isinstance(uid_entry, str):
                                    try:
                                        return ast.literal_eval(uid_entry)
                                    except (SyntaxError, ValueError):
                                        return []
                                return uid_entry if isinstance(uid_entry, list) else []
                            
                            ref_sunburst_df['UID_LIST'] = ref_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            comp_sunburst_df['UID_LIST'] = comp_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            
                            # Extract all unique events from both datasets to ensure consistent colors
                            all_events = set()
                            for _, row in ref_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            for _, row in comp_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            
                            # Generate consistent color map for all events
                            shared_color_map = generate_colors(all_events)
                            
                            # Display sunburst charts side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{reference_label}</div>", unsafe_allow_html=True)
                                 # Add space to prevent tooltip truncation
                                 st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(ref_sunburst_df, direction="to", color_map=shared_color_map)
                             
                            with col2:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{compared_label}</div>", unsafe_allow_html=True)
                                 # Add space to prevent tooltip truncation
                                 st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(comp_sunburst_df, direction="to", color_map=shared_color_map)
                            
                            # Display shared legend in expander
                            with st.expander("Event legend", expanded=False, icon=":material/palette:"):
                                legend_columns = st.columns(4)  
                                col_idx = 0
                                for event, color in shared_color_map.items():
                                    with legend_columns[col_idx]:
                                        st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                                    
                                    col_idx = (col_idx + 1) % 4
                        else:
                            st.info("No path data available for sunburst visualization", icon=":material/info:")
                    
                    except Exception as e:
                        st.error(f"Error generating sunburst charts: {e}", icon=":material/chat_error:")
                                        
                    # Robust cleanup of temporary tables
                    temp_tables = [unique_reftable_name, unique_comptable_name, unique_comptftable_name, unique_reftftable_name]
                    for table_name in temp_tables:
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
                        except Exception as e:
                            st.warning(f"Could not drop table {table_name}: {str(e)}")
                            pass
                else:
                        st.write("") 
    
            # Separate block for PATH FROM 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
    
                if st.toggle("Show me!", key='complementfrom'):
                    with st.spinner("Analyzing path comparison..."):
                       # Generate a unique ref table name
                        def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                            unique_refid = uuid.uuid4().hex  # Generate a random UUID
                            return f"{base_name}_{unique_refid}"
                        unique_reftable_name = generate_unique_reftable_name()
                         # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        st.write(crttblrawseventsrefsql)
                    
                    elif unitoftime != None and timeout !=None :
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        st.write(crttblrawseventsrefsql)
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                     # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime==None and timeout ==None :
                    
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3})and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({fromevt})
                        )  {groupby}) """
                       
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({fromevt})
                        )  {groupby}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
            
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                # Reshape the data for plotting
                    restfidf_pivot = restfidf.pivot(index="EVENT", columns="GRP_Labeled", values="RANKING").reset_index().melt(id_vars="EVENT", value_name="RANKING",var_name="GRP_Labeled")
                # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["RANKING"].nunique()  # Number of unique ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                    
                    # Add adjusted ranking to avoid overlap
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING'] + restfidf_pivot.groupby('RANKING').cumcount() * 0.3
                    
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q', scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False), title='Ranking'),
                        color=alt.Color('EVENT:N', legend=None),
                        detail='EVENT:N',
                        tooltip=[alt.Tooltip('RANKING:Q', title="Original Ranking")]  # Tooltip for non-adjusted ranking
                     ).properties(
                         width=800,
                         height=dynamic_height,
                         title=""
                     )
                     
                     # Add labels for each point
                    # For the "COMPARED" group, we place labels on the right
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),  # Explicitly set text to EVENT column
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the "REFERENCE" group, we place labels on the left
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
                     # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                     
                     # Add styled title for slope chart
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Event Ranking Comparison</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                     # Display in Streamlit
                    with st.container(border=True):
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    # Add legend in expander
                    with st.expander("Event Legend", expanded=False,icon=":material/palette:"):
                        # Get unique events and their colors from the chart
                        unique_events = sorted(restfidf_pivot['EVENT'].unique())
                        legend_columns = st.columns(4)
                        col_idx = 0
                        
                        # Use Altair's default color scheme
                        altair_colors = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
                        ]
                        
                        for i, event in enumerate(unique_events):
                            color = altair_colors[i % len(altair_colors)]
                            with legend_columns[col_idx]:
                                st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                            col_idx = (col_idx + 1) % 4
                                        
                     # Add styled title for sunburst charts
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Path Comparison - Sunburst Charts</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                    # Create aggregations for sunburst charts - use same structure as analyze tab
                    ref_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_reftable_name} GROUP BY path ORDER BY COUNT DESC"
                    comp_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_comptable_name} GROUP BY path ORDER BY COUNT DESC"
                    
                    try:
                        ref_sunburst_data = session.sql(ref_sunburst_sql).collect()
                        comp_sunburst_data = session.sql(comp_sunburst_sql).collect()
                        
                        ref_sunburst_df = pd.DataFrame(ref_sunburst_data)
                        comp_sunburst_df = pd.DataFrame(comp_sunburst_data)
                        
                        if not ref_sunburst_df.empty and not comp_sunburst_df.empty:
                            # Columns are already named correctly: PATH, COUNT, UID_LIST
                            # Convert UID_LIST from string to actual list (same as analyze tab)
                            import ast
                            def convert_uid_list(uid_entry):
                                if isinstance(uid_entry, str):
                                    try:
                                        return ast.literal_eval(uid_entry)
                                    except (SyntaxError, ValueError):
                                        return []
                                return uid_entry if isinstance(uid_entry, list) else []
                            
                            ref_sunburst_df['UID_LIST'] = ref_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            comp_sunburst_df['UID_LIST'] = comp_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            
                            # Extract all unique events from both datasets to ensure consistent colors
                            all_events = set()
                            for _, row in ref_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            for _, row in comp_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            
                            # Generate consistent color map for all events
                            shared_color_map = generate_colors(all_events)
                            
                            # Display sunburst charts side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{reference_label}</div>", unsafe_allow_html=True)
                                 # Add space to prevent tooltip truncation
                                 st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(ref_sunburst_df, direction="to", color_map=shared_color_map)
                             
                            with col2:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{compared_label}</div>", unsafe_allow_html=True)
                                 # Add space to prevent tooltip truncation
                                 st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(comp_sunburst_df, direction="to", color_map=shared_color_map)
                            
                            # Display shared legend in expander
                            with st.expander("Event legend", expanded=False, icon=":material/palette:"):
                                legend_columns = st.columns(4)  
                                col_idx = 0
                                for event, color in shared_color_map.items():
                                    with legend_columns[col_idx]:
                                        st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                                    
                                    col_idx = (col_idx + 1) % 4
                        else:
                            st.info("No path data available for sunburst visualization", icon=":material/info:")
                    
                    except Exception as e:
                        st.error(f"Error generating sunburst charts: {e}", icon=":material/chat_error:")
                                        
                    # Robust cleanup of temporary tables
                    temp_tables = [unique_reftable_name, unique_comptable_name, unique_comptftable_name, unique_reftftable_name]
                    for table_name in temp_tables:
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
                        except Exception as e:
                            st.warning(f"Could not drop table {table_name}: {str(e)}")
                            pass
        
                else:
                    st.write("")
                    
            # Separate block for PATH BETWEEN 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any':
                st.warning("Not a valid pattern for comparison",icon=":material/warning:")
    
            elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any':
                st.warning("This is tuple generator - Not a valid pattern for comparison",icon=":material/warning:")
                
            else:
                st.write("Please select appropriate options for 'from' and 'to'")
        else:
            st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                    Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)
                </h5>
            </div>
            """, unsafe_allow_html=True)
            #st.warning("Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)")
   
    elif mode == 'Union':
        if all([uid, evt, tmstp,fromevt, toevt,uid1, evt1, tmstp1,fromevt1, toevt1]):
            with st.expander("Group Labels", icon=":material/label:"):
                    #Name Reference and Compared Group
                rename_groups= st.checkbox("Label Reference and Compared groups", key="disabled")
                
                reference_label = 'REFERENCE'
                compared_label = 'COMPARED'
                
                if rename_groups:
                    col1, col2 = st.columns(2)
                    with col1:
                            reference_label = st.text_input(
                            "Reference Group Name",
                            placeholder="Please name the reference group",
                            )     
                    with col2:
                             compared_label = st.text_input(
                             "Compared Group Name",
                            placeholder="Please name the compared group",
                            )
            # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
            if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any' and fromevt1.strip("'") == 'Any' and toevt1.strip("'") != 'Any':     
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.toggle("Show me!", key='unionto'):
                    with st.spinner("Analyzing path comparison..."):
                            # Generate a unique ref table name
                     def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                     unique_reftable_name = generate_unique_reftable_name()
                    
                        # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                    # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime1==None and timeout1 ==None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (select * from {database1}.{schema1}.{tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}') {sql_where_clause_instance}) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A{{{minnbbevt1},{maxnbbevt1}}} B) 
                            define A as true, B AS {evt1} IN ({toevt1})
                        )  {groupby1}) """
                    elif unitoftime1 != None and timeout1 !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid1},{tmstp1},{evt1},TIMESTAMPDIFF({unitoftime1}, LAG({tmstp1}) OVER (PARTITION BY  {uid1} ORDER BY {tmstp1}),
                        {tmstp1}) AS TIMEWINDOW FROM {database1}.{schema1}.{tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}'){sql_where_clause_instance})
                        ,sessions AS (SELECT {uid1},{tmstp1},{evt1},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout1} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid1} ORDER BY {tmstp1}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A{{{minnbbevt1},{maxnbbevt1}}} B) 
                            define A as true, B AS {evt1} IN ({toevt1})
                        )  {groupby1}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                 # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid1},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                    
                    # Reshape the data for plotting
                    restfidf_pivot = (
                        restfidf
                        .pivot(index="EVENT", columns="GRP_Labeled", values="RANKING")
                        .reset_index()
                        .melt(id_vars="EVENT", value_name="RANKING", var_name="GRP_Labeled")
                    )
                    
                    # Identify duplicates within each group (GRP_Labeled) and adjust ranking for duplicates only
                    restfidf_pivot['is_duplicate'] = restfidf_pivot.duplicated(subset=['GRP_Labeled', 'RANKING'], keep=False)
                    
                    # Create the adjusted ranking column
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING']
                    restfidf_pivot.loc[restfidf_pivot['is_duplicate'], 'adjusted_ranking'] = (
                        restfidf_pivot.loc[restfidf_pivot['is_duplicate']]
                        .groupby(['GRP_Labeled', 'RANKING'])
                        .cumcount() * 0.3 + restfidf_pivot['RANKING']
                    )
                    
                    # Drop the temporary 'is_duplicate' column if no longer needed
                    restfidf_pivot.drop(columns='is_duplicate', inplace=True)
                    
                    # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["adjusted_ranking"].nunique()  # Number of unique adjusted ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                                            
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', axis=alt.Axis(grid=False, title='Ranking', labelAngle=0), sort='descending'),
                        color=alt.Color('EVENT:N', legend=None),
                        detail='EVENT:N',
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                     ).properties(
                         width=800,
                         height=dynamic_height,
                         title=""
                     )
                     
                     # Add labels for each point
                    # For the COMPARED group (right-side labels)
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the REFERENCE group (left-side labels)
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0),sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
           
                    
                     # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                     
                     # Add styled title for slope chart
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Event Ranking Comparison</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                     # Display in Streamlit
                    with st.container(border=True):
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    # Add legend in expander
                    with st.expander("Event Legend", expanded=False,icon=":material/palette:"):
                        # Get unique events and their colors from the chart
                        unique_events = sorted(restfidf_pivot['EVENT'].unique())
                        legend_columns = st.columns(4)
                        col_idx = 0
                        
                        # Use Altair's default color scheme
                        altair_colors = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
                        ]
                        
                        for i, event in enumerate(unique_events):
                            color = altair_colors[i % len(altair_colors)]
                            with legend_columns[col_idx]:
                                st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                            col_idx = (col_idx + 1) % 4
                                        
                     # Add styled title for sunburst charts
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Path Comparison - Sunburst Charts</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    
                    # Create aggregations for sunburst charts - use same structure as analyze tab
                    ref_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_reftable_name} GROUP BY path ORDER BY COUNT DESC"
                    comp_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_comptable_name} GROUP BY path ORDER BY COUNT DESC"
                    
                    try:
                        ref_sunburst_data = session.sql(ref_sunburst_sql).collect()
                        comp_sunburst_data = session.sql(comp_sunburst_sql).collect()
                        
                        ref_sunburst_df = pd.DataFrame(ref_sunburst_data)
                        comp_sunburst_df = pd.DataFrame(comp_sunburst_data)
                        
                        if not ref_sunburst_df.empty and not comp_sunburst_df.empty:
                            # Columns are already named correctly: PATH, COUNT, UID_LIST
                            # Convert UID_LIST from string to actual list (same as analyze tab)
                            import ast
                            def convert_uid_list(uid_entry):
                                if isinstance(uid_entry, str):
                                    try:
                                        return ast.literal_eval(uid_entry)
                                    except (SyntaxError, ValueError):
                                        return []
                                return uid_entry if isinstance(uid_entry, list) else []
                            
                            ref_sunburst_df['UID_LIST'] = ref_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            comp_sunburst_df['UID_LIST'] = comp_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            
                            # Extract all unique events from both datasets to ensure consistent colors
                            all_events = set()
                            for _, row in ref_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            for _, row in comp_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            
                            # Generate consistent color map for all events
                            shared_color_map = generate_colors(all_events)
                            
                            # Display sunburst charts side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{reference_label}</div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(ref_sunburst_df, direction="from", color_map=shared_color_map)
                            
                            with col2:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{compared_label}</div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(comp_sunburst_df, direction="from", color_map=shared_color_map)
                            
                            # Display shared legend in expander
                            with st.expander("Event legend", expanded=False, icon=":material/palette:"):
                                legend_columns = st.columns(4)  
                                col_idx = 0
                                for event, color in shared_color_map.items():
                                    with legend_columns[col_idx]:
                                        st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                                    
                                    col_idx = (col_idx + 1) % 4
                        else:
                            st.info("No path data available for sunburst visualization", icon=":material/info:")
                    
                    except Exception as e:
                        st.error(f"Error generating sunburst charts: {e}", icon=":material/chat_error:")
                                        
                    # Robust cleanup of temporary tables
                    temp_tables = [unique_reftable_name, unique_comptable_name, unique_comptftable_name, unique_reftftable_name]
                    for table_name in temp_tables:
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
                        except Exception as e:
                            st.warning(f"Could not drop table {table_name}: {str(e)}")
                            pass
                else:
                        st.write("") 
    
            # Separate block for PATH FROM 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any'and fromevt1.strip("'") != 'Any' and toevt1.strip("'")== 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.toggle("Show me!", key='unionfrom'):
                    with st.spinner("Analyzing path comparison..."):
                       # Generate a unique ref table name
                        def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                            unique_refid = uuid.uuid4().hex  # Generate a random UUID
                            return f"{base_name}_{unique_refid}"
                        unique_reftable_name = generate_unique_reftable_name()
                         # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        
                        
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                     # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime1==None and timeout1 ==None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (select * from {database1}.{schema1}.{tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}') {sql_where_clause_instance}) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A B{{{minnbbevt1},{maxnbbevt1}}}) 
                            define B as true, A AS {evt1} IN ({fromevt1})
                        )  {groupby1}) """
                    elif unitoftime1 != None and timeout1 !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid1},{tmstp1},{evt1},TIMESTAMPDIFF({unitoftime1}, LAG({tmstp1}) OVER (PARTITION BY  {uid1} ORDER BY {tmstp1}),
                        {tmstp1}) AS TIMEWINDOW FROM {database1}.{schema1}.{tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}'){sql_where_clause_instance})
                        ,sessions AS (SELECT {uid1},{tmstp1},{evt1},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout1} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid1} ORDER BY {tmstp1}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A B{{{minnbbevt1},{maxnbbevt1}}}) 
                            define B as true, A AS {evt1} IN ({fromevt1})
                        )  {groupby1}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                 # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid1},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                 # Reshape the data for plotting
                    restfidf_pivot = restfidf.pivot(index="EVENT", columns="GRP_Labeled", values="RANKING").reset_index().melt(id_vars="EVENT", value_name="RANKING",var_name="GRP_Labeled")
                 # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["RANKING"].nunique()  # Number of unique ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                    
                    # Add adjusted ranking to avoid overlap
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING'] + restfidf_pivot.groupby('RANKING').cumcount() * 0.3
                    
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q', scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False), title='Ranking'),
                        color=alt.Color('EVENT:N', legend=None),
                        detail='EVENT:N',
                        tooltip=[alt.Tooltip('RANKING:Q', title="Original Ranking")]  # Tooltip for non-adjusted ranking
                     ).properties(
                         width=800,
                         height=dynamic_height,
                         title=""
                     )
                     
                     # Add labels for each point
                    # For the "COMPARED" group, we place labels on the right
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),  # Explicitly set text to EVENT column
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the "REFERENCE" group, we place labels on the left
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
                     # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                     
                     # Add styled title for slope chart
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Event Ranking Comparison</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    st.write("")
                     # Display in Streamlit
                    with st.container(border=True):
                        st.altair_chart(final_chart, use_container_width=True)
                    
                    # Add legend in expander
                    with st.expander("Event Legend", expanded=False,icon=":material/palette:"):
                        # Get unique events and their colors from the chart
                        unique_events = sorted(restfidf_pivot['EVENT'].unique())
                        legend_columns = st.columns(4)
                        col_idx = 0
                        
                        # Use Altair's default color scheme
                        altair_colors = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
                        ]
                        
                        for i, event in enumerate(unique_events):
                            color = altair_colors[i % len(altair_colors)]
                            with legend_columns[col_idx]:
                                st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                            col_idx = (col_idx + 1) % 4
                                        
                     # Add styled title for sunburst charts
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Path Comparison - Sunburst Charts</h2>
                         <hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>
                         """, unsafe_allow_html=True)
                    
                    # Create aggregations for sunburst charts - use same structure as analyze tab
                    ref_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_reftable_name} GROUP BY path ORDER BY COUNT DESC"
                    comp_sunburst_sql = f"SELECT path as PATH, COUNT(*) as COUNT, ARRAY_AGG({uid}) as UID_LIST FROM {unique_comptable_name} GROUP BY path ORDER BY COUNT DESC"
                    
                    try:
                        ref_sunburst_data = session.sql(ref_sunburst_sql).collect()
                        comp_sunburst_data = session.sql(comp_sunburst_sql).collect()
                        
                        ref_sunburst_df = pd.DataFrame(ref_sunburst_data)
                        comp_sunburst_df = pd.DataFrame(comp_sunburst_data)
                        
                        if not ref_sunburst_df.empty and not comp_sunburst_df.empty:
                            # Columns are already named correctly: PATH, COUNT, UID_LIST
                            # Convert UID_LIST from string to actual list (same as analyze tab)
                            import ast
                            def convert_uid_list(uid_entry):
                                if isinstance(uid_entry, str):
                                    try:
                                        return ast.literal_eval(uid_entry)
                                    except (SyntaxError, ValueError):
                                        return []
                                return uid_entry if isinstance(uid_entry, list) else []
                            
                            ref_sunburst_df['UID_LIST'] = ref_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            comp_sunburst_df['UID_LIST'] = comp_sunburst_df['UID_LIST'].apply(convert_uid_list)
                            
                            # Extract all unique events from both datasets to ensure consistent colors
                            all_events = set()
                            for _, row in ref_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            for _, row in comp_sunburst_df.iterrows():
                                path_str = row['PATH']
                                # Handle both comma-space and comma-only separators
                                if ", " in path_str:
                                    path_events = path_str.split(", ")
                                else:
                                    path_events = path_str.split(",")
                                path_events = [event.strip() for event in path_events]
                                all_events.update(path_events)
                            
                            # Generate consistent color map for all events
                            shared_color_map = generate_colors(all_events)
                            
                            # Display sunburst charts side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{reference_label}</div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(ref_sunburst_df, direction="from", color_map=shared_color_map)
                            
                            with col2:
                                with st.container(border=True):
                                 st.markdown(f"<div style='text-align: center; font-size: 14px; font-weight: normal;'>{compared_label}</div>", unsafe_allow_html=True)
                                 process_and_generate_sunburst_with_colors(comp_sunburst_df, direction="from", color_map=shared_color_map)
                            
                            # Display shared legend in expander
                            with st.expander("Event legend", expanded=False,icon=":material/palette:"):
                                legend_columns = st.columns(4)  
                                col_idx = 0
                                for event, color in shared_color_map.items():
                                    with legend_columns[col_idx]:
                                        st.markdown(f"<span style='color:{color};font-weight:bold;font-size:14px'>●</span> {event}", unsafe_allow_html=True)
                                    
                                    col_idx = (col_idx + 1) % 4
                        else:
                            st.info("No path data available for sunburst visualization", icon=":material/info:")
                    
                    except Exception as e:
                        st.error(f"Error generating sunburst charts: {e}", icon=":material/chat_error:")
                                        
                    # Robust cleanup of temporary tables
                    temp_tables = [unique_reftable_name, unique_comptable_name, unique_comptftable_name, unique_reftftable_name]
                    for table_name in temp_tables:
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
                        except Exception as e:
                            st.warning(f"Could not drop table {table_name}: {str(e)}")
                            pass
        
                else:
                    st.write("")
                    
            # Separate block for PATH BETWEEN 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any' and fromevt1.strip("'") != 'Any' and toevt1.strip("'") != 'Any':
                st.warning("Not a valid pattern for comparison",icon=":material/warning:")
    
            elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any' and fromevt1.strip("'") == 'Any' and toevt1.strip("'") == 'Any':
                st.warning("This is tuple generator - Not a valid pattern for comparison",icon=":material/warning:")
                
            else:
                st.write("Please select appropriate options for 'from' and 'to'")
        else:
            st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200; margin-top: 0px; margin-bottom: -15px;">
                    Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)
                </h5>
            </div>
            """, unsafe_allow_html=True)
            #st.warning("Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)")
