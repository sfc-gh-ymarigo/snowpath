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

#st.set_page_config(layout="wide")

st.markdown("""
        <style>
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
#ASSOCIATION ANALYSIS
#--------------------------------------

@st.cache_data(show_spinner=False)
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
            ai_enabled = st.toggle("ExpAIn Me!", key=f"{key_prefix}_toggle", help="Generate AI insights and recommendations for your association analysis results. **Auto**: Pre-built analysis with structured insights. **Custom**: Enter your own questions and prompts for personalized analysis.")
        
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
def rgba_to_str(rgba):
    return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"
        
def sigma_graph(df, metric, height=900):
    # Step 1: Extract event pairs and calculate weights
    event_pairs = []
    event_counts = Counter()

    for _, row in df.iterrows():
        antecedent = row["ANTECEDENT"]
        consequent = row["CONSEQUENT"]
        cnt1 = row["CNT1"]
        cntb = row["CNTB"]
        event_pairs.append((antecedent, consequent, cntb))
        event_counts[antecedent] += cnt1
        event_counts[consequent] += cnt1

    pair_counts = Counter({(pair[0], pair[1]): pair[2] for pair in event_pairs})
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

    # Step 3: Create edges with color gradient and normalized thickness
    # **Normalize color scaling based on selected metric (heat-map style: yellow → orange → red)**
    min_color_value = df[metric].min()
    max_color_value = df[metric].max()
    norm = Normalize(vmin=min_color_value, vmax=max_color_value)  # Dynamically scale colors
    
    # **Normalize thickness: always range from 1 to 4 regardless of actual values**
    min_cntb = df['CNTB'].min()
    max_cntb = df['CNTB'].max()
    min_width = 1.0
    max_width = 4.0
        
    edges = [
    {
        "source": src,
        "target": tgt,
        "lineStyle": {
            # Normalize thickness to range [1, 3]
            "width": min_width + (cntb - min_cntb) / (max_cntb - min_cntb) * (max_width - min_width) if max_cntb > min_cntb else 2.0,
            # Heat-map color: YlOrRd (Yellow-Orange-Red)
            "color": rgba_to_str(plt.cm.YlOrRd(norm(row[metric]))),
            "opacity": 0.85
        },
        "tooltip": {
            "formatter": f"{src} → {tgt}<br>Association Count: {cntb}<br>{metric}: {row[metric]:.4f}"
        }
    }
    for _, row in df.iterrows()  # Ensure row is a Pandas Series, not a tuple
    for src, tgt, cntb in [(row["ANTECEDENT"], row["CONSEQUENT"], row["CNTB"])]
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
                        "color": "#ffffff",  # White label on hover
                        "fontWeight": "bold",
                    }
                },
            }
        ],
    }

    # Step 5: Render the ECharts graph in Streamlit
    st_echarts(options=options, height=f"{height}px")
    

    
st.sidebar.markdown("")
#Page Title

st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
        ASSOCIATION ANALYSIS
    </h5>
</div>
""", unsafe_allow_html=True)
# Get the current credentials
session = get_active_session()
#Initialize variables
uid = None
evt = None
tbl = None
partitionby = None
startdt_input = None
enddt_input = None
sess = None
excl3 = "''"
cols=''
colsdf=pd.DataFrame()
with st.expander("Input parameters", icon=":material/settings:"):
        
        # DATA SOURCE 
        st.markdown("""
    <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
    <hr style='margin-top: -8px;margin-bottom: 5px;'>
    """, unsafe_allow_html=True)
        
            # Get list of databases
        sqldb = "SHOW DATABASES"
        databases = session.sql(sqldb).collect()
        db0 = pd.DataFrame(databases)
            
        col1, col2, col3 = st.columns(3)
            
            # **Database Selection**
        with col1:
            database = st.selectbox('Select Database', key='assodb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
        if database:
            sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
            schemas = session.sql(sqlschemas).collect()
            schema0 = pd.DataFrame(schemas)
            
            with col2:
                schema = st.selectbox('Select Schema', key='assosch', index=None, 
                                          placeholder="Choose from list...", options=schema0['name'].unique())
        else:
            schema = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected)**
        if database and schema:
            sqltables = f"""
                    SELECT TABLE_NAME 
                    FROM {database}.INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                """
            tables = session.sql(sqltables).collect()
            table0 = pd.DataFrame(tables)
            
            with col3:
                tbl = st.selectbox('Select Event Table or View', key='assotbl', index=None, 
                                       placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
        else:
            tbl = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected)**
        if database and schema and tbl:
            cols = f"""
                    SELECT COLUMN_NAME
                    FROM {database}.INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{tbl}'
                    ORDER BY ORDINAL_POSITION;
                """

            colssql = session.sql(cols).collect()
            colsdf = pd.DataFrame(colssql)
            
            col3, col4, col5 = st.columns(3)
            with col3:
                uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...",help="The identifier column is the unique identifier for each record. This may be a user id, a customer id, a process id, a basket id, a sale id, etc")
            with col4: 
                item = st.selectbox('Select item column', colsdf, index=None, placeholder="Choose from list...", help="The item column : a product, an event, a channel, a service or even a person,...)")
            
            # FILTERS SECTION
            if (uid != None and item != None):
                # Get Distinct Items from Table for exclusion filter
                EOI = f"SELECT DISTINCT {item} FROM {database}.{schema}.{tbl} ORDER BY {item}"
                excl = session.sql(EOI).collect()
                excl0 = pd.DataFrame(excl)
                
                # Initialize sql_where_clause
                sql_where_clause = ""
                
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 10px;'>""", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Show only specific items
                        show_only_items_input = st.multiselect(
                            'Show only associations for specific item(s) - optional', 
                            excl0, 
                            placeholder="Select item(s)...", 
                            help="Analyze only associations involving these specific item(s). This will significantly speed up the query by filtering the data before computing associations. Leave empty to analyze all items."
                        )
                        
                    with col2:
                        # Exclude Items
                        excl1 = st.multiselect('Exclude item(s) - optional', excl0, placeholder="Select item(s)...", help="Item(s) to be excluded from the association analysis and the output.")
                        
                    # Build SQL filter clauses for items (both are optional)
                    # Exclude filter: if empty, use '' which means "exclude nothing"
                    if not excl1:
                        excl3 = "''"  # NOT IN ('') - effectively no exclusion
                    else:
                        excl3 = ', '.join([f"'{excl2}'" for excl2 in excl1])
                    
                    # Show only filter: if empty, no filter is applied (show all items)
                    if show_only_items_input:
                        show_only_sql = ' AND ' + item + ' IN (' + ', '.join([f"'{i}'" for i in show_only_items_input]) + ')'
                    else:
                        show_only_sql = ''  # Empty string - no filter applied
            
                # ADDITIONAL FILTERS
                # Check if there are any filterable columns (all columns except item)
                filterable_columns = colsdf['COLUMN_NAME']
            
                if not filterable_columns.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        checkfilters = st.toggle("Additional filters", help="Apply conditional filters to any available columns. Item filtering is handled in the dedicated section above.")
                    with col2:
                        st.write("")
                else:
                    st.markdown("""
                    <div class="custom-container-1">
                        <h5 style="font-size: 14px; font-weight: 200 ; color: #0f0f0f; margin-top: 0px; margin-bottom: -15px;">
                            No additional columns available for filtering. Item filtering is handled in the dedicated section above.
                        </h5>
                    </div>
                    """, unsafe_allow_html=True)
                    checkfilters = False
                
                if checkfilters:
                
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
                                distinct_values = fetch_distinct_values(col_name)
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
                        filter_index = 0  # A unique index to assign unique keys to each filter
                    
                        # Dynamic filter loop with one selectbox at a time
                        while True:
                            # Get all available columns for filtering
                            available_columns = filterable_columns
                            
                            if available_columns.empty:
                                st.write("No columns available for filtering.")
                                break
                    
                            # Create 3 columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])  # Adjust width ratios for better layout
                    
                            with col1:
                                # Select a column to filter on
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns, key=f"asso_column_{filter_index}")
                    
                            # Determine column data type by querying the INFORMATION_SCHEMA
                            column_info_query = f"""
                                SELECT DATA_TYPE
                                FROM INFORMATION_SCHEMA.COLUMNS
                                WHERE TABLE_NAME = '{tbl}' AND COLUMN_NAME = '{selected_column}';
                            """
                            column_info = session.sql(column_info_query).collect()
                            col_data_type = column_info[0]['DATA_TYPE']  # Get the data type of the selected column
                            
                            with col2:
                                # Display operator selection based on column data type
                                operator = get_operator_input(selected_column, col_data_type, filter_index)
                        
                            with col3:
                                # Display value input based on column data type and selected operator
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
                        
                            # Ask user if they want to add another filter
                            add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"asso_add_filter_{filter_index}")
                        
                            if add_filter == 'Yes':
                                # If another filter is to be added, ask for AND/OR logic
                                col1, col2 = st.columns([2, 13])
                                with col1: 
                                    logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"asso_logic_operator_{filter_index}")
                                    filter_index += 1  # Increment the index for the next filter
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

if (uid != None and item != None):
    # Display active filters summary
    active_filters = []
    if show_only_items_input:
        active_filters.append(f"Show associations involving: {', '.join(show_only_items_input)}")
    if excl1:
        active_filters.append(f"Exclude from base data: {', '.join(excl1)}")
    

    
    # Show toggle first, before executing expensive query
    show_details = st.toggle("Show me!", help='Toggle ON to run the analysis and choose a visualization: Detailed Table, Heatmap, or Forced Layout Graph')

    if not show_details:
        st.markdown("""
        <div class="custom-container-1">
            <h5 style="font-size: 14px; font-weight: 200 ; color: #0f0f0f; margin-top: 0px; margin-bottom: -15px;">
                Toggle 'Show me!' above to run the association analysis. You can adjust filters in the Input Parameters section before running.
            </h5>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Build WHERE clause for filtering final results (not base data)
    # This allows us to see associations INVOLVING the selected items
    if show_only_items_input:
        show_only_where = ' WHERE (ANTECEDENT IN (' + ', '.join([f"'{i}'" for i in show_only_items_input]) + ') OR CONSEQUENT IN (' + ', '.join([f"'{i}'" for i in show_only_items_input]) + '))'
    else:
        show_only_where = ''
    
    # SQL query only runs when toggle is ON
    associationsql=f"""WITH
-- Base filtered data (scan table only once)
base_data AS (
    SELECT DISTINCT {item}, {uid}
    FROM {database}.{schema}.{tbl}
    WHERE {item} NOT IN ({excl3}){sql_where_clause}
),
-- Item frequencies (used as both CNT1 and CNT2)
item_counts AS (
    SELECT COUNT(*) AS item_cnt, {item}
    FROM base_data
    GROUP BY {item}
),
-- Item pair co-occurrences (reuse base_data)
CNTB AS (
    SELECT A.{item} AS item1, B.{item} AS item2, COUNT(DISTINCT A.{uid}) AS cntb
    FROM base_data A
    JOIN base_data B
        ON A.{uid} = B.{uid}
        AND A.{item} > B.{item}
    GROUP BY A.{item}, B.{item}
),
-- Global statistics (from base_data)
NUMPART AS (
    SELECT COUNT(DISTINCT {uid}) AS N
    FROM base_data
),
TOTAL_EVENTS AS (
    SELECT COUNT(*) AS total_events
    FROM base_data
),
ASSOCIATION_RULES AS (
    SELECT
        CNTB.item1 as ANTECEDENT,
        CNTB.item2 as CONSEQUENT,
        CNTB.cntb as CNTB,
        IC1.item_cnt as CNT1,
        IC2.item_cnt as CNT2,
        N,
        total_events,
        ((CNTB.cntb * CNTB.cntb) / (IC1.item_cnt * IC2.item_cnt))::decimal(17,4) AS SCORE,
        100*(CNTB.cntb / IC1.item_cnt)::decimal(17,4) AS CONFIDENCE,
        100*(CNTB.cntb / N)::decimal(17,4) AS SUPPORT,
        CASE WHEN (1 - (CNTB.cntb / IC1.item_cnt)) = 0 THEN NULL ELSE (1 - (IC2.item_cnt / total_events)) / NULLIFZERO(1 - (CNTB.cntb / IC1.item_cnt)) END AS CONVICTION,
        (CNTB.cntb / IC1.item_cnt) / NULLIFZERO(IC2.item_cnt / N) AS LIFT
    FROM
        CNTB
    JOIN item_counts IC1 ON CNTB.item1 = IC1.{item}
    JOIN item_counts IC2 ON CNTB.item2 = IC2.{item}
    CROSS JOIN NUMPART
    CROSS JOIN TOTAL_EVENTS
)
SELECT
    ANTECEDENT,
    CONSEQUENT,
    CNTB,
    CNT1,
    CNT2,
    SCORE,
    CONFIDENCE,
    SUPPORT,
    CONVICTION,
    LIFT,
    -- Statistical Z-Score: Measures if co-occurrence is significantly different from random chance
    -- Expected = (CNT1 × CNT2) / N (expected co-occurrence under independence)
    -- Standard_Error = sqrt(Expected × (1 - CNT1/N) × (1 - CNT2/N))
    -- Z = (observed - expected) / standard_error
    CASE 
        WHEN (CNT1::float * CNT2::float / N::float) > 0 THEN
            (CNTB - (CNT1::float * CNT2::float / N::float)) / NULLIFZERO(
                SQRT((CNT1::float * CNT2::float / N::float) * (1.0 - CNT1::float / N::float) * (1.0 - CNT2::float / N::float))
            )
        ELSE NULL
    END AS Z_SCORE
FROM ASSOCIATION_RULES{show_only_where};
            """ 
    
    # Cache the SQL query execution to prevent re-running on every parameter change
    # Note: Cache key includes query SQL to ensure cache invalidation when SQL changes
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_association_query(query_sql, cache_version="v3"):
        """Execute association query and cache results for 1 hour"""
        return session.sql(query_sql).collect()
    
    with st.spinner("Computing association rules... This may take a few moments."):
        association = run_association_query(associationsql, cache_version="v3")

        #export output as a dataframe
        dfasso = pd.DataFrame(association)
        dfnetwork = pd.DataFrame(association)
    
    # Check if DataFrame is empty (outside spinner for proper rendering)
    if dfasso.empty:
        st.info("No association rules found with the current filters. Try adjusting your filter criteria.", icon=":material/warning:")
        st.stop()
    
    # Verify expected columns exist
    expected_cols = ['ANTECEDENT', 'CONSEQUENT', 'CNTB', 'CNT1', 'CNT2', 'SCORE', 'CONFIDENCE', 'SUPPORT', 'CONVICTION', 'LIFT', 'Z_SCORE']
    missing_cols = [col for col in expected_cols if col not in dfasso.columns]
    if missing_cols:
        st.error(f"Missing expected columns: {missing_cols}")
        st.write("Available columns:", list(dfasso.columns))
        st.stop()
    
    with st.expander("Metrics & Filters"):
        st.caption("The association analysis produces association rules (if-then statements) and measures of frequency, relationship, and statistical significance associated with these rules:")
        st.caption("**Antecedent**: represents the 'if' part of the rule, which is an item in the dataset.")
        st.caption("**Consequent**: represents the 'then' part of the rule, which is another item in the dataset.")
        st.caption("**Cntb**: represents the total co-occurences of both antecedent and consequent items.")
        st.caption("**Cnt1**: represents the total occurrences of antecedent item")
        st.caption("**Cnt2**: represents the total occurences of consequent item")
        st.caption("**Score**: represents a calculated value that indicates the strength of association between two items (antecedent and consequent). It helps measure the strength of the association between two items based on how often they co-occur compared to their individual occurrences. ")
        st.caption("**Confidence** (expressed as a percentage): represents the probability that consequent occurs given that antecedent has occurred. A high confidence indicates that when the antecedent is present, the consequent is very likely to also be present.")
        st.caption("**Support** (expressed as a percentage): represents the proportion of transactions that contain both the antecedent and consequent items.")
        st.caption("**Conviction** (expressed as a ratio): measures how strongly the absence of the antecedent implies the absence of the consequent.")
        st.caption("**Lift** (expressed as a ratio): measures how much more likely the consequent is to occur when the antecedent is present compared to if they were independent.")
        st.caption("**Z_Score** (expressed in standard deviations): measures how statistically significant the observed co-occurrence is compared to what would be expected by chance. A higher absolute value of the Z_Score indicates a stronger and more statistically significant association between the items in the rule.")
        
        st.markdown("---")
        
        # Metric Filters section
        st.markdown("**Metric Filters**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            # Dynamically adjust slider range based on actual values in the DataFrame
            min_score = st.number_input('Minimum Score', 
            min_value=float(dfasso['SCORE'].min()), 
            max_value=float(dfasso['SCORE'].max()), 
            value=float(dfasso['SCORE'].min()))
   
        with col2:
            min_confidence = st.number_input('Minimum Confidence', 
            min_value=float(dfasso['CONFIDENCE'].min()), 
            max_value=float(dfasso['CONFIDENCE'].max()), 
            value=float(dfasso['CONFIDENCE'].min()))
        
        with col3:
            min_support = st.number_input('Minimum Support', 
            min_value=float(dfasso['SUPPORT'].min()), 
            max_value=float(dfasso['SUPPORT'].max()), 
            value=float(dfasso['SUPPORT'].min()))
            
        with col4:
            # Handle case where Conviction might be all NULL
            conv_min = dfasso['CONVICTION'].min()
            conv_max = dfasso['CONVICTION'].max()
            if pd.isna(conv_min) or pd.isna(conv_max):
                min_conviction = st.number_input('Minimum Conviction', value=0.0, help="Conviction is NULL for all rows")
            else:
                min_conviction = st.number_input('Minimum Conviction', 
                min_value=float(conv_min), 
                max_value=float(conv_max), 
                value=float(conv_min))
        with col5:
            min_lift = st.number_input('Minimum Lift', 
            min_value=float(dfasso['LIFT'].min()), 
            max_value=float(dfasso['LIFT'].max()), 
            value=float(dfasso['LIFT'].min()))
        with col6:
            # Handle case where Z_Score might be all NULL
            z_min = dfasso['Z_SCORE'].min()
            z_max = dfasso['Z_SCORE'].max()
            if pd.isna(z_min) or pd.isna(z_max):
                min_zscore = st.number_input('Minimum Z_Score', value=0.0, help="Z_Score is NULL for all rows (likely due to very frequent items)")
            else:
                min_zscore = st.number_input('Minimum Z_Score', 
                min_value=float(z_min), 
                max_value=float(z_max), 
                value=float(z_min))
     
        # Apply filters to the DataFrame based on the minimum values
        filtered_df = dfasso[
                (dfasso['SCORE'] >= min_score) &
                (dfasso['CONFIDENCE'] >= min_confidence) &
                (dfasso['SUPPORT'] >= min_support) &
                ((dfasso['CONVICTION'] >= min_conviction) | (dfasso['CONVICTION'].isna())) &  # Include NULL conviction
                (dfasso['LIFT'] >= min_lift)&
                ((dfasso['Z_SCORE'] >= min_zscore) | (dfasso['Z_SCORE'].isna()))  # Include NULL Z_Scores
            ]
    
    # Apply filtered data outside the expander
    
    dfasso = filtered_df.copy()
    
    if not dfasso.empty:
        # Visualization selection
        genre = st.pills(
            "Choose a visualization:",
            ["Detailed Table", "Heatmap", "Graph"],
            selection_mode="single",
            default="Detailed Table"
        )
        
        # Ensure genre has a value (fallback to Detailed Table if None)
        if genre is None:
            genre = "Detailed Table"
        
        # Display the selected visualization
        if genre == 'Detailed Table':
            with st.container(border=True):
                with st.spinner("Loading detailed table..."):
                    st.dataframe(filtered_df, use_container_width=True)
        elif genre == 'Heatmap':
            with st.container(border=True):
                # Allow the user to select the metric to display in the heatmap
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    metric = st.selectbox('Choose a metric', dfasso.columns[5:])
                    st.write("")
                
                with st.spinner("Generating heatmap visualization..."):
        
                    dfasso[metric] = pd.to_numeric(dfasso[metric], errors='coerce')
                    # Create a heatmap
                    heatmap_data = dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric)
                    n_rows = heatmap_data.shape[0]  # Number of unique ANTECEDENT values (rows)
                    n_cols = heatmap_data.shape[1]  # Number of unique CONSEQUENT values (columns)
                    font_size = max(5, 20 - max(n_rows, n_cols))
                    plt.figure(figsize=(14, 8))
                    sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".2f", annot_kws={"fontsize": font_size}, cbar=False)
        
                    # Set the title and labels
                    plt.xlabel("CONSEQUENT", fontsize=12)
                    plt.ylabel("ANTECEDENT", fontsize=12)
        
                    # Show the plot
                    #st.pyplot(plt)
                
                    # Create pivot table for heatmap data
                    # Step 1: Create heatmap data and round values to 2 decimal places
                    all_labels = sorted(set(dfasso['ANTECEDENT']).union(set(dfasso['CONSEQUENT'])))
                    heatmap_data = (dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric).reindex(index=all_labels, columns=all_labels).round(2))
                    #heatmap_data = dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric).fillna(0).round(2)
                    #st.write(heatmap_data)
                    #st.write("Unique ANTECEDENTs:", dfasso['ANTECEDENT'].nunique())
                    #st.write("Unique CONSEQUENTs:", dfasso['CONSEQUENT'].nunique())
                    #st.write("Missing CONSEQUENTs in rows:", set(dfasso['CONSEQUENT'].unique()) - set(dfasso['ANTECEDENT'].unique()))
                    # Step 2: Remove diagonal values by setting them to None
                     # Step 2: Remove diagonal values by setting them to None
                    for idx in heatmap_data.index:
                        if idx in heatmap_data.columns:
                            heatmap_data.at[idx, idx] = None
                    # Step 3: Convert to ECharts-compatible list format (skip None values)
                    #heatmap_list = [
                    #    [i, j, float(heatmap_data.iloc[i, j])]  # Use float to avoid JSON errors
                    #    for i in range(heatmap_data.shape[0])
                    #    for j in range(heatmap_data.shape[1])
                    #    if pd.notna(heatmap_data.iloc[i, j])
                    #]
                    heatmap_list = [
                        [i, j, float(heatmap_data.iloc[i, j])]
                        for i in range(len(all_labels))
                        for j in range(len(all_labels))
                        if pd.notna(heatmap_data.iloc[i, j])]
                    # Step 4: Define options for ECharts heatmap
                    options = {
                        "tooltip": {"trigger": "item"},
                        "grid": {
                            "height": "80%",
                            "top": "10%",
                            "left": "10%",
                            "right": "10%",
                            "containLabel": True
                        },
                        "xAxis": {
                            "type": "category",
                            #"data": heatmap_data.columns.tolist(),
                            "data": all_labels,
                            "splitArea": {"show": True},
                            "axisLabel": {
                                "interval": 0,
                                "rotate": 45
                            }
                        },
                        "yAxis": {
                            "type": "category",
                            "data": all_labels,
                            #"data": heatmap_data.index.tolist(),
                            "splitArea": {"show": True}
                        },
                        "visualMap": {
                            "min": heatmap_data.min().min(skipna=True),
                            "max": heatmap_data.max().max(skipna=True),
                            "calculable": True,
                            "orient": "horizontal",
                            "left": "center",
                            "top": "0%",
                            "inRange": {
                                "color": ["#ffffff", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
                            },
                            "outOfRange": {
                                "color": "#ffffff"  # White background for 0 or None
                            }
                        },
                        "series": [{
                            "type": "heatmap",
                            "data": heatmap_list,
                            "label": {
                                "show": True,
                            },
                            "emphasis": {
                                "itemStyle": {
                                    "shadowBlur": 10,
                                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                                }
                            }
                        }]
                    }
                    # Step 5: Render in Streamlit
                    st_echarts(
                        options=options,
                        height="700px",
                        key=f"heatmap_{metric}"
                    )
        elif genre == 'Graph':
            with st.container(border=True):
                # Select a metric for edge color gradient and filter threshold
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    metric = st.selectbox('Choose a metric', dfasso.columns[5:])
                with col2:
                    # Add slider to filter by top % of selected metric
                    filter_percentage = st.slider("Display Top %", 1, 100, 100, 
                                                 help=f"Show top X% of associations based on {metric if 'metric' in locals() else 'selected metric'}")
                with col3:
                    # Add chart height control
                    chart_height = st.number_input("Chart Height (px)", min_value=400, max_value=2000, value=900, step=50,
                                                  help="Adjust the height of the network graph")
                with col4:
                    st.write("")
                with col5:
                    st.write("")
                
                with st.spinner("Generating graph visualization..."):
                    dfasso[metric] = dfasso[metric].astype(float)
                    
                    # Filter dataframe based on slider
                    if filter_percentage < 100:
                        # Sort by selected metric and get top X%
                        sorted_df = dfasso.sort_values(by=metric, ascending=False)
                        num_rows = int(len(sorted_df) * (filter_percentage / 100))
                        num_rows = max(1, num_rows)  # Ensure at least 1 row
                        filtered_dfasso = sorted_df.head(num_rows)
                        
                        st.caption(f"Showing top {filter_percentage}% ({num_rows:,} out of {len(dfasso):,} associations) based on {metric}")
                    else:
                        filtered_dfasso = dfasso
                    
                    # Visualize the graph with filtered data
                    sigma_graph(filtered_dfasso, metric, chart_height)
    
    # AI-Powered Insights with model selection (only show if toggle is on)
    if show_details:
        def association_ai_analysis_callback(selected_model, prompt_type):
            """Callback function for association analysis AI insights"""
            
            # Show custom prompt input if Custom is selected
            if prompt_type == "Custom":
                custom_prompt = st.text_area(
                    "Enter your custom prompt:",
                    value="",
                    key="association_custom_prompt",
                    help="Enter your custom analysis prompt. The association rules data will be automatically included.",
                    placeholder="Type your custom prompt here and press Ctrl+Enter or Cmd+Enter to submit..."
                )
                
                # Only proceed if custom prompt is not empty
                if not custom_prompt or custom_prompt.strip() == "":
                    st.markdown("""
                    <div class="custom-container-1">
                        <h5 style="font-size: 14px; font-weight: 200 ; color: #0f0f0f; margin-top: 0px; margin-bottom: -15px;">
                            Please enter your custom prompt above to generate AI insights.
                        </h5>
                    </div>
                    """, unsafe_allow_html=True)
                    return
            
            
            try:
                with st.spinner(f"Generating AI insights using {selected_model}..."):
                    # Use top association rules (limit to 50 for better performance)
                    top_rules = filtered_df.head(50)
                    
                    # Transform rules into readable text format
                    def rule_row_to_text(row):
                        """Transforms a row of association rule data into human-readable text."""
                        z_score_str = f"{row['Z_SCORE']:.2f}" if pd.notna(row['Z_SCORE']) else "N/A"
                        conviction_str = f"{row['CONVICTION']:.2f}" if pd.notna(row['CONVICTION']) else "N/A"
                        return (
                            f"• {row['ANTECEDENT']} → {row['CONSEQUENT']}: "
                            f"Support={row['SUPPORT']:.2f}%, Confidence={row['CONFIDENCE']:.2f}%, "
                            f"Lift={row['LIFT']:.2f}, Z-Score={z_score_str}, Conviction={conviction_str}"
                        )
                    
                    rules_text = "\n".join([rule_row_to_text(row) for _, row in top_rules.iterrows()])
                    
                    # Get summary statistics
                    total_rules = len(filtered_df)
                    avg_support = filtered_df['SUPPORT'].mean()
                    avg_confidence = filtered_df['CONFIDENCE'].mean()
                    avg_lift = filtered_df['LIFT'].mean()
                    unique_antecedents = filtered_df['ANTECEDENT'].nunique()
                    unique_consequents = filtered_df['CONSEQUENT'].nunique()
                    
                    if prompt_type == "Auto":
                        ai_prompt = f"""
Analyze these top 50 association rules from a dataset containing {total_rules} total rules:

{rules_text}

Dataset Summary:
- Total Association Rules: {total_rules}
- Unique Antecedents (Items): {unique_antecedents}
- Unique Consequents (Items): {unique_consequents}
- Average Support: {avg_support:.2f}%
- Average Confidence: {avg_confidence:.2f}%
- Average Lift: {avg_lift:.2f}

Association Rules Metrics Explanation:
- **Support**: Frequency of item co-occurrence (how often items appear together)
- **Confidence**: Probability of Consequent given Antecedent (reliability of the rule)
- **Lift**: Strength of association compared to random chance (>1 means positive correlation)
- **Z-Score**: Statistical significance of the association (higher absolute values = more significant)
- **Conviction**: How much more often Antecedent occurs without Consequent than expected (if rule was independent)

Please provide comprehensive insights on:

1. **Strongest Associations**: Identify the most significant patterns based on high Lift, Z-Score, and Confidence. What do these strong associations reveal about item relationships?

2. **Business Implications**: What do these associations suggest about:
   - Cross-selling opportunities (items frequently purchased together)
   - Sequential patterns (if items represent user actions/events)
   - Behavioral insights (if items represent user activities)
   - Product/service bundling strategies

3. **Statistical Significance**: Highlight rules with high Z-Scores that indicate statistically significant associations beyond random chance.

4. **Actionable Recommendations**: Based on the identified patterns:
   - What specific actions could be taken to leverage these associations?
   - Are there any surprising or counter-intuitive patterns that warrant investigation?
   - Which associations should be prioritized for strategic initiatives?

5. **Pattern Categories**: Group similar rules into categories (e.g., complementary products, sequential behaviors, substitute products) if patterns emerge.

Keep your analysis structured, concise, and actionable. Focus on insights that can drive decision-making.
                        """
                    else:  # Custom
                        ai_prompt = f"""
{custom_prompt}

Data to analyze - Top 50 association rules from {total_rules} total rules:

{rules_text}

Dataset Summary:
- Total Association Rules: {total_rules}
- Unique Items (Antecedents): {unique_antecedents}
- Unique Items (Consequents): {unique_consequents}
- Average Support: {avg_support:.2f}%
- Average Confidence: {avg_confidence:.2f}%
- Average Lift: {avg_lift:.2f}

Metrics Reference:
- Support: Frequency of co-occurrence
- Confidence: Reliability of the rule
- Lift: Strength vs random chance (>1 = positive correlation)
- Z-Score: Statistical significance
- Conviction: Independence measure
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
            "association_ai", 
            "Select the LLM model for AI analysis of association rules",
            ai_content_callback=association_ai_analysis_callback
        )
    if st.toggle("Export network data as Gefx", help="Export network data as a Gefx file for further exploration into Gephi Open Graph Platform."):
   
        # Create a network graph
        G = nx.Graph()
    
    # Add nodes
        #nodes = set(dfasso['GAME1']).union(set(dfasso['GAME2']))
        #G.add_nodes_from(nodes)
        for _, row in dfnetwork.iterrows():
            if not G.has_node(row['ANTECEDENT']):
                G.add_node(row['ANTECEDENT'], cnt=row['CNT1'])
        
            if not G.has_node(row['CONSEQUENT']):
                G.add_node(row['CONSEQUENT'], cnt=row['CNT2'])
       
        
    # Add edges with lift as edge weight
        for i, row in dfnetwork.iterrows():
            G.add_edge(row['ANTECEDENT'], row['CONSEQUENT'], cntb=row['CNTB'],cnt1=row['CNT1'],
            score=float(row['SCORE']),
            confidence=float(row['CONFIDENCE']),
            support=float(row['SUPPORT']),
            #conviction=float(row['CONVICTION']),
            lift=float(row['LIFT'] ))
    
    # Draw the graph
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)  # For consistent layout
        #plt.figure(figsize=(12, 12))
    
    # Draw edges with thickness representing lift
        edges = nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='black', width=[G[u][v]['cntb'] for u, v in G.edges()])
        
    # Node sizes based on the sum of lift values of connected edges
        #node_size = [0.3*sum([G[u][v]['cnt1'] for u, v in G.nodes(node)]) for node in G.nodes()]
    
    # Draw nodes
        #nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue")
        #nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")
    
    # Show the plot  
        #st.pyplot(plt)
    
    
    # Save the GEXF data to a BytesIO object
        from io import BytesIO
        output = BytesIO()
        nx.write_gexf(G, output)
        output.seek(0) 
    
    # Provide a download button in Streamlit
        st.download_button(
            label="Download Graph as GEXF",
            data=output,
            file_name="my_network_graph.gexf",
            mime="application/octet-stream")
else:
    st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 14px; font-weight: 200 ; color: #0f0f0f; margin-top: 0px; margin-bottom: -15px;">
            Please ensure all required inputs are selected before running the app.
        </h5>
    </div>
    """, unsafe_allow_html=True)
