# Import python packages
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime 
import uuid
import altair as alt
import re
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, when
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast
import pickle


# Defer all library availability checks until runtime
# This avoids any import conflicts during Streamlit page loading
SNOWPARK_ML_AVAILABLE = None  # Will be checked when needed
SKLEARN_AVAILABLE = None       # Will be checked when needed

def check_snowpark_ml():
    """Check if Snowpark ML is available without importing at module level"""
    try:
        import importlib
        importlib.import_module('snowflake.ml.modeling.naive_bayes')
        return True
    except ImportError:
        return False

def check_sklearn():
    """Check if scikit-learn is available without importing at module level"""
    try:
        import importlib
        importlib.import_module('sklearn.naive_bayes')
        return True
    except ImportError:
        return False

# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

# Define emergency cleanup function for error scenarios
def emergency_cleanup_tables():
    """Emergency cleanup function that runs when errors occur"""
    try:
        tables_to_cleanup = st.session_state.get('temp_tables_for_cleanup', [])
        if tables_to_cleanup:
            st.warning(f"🧹 Emergency cleanup: Removing {len(tables_to_cleanup)} intermediate tables...", icon=":material/cleaning_services:")
            cleanup_count = 0
            for table_name in tables_to_cleanup:
                try:
                    cleanup_sql = f"DROP TABLE IF EXISTS {database}.{schema}.{table_name}"
                    session.sql(cleanup_sql).collect()
                    cleanup_count += 1
                except:
                    pass  # Ignore individual cleanup errors
            
            # Also clean up any orphaned tables by pattern
            cleanup_patterns = ["RAWEVENTSREF_%", "RAWEVENTSCOMP_%", "TFIDFREF_%", "TFIDFCOMP_%", "FULLTFIDF_%", "TRAINING_%","%TEMP%"]
            for pattern in cleanup_patterns:
                try:
                    list_tables_sql = f"SHOW TABLES LIKE '{pattern}' IN {database}.{schema}"
                    result = session.sql(list_tables_sql).collect()
                    for row in result:
                        try:
                            cleanup_sql = f"DROP TABLE IF EXISTS {database}.{schema}.{row['name']}"
                            session.sql(cleanup_sql).collect()
                            cleanup_count += 1
                        except:
                            pass
                except:
                    pass
            
            st.success(f"Emergency cleanup completed: {cleanup_count} tables removed", icon=":material/check:")
            # Clear the cleanup list
            st.session_state['temp_tables_for_cleanup'] = []
        else:
            st.info("No intermediate tables to clean up", icon=":material/info:")
    except Exception as e:
        st.error(f"Emergency cleanup failed: {str(e)}", icon=":material/error:")

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

#st.set_page_config(layout="wide")

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

.custom-container-1 {
    background-color: #f0f2f6 !important;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.custom-container-1 h5 {
    color: #0f0f0f !important;
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

st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
        PREDICTIVE MODELING
    </h5>
</div>
""", unsafe_allow_html=True)
# Use the existing session (already created at line 60)
#--------------------------------------
#VAR INIT
#--------------------------------------
#Initialize variables
fromevt = None
toevt = None
minnbbevt = 0
maxnbbevt = 5
overlap = 'PAST LAST ROW'
uid = None
evt = None
tmstp = None
tbl = None
partitionby = None
groupby = None
startdt_input = None
enddt_input = None
excl3_instance = "''"
timeout = None
unitoftime= None
cols = ''
colsdf = pd.DataFrame()
def init_session_state():
    """Initialize session state variables"""
    if "model_created" not in st.session_state:
        st.session_state["model_created"] = False
    if "created_model_name" not in st.session_state:
        st.session_state["created_model_name"] = None
    if "test_model_created" not in st.session_state:
        st.session_state["test_model_created"] = False
    if "test_model_success" not in st.session_state:
        st.session_state["test_model_success"] = False
    if "unique_testtable_name" not in st.session_state:
        st.session_state["unique_testtable_name"] = None

# Initialize the app
init_session_state()

# Initialize variables that will be set in the expander
primary_label = None
complementary_label = None
model_name = None

with st.expander("Input Parameters (Primary Class)", icon=":material/settings:"):
         
         # DATA SOURCE 
         st.markdown("""
     <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
     <hr style='margin-top: -8px;margin-bottom: 5px;'>
     """, unsafe_allow_html=True)
         
         # Database, Schema, Table Selection (cached)
         db0 = fetch_databases(session)
         
         col1, col2, col3 = st.columns(3)
         
         # **Database Selection**
         with col1:
             # Handle different possible column names from SHOW DATABASES
             db_column = 'name' if 'name' in db0.columns else db0.columns[1] if len(db0.columns) > 1 else db0.columns[0]
             database = st.selectbox('Select Database', key='predictrefdb', index=None, 
                                     placeholder="Choose from list...", options=db0[db_column].unique())
         
         # **Schema Selection (Only if a database is selected)**
         if database:
             schema0 = fetch_schemas(session, database)
         
             with col2:
                 # Handle different possible column names from SHOW SCHEMAS
                 schema_column = 'name' if 'name' in schema0.columns else schema0.columns[1] if len(schema0.columns) > 1 else schema0.columns[0]
                 schema = st.selectbox('Select Schema', key='comparerefsch', index=None, 
                                       placeholder="Choose from list...", options=schema0[schema_column].unique())
         else:
             schema = None  # Prevents SQL execution
         
         # **Table Selection (Only if a database & schema are selected)**
         if database and schema:
             table0 = fetch_tables(session, database, schema)
         
             with col3:
                 tbl = st.selectbox('Select Event Table or View', key='comparereftbl', index=None, 
                                    placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                    help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
         else:
             tbl = None  # Prevents SQL execution
         
         # **Column Selection (Only if a database, schema, and table are selected)**
         if database and schema and tbl:
             colsdf = fetch_columns(session, database, schema, tbl)

         col1, col2, col3 = st.columns([4,4,4])
         with col1:
             uid = st.selectbox('Select identifier column', colsdf, index=None,  key='uidpredictref',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
         with col2: 
             evt = st.selectbox('Select event column', colsdf, index=None, key='evtpredictref', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
         with col3:
             tmstp = st.selectbox('Select timestamp column', colsdf, index=None, key='tsmtppredictref',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
         
        # Clear previous results if key parameters have changed
         current_params = {'database': database, 'schema': schema, 'table': tbl, 'uid': uid, 'evt': evt, 'tmstp': tmstp}
         if 'previous_params' not in st.session_state:
            st.session_state['previous_params'] = current_params
         elif st.session_state['previous_params'] != current_params:
            if 'predictive_results' in st.session_state:
                del st.session_state['predictive_results']
            st.session_state['previous_params'] = current_params
         
         #Get Distinct Events Of Interest from Event Table
         if (uid != None and evt != None and tmstp != None):
         # Get Distinct Events Of Interest from Event Table
             EOI = f"SELECT DISTINCT {evt} FROM {database}.{schema}.{tbl} ORDER BY {evt}"
                     # Get start EOI :
             start = session.sql(EOI).collect()
         # Get end EOI :
             end = session.sql(EOI).collect()
         # Get excluded EOI :
             excl =session.sql(EOI).collect()
     
             # Write query output in a pandas dataframe
             startdf0 = pd.DataFrame(start)
             enddf0 = pd.DataFrame(end)
             excl0 =pd.DataFrame(excl)   


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
         
     # Add 'Any' to preceding events but NOT to target event
             options_with_placeholder_from = ["Any"] + startdf1[evt].unique().tolist()
             options_with_placeholder_to = [e for e in enddf1[evt].unique().tolist() if e != "Any"]  # Filter out 'Any'
             col1, col2, col3, col4 = st.columns([4, 4, 2, 2])
             with col1:
                 frm = st.multiselect(
                     'Select preceding events:', 
                     options=options_with_placeholder_from,
                     key='evtfrmref',
                     default=["Any"],
                     help="Select one or more preceding events. 'Any' (default) matches all values."
                 )
                 if "Any" in frm:
                     fromevt = "Any"  
                 else:
                     fromevt = ", ".join([f"'{value}'" for value in frm])
             with col2:
                 to = st.selectbox(
                     'Select target event (primay class):',
                     options=options_with_placeholder_to,
                     index=None,
                     key='evttoref',
                     help="Select one target event of interest as the primary class of the model."
                 )
                 if to:
                     toevt = f"'{to}'"
                 else:
                     toevt = None
             
             with col3:
                 minnbbevt = st.number_input(
                     "Min # preceding events", 
                     value=0,  
                     key='minnbbevtref',
                     placeholder="Type a number...",
                     help="Select the minimum number of events preceding the event of interest."
                 )
             with col4:
                 maxnbbevt = st.number_input(
                     "Max # preceding events", 
                     value=5, 
                     min_value=1, 
                     key='maxnbbevtref',
                     placeholder="Type a number...",
                     help="Select the maximum number of events preceding the event of interest."
                 )
             
 
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
             
                         if col_data_type in ['NUMBER', 'FLOAT', 'INT']:
                             operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN'], key=operator_key)
                         elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                             operator = st.selectbox("Operator", ['<=', '>=', '='], key=operator_key)
                         else:  # For string or categorical columns
                             operator = st.selectbox("Operator", ['=', '!=', 'IN'], key=operator_key)
                         return operator
             
                     # Helper function to display value input based on column data type
                     def get_value_input(col_name, col_data_type, operator, filter_index):
                         """ Returns the value for filtering based on column type """
                         value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
             
                         if operator == 'IN':
                             distinct_values = fetch_distinct_values(col_name)
                             value = st.multiselect(f"Values for {col_name}", distinct_values, key=value_key)
                         elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                             value = st.date_input(f"Value for {col_name}", key=value_key)
                         else:
                             distinct_values = fetch_distinct_values(col_name)
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
             
                         # Determine column data type
                         column_info_query = f"""
                             SELECT DATA_TYPE
                             FROM INFORMATION_SCHEMA.COLUMNS
                             WHERE TABLE_NAME = '{tbl}' AND COLUMN_NAME = '{selected_column}';
                         """
                         column_info = session.sql(column_info_query).collect()
                         col_data_type = column_info[0]['DATA_TYPE']  # Get the data type
             
                         with col2:
                             operator = get_operator_input(selected_column, col_data_type, filter_index)
             
                         with col3:
                             value = get_value_input(selected_column, col_data_type, operator, filter_index)
             
                         # Append filter if valid
                         if operator and (value is not None or value == 0):
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
                     sql_where_clause = " AND "
                     #st.write(filters)
                     for i, (col, operator, value) in enumerate(filters):
                         if i > 0 and logic_operator:
                             sql_where_clause += f" {logic_operator} "
                         
                         if operator == 'IN':
                             # Handle IN operator where the value is a list
                             sql_where_clause += f"{col} IN {tuple(value)}"
                         else:
                             # Check if value is numeric
                                 if isinstance(value, (int, float)):
                                     sql_where_clause += f"{col} {operator} {value}"
                                 else:
                              # For non-numeric values (strings, dates), enclose the value in quotes
                                     sql_where_clause += f"{col} {operator} '{value}'"        
                     else:
                         st.write("")
             if all([uid, evt, tmstp,fromevt, toevt]):      
                     with st.container():
                         st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Model</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                     overlap= 'PAST LAST ROW'
                 
                 
                     col1, col2, col3 = st.columns(3)
                     with col1:
                                 primary_label = st.text_input(
                                 "Primary Class Name",
                                 placeholder='Please type the Primary Class Name',
                                 )     
                     with col2:
                                  complementary_label = st.text_input(
                                  "Complementary Class Name",
                                 placeholder='Please type the Complementary Class Name',
                                 )
                     with col3:
                                  model_name = st.text_input(
                                  "Model Name",
                                 placeholder='Please type the Model Name',
                                )
                     
                     # Build classifier options based on available libraries (check at runtime)
                     classifier_options = []
                     
                     # Check library availability only when UI is being built
                     snowpark_ml_available = check_snowpark_ml()
                     sklearn_available = check_sklearn()
                     
                     if snowpark_ml_available:
                        classifier_options.extend([
                            "Snowpark ML Naive Bayes (Bernoulli)", 
                            "Snowpark ML Naive Bayes (Multinomial)",
                            "Snowpark ML Random Forest"
                        ])
                    
                     if sklearn_available:
                        classifier_options.extend([
                            "Scikit-learn Naive Bayes (Bernoulli)", 
                            "Scikit-learn Naive Bayes (Multinomial)",
                            "Scikit-learn Random Forest"
                        ])
                    
                     # Classifier Selection
                     st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Classifier Algorithm</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                     
                     # Row 1: Create 3 columns - Classifier always, RF params if Random Forest
                     col_algo, col_param1, col_param2 = st.columns([1, 1, 1])
                     
                     with col_algo:
                         classifier_type = st.selectbox(
                             "Classification Algorithm",
                             options=classifier_options,
                             index=0,
                             help="Choose the classification algorithm. Bernoulli NB uses binary features (event presence/absence), Multinomial NB preserves TF-IDF values (event importance), Random Forest uses ensemble learning with multiple decision trees. Use appropriate compute resources for the selected algorithm. Random Forest is more compute intensive than Naive Bayes and Snowpark optimized warehouse may be considered for large datasets if using Snowpark ML."
                         )
                     
                     # Check if Random Forest is selected
                     is_random_forest = "Random Forest" in classifier_type
                     
                     if is_random_forest:
                         # Random Forest parameters in Col 2 and Col 3
                         with col_param1:
                             n_estimators = st.number_input(
                                 "N Estimators",
                                 min_value=10,
                                 max_value=1000,
                                 value=100,
                                 step=10,
                                 help="Number of trees in the forest. More trees = better accuracy but slower training."
                             )
                         
                         with col_param2:
                             max_depth = st.number_input(
                                 "Max Depth",
                                 min_value=1,
                                 max_value=50,
                                 value=10,
                                 step=1,
                                 help="Maximum depth of the decision trees. Controls model complexity and overfitting."
                             )
                     else:
                         # Col 2 and Col 3 empty for Naive Bayes
                         with col_param1:
                             st.empty()
                         with col_param2:
                             st.empty()
                         # Set defaults for Random Forest params (not used)
                         n_estimators = 100
                         max_depth = 10
                     
                     # Row 2: Sample Settings - always shown
                     st.markdown("""<h3 style='font-size: 13px; margin-top: 10px;'>Sample Settings</h3>""", unsafe_allow_html=True)
                     col_sample1, col_sample2, col_empty = st.columns([1, 1, 1])
                     
                     with col_sample1:
                         sample_percent = st.slider(
                             "Sample % of Predictions",
                             min_value=1,
                             max_value=100,
                             value=50,
                             help="Percentage of prediction data to sample for performance metrics evaluation. Samples this % of rows from the full prediction set. Lower values = faster processing but less representative metrics."
                         )
                     
                     with col_sample2:
                         max_sample_rows = st.number_input(
                             "Max Rows from Sample",
                             min_value=1000,
                             max_value=100000,
                             value=10000,
                             step=1000,
                             help="Maximum number of sampled rows to use for metrics computation. Caps the final evaluation dataset size after sampling to control processing time."
                        )
                     
                     with col_empty:
                         # Empty column
                         st.empty()
                     
                     # Cleanup section - Manual temp table cleanup
                     st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Cleanup</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                     st.write("")
                     cleanup_button = st.button(
                         "Cleanup Temp Tables", 
                         key='cleanup_temp_tables_button',
                         help="Cleanup temporary intermediate tables before creating a new model. Temporary intermediate tables may not be dropped in case of model creation failure or abortion."
                     )
                     
                     # Show cleanup status from session state
                     if 'cleanup_status' in st.session_state and st.session_state['cleanup_status']:
                         status = st.session_state['cleanup_status']
                         if status.get('success'):
                             st.success(status['message'], icon=":material/check:")
                         elif status.get('info'):
                             st.info(status['message'], icon=":material/info:")
                         elif status.get('warning'):
                             st.warning(status['message'], icon=":material/warning:")
                     
                     # Execute cleanup when button is clicked
                     if cleanup_button:
                         # Manual cleanup function
                         try:
                             cleanup_count = 0
                             temp_table_prefixes = ["RAWEVENTSREF_", "RAWEVENTSCOMP_", "TFIDFREF_", "TFIDFCOMP_", "FULLTFIDF_", "TRAINING_","%TEMP%"]
                             
                             # Clean up tables from session state if available
                             tables_to_cleanup = st.session_state.get('temp_tables_for_cleanup', [])
                             for table_name in tables_to_cleanup:
                                 try:
                                     cleanup_sql = f"DROP TABLE IF EXISTS {database}.{schema}.{table_name}"
                                     session.sql(cleanup_sql).collect()
                                     cleanup_count += 1
                                 except:
                                     pass
                             
                             # Clean up any orphaned tables by pattern
                             for pattern in temp_table_prefixes:
                                 try:
                                     list_tables_sql = f"SHOW TABLES LIKE '{pattern}%' IN {database}.{schema}"
                                     result = session.sql(list_tables_sql).collect()
                                     for row in result:
                                         try:
                                             cleanup_sql = f"DROP TABLE IF EXISTS {database}.{schema}.{row['name']}"
                                             session.sql(cleanup_sql).collect()
                                             cleanup_count += 1
                                         except:
                                             pass
                                 except:
                                     pass
                             
                             # Clear the cleanup list
                             st.session_state['temp_tables_for_cleanup'] = []
                             
                             # Store status in session state for display
                             if cleanup_count > 0:
                                 st.session_state['cleanup_status'] = {
                                     'success': True,
                                     'message': f"Cleanup completed: {cleanup_count} temporary tables removed"
                                 }
                             else:
                                 st.session_state['cleanup_status'] = {
                                     'info': True,
                                     'message': "No temporary tables found to clean up"
                                 }
                         except Exception as e:
                             st.session_state['cleanup_status'] = {
                                 'warning': True,
                                 'message': f"Cleanup completed with some issues: {str(e)}"
                             }
                         # Store a flag to indicate cleanup was just clicked
                         st.session_state['cleanup_just_clicked'] = True
            
                    # Continue with SQL generation and execution based on inputs...

# Determine if inputs are ready
inputs_ready = all([uid, evt, tmstp, fromevt, toevt, primary_label, complementary_label, model_name]) and fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any'

# Validation message
if not inputs_ready:
    st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 14px; font-weight: 200 ; margin-top: 0px; margin-bottom: -15px;">
            Please ensure all required input parameters are selected and configured correctly before creating the model.
        </h5>
    </div>
    """, unsafe_allow_html=True)

# Create Model button - only show when inputs are ready
run_model_button = False
if inputs_ready:
    run_model_button = st.button("Create Model", help="Click to train and evaluate the selected classification model")

# Check if cleanup was just clicked - if so, skip model creation this run to show cleanup status
cleanup_just_clicked = st.session_state.get('cleanup_just_clicked', False)
# Reset the flag after checking
if cleanup_just_clicked:
    st.session_state['cleanup_just_clicked'] = False

# Check if AI model selection is active to prevent model recreation
ai_model_selection_active = st.session_state.get("ai_model_selection_active", False)

if inputs_ready and run_model_button and not cleanup_just_clicked and not ai_model_selection_active:
    # Clear cleanup status when starting new model creation
    if 'cleanup_status' in st.session_state:
        del st.session_state['cleanup_status']
    
    tab1, tab2, tab3 = st.tabs(["Model Training", "Results", "Model Logging"])
    
    with tab1:
        with st.spinner("Creating model..."):
            # Clear any previous results when creating a new model
            if 'predictive_results' in st.session_state:
                del st.session_state['predictive_results']
            
            crttblrawseventsrefsql= None
            crttblrawseventsref = None
            crttblrawseventscompsql = None
            crttblrawseventscomp = None
                             
              # Generate unique table names
            def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                  unique_refid = uuid.uuid4().hex  # Generate a random UUID
                  return f"{base_name}_{unique_refid}"
     
            unique_reftable_name = generate_unique_reftable_name()
             
             # Store table names in session state for cleanup
            if 'temp_tables_for_cleanup' not in st.session_state:
                 st.session_state['temp_tables_for_cleanup'] = []
            st.session_state['temp_tables_for_cleanup'].append(unique_reftable_name)
             
             # CREATE TABLE individual reference Paths 
            if unitoftime==None and timeout ==None :
                  
                  crttblrawseventsrefsql = f"""CREATE TABLE {database}.{schema}.{unique_reftable_name} AS (
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
                  
                  crttblrawseventsrefsql = f"""CREATE TABLE {database}.{schema}.{unique_reftable_name} AS (
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
            st.session_state['temp_tables_for_cleanup'].append(unique_comptable_name)
              
              # CREATE TABLE individiual compared (complement set) Paths 
            if unitoftime==None and timeout ==None :
                  crttblrawseventscompsql = f"""CREATE TABLE {database}.{schema}.{unique_comptable_name} AS (
                  select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                  from  (select * from {database}.{schema}.{tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {database}.{schema}.{unique_reftable_name} ) AND
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
                  crttblrawseventscompsql = f"""CREATE TABLE {database}.{schema}.{unique_comptable_name} AS (
                  select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                  from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                  {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {database}.{schema}.{unique_reftable_name} ) AND
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
              #st.success(f"✅ Complementary Paths Table `{unique_comptable_name}` created.")
     
     
              # Generate a unique ref tfidf table name
            def generate_unique_reftftable_name(base_name="TFIDFREF"):
                  unique_refid = uuid.uuid4().hex  # Generate a random UUID
                  return f"{base_name}_{unique_refid}"
     
            unique_reftftable_name = generate_unique_reftftable_name()
            st.session_state['temp_tables_for_cleanup'].append(unique_reftftable_name)
             
             #CREATE TABLE TF-IDF Reference
            crttbltfidfrefsql=f"""CREATE TABLE {database}.{schema}.{unique_reftftable_name} AS
              (
                  Select
                  {uid},SEQ,INDEX,
                  count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                  LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                  (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                 VALUE AS EVENT
                  FROM
                  (select * 
                  from {database}.{schema}.{unique_reftable_name}, lateral strtok_split_to_table({database}.{schema}.{unique_reftable_name}.path, ',')
                  order by seq, index)
                  )"""
              
              #st.write(crttbltfidfrefsql)
            crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
              #st.success(f"✅ TF-IDF Reference Table `{unique_reftftable_name}` created.")
     
            #Generate a unique comp tfidf table name
            def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                  unique_refid = uuid.uuid4().hex  # Generate a random UUID
                  return f"{base_name}_{unique_refid}"
     
            unique_comptftable_name = generate_unique_comptftable_name()
            st.session_state['temp_tables_for_cleanup'].append(unique_comptftable_name)
             
             #CREATE TABLE TF-IDF Compared
            crttbltfidfcompsql=f"""CREATE TABLE {database}.{schema}.{unique_comptftable_name} AS
              (
                  Select
                  {uid},SEQ,INDEX,
                  count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                  LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                  (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                 VALUE AS EVENT
                  FROM
                  (select * 
                  from {database}.{schema}.{unique_comptable_name}, lateral strtok_split_to_table({database}.{schema}.{unique_comptable_name}.path, ',')
                  order by seq, index)
                  )"""
              
              #st.write(crttbltfidfcompsql)
            crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
              #st.success(f"✅ TF-IDF Complementary Table `{unique_comptftable_name}` created.")
     
            #Generate a unique full tfidf table name
            def generate_unique_fulltftable_name(base_name="FULLTFIDF"):
                  unique_refid = uuid.uuid4().hex  # Generate a random UUID
                  return f"{base_name}_{unique_refid}"
     
            unique_fulltftable_name = generate_unique_fulltftable_name()
            st.session_state['temp_tables_for_cleanup'].append(unique_fulltftable_name)
             
             #CREATE TABLE TF-IDF Full (Primary+Complementary)
            crttbltfidffullsql=f"""CREATE TABLE {database}.{schema}.{unique_fulltftable_name} AS
                 (
                  SELECT * FROM (
                  WITH 
                  COMP_TFIDF AS (
                      SELECT
                          {uid},
                          seq,
                          TFIDF,
                          EVENT,
                          '{complementary_label}' AS LABEL
                      FROM {database}.{schema}.{unique_comptftable_name}
                  ),
                  COMP_TFIDF_NORM AS (
                      SELECT
                          {uid},
                          seq,
                          EVENT,
                          TFIDF,
                          CASE 
                              WHEN TFIDF = 0 THEN 0
                              ELSE TFIDF / NULLIF(MAX(TFIDF) OVER (PARTITION BY EVENT), 0)
                          END AS TFIDF_NORMALIZED,
                          LABEL
                      FROM COMP_TFIDF
                  ),
                  REF_TFIDF AS (
                      SELECT
                          {uid},
                          seq,
                          TFIDF,
                          EVENT,
                          '{primary_label}' AS LABEL
                      FROM {database}.{schema}.{unique_reftftable_name}
                  ),
                  REF_TFIDF_NORM AS (
                      SELECT
                          {uid},
                          seq,
                          EVENT,
                          TFIDF,
                          CASE 
                              WHEN TFIDF = 0 THEN 0
                              ELSE TFIDF / NULLIF(MAX(TFIDF) OVER (PARTITION BY EVENT), 0)
                          END AS TFIDF_NORMALIZED,
                          LABEL
                      FROM REF_TFIDF
                  )
                  SELECT {uid}, SEQ, EVENT, SUM(TFIDF_NORMALIZED) as TFIDF, LABEL
                  FROM (
                      SELECT * FROM COMP_TFIDF_NORM
                      UNION ALL
                      SELECT * FROM REF_TFIDF_NORM
                  )
                  GROUP BY {uid}, SEQ, EVENT, LABEL)
                  PIVOT (
                      SUM(TFIDF)
                      FOR EVENT IN (ANY ORDER BY {uid}, SEQ)
                      DEFAULT ON NULL (0)
                  )
                  ORDER BY {uid}, SEQ)"""
         
              #st.write (crttbltfidffullsql)
            crttbltfidffull = session.sql(crttbltfidffullsql).collect()
              #st.success(f"✅ Full TF-IDF Table `{unique_fulltftable_name}` created.")
              # Generate a unique training table name
            def generate_unique_trainingtable_name(base_name="TRAINING"):
                  unique_refid = uuid.uuid4().hex  # Generate a random UUID
                  return f"{base_name}_{unique_refid}"
     
            unique_trainingtable_name = generate_unique_trainingtable_name()
            st.session_state['temp_tables_for_cleanup'].append(unique_trainingtable_name)
             
             #CREATE TRAINING TABLE
            crttbltrainingsql=f"""CREATE TABLE {database}.{schema}.{unique_trainingtable_name} AS (SELECT * exclude ({uid},seq), ABS(HASH({uid})) % 100 AS hash_value
             from {database}.{schema}.{unique_fulltftable_name} where hash_value <80)"""
            crttbltraining = session.sql(crttbltrainingsql).collect()
              #st.success(f"✅ Training Table `{unique_trainingtable_name}` created.")
              
              # VALIDATE TARGET COLUMN BEFORE MODEL CREATION
            label_validation_sql = f"""SELECT COUNT(DISTINCT LABEL) as unique_labels, 
                                        COUNT(*) as total_rows,
                                        ARRAY_AGG(DISTINCT LABEL) as label_values
                                        FROM {database}.{schema}.{unique_trainingtable_name}"""
            label_validation = session.sql(label_validation_sql).collect()
              
            unique_labels = label_validation[0]['UNIQUE_LABELS']
            total_rows = label_validation[0]['TOTAL_ROWS']
            label_values = label_validation[0]['LABEL_VALUES']
              
            if unique_labels < 2:
                st.error(f"Cannot create classification model: Target column 'LABEL' has only {unique_labels} unique value(s): {label_values}", icon=":material/chat_error:")
                st.warning(f"Solutions:\n• Check that both primary class ('{primary_label}') and complementary class ('{complementary_label}') have data\n• Verify your path patterns return results for both classes\n• Consider different event filters or date ranges\n• Review your data to ensure balanced classes", icon=":material/lightbulb:")
                st.info(f"Data Summary:\n• Total training rows: {total_rows:,}\n• Unique labels found: {unique_labels}\n• Label values: {label_values}", icon=":material/info:")
                
                # Emergency cleanup before stopping
                emergency_cleanup_tables()
                st.stop()
            
            #CREATE MODEL
            if "Naive Bayes" in classifier_type:
                # Create Naive Bayes model with TF-IDF features
                nb_type = "Bernoulli" if "Bernoulli" in classifier_type else "Multinomial"
                is_snowpark_ml = "Snowpark ML" in classifier_type
                library_name = "Snowpark ML" if is_snowpark_ml else "Scikit-learn"
                st.markdown(f"""<h2 style='font-size: 14px; margin-bottom: 0px;'>Training {library_name} {nb_type} Naive Bayes Model</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                st.write("")
                
                # Double-check availability at runtime
                if is_snowpark_ml and not check_snowpark_ml():
                    st.error("Snowpark ML not available. Please select a different classifier.", icon=":material/error:")
                    emergency_cleanup_tables()
                    st.stop()
                elif not is_snowpark_ml and not check_sklearn():
                    st.error("Scikit-learn not available. Please select a different classifier.", icon=":material/error:")
                    emergency_cleanup_tables()
                    st.stop()
                
                # Libraries already imported at top of script
                # Explicitly import Snowpark functions to avoid any scoping issues
                from snowflake.snowpark.functions import col, when
                
                # Use pre-computed TF-IDF data
                st.info(f"Preparing TF-IDF data for {library_name} {nb_type}NB...", icon=":material/download:")
                
                # Create Snowpark DataFrame from the TF-IDF table
                tfidf_snowpark_df = session.table(f"{database}.{schema}.{unique_fulltftable_name}")
                
               # Get column information
                columns_info = tfidf_snowpark_df.columns
                feature_columns = [col_name for col_name in columns_info if col_name not in [uid.upper(), 'SEQ', 'LABEL']]
               
               # Clean feature column names by removing extra quotes if present
                cleaned_feature_columns = []
                for col_name in feature_columns:
                   # Handle complex quoting patterns like "'ACCOUNT_CLOSED'"
                   cleaned_name = col_name
                   # Remove outer double quotes: "'ACCOUNT_CLOSED'" -> 'ACCOUNT_CLOSED'
                   while cleaned_name.startswith('"') and cleaned_name.endswith('"'):
                       cleaned_name = cleaned_name[1:-1]
                   # Remove inner single quotes: 'ACCOUNT_CLOSED' -> ACCOUNT_CLOSED
                   while cleaned_name.startswith("'") and cleaned_name.endswith("'"):
                       cleaned_name = cleaned_name[1:-1]
                   cleaned_feature_columns.append(cleaned_name)
               
               # Debug: Show sample column names
                st.info(f"Found {len(feature_columns)} TF-IDF features (events)", icon=":material/analytics:")
                # Debug information removed - feature columns processed successfully
                
                if nb_type == "Bernoulli":
                    # Convert TF-IDF values to binary features for Bernoulli NB (in Snowflake)
                    # Create binary columns: any non-zero TF-IDF value becomes 1
                    processed_df = tfidf_snowpark_df.select(
                        col(uid.upper()).alias("ID"),
                        col("LABEL"),
                        *[when(col(feat_col) > 0, 1).otherwise(0).alias(feat_col) for feat_col in feature_columns]
                    )
                    st.info("Converted TF-IDF to binary features (presence/absence)", icon=":material/transform:")
                else:  # Multinomial
                    # Keep original TF-IDF values for Multinomial NB
                    processed_df = tfidf_snowpark_df.select(
                        col(uid.upper()).alias("ID"),
                        col("LABEL"),
                        *[col(feat_col) for feat_col in feature_columns]
                    )
                    st.info("Using original TF-IDF values (event importance)", icon=":material/transform:")
                
                # Get data dimensions for display
                total_rows = tfidf_snowpark_df.count()
                total_features = len(feature_columns)
                
                st.success(f"Prepared TF-IDF features for Snowpark ML: {total_features} features from {total_rows} samples", icon=":material/check:")
                
                # Train Snowpark ML Bernoulli Naive Bayes
                st.info(f"Training {library_name} {nb_type} Naive Bayes classifier...", icon=":material/model_training:")
                
                # Initialize Naive Bayes model with lazy imports and fallback
                nb_model = None
                if is_snowpark_ml:
                    # Try Snowpark ML first
                    try:
                        from snowflake.ml.modeling.naive_bayes import BernoulliNB, MultinomialNB
                        
                        # Snowpark ML implementation - use proper parameters
                        if nb_type == "Bernoulli":
                            nb_model = BernoulliNB(
                                input_cols=feature_columns,
                                label_cols=["LABEL"],
                                output_cols=["PREDICTED_LABEL"],
                                alpha=1.0
                            )
                        else:  # Multinomial
                            nb_model = MultinomialNB(
                                input_cols=feature_columns,
                                label_cols=["LABEL"],
                                output_cols=["PREDICTED_LABEL"],
                                alpha=1.0
                            )
                    except Exception as snowpark_error:
                        # Automatic fallback to Scikit-learn
                        st.warning(f"Snowpark ML not available: {type(snowpark_error).__name__}. Automatically switching to Scikit-learn {nb_type} Naive Bayes.", icon=":material/swap_horiz:")
                        is_snowpark_ml = False  # Switch flag for fit/predict logic
                        library_name = "Scikit-learn"
                        nb_model = None  # Reset to attempt sklearn
                
                if nb_model is None:  # Either sklearn selected or snowpark ML failed
                    try:
                        # Lazy import scikit-learn
                        from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
                        
                        # Scikit-learn implementation - will need pandas data
                        if nb_type == "Bernoulli":
                            nb_model = SklearnBernoulliNB(alpha=1.0)
                        else:  # Multinomial
                            nb_model = SklearnMultinomialNB(alpha=1.0)
                    except Exception as sklearn_error:
                        # Handle complete import failure (both libraries unavailable)
                        st.error(f"Model initialization failed: Both Snowpark ML and Scikit-learn are unavailable.", icon=":material/error:")
                        st.error(f"Scikit-learn error: {type(sklearn_error).__name__}: {str(sklearn_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                # Fit the model with error handling and automatic fallback
                fit_success = False
                retry_with_sklearn = False
                
                try:
                    if is_snowpark_ml:
                        # Snowpark ML - use full DataFrame (columns specified in model init)
                        nb_model.fit(processed_df)
                        fit_success = True
                    else:
                        # Scikit-learn - convert to pandas
                        processed_pandas_df = processed_df.to_pandas()
                        
                        # Get actual pandas column names and create mapping
                        pandas_columns = processed_pandas_df.columns.tolist()
                        pandas_feature_columns = [col for col in pandas_columns if col not in ['ID', 'LABEL']]
                        
                        # Create mapping from pandas columns to cleaned names
                        col_mapping = {}
                        for pandas_col in pandas_feature_columns:
                            cleaned_name = pandas_col.strip("'").strip('"')
                            col_mapping[pandas_col] = cleaned_name
                        
                        # Rename columns to use cleaned names
                        processed_pandas_df = processed_pandas_df.rename(columns=col_mapping)
                        cleaned_feature_columns_sklearn = list(col_mapping.values())
                        
                        X = processed_pandas_df[cleaned_feature_columns_sklearn]
                        y = processed_pandas_df["LABEL"]
                        nb_model.fit(X, y)
                        fit_success = True
                        
                except Exception as model_error:
                    # Check if this is a Snowpark ML dependency error that should trigger fallback
                    error_str = str(model_error).lower()
                    if is_snowpark_ml and ('xgboost' in error_str or 'package' in error_str or 'conda' in error_str):
                        st.warning(f"Snowpark ML training failed due to dependency issues. Automatically switching to Scikit-learn {nb_type} Naive Bayes.", icon=":material/swap_horiz:")
                        retry_with_sklearn = True
                        is_snowpark_ml = False
                        library_name = "Scikit-learn"
                    else:
                        # Handle any other model training errors
                        st.error(f"Model training failed: {str(model_error)}", icon=":material/error:")
                        st.error(f"Error details: {type(model_error).__name__}: {str(model_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                # Retry with scikit-learn if Snowpark ML failed
                if retry_with_sklearn and not fit_success:
                    try:
                        st.info(f"Training {library_name} {nb_type} Naive Bayes classifier...", icon=":material/model_training:")
                        
                        # Reinitialize with sklearn
                        from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
                        
                        if nb_type == "Bernoulli":
                            nb_model = SklearnBernoulliNB(alpha=1.0)
                        else:  # Multinomial
                            nb_model = SklearnMultinomialNB(alpha=1.0)
                        
                        # Convert to pandas
                        processed_pandas_df = processed_df.to_pandas()
                        
                        # Get actual pandas column names and create mapping
                        pandas_columns = processed_pandas_df.columns.tolist()
                        pandas_feature_columns = [col for col in pandas_columns if col not in ['ID', 'LABEL']]
                        
                        # Create mapping from pandas columns to cleaned names
                        col_mapping = {}
                        for pandas_col in pandas_feature_columns:
                            cleaned_name = pandas_col.strip("'").strip('"')
                            col_mapping[pandas_col] = cleaned_name
                        
                        # Rename columns to use cleaned names
                        processed_pandas_df = processed_pandas_df.rename(columns=col_mapping)
                        cleaned_feature_columns_sklearn = list(col_mapping.values())
                        
                        X = processed_pandas_df[cleaned_feature_columns_sklearn]
                        y = processed_pandas_df["LABEL"]
                        nb_model.fit(X, y)
                        fit_success = True
                    except Exception as sklearn_error:
                        st.error(f"Scikit-learn training also failed: {str(sklearn_error)}", icon=":material/error:")
                        st.error(f"Error details: {type(sklearn_error).__name__}: {str(sklearn_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                st.success(f"{library_name} {nb_type}NB model trained successfully!", icon=":material/check:")
                
                # Evaluate model performance
                st.info("Evaluating model performance...", icon=":material/analytics:")
                
                # Create train/test split for evaluation (80/20)
                if is_snowpark_ml:
                    train_df, test_df = processed_df.random_split([0.8, 0.2], seed=42)
                    # Make predictions on test set
                    predictions_df = nb_model.predict(test_df)
                else:
                   # Lazy import train_test_split
                   from sklearn.model_selection import train_test_split
                   
                   # Scikit-learn evaluation - use cleaned column names
                   X_train, X_test, y_train, y_test = train_test_split(
                       processed_pandas_df[cleaned_feature_columns_sklearn], 
                       processed_pandas_df["LABEL"], 
                       test_size=0.2, 
                       random_state=42
                   )
                   y_pred = nb_model.predict(X_test)
                
                # Calculate accuracy and other metrics
                # Lazy import sklearn metrics
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                if is_snowpark_ml:
                    # Smart sampling for evaluation metrics calculation to avoid scanning whole dataset
                    # Use a regular table and SQL TABLESAMPLE for efficient database-level sampling
                    temp_table_name = f"TEMP_PREDICTIONS_{uuid.uuid4().hex[:8].upper()}"
                    
                    try:
                        # Save predictions to a regular table for TABLESAMPLE
                        predictions_df.write.mode("overwrite").save_as_table(temp_table_name)
                        
                        # Sample using SQL TABLESAMPLE with user-defined percentage
                        sample_sql = f"""
                            SELECT "LABEL", "PREDICTED_LABEL"
                            FROM {temp_table_name} TABLESAMPLE BERNOULLI ({sample_percent})
                            LIMIT {max_sample_rows}
                        """
                        
                        # Get sampled data
                        sampled_df = session.sql(sample_sql)
                        eval_df = sampled_df.to_pandas()
                        
                        # Drop the temp table immediately after use
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {temp_table_name}").collect()
                        except:
                            pass
                        
                        # If we got more than expected, cap it
                        if len(eval_df) > max_sample_rows:
                            eval_df = eval_df.sample(n=max_sample_rows, random_state=42)
                        
                        st.info(f"Evaluation metrics computed on {len(eval_df):,} sampled rows", icon=":material/analytics:")
                    except Exception as sample_error:
                        # Clean up temp table if it exists
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {temp_table_name}").collect()
                        except:
                            pass
                        
                        # Fallback: use DataFrame sample() method
                        st.warning(f"SQL sampling failed, using DataFrame sample: {str(sample_error)}", icon=":material/warning:")
                        sample_fraction = sample_percent / 100.0
                        eval_df = predictions_df.sample(frac=sample_fraction).select("LABEL", "PREDICTED_LABEL").to_pandas()
                        if len(eval_df) > max_sample_rows:
                            eval_df = eval_df.sample(n=max_sample_rows, random_state=42)
                    
                    accuracy = accuracy_score(eval_df["LABEL"], eval_df["PREDICTED_LABEL"])
                    class_report = classification_report(eval_df["LABEL"], eval_df["PREDICTED_LABEL"], output_dict=True)
                    conf_matrix = confusion_matrix(eval_df["LABEL"], eval_df["PREDICTED_LABEL"])
                else:
                    # Scikit-learn metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    # Create eval_df for consistency
                    eval_df = pd.DataFrame({"LABEL": y_test, "PREDICTED_LABEL": y_pred})
                
                st.success(f"Model evaluation complete! Accuracy: {accuracy:.4f}", icon=":material/check:")
                
               # Model registry functionality removed - will be added as optional feature later
                
                model_type_display = "Scikit-learn Naive Bayes (Bernoulli)"
                
                # Compute Naive Bayes feature analysis
                st.info("Computing feature importance analysis...", icon=":material/analytics:")
               
                if is_snowpark_ml:
                   # Snowpark ML feature analysis - basic statistical approach
                   try:
                       # Get unique classes from the data
                       classes_df = processed_df.select("LABEL").distinct().to_pandas()
                       class_names = sorted(classes_df["LABEL"].unique())
                       
                       # Create enhanced statistical feature analysis for Snowpark ML
                       feature_analysis = {
                           'model_type': f'snowpark_ml_{nb_type.lower()}_nb',
                           'total_features': total_features,
                           'class_names': class_names,
                           'feature_columns': cleaned_feature_columns  # Use cleaned names
                       }
                       
                       # Calculate enhanced feature statistics per class
                       class_statistics = {}
                       discriminative_features = []
                       
                       # Convert to pandas for statistical analysis
                       analysis_df = processed_df.to_pandas()
                       
                       # Get actual pandas column names (they might differ from Snowpark column names)
                       pandas_columns = analysis_df.columns.tolist()
                       pandas_feature_columns = [col for col in pandas_columns if col not in ['ID', 'LABEL']]
                       
                       # Create mapping from actual pandas columns to cleaned names
                       # Match by removing quotes from both sides
                       col_mapping = {}
                       for pandas_col in pandas_feature_columns:
                           # Find the corresponding cleaned name
                           pandas_cleaned = pandas_col.strip("'").strip('"')
                           col_mapping[pandas_col] = pandas_cleaned
                       
                       # Rename columns in pandas DataFrame to use cleaned names
                       analysis_df = analysis_df.rename(columns=col_mapping)
                       
                       # Update cleaned_feature_columns to match what we actually have
                       cleaned_feature_columns = list(col_mapping.values())
                       
                       for class_name in class_names:
                           class_data = analysis_df[analysis_df["LABEL"] == class_name]
                           class_count = len(class_data)
                           
                           # Calculate feature activity for this class (use cleaned names)
                           active_features = 0
                           for cleaned_col in cleaned_feature_columns:
                               if (class_data[cleaned_col] > 0).sum() > 0:
                                   active_features += 1
                           
                           class_statistics[class_name] = {
                               'total_features': total_features,
                               'sample_count': class_count,
                               'class_proportion': class_count / total_rows if total_rows > 0 else 0,
                               'active_features': active_features,
                               'feature_density': active_features / total_features if total_features > 0 else 0
                           }
                       
                       # Enhanced discriminative power analysis for Snowpark ML
                       if len(class_names) == 2:  # Binary classification
                           import numpy as np
                           from scipy.stats import chi2_contingency
                           
                           class_0_data = analysis_df[analysis_df["LABEL"] == class_names[0]]
                           class_1_data = analysis_df[analysis_df["LABEL"] == class_names[1]]
                           
                           for cleaned_col in cleaned_feature_columns:
                               try:
                                   # Method 1: Enhanced log odds ratio with TF-IDF weighting
                                   class_0_values = class_0_data[cleaned_col]
                                   class_1_values = class_1_data[cleaned_col]
                                   
                                   # Calculate weighted means (TF-IDF aware)
                                   class_0_mean = class_0_values.mean()
                                   class_1_mean = class_1_values.mean()
                                   
                                   # Method 2: Chi-square test for independence
                                   # Create contingency table: [present/absent] x [class0/class1]
                                   class_0_present = (class_0_values > 0).sum()
                                   class_0_absent = (class_0_values == 0).sum()
                                   class_1_present = (class_1_values > 0).sum()
                                   class_1_absent = (class_1_values == 0).sum()
                                   
                                   contingency_table = np.array([
                                       [class_0_present, class_0_absent],
                                       [class_1_present, class_1_absent]
                                   ])
                                   
                                   # Calculate chi-square statistic
                                   if contingency_table.sum() > 0 and np.all(contingency_table.sum(axis=0) > 0):
                                       chi2, p_value, _, _ = chi2_contingency(contingency_table)
                                       
                                       # Method 3: Mutual information approximation
                                       # Calculate conditional probabilities
                                       total_samples = len(analysis_df)
                                       p_class_0 = len(class_0_data) / total_samples
                                       p_class_1 = len(class_1_data) / total_samples
                                       
                                       # Feature presence probabilities
                                       p_feature_present = (analysis_df[cleaned_col] > 0).sum() / total_samples
                                       p_feature_absent = 1 - p_feature_present
                                       
                                       # Conditional probabilities
                                       if len(class_0_data) > 0 and len(class_1_data) > 0:
                                           p_feature_given_class0 = (class_0_values > 0).mean()
                                           p_feature_given_class1 = (class_1_values > 0).mean()
                                           
                                           # Enhanced discriminative score combining multiple methods
                                           # 1. TF-IDF weighted difference
                                           tfidf_score = class_1_mean - class_0_mean
                                           
                                           # 2. Log likelihood ratio (more robust than simple log odds)
                                           epsilon = 1e-6
                                           p_feature_given_class0 = max(p_feature_given_class0, epsilon)
                                           p_feature_given_class1 = max(p_feature_given_class1, epsilon)
                                           
                                           log_likelihood_ratio = np.log(p_feature_given_class1 / p_feature_given_class0)
                                           
                                           # 3. Chi-square normalized score
                                           chi2_score = chi2 * np.sign(tfidf_score) if chi2 > 0 else 0
                                           
                                           # Combined discriminative score (weighted combination)
                                           discriminative_score = (
                                               0.4 * log_likelihood_ratio +  # Probabilistic component
                                               0.3 * tfidf_score * 10 +      # TF-IDF component (scaled)
                                               0.3 * chi2_score / 100        # Statistical significance (scaled)
                                           )
                                           
                                           interpretation = f"Favors {class_names[1]}" if discriminative_score > 0 else f"Favors {class_names[0]}"
                                           discriminative_features.append((cleaned_col, discriminative_score, interpretation))
                                   
                               except Exception as feature_error:
                                   # Skip problematic features but continue with others
                                   continue
                       
                       # Sort by absolute discriminative power
                       discriminative_features.sort(key=lambda x: abs(x[1]), reverse=True)
                       
                       feature_analysis['class_statistics'] = class_statistics
                       feature_analysis['top_discriminative_features'] = discriminative_features
                       
                       st.success("Enhanced feature analysis completed for Snowpark ML model", icon=":material/check:")
                       
                   except Exception as e:
                       st.warning(f"Enhanced feature analysis failed: {str(e)}", icon=":material/warning:")
                       st.write(f"**Debug - Error details:**")
                       st.write(f"Error type: {type(e).__name__}")
                       st.write(f"Error message: {str(e)}")
                       if len(feature_columns) > 0:
                           st.write(f"Problematic column names: {feature_columns[:5]}")
                           st.write(f"Cleaned column names: {cleaned_feature_columns[:5]}")
                       
                       # Fallback to basic analysis
                       feature_analysis = {
                           'model_type': f'snowpark_ml_{nb_type.lower()}_nb',
                           'total_features': total_features,
                           'note': 'Enhanced feature analysis not available'
                       }
                else:
                   # Scikit-learn feature analysis - use actual log probabilities
                   try:
                       import numpy as np
                       
                       # Get the trained model's log probabilities
                       log_prob_class_0 = nb_model.feature_log_prob_[0]  # Log probabilities for class 0
                       log_prob_class_1 = nb_model.feature_log_prob_[1]  # Log probabilities for class 1
                       
                       # Calculate log probability ratios (discriminative power)
                       log_prob_ratios = log_prob_class_1 - log_prob_class_0
                       
                       # Get class names
                       class_names = nb_model.classes_
                       
                       # Create discriminative features list with actual log probability ratios
                       discriminative_features = []
                       for i, cleaned_name in enumerate(cleaned_feature_columns_sklearn):
                           ratio = log_prob_ratios[i]
                           interpretation = f"Favors {class_names[1]}" if ratio > 0 else f"Favors {class_names[0]}"
                           discriminative_features.append((cleaned_name, float(ratio), interpretation))
                       
                       # Sort by absolute discriminative power
                       discriminative_features.sort(key=lambda x: abs(x[1]), reverse=True)
                       
                       # Calculate class statistics from the pandas data
                       class_statistics = {}
                       
                       # Use the already cleaned DataFrame from training
            
                       for class_idx, class_name in enumerate(class_names):
                           class_data = processed_pandas_df[processed_pandas_df["LABEL"] == class_name]
                           class_count = len(class_data)
                           
                           # Calculate feature activity for this class (use cleaned names)
                           active_features = 0
                           for cleaned_col in cleaned_feature_columns_sklearn:
                               if (class_data[cleaned_col] > 0).sum() > 0:
                                   active_features += 1
                           
                           # Get average log probability for this class
                           avg_log_prob = np.mean(nb_model.feature_log_prob_[class_idx])
                           
                           class_statistics[class_name] = {
                               'total_features': total_features,
                               'sample_count': class_count,
                               'class_proportion': class_count / len(processed_pandas_df),
                               'active_features': active_features,
                               'feature_density': active_features / total_features if total_features > 0 else 0,
                               'avg_log_prob': float(avg_log_prob)
                           }
                       
                       feature_analysis = {
                           'model_type': f'sklearn_{nb_type.lower()}_nb',
                           'total_features': len(cleaned_feature_columns_sklearn),
                           'class_names': class_names.tolist(),
                           'feature_columns': cleaned_feature_columns_sklearn,  # Use cleaned names
                           'top_discriminative_features': discriminative_features,
                           'class_statistics': class_statistics
                       }
                       
                       st.success(f"Scikit-learn feature analysis completed with {len(discriminative_features)} discriminative features", icon=":material/check:")
                       
                   except Exception as e:
                       st.warning(f"Scikit-learn feature analysis failed: {str(e)}", icon=":material/warning:")
                       st.write(f"**Debug - Error details:**")
                       st.write(f"Error type: {type(e).__name__}")
                       st.write(f"Error message: {str(e)}")
                       if len(feature_columns) > 0:
                           st.write(f"Problematic column names: {feature_columns[:5]}")
                           st.write(f"Cleaned column names: {cleaned_feature_columns[:5]}")
                       
                       feature_analysis = {
                           'model_type': f'sklearn_{nb_type.lower()}_nb',
                           'total_features': total_features,
                           'note': 'Feature analysis not available'
                       }
                
                # Store evaluation metrics for display
                evaluation_metrics = {
                    'accuracy': accuracy,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_count': total_features,
                    'training_samples': int(total_rows * 0.8),  # Approximate training size
                    'test_samples': int(total_rows * 0.2),     # Approximate test size
                    'feature_analysis': feature_analysis
               }
               
            elif "Random Forest" in classifier_type:
                # Create Random Forest model with TF-IDF features
                is_snowpark_ml = "Snowpark ML" in classifier_type
                library_name = "Snowpark ML" if is_snowpark_ml else "Scikit-learn"
                st.markdown(f"""<h2 style='font-size: 14px; margin-bottom: 0px;'>Training {library_name} Random Forest Model</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                
                # Double-check availability at runtime
                if is_snowpark_ml and not check_snowpark_ml():
                    st.error("Snowpark ML not available. Please select a different classifier.", icon=":material/error:")
                    emergency_cleanup_tables()
                    st.stop()
                elif not is_snowpark_ml and not check_sklearn():
                    st.error("Scikit-learn not available. Please select a different classifier.", icon=":material/error:")
                    emergency_cleanup_tables()
                    st.stop()
                
                # Use pre-computed TF-IDF data
                st.info(f"Preparing TF-IDF data for {library_name} Random Forest...", icon=":material/download:")
                
                # Get the processed DataFrame with TF-IDF features
                processed_df = session.table(f"{database}.{schema}.{unique_fulltftable_name}")
                
                # Create Random Forest model with automatic fallback
                rf_model = None
                if is_snowpark_ml:
                    # Try Snowpark ML first
                    try:
                        from snowflake.ml.modeling.ensemble import RandomForestClassifier as SnowparkRandomForest
                        
                        # Snowpark ML implementation
                        rf_model = SnowparkRandomForest(
                            input_cols=cleaned_feature_columns,
                            label_cols=["LABEL"],
                            output_cols=["PREDICTED_LABEL"],
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                    except Exception as snowpark_error:
                        # Automatic fallback to Scikit-learn
                        st.warning(f"Snowpark ML not available: {type(snowpark_error).__name__}. Automatically switching to Scikit-learn Random Forest.", icon=":material/swap_horiz:")
                        is_snowpark_ml = False  # Switch flag for fit/predict logic
                        library_name = "Scikit-learn"
                        rf_model = None  # Reset to attempt sklearn
                
                if rf_model is None:  # Either sklearn selected or snowpark ML failed
                    try:
                        # Lazy import scikit-learn
                        from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
                        
                        # Scikit-learn implementation
                        rf_model = SklearnRandomForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1
                        )
                    except Exception as sklearn_error:
                        # Handle complete import failure (both libraries unavailable)
                        st.error(f"Model initialization failed: Both Snowpark ML and Scikit-learn are unavailable.", icon=":material/error:")
                        st.error(f"Scikit-learn error: {type(sklearn_error).__name__}: {str(sklearn_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                # Fit the model with error handling and automatic fallback
                fit_success = False
                retry_with_sklearn = False
                
                try:
                    if is_snowpark_ml:
                        # Snowpark ML - use full DataFrame (columns specified in model init)
                        rf_model.fit(processed_df)
                        fit_success = True
                    else:
                        # Scikit-learn - convert to pandas
                        processed_pandas_df = processed_df.to_pandas()
                        
                        # Get actual pandas column names and create mapping
                        pandas_columns = processed_pandas_df.columns.tolist()
                        pandas_feature_columns = [col for col in pandas_columns if col not in ['ID', 'LABEL']]
                        
                        # Create mapping from pandas columns to cleaned names
                        col_mapping = {}
                        for pandas_col in pandas_feature_columns:
                            cleaned_name = pandas_col.strip("'").strip('"')
                            col_mapping[pandas_col] = cleaned_name
                        
                        # Rename columns to use cleaned names
                        processed_pandas_df = processed_pandas_df.rename(columns=col_mapping)
                        cleaned_feature_columns_sklearn = list(col_mapping.values())
                        
                        X = processed_pandas_df[cleaned_feature_columns_sklearn]
                        y = processed_pandas_df["LABEL"]
                        rf_model.fit(X, y)
                        fit_success = True
                        
                except Exception as model_error:
                    # Check if this is a Snowpark ML dependency error that should trigger fallback
                    error_str = str(model_error).lower()
                    if is_snowpark_ml and ('xgboost' in error_str or 'package' in error_str or 'conda' in error_str):
                        st.warning(f"Snowpark ML training failed due to dependency issues. Automatically switching to Scikit-learn Random Forest.", icon=":material/swap_horiz:")
                        retry_with_sklearn = True
                        is_snowpark_ml = False
                        library_name = "Scikit-learn"
                    else:
                        # Handle any other model training errors
                        st.error(f"Model training failed: {str(model_error)}", icon=":material/error:")
                        st.error(f"Error details: {type(model_error).__name__}: {str(model_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                # Retry with scikit-learn if Snowpark ML failed
                if retry_with_sklearn and not fit_success:
                    try:
                        st.info(f"Training {library_name} Random Forest classifier...", icon=":material/model_training:")
                        
                        # Reinitialize with sklearn
                        from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
                        
                        rf_model = SklearnRandomForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        # Convert to pandas
                        processed_pandas_df = processed_df.to_pandas()
                        
                        # Get actual pandas column names and create mapping
                        pandas_columns = processed_pandas_df.columns.tolist()
                        pandas_feature_columns = [col for col in pandas_columns if col not in ['ID', 'LABEL']]
                        
                        # Create mapping from pandas columns to cleaned names
                        col_mapping = {}
                        for pandas_col in pandas_feature_columns:
                            cleaned_name = pandas_col.strip("'").strip('"')
                            col_mapping[pandas_col] = cleaned_name
                        
                        # Rename columns to use cleaned names
                        processed_pandas_df = processed_pandas_df.rename(columns=col_mapping)
                        cleaned_feature_columns_sklearn = list(col_mapping.values())
                        
                        X = processed_pandas_df[cleaned_feature_columns_sklearn]
                        y = processed_pandas_df["LABEL"]
                        rf_model.fit(X, y)
                        fit_success = True
                    except Exception as sklearn_error:
                        st.error(f"Scikit-learn training also failed: {str(sklearn_error)}", icon=":material/error:")
                        st.error(f"Error details: {type(sklearn_error).__name__}: {str(sklearn_error)}", icon=":material/bug_report:")
                        
                        # Emergency cleanup before stopping
                        emergency_cleanup_tables()
                        st.stop()
                
                st.success(f"{library_name} Random Forest model trained successfully!", icon=":material/check:")
                
                # Evaluate model performance
                st.info("Evaluating model performance...", icon=":material/analytics:")
                
                # Create train/test split for evaluation (80/20)
                if is_snowpark_ml:
                    train_df, test_df = processed_df.random_split([0.8, 0.2], seed=42)
                    # Make predictions on test set
                    predictions_df = rf_model.predict(test_df)
                    
                    # Smart sampling for evaluation to avoid scanning whole dataset
                    # Use a regular table and SQL TABLESAMPLE for efficient database-level sampling
                    temp_table_name = f"TEMP_PREDICTIONS_{uuid.uuid4().hex[:8].upper()}"
                    
                    try:
                        # Save predictions to a regular table for TABLESAMPLE
                        predictions_df.write.mode("overwrite").save_as_table(temp_table_name)
                        
                        # Sample using SQL TABLESAMPLE with user-defined percentage
                        sample_sql = f"""
                            SELECT "LABEL", "PREDICTED_LABEL"
                            FROM {temp_table_name} TABLESAMPLE BERNOULLI ({sample_percent})
                            LIMIT {max_sample_rows}
                        """
                        
                        # Get sampled data
                        sampled_df = session.sql(sample_sql)
                        predictions_pandas = sampled_df.to_pandas()
                        
                        # Drop the temp table immediately after use
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {temp_table_name}").collect()
                        except:
                            pass
                        
                        # If we got more than expected, cap it
                        if len(predictions_pandas) > max_sample_rows:
                            predictions_pandas = predictions_pandas.sample(n=max_sample_rows, random_state=42)
                        
                        st.info(f"Evaluation metrics computed on {len(predictions_pandas):,} sampled rows", icon=":material/analytics:")
                    except Exception as sample_error:
                        # Clean up temp table if it exists
                        try:
                            session.sql(f"DROP TABLE IF EXISTS {temp_table_name}").collect()
                        except:
                            pass
                        
                        # Fallback: use DataFrame sample() method
                        st.warning(f"SQL sampling failed, using DataFrame sample: {str(sample_error)}", icon=":material/warning:")
                        sample_fraction = sample_percent / 100.0
                        predictions_pandas = predictions_df.sample(frac=sample_fraction).select("LABEL", "PREDICTED_LABEL").to_pandas()
                        if len(predictions_pandas) > max_sample_rows:
                            predictions_pandas = predictions_pandas.sample(n=max_sample_rows, random_state=42)
                    
                    y_true = predictions_pandas["LABEL"]
                    y_pred = predictions_pandas["PREDICTED_LABEL"]
                else:
                    # Scikit-learn evaluation
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Make predictions
                    y_pred = rf_model.predict(X_test)
                    y_true = y_test
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                accuracy = accuracy_score(y_true, y_pred)
                class_report = classification_report(y_true, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                st.success(f"Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)", icon=":material/check:")
                
                # Feature importance analysis
                st.info("Analyzing feature importance...", icon=":material/analytics:")
                
                feature_analysis = {}
                
                if is_snowpark_ml:
                    try:
                        # Snowpark ML Random Forest feature importance
                        # Note: Snowpark ML Random Forest may not have direct feature_importances_ 
                        # We'll use a simplified approach
                        st.info("Snowpark ML Random Forest feature importance analysis in development", icon=":material/info:")
                        feature_analysis = {"method": "Snowpark ML Random Forest", "status": "limited"}
                        
                    except Exception as e:
                        st.warning(f"Snowpark ML feature analysis failed: {str(e)}", icon=":material/warning:")
                        feature_analysis = {"method": "Snowpark ML Random Forest", "status": "failed", "error": str(e)}
                else:
                    try:
                        # Scikit-learn Random Forest feature importance
                        feature_importances = rf_model.feature_importances_
                        
                        # Create feature importance DataFrame
                        import pandas as pd
                        import numpy as np
                        
                        feature_importance_data = []
                        for i, (feature, importance) in enumerate(zip(cleaned_feature_columns_sklearn, feature_importances)):
                            feature_importance_data.append({
                                'Feature': feature,
                                'Importance': importance,
                                'Rank': i + 1
                            })
                        
                        # Sort by importance
                        feature_importance_data.sort(key=lambda x: x['Importance'], reverse=True)
                        
                        # Update ranks
                        for i, item in enumerate(feature_importance_data):
                            item['Rank'] = i + 1
                        
                        # Create DataFrame for display
                        importance_df = pd.DataFrame(feature_importance_data)
                        importance_df.index = range(1, len(importance_df) + 1)
                        
                        # Store feature analysis
                        feature_analysis = {
                            'method': 'Random Forest Feature Importance',
                            'features': feature_importance_data[:10],  # Top 10 features
                            'total_features': len(feature_importance_data)
                        }
                        
                        st.success("Random Forest feature importance analysis completed", icon=":material/check:")
                        
                    except Exception as e:
                        st.warning(f"Random Forest feature analysis failed: {str(e)}", icon=":material/warning:")
                        feature_analysis = {"method": "Random Forest", "status": "failed", "error": str(e)}
                
                # Store Random Forest results
                rf_results = {
                    'model_type': 'Random Forest',
                    'library': library_name,
                    'accuracy': accuracy,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_count': total_features,
                    'training_samples': int(total_rows * 0.8),  # Approximate training size
                    'test_samples': int(total_rows * 0.2),     # Approximate test size
                    'feature_analysis': feature_analysis,
                    'n_estimators': 100,
                    'max_depth': 10
                }
               
            else:
              # No other classifier types supported
              st.error("Please select a valid classifier.", icon=":material/error:")
              emergency_cleanup_tables()
              st.stop()
            
            # Store model results in session state for persistence
            if "Naive Bayes" in classifier_type:
               # Naive Bayes models (both Bernoulli and Multinomial)
               st.session_state["predictive_results"] = {
                   'completed': True,
                   'model_name': model_name,
                   'num_classes': len(class_names),
                   'num_samples': total_rows,
                   'model_type': f'Binary Classification ({library_name} {nb_type}NB)',
                   'classifier_algorithm': classifier_type,
                   'model_accuracy': f"{accuracy:.4f}",
                   'accuracy': accuracy,
                   'evaluation_metrics': evaluation_metrics,
                   'is_sklearn_model': not is_snowpark_ml,
                   'model': nb_model
               }
               st.success(f"{library_name} {nb_type}NB model `{model_name}` trained successfully with {len(class_names)} classes ({total_rows:,} training samples, {accuracy:.4f} accuracy)", icon=":material/check:")
            elif "Random Forest" in classifier_type:
               # Random Forest models
               st.session_state["predictive_results"] = {
                   'completed': True,
                   'model_name': model_name,
                   'num_classes': len(class_names),
                   'num_samples': total_rows,
                   'model_type': f'Binary Classification ({library_name} Random Forest)',
                   'classifier_algorithm': classifier_type,
                   'model_accuracy': f"{accuracy:.4f}",
                   'accuracy': accuracy,
                   'evaluation_metrics': rf_results,
                   'is_sklearn_model': not is_snowpark_ml,
                   'model': rf_model
               }
               st.success(f"{library_name} Random Forest model `{model_name}` trained successfully with {len(class_names)} classes ({total_rows:,} training samples, {accuracy:.4f} accuracy)", icon=":material/check:")
                
            # Cleanup intermediate tables with better error handling
            with st.expander("Process Details & Cleanup", expanded=False, icon=":material/cleaning_services:"):
                st.info("Cleaning up intermediate tables...", icon=":material/cleaning_services:")
                st.write(f"**Database:** {database}")
                st.write(f"**Schema:** {schema}")
                cleanup_errors = []
                
                # List of specific table names to clean up (from this session)
                try:
                    specific_tables = [
                        unique_reftable_name,
                        unique_comptable_name,
                        unique_reftftable_name,
                        unique_comptftable_name,
                        unique_fulltftable_name,
                        unique_trainingtable_name
                    ]
                    st.write(f"**Tables to clean up (local variables):** {specific_tables}")
                except NameError as e:
                    st.warning(f"Table name variables not defined: {str(e)}")
                    # Fallback to session state
                    specific_tables = st.session_state.get('temp_tables_for_cleanup', [])
                    if specific_tables:
                        st.write(f"**Tables to clean up (from session state):** {specific_tables}")
                    else:
                        st.error("No table names available for cleanup!")
                        specific_tables = []
                
                # Clean up specific tables first
                if specific_tables:
                    st.write(f"**Cleaning up {len(specific_tables)} intermediate tables...**")
                    for table_name in specific_tables:
                        try:
                            cleanup_sql = f"""DROP TABLE IF EXISTS {database}.{schema}.{table_name}"""
                            result = session.sql(cleanup_sql).collect()
                            st.success(f"Dropped table: {table_name}", icon=":material/delete:")
                        except Exception as cleanup_error:
                            cleanup_errors.append(f"{table_name}: {str(cleanup_error)}")
                            st.warning(f"Could not drop {table_name}: {str(cleanup_error)}", icon=":material/warning:")
                else:
                    st.warning("No tables to clean up (table names not available)", icon=":material/warning:")
                
                # Also clean up any other tables matching patterns (for safety)
                table_patterns = [
                    f"RAWEVENTSREF_%",
                    f"RAWEVENTSCOMP_%", 
                    f"TFIDFREF_%",
                    f"TFIDFCOMP_%",
                    f"FULLTFIDF_%",
                    f"TRAINING_%"
                ]
                
                # Get list of tables matching our patterns
                for pattern in table_patterns:
                    try:
                        list_tables_sql = f"""
                        SHOW TABLES LIKE '{pattern}' IN {database}.{schema}
                        """
                        result = session.sql(list_tables_sql).collect()
                        
                        for row in result:
                            table_name = row['name']
                            try:
                                cleanup_sql = f"""DROP TABLE IF EXISTS {database}.{schema}.{table_name}"""
                                session.sql(cleanup_sql).collect()
                            except Exception as cleanup_error:
                                cleanup_errors.append(f"{table_name}: {str(cleanup_error)}")
                                
                    except Exception as pattern_error:
                        # If SHOW TABLES fails, fall back to trying specific names
                        pass
                
                # Summary
                total_tables = len(specific_tables)
                failed_tables = len(cleanup_errors)
                successful_tables = total_tables - failed_tables
                
                if cleanup_errors:
                    st.warning(f"Cleanup completed: {successful_tables}/{total_tables} tables cleaned successfully. Failed: {', '.join(cleanup_errors)}", icon=":material/warning:")
                else:
                    st.success(f"All {total_tables} intermediate tables cleaned up successfully!", icon=":material/check:")
                
                # Clear the session state cleanup list
                if 'temp_tables_for_cleanup' in st.session_state:
                    st.session_state['temp_tables_for_cleanup'] = []
                    st.write("**Session state cleanup list cleared.**")
        
        with tab2:
            # Results tab - show results if available
            if 'predictive_results' in st.session_state and st.session_state['predictive_results'].get('completed'):
                
                # Get results from session state
                results = st.session_state['predictive_results']
                model_name = results.get('model_name', '')
                is_sklearn_model = results.get('is_sklearn_model', False)
                
                if model_name:
                    if is_sklearn_model:
                        # Handle sklearn model results
                        st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Scikit-learn Model Evaluation Metrics</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                        st.write("")
                        evaluation_metrics = results.get('evaluation_metrics', {})
                        if evaluation_metrics:
                            # Display accuracy
                            accuracy = evaluation_metrics.get('accuracy', 'N/A')
                            with st.container(border=True):
                                st.metric("Test Accuracy", f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy)
                            
                            # Display classification report
                            classification_report = evaluation_metrics.get('classification_report', {})
                            if classification_report:
                                st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Classification Report</h3>""", unsafe_allow_html=True)

                                # Convert classification report to DataFrame for better display
                                import pandas as pd
                                report_df = pd.DataFrame(classification_report).transpose()
                                with st.container(border=True):
                                    st.dataframe(report_df, use_container_width=True)
                            
                            # Display confusion matrix
                            confusion_matrix = evaluation_metrics.get('confusion_matrix', [])
                            if confusion_matrix:
                                st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Confusion Matrix</h3>""", unsafe_allow_html=True)

                                import pandas as pd
                                import numpy as np
                                
                                # Get class names from results
                                classes = ['Class 0', 'Class 1']  # Default for binary classification
                                cm_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
                                with st.container(border=True):
                                    st.dataframe(cm_df, use_container_width=True)
                            
                            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Feature Analysis</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                            st.write("")
                            # Display feature information
                            feature_count = evaluation_metrics.get('feature_count', 'N/A')
                            training_samples = evaluation_metrics.get('training_samples', 'N/A')
                            test_samples = evaluation_metrics.get('test_samples', 'N/A')
                            
                            with st.container(border=True):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("TF-IDF Features", feature_count)
                                with col2:
                                    st.metric("Training Samples", training_samples)
                                with col3:
                                    st.metric("Test Samples", test_samples)
                            
                            # Display Naive Bayes Feature Analysis
                            feature_analysis = evaluation_metrics.get('feature_analysis', {})
                            if feature_analysis:
                                
                                # Most discriminative features
                                top_features = feature_analysis.get('top_discriminative_features', [])
                                if top_features:
                                    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Most Discriminative Events</h3>""", unsafe_allow_html=True, help="Positive values indicate the event is more likely in the second class, negative values indicate it's more likely in the first class. Higher absolute values indicate stronger discriminative power.")

                                    
                                    # Create DataFrame for better display
                                    import pandas as pd
                                    features_df = pd.DataFrame(top_features)
                                    features_df.columns = ['Event', 'Log Probability Ratio', 'Interpretation']
                                    features_df.index = range(1, len(features_df) + 1)  # Start index from 1
                                    with st.container(border=True):
                                        st.dataframe(features_df, use_container_width=True)
                                    
                                    
                                    # Create a bar chart for all discriminative features
                                    if len(top_features) > 0:
                                        import altair as alt
                                        
                                        # Prepare data for chart (all features)
                                        chart_data = []
                                        for i, (event, ratio_str, interpretation) in enumerate(top_features):
                                            chart_data.append({
                                                'Event': event,
                                                'Log_Probability_Ratio': float(ratio_str),
                                                'Rank': i + 1
                                            })
                                        
                                        chart_df = pd.DataFrame(chart_data)
                                        
                                        # Create horizontal bar chart
                                        chart = alt.Chart(chart_df).mark_bar().add_selection(
                                            alt.selection_single()
                                        ).encode(
                                            y=alt.Y('Event:N', sort='-x', title='Event'),
                                            x=alt.X('Log_Probability_Ratio:Q', title='Log Probability Ratio'),
                                            color=alt.condition(
                                                alt.datum.Log_Probability_Ratio > 0,
                                                alt.value('#1f77b4'),  # Blue for positive (primary class)
                                                alt.value('#ff7f0e')   # Orange for negative (complementary class)
                                            ),
                                            tooltip=['Event:N', 'Log_Probability_Ratio:Q', 'Rank:O']
                                        ).properties(
                                            width=600,
                                            height=max(300, len(top_features) * 20),  # Dynamic height
                                            title=""
                                        )
                                        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Feature Discriminative Power (Log Probability Ratios)</h3>""", unsafe_allow_html=True, help="Positive values indicate the event is more likely in the second class, negative values indicate it's more likely in the first class. Higher absolute values indicate stronger discriminative power.")
                                        
                                        with st.container(border=True):
                                            st.altair_chart(chart, use_container_width=True)
                                
                                # Feature statistics by class
                                class_stats = feature_analysis.get('class_statistics', {})
                                if class_stats:
                                    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Class Statistics</h3>""", unsafe_allow_html=True, help="Class statistics for the Naive Bayes model.")
                                    
                                    # Display class-specific feature counts
                                    for class_name, stats in class_stats.items():
                                        
                                        with st.container(border=True):
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(f"{class_name} - Sample Count", stats.get('sample_count', 'N/A'))
                                            with col2:
                                                st.metric(f"{class_name} - Class Proportion", f"{stats.get('class_proportion', 0):.3f}")
                                            with col3:
                                                st.metric(f"{class_name} - Active Features", f"{stats.get('active_features', 'N/A')}/{stats.get('total_features', 'N/A')}" if 'active_features' in stats else 'N/A')
                        else:
                            st.info("Evaluation metrics not available", icon=":material/info:")
                    
                    else:
                        # Handle Snowpark ML model results
                        classifier_algorithm = results.get("classifier_algorithm", "Snowpark ML")
                        
                        if 'Naive Bayes' in classifier_algorithm:
                            # For Naive Bayes models, display the evaluation metrics from session state
                            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Snowpark ML Naive Bayes Model Evaluation Metrics</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                            st.write("")
                            evaluation_metrics = results.get('evaluation_metrics', {})
                            if evaluation_metrics:
                                # Display accuracy
                                accuracy = evaluation_metrics.get('accuracy', 'N/A')
                                with st.container(border=True):
                                    st.metric("Test Accuracy", f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy)
                                
                                # Display classification report
                                classification_report = evaluation_metrics.get('classification_report', {})
                                if classification_report:
                                    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Classification Report</h3>""", unsafe_allow_html=True)

                                    # Convert classification report to DataFrame for better display
                                    import pandas as pd
                                    report_df = pd.DataFrame(classification_report).transpose()
                                    with st.container(border=True):
                                        st.dataframe(report_df, use_container_width=True)
                                
                                # Display confusion matrix
                                confusion_matrix = evaluation_metrics.get('confusion_matrix', [])
                                if confusion_matrix:
                                    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Confusion Matrix</h3>""", unsafe_allow_html=True)

                                    import pandas as pd
                                    import numpy as np
                                    
                                    # Get class names from results
                                    classes = ['Class 0', 'Class 1']  # Default for binary classification
                                    cm_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
                                    with st.container(border=True):
                                        st.dataframe(cm_df, use_container_width=True)
                                
                                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Feature Analysis</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                                st.write("")
                                # Display feature information
                                feature_count = evaluation_metrics.get('feature_count', 'N/A')
                                training_samples = evaluation_metrics.get('training_samples', 'N/A')
                                test_samples = evaluation_metrics.get('test_samples', 'N/A')
                                
                                with st.container(border=True):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("TF-IDF Features", feature_count)
                                    with col2:
                                        st.metric("Training Samples", training_samples)
                                    with col3:
                                        st.metric("Test Samples", test_samples)
                                
                                # Display Naive Bayes Feature Analysis
                                feature_analysis = evaluation_metrics.get('feature_analysis', {})
                                if feature_analysis:
                    
                                    
                                    # Most discriminative features
                                    top_features = feature_analysis.get('top_discriminative_features', [])
                                    if top_features:
                                        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Most Discriminative Events</h3>""", unsafe_allow_html=True, help="Positive values indicate the event is more likely in the second class, negative values indicate it's more likely in the first class. Higher absolute values indicate stronger discriminative power.")

                                        # Create DataFrame for better display
                                        import pandas as pd
                                        features_df = pd.DataFrame(top_features)
                                        features_df.columns = ['Event', 'Log Probability Ratio', 'Interpretation']
                                        features_df.index = range(1, len(features_df) + 1)  # Start index from 1
                                        with st.container(border=True):
                                            st.dataframe(features_df, use_container_width=True)
                                        
                                        
                                        # Create a bar chart for all discriminative features
                                        if len(top_features) > 0:
                                            import altair as alt
                                            
                                            # Prepare data for chart
                                            chart_data = pd.DataFrame(top_features, columns=['Event', 'Log_Prob_Ratio', 'Interpretation'])
                                            
                                            # Create bar chart
                                            chart = alt.Chart(chart_data).mark_bar().encode(
                                                x=alt.X('Log_Prob_Ratio:Q', title='Log Probability Ratio'),
                                                y=alt.Y('Event:N', sort='-x', title='Customer Journey Events'),
                                                color=alt.Color('Log_Prob_Ratio:Q', 
                                                            scale=alt.Scale(scheme='redblue', reverse=True),
                                                            title='Discriminative Power'),
                                                tooltip=['Event:N', 'Log_Prob_Ratio:Q', 'Interpretation:N']
                                            ).properties(
                                                width=600,
                                                height=max(300, len(top_features) * 20),  # Dynamic height
                                                title=""
                                            )
                                            st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Feature Discriminative Power (Log Probability Ratios)</h3>""", unsafe_allow_html=True, help="Positive values indicate the event is more likely in the second class, negative values indicate it's more likely in the first class. Higher absolute values indicate stronger discriminative power.")
                                            
                                            with st.container(border=True):
                                                st.altair_chart(chart, use_container_width=True)
                                    else:
                                        st.info("Detailed discriminative features analysis not available for this model type.", icon=":material/info:")
                                    
                                    # Class statistics
                                    class_stats = feature_analysis.get('class_statistics', {})
                                    if class_stats:
                                        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Class Statistics</h3>""", unsafe_allow_html=True, help="Class statistics for the Naive Bayes model.")
                                        
                                        for class_name, stats in class_stats.items():
                                            
                                            with st.container(border=True):
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric(f"{class_name} - Sample Count", stats.get('sample_count', 'N/A'))
                                                with col2:
                                                    st.metric(f"{class_name} - Class Proportion", f"{stats.get('class_proportion', 0):.3f}")
                                                with col3:
                                                    st.metric(f"{class_name} - Active Features", f"{stats.get('active_features', 'N/A')}/{stats.get('total_features', 'N/A')}" if 'active_features' in stats else 'N/A')
                                else:
                                    st.info("Feature analysis not available for this model.", icon=":material/info:")
                            else:
                                st.info("Evaluation metrics not available", icon=":material/info:")
                else:
                    st.warning("Model name not found in session state", icon=":material/warning:")

                # AI Model Insights & Recommendations Section
                with st.expander("AI Model Insights & Recommendations", expanded=False, icon=":material/network_intel_node:"):
                    # Initialize session state for AI
                    if "ai_insights_generated" not in st.session_state:
                        st.session_state.ai_insights_generated = ""
                    
                    models_result = get_available_cortex_models()
                    available_models = models_result["models"]
                    status = models_result["status"]
                    
                    # Show status message if needed
                    if status == "not_found":
                        st.warning("No Cortex models found in your Snowflake account. Using default models.", icon=":material/warning:")
                    elif status == "error":
                        st.warning("Error accessing Cortex models. Using fallback models.", icon=":material/warning:")
                    elif status == "found":
                        st.success("Successfully loaded Cortex models from Snowflake", icon=":material/check:")
                    
                    # Model selection in 3-column layout
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if available_models:
                            # Use a session state key to track if we should prevent auto-triggering
                            if "ai_model_selection_active" not in st.session_state:
                                st.session_state["ai_model_selection_active"] = False
                            
                            selected_model = st.selectbox(
                                "Select AI Model", 
                                options=available_models,
                                index=0 if available_models else None,
                                key="ai_model_final",
                                on_change=lambda: st.session_state.update({"ai_model_selection_active": True})
                            )
                            # Reset the flag after selection
                            if st.session_state.get("ai_model_selection_active"):
                                st.session_state["ai_model_selection_active"] = False
                        else:
                            st.error("No models available", icon=":material/chat_error:")
                            selected_model = None
                    
                    # Use button instead of toggle to prevent auto-rerun
                    if st.button("Generate AI Insights", key="generate_ai_final", help="Click to generate AI analysis of your model", icon=":material/network_intel_node:"):
                        if selected_model:
                            # Get model metrics from results structure
                            results = st.session_state['predictive_results']
                            accuracy = results.get("model_accuracy", "Not available")
                            num_classes = results.get("num_classes", "Not available") 
                            num_samples = results.get("num_samples", "Not available")
                            classifier_algorithm = results.get("classifier_algorithm", "Naive Bayes")
                            model_name = results.get("model_name", "")
                            
                            # Create algorithm-specific prompt for AI analysis
                            if 'Naive Bayes' in classifier_algorithm:
                                # Get Naive Bayes specific data from session state
                                results = st.session_state.get('predictive_results', {})
                                evaluation_metrics = results.get('evaluation_metrics', {})
                                feature_analysis = evaluation_metrics.get('feature_analysis', {})
                                
                                # Format feature analysis data for AI
                                discriminative_features_data = ""
                                if feature_analysis:
                                    top_features = feature_analysis.get('top_discriminative_features', [])
                                    if top_features:
                                        discriminative_features_data = "TOP DISCRIMINATIVE EVENTS (Log Probability Ratios):\n"
                                        for event, ratio, interpretation in top_features[:15]:  # Show top 15 for AI
                                            discriminative_features_data += f"- {event}: {ratio} ({interpretation})\n"
                                    
                                    class_stats = feature_analysis.get('class_statistics', {})
                                    if class_stats:
                                        discriminative_features_data += "\nCLASS STATISTICS:\n"
                                        for class_name, stats in class_stats.items():
                                            discriminative_features_data += f"- {class_name}: {stats['active_features']}/{stats['total_features']} active features (density: {stats['feature_density']:.3f})\n"
                                
                                ai_prompt = f"""As an expert data scientist specializing in Naive Bayes classification, analyze this customer journey predictive model using the actual outputs:

            MODEL PERFORMANCE:
            - Accuracy: {accuracy}
            - Number of Classes: {num_classes}
            - Training Samples: {num_samples}
            - Model Type: {classifier_algorithm} on TF-IDF customer journey features
            - Algorithm: {classifier_algorithm} with Laplace smoothing - processes everything in Snowflake for maximum scalability
            - Feature Type: {'Binary (presence/absence)' if 'Bernoulli' in classifier_algorithm else 'Continuous (TF-IDF values)'}

            NAIVE BAYES FEATURE ANALYSIS:
            {discriminative_features_data}

            ANALYSIS REQUESTED (NAIVE BAYES SPECIFIC):
            1. PERFORMANCE EVALUATION: Analyze the classification metrics. How well does the Naive Bayes model perform on this customer journey data? Comment on precision, recall, and F1-scores for each class.

            2. DISCRIMINATIVE EVENTS ANALYSIS: Examine the log probability ratios for customer journey events:
            - Which events are most predictive of each class?
            - Are the discriminative events business-logical (e.g., do conversion events favor the converted class)?
            - Are there any surprising event patterns that warrant investigation?

            3. NAIVE BAYES ASSUMPTIONS: Assess how well the Naive Bayes assumptions hold:
            - Feature independence: Are customer journey events truly independent?
            - Class distribution: How balanced are the classes?
            - Feature sparsity: How does the sparse nature of TF-IDF features affect performance?

            4. CUSTOMER JOURNEY INSIGHTS: Based on the discriminative events, what can we learn about customer behavior:
            - What journey patterns lead to conversion vs. non-conversion?
            - Which touchpoints are most critical in the customer journey?
            - Are there early indicators of customer intent?

            5. IMPROVEMENT RECOMMENDATIONS: Provide specific strategies for this Naive Bayes customer journey model:
            - Journey feature engineering (sequence patterns, timing, frequency)
            - Data collection strategies for better journey tracking
            - Alternative probabilistic models that might capture journey dependencies better
            - Business process improvements based on discriminative events

            6. BUSINESS IMPLICATIONS: Translate findings into actionable business insights:
            - Which customer touchpoints should be optimized?
            - What early warning signals can be monitored?
            - How can this model be deployed for real-time customer journey optimization?

            Provide a comprehensive analysis focusing on the customer journey context and Naive Bayes-specific insights."""
                                            
                                try:
                                    with st.spinner("Generating AI insights..."):
                                        ai_result = session.sql(f"""
                                            SELECT SNOWFLAKE.CORTEX.COMPLETE('{selected_model}', $${ai_prompt}$$) as insights
                                        """).collect()
                                        
                                        if ai_result and len(ai_result) > 0:
                                            insights = ai_result[0]['INSIGHTS']
                                            st.session_state.ai_insights_generated = insights
                                        else:
                                            st.error("Failed to generate AI insights", icon=":material/chat_error:")
                                            
                                except Exception as e:
                                    st.error(f"Error generating AI insights: {str(e)}", icon=":material/chat_error:")
                        else:
                            st.warning("Please select an AI model to generate insights", icon=":material/warning:")
                    
                    # Display stored insights if available
                    if st.session_state.ai_insights_generated:
                        st.markdown("**AI Model Analysis & Recommendations**")
                        st.markdown(st.session_state.ai_insights_generated)
                        
                        # Add a clear button
                        if st.button("Clear AI Insights", key="clear_ai_final"):
                            st.session_state.ai_insights_generated = ""
                            st.rerun()
            else:
                # No results yet - show info message
                st.info("No results yet. Click the 'Create Model' button above to train and evaluate a model.", icon=":material/info:")
        
        with tab3:
            # Model Logging tab - only show if model was successfully created
            if 'predictive_results' in st.session_state and st.session_state['predictive_results'].get('completed'):
                results = st.session_state['predictive_results']
                trained_model = results.get('model')
                accuracy = results.get('accuracy')
                model_type = results.get('model_type', 'Unknown')
                is_sklearn = results.get('is_sklearn_model', False)
                
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;' title='Register your trained model in the Snowflake Model Registry for versioning, deployment, and tracking.'>Log Model to Snowflake Model Registry</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
                st.write("")
                
                # Clear status from previous runs when tab first loads
                if 'model_logging_status' in st.session_state and 'model_logging_just_completed' not in st.session_state:
                    del st.session_state['model_logging_status']
                
                # Model Information section
                st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Model Information</h3>""", unsafe_allow_html=True)
                with st.container(border=True):
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.markdown(f"<p style='font-size: 14px; margin-bottom: 0px; '>Model Name</p><p style='font-size: 18px; font-weight: normal; margin-top: 0px;'>{model_name}</p>", unsafe_allow_html=True)
                    with col_info2:
                        st.markdown(f"<p style='font-size: 14px; margin-bottom: 0px; '>Model Type</p><p style='font-size: 18px; font-weight: normal; margin-top: 0px;'>{model_type}</p>", unsafe_allow_html=True)
                    with col_info3:
                        st.markdown(f"<p style='font-size: 14px; margin-bottom: 0px; '>Accuracy</p><p style='font-size: 18px; font-weight: normal; margin-top: 0px;'>{f'{accuracy:.4f}' if accuracy else 'N/A'}</p>", unsafe_allow_html=True)
                
                # Registry configuration in container
                with st.container(border=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Get list of databases (cached)
                        databases_df = fetch_databases(session)
                        databases_list = databases_df['name'].tolist() if 'name' in databases_df.columns else databases_df[databases_df.columns[1]].tolist()
                        
                        registry_database = st.selectbox(
                            "Registry Database",
                            options=databases_list,
                            index=databases_list.index(database) if database in databases_list else 0,
                            help="Database where the model registry will be stored"
                        )
                    
                    with col2:
                        # Get schemas for selected database (cached)
                        schemas_df = fetch_schemas(session, registry_database)
                        schemas_list = schemas_df['name'].tolist() if 'name' in schemas_df.columns else schemas_df[schemas_df.columns[1]].tolist()
                        
                        registry_schema = st.selectbox(
                            "Registry Schema",
                            options=schemas_list,
                            index=schemas_list.index(schema) if schema in schemas_list else 0,
                            help="Schema where the model registry will be stored"
                        )
                    
                    with col3:
                        version_name = st.text_input(
                            "Version Name",
                            value="V1",
                            help="Version identifier for this model (e.g., V1, V2, PROD, etc.)"
                        )
                    
                    # Model comments
                    accuracy_str = f"{accuracy:.4f}" if accuracy else "N/A"
                    model_comments = st.text_area(
                        "Model Comments",
                        value=f"Model Type: {model_type}\nAccuracy: {accuracy_str}",
                        height=100,
                        help="Add comments about this model version"
                    )
                
                # Log model button (transparent)
                if st.button("Log Model to Registry", type="secondary"):
                    if trained_model and version_name:
                        try:
                            with st.spinner("Logging model to registry..."):
                                # Import registry
                                from snowflake.ml.registry import Registry
                                from snowflake.ml._internal.utils import identifier
                                
                                # Get unescaped names
                                reg_db = identifier._get_unescaped_name(registry_database)
                                reg_schema = identifier._get_unescaped_name(registry_schema)
                                
                                # Create registry
                                model_registry = Registry(
                                    session=session,
                                    database_name=reg_db,
                                    schema_name=reg_schema
                                )
                                
                                # Log the model with sample_input_data for sklearn models
                                if is_sklearn:
                                    # For sklearn models, create dummy sample input data
                                    import pandas as pd
                                    import numpy as np
                                    # Create dummy data with expected number of features
                                    n_features = trained_model.n_features_in_ if hasattr(trained_model, 'n_features_in_') else 10
                                    sample_input = pd.DataFrame(np.zeros((1, n_features)), columns=[f'feature_{i}' for i in range(n_features)])
                                    
                                    model_ver = model_registry.log_model(
                                        model_name=model_name,
                                        version_name=version_name,
                                        model=trained_model,
                                        sample_input_data=sample_input
                                    )
                                else:
                                    # For Snowpark ML models, no sample_input_data needed
                                    model_ver = model_registry.log_model(
                                        model_name=model_name,
                                        version_name=version_name,
                                        model=trained_model
                                    )
                                
                                # Set metrics if available
                                if accuracy:
                                    model_ver.set_metric(metric_name="accuracy", value=accuracy)
                                
                                # Add comments
                                if model_comments:
                                    model_ver.comment = model_comments
                                
                                # Get registered versions
                                versions_df = model_registry.get_model(model_name).show_versions()
                                
                                # Store success status in session state
                                st.session_state['model_logging_status'] = {
                                    'success': True,
                                    'message': f"Model '{model_name}' version '{version_name}' successfully logged to registry!",
                                    'versions_df': versions_df
                                }
                                st.session_state['model_logging_just_completed'] = True
                                st.rerun()
                                
                        except Exception as e:
                            # Store error status in session state
                            st.session_state['model_logging_status'] = {
                                'error': True,
                                'message': f"Failed to log model to registry: {str(e)}"
                            }
                            st.session_state['model_logging_just_completed'] = True
                            st.rerun()
                    else:
                        # Store warning status in session state
                        st.session_state['model_logging_status'] = {
                            'warning': True,
                            'message': "Please provide a version name to log the model."
                        }
                        st.session_state['model_logging_just_completed'] = True
                        st.rerun()
                
                # Show logging status if available (below button)
                if 'model_logging_status' in st.session_state and st.session_state['model_logging_status']:
                    status = st.session_state['model_logging_status']
                    if status.get('success'):
                        st.success(status['message'], icon=":material/check_circle:")
                    elif status.get('error'):
                        st.error(status['message'], icon=":material/error:")
                    elif status.get('warning'):
                        st.warning(status['message'], icon=":material/warning:")
                    
                    # Show registered versions if available
                    if status.get('success') and status.get('versions_df') is not None:
                        st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Registered Model Versions</h3>""", unsafe_allow_html=True)
                        with st.container(border=True):
                            st.dataframe(status['versions_df'], use_container_width=True)
                    
                    # Clear the flag after displaying
                    if 'model_logging_just_completed' in st.session_state:
                        del st.session_state['model_logging_just_completed']
            else:
                st.info("Train a model first to enable model logging.", icon=":material/info:")

# Show results tab if we have cached session state data and button wasn't clicked
elif inputs_ready and not run_model_button and 'predictive_results' in st.session_state and st.session_state['predictive_results'].get('completed'):
    # Show tabs with cached results
    tab1, tab2, tab3 = st.tabs(["Model Training", "Results", "Model Logging"])
    
    with tab1:
        st.info("Model training completed. Check the Results tab to view your model performance.", icon=":material/check:")
    
    with tab2:
        # Results tab - show results if available
        if 'predictive_results' in st.session_state and st.session_state['predictive_results'].get('completed'):
            
            # Get results from session state
            results = st.session_state['predictive_results']
            model_name = results.get('model_name', '')
            is_sklearn_model = results.get('is_sklearn_model', False)
            
            if model_name:
                st.info("Results are available above in the main Results tab.", icon=":material/info:")
        else:
            st.info("No results yet. Click the 'Create Model' button above to train and evaluate a model.", icon=":material/info:")
    
    with tab3:
        # Model Logging tab - only show if model was successfully created
        if 'predictive_results' in st.session_state and st.session_state['predictive_results'].get('completed'):
            results = st.session_state['predictive_results']
            trained_model = results.get('model')
            accuracy = results.get('accuracy')
            model_type = results.get('model_type', 'Unknown')
            model_name = results.get('model_name', '')
            is_sklearn = results.get('is_sklearn_model', False)
            
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;' title='Register your trained model in the Snowflake Model Registry for versioning, deployment, and tracking.'>Log Model to Snowflake Model Registry</h2><hr style='margin-top: -8px;margin-bottom: 10px;height: 3px; border: 0; background-color: #d1d5db;'>""", unsafe_allow_html=True)
            st.write("")
            
            # Clear status from previous runs when tab first loads
            if 'model_logging_status_cached' in st.session_state and 'model_logging_just_completed_cached' not in st.session_state:
                del st.session_state['model_logging_status_cached']
            
            # Model Information section
            st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Model Information</h3>""", unsafe_allow_html=True)
            with st.container(border=True):
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.markdown(f"<p style='font-size: 12px; margin-bottom: 0px; color: #888;'>Model Name</p><p style='font-size: 14px; font-weight: bold; margin-top: 0px;'>{model_name}</p>", unsafe_allow_html=True)
                with col_info2:
                    st.markdown(f"<p style='font-size: 12px; margin-bottom: 0px; color: #888;'>Model Type</p><p style='font-size: 14px; font-weight: bold; margin-top: 0px;'>{model_type}</p>", unsafe_allow_html=True)
                with col_info3:
                    st.markdown(f"<p style='font-size: 12px; margin-bottom: 0px; color: #888;'>Accuracy</p><p style='font-size: 14px; font-weight: bold; margin-top: 0px;'>{f'{accuracy:.4f}' if accuracy else 'N/A'}</p>", unsafe_allow_html=True)
            
            # Registry configuration in container
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Get list of databases (cached)
                    databases_df = fetch_databases(session)
                    databases_list = databases_df['name'].tolist() if 'name' in databases_df.columns else databases_df[databases_df.columns[1]].tolist()
                    
                    registry_database = st.selectbox(
                        "Registry Database",
                        options=databases_list,
                        index=databases_list.index(database) if database in databases_list else 0,
                        help="Database where the model registry will be stored",
                        key="registry_db_cached"
                    )
                
                with col2:
                    # Get schemas for selected database (cached)
                    schemas_df = fetch_schemas(session, registry_database)
                    schemas_list = schemas_df['name'].tolist() if 'name' in schemas_df.columns else schemas_df[schemas_df.columns[1]].tolist()
                    
                    registry_schema = st.selectbox(
                        "Registry Schema",
                        options=schemas_list,
                        index=schemas_list.index(schema) if schema in schemas_list else 0,
                        help="Schema where the model registry will be stored",
                        key="registry_schema_cached"
                    )
                
                with col3:
                    version_name = st.text_input(
                        "Version Name",
                        value="V1",
                        help="Version identifier for this model (e.g., V1, V2, PROD, etc.)",
                        key="version_name_cached"
                    )
                
                # Model comments
                accuracy_str = f"{accuracy:.4f}" if accuracy else "N/A"
                model_comments = st.text_area(
                    "Model Comments",
                    value=f"Model Type: {model_type}\nAccuracy: {accuracy_str}",
                    height=100,
                    help="Add comments about this model version",
                    key="model_comments_cached"
                )
            
            # Log model button (transparent)
            if st.button("Log Model to Registry", type="secondary", key="log_model_cached"):
                if trained_model and version_name:
                    try:
                        with st.spinner("Logging model to registry..."):
                            # Import registry
                            from snowflake.ml.registry import Registry
                            from snowflake.ml._internal.utils import identifier
                            
                            # Get unescaped names
                            reg_db = identifier._get_unescaped_name(registry_database)
                            reg_schema = identifier._get_unescaped_name(registry_schema)
                            
                            # Create registry
                            model_registry = Registry(
                                session=session,
                                database_name=reg_db,
                                schema_name=reg_schema
                            )
                            
                            # Log the model with sample_input_data for sklearn models
                            if is_sklearn:
                                # For sklearn models, create dummy sample input data
                                import pandas as pd
                                import numpy as np
                                # Create dummy data with expected number of features
                                n_features = trained_model.n_features_in_ if hasattr(trained_model, 'n_features_in_') else 10
                                sample_input = pd.DataFrame(np.zeros((1, n_features)), columns=[f'feature_{i}' for i in range(n_features)])
                                
                                model_ver = model_registry.log_model(
                                    model_name=model_name,
                                    version_name=version_name,
                                    model=trained_model,
                                    sample_input_data=sample_input
                                )
                            else:
                                # For Snowpark ML models, no sample_input_data needed
                                model_ver = model_registry.log_model(
                                    model_name=model_name,
                                    version_name=version_name,
                                    model=trained_model
                                )
                            
                            # Set metrics if available
                            if accuracy:
                                model_ver.set_metric(metric_name="accuracy", value=accuracy)
                            
                            # Add comments
                            if model_comments:
                                model_ver.comment = model_comments
                            
                            # Get registered versions
                            versions_df = model_registry.get_model(model_name).show_versions()
                            
                            # Store success status in session state
                            st.session_state['model_logging_status_cached'] = {
                                'success': True,
                                'message': f"Model '{model_name}' version '{version_name}' successfully logged to registry!",
                                'versions_df': versions_df
                            }
                            st.session_state['model_logging_just_completed_cached'] = True
                            st.rerun()
                            
                    except Exception as e:
                        # Store error status in session state
                        st.session_state['model_logging_status_cached'] = {
                            'error': True,
                            'message': f"Failed to log model to registry: {str(e)}"
                        }
                        st.session_state['model_logging_just_completed_cached'] = True
                        st.rerun()
                else:
                    # Store warning status in session state
                    st.session_state['model_logging_status_cached'] = {
                        'warning': True,
                        'message': "Please provide a version name to log the model."
                    }
                    st.session_state['model_logging_just_completed_cached'] = True
                    st.rerun()
            
            # Show logging status if available (below button)
            if 'model_logging_status_cached' in st.session_state and st.session_state['model_logging_status_cached']:
                status = st.session_state['model_logging_status_cached']
                if status.get('success'):
                    st.success(status['message'], icon=":material/check_circle:")
                elif status.get('error'):
                    st.error(status['message'], icon=":material/error:")
                elif status.get('warning'):
                    st.warning(status['message'], icon=":material/warning:")
                
                # Show registered versions if available
                if status.get('success') and status.get('versions_df') is not None:
                    st.markdown("""<h3 style='font-size: 14px; margin-bottom: 0px;'>Registered Model Versions</h3>""", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.dataframe(status['versions_df'], use_container_width=True)
                
                # Clear the flag after displaying
                if 'model_logging_just_completed_cached' in st.session_state:
                    del st.session_state['model_logging_just_completed_cached']
        else:
            st.info("Train a model first to enable model logging.", icon=":material/info:")
