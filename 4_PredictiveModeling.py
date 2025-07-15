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
from snowflake.snowpark import Session#
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast
from streamlit_extras.app_logo import add_logo

# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

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
#PREDICTION MODELING
#--------------------------------------

st.sidebar.markdown("")
#Page Title
st.markdown("""
<style>
.custom-container-1 {
    padding: 10px 10px 10px 10px;
    border-radius: 10px;
    background-color: #f0f2f6;  /* Light blue background */
    border: 1px solid #29B5E8;  /* Blue border */
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="custom-container-1">
    <h5 style="font-size: 18px; font-weight: normal; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
        PREDICTIVE MODELING
    </h5>
</div>
""", unsafe_allow_html=True)
# Get the current credentials
session = get_active_session()
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
#if model_created" not in st.session_state:
st.session_state["model_created"] = False
#if "created_model_name" not in st.session_state:
st.session_state["created_model_name"] = None
#if "test_model_created" not in st.session_state:
st.session_state["test_model_created"] = False
if "test_model_success" not in st.session_state:
    st.session_state["test_model_success"] = False
#if "unique_testtable_name" not in st.session_state:
st.session_state["unique_testtable_name"] = None

with st.expander("Input Parameters (Primary Class)"):
         
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
             database = st.selectbox('Select Database', key='predictrefdb', index=None, 
                                     placeholder="Choose from list...", options=db0['name'].unique())
         
         # **Schema Selection (Only if a database is selected)**
         if database:
             sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
             schemas = session.sql(sqlschemas).collect()
             schema0 = pd.DataFrame(schemas)
         
             with col2:
                 schema = st.selectbox('Select Schema', key='comparerefsch', index=None, 
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
                 tbl = st.selectbox('Select Event Table or View', key='comparereftbl', index=None, 
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

         col1, col2, col3 = st.columns([4,4,4])
         with col1:
             uid = st.selectbox('Select identifier column', colsdf, index=None,  key='uidpredictref',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
         with col2: 
             evt = st.selectbox('Select event column', colsdf, index=None, key='evtpredictref', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
         with col3:
             tmstp = st.selectbox('Select timestamp column', colsdf, index=None, key='tsmtppredictref',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
         
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

                     model_name=None
                     primary_label = None
                     complementary_label = None
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
             
                     # Continue with SQL generation and execution based on inputs...
             
                     # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
                     if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any' and primary_label.strip() and complementary_label.strip() and model_name.strip():
                         
                         crttblrawseventsrefsql= None
                         crttblrawseventsref = None
                         crttblrawseventscompsql = None
                         crttblrawseventscomp = None
                         # Initialiser l'état de la session
                         
                         if st.button("Create Model", key='createmodel'):
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
                             #st.success(f"✅ Reference Paths Table `{unique_reftable_name}` created.")
     
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
                             #st.success(f"✅ Complementary Paths Table `{unique_comptable_name}` created.")
     
     
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
                             #st.success(f"✅ TF-IDF Reference Table `{unique_reftftable_name}` created.")
     
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
                             #st.success(f"✅ TF-IDF Complementary Table `{unique_comptftable_name}` created.")
     
                         # Generate a unique full tfidf table name
                             def generate_unique_fulltftable_name(base_name="FULLTFIDF"):
                                 unique_refid = uuid.uuid4().hex  # Generate a random UUID
                                 return f"{base_name}_{unique_refid}"
     
                             unique_fulltftable_name = generate_unique_fulltftable_name()
                         
                             #CREATE TABLE TF-IDF Full (Primary+Complementary)
                             crttbltfidffullsql=f"""CREATE TABLE {unique_fulltftable_name} AS
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
                                     FROM {unique_comptftable_name}
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
                                     FROM {unique_reftftable_name}
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
                             #CREATE TRAINING TABLE
                             crttbltrainingsql=f"""CREATE TABLE {unique_trainingtable_name} AS (SELECT * exclude ({uid},seq), ABS(HASH({uid})) % 100 AS hash_value
                             from {unique_fulltftable_name} where hash_value <80)"""
                             crttbltraining = session.sql(crttbltrainingsql).collect()
                             #st.success(f"✅ Training Table `{unique_trainingtable_name}` created.")
                             
                             #CREATE MODEL
                             crtmodelsql= f"""CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION  {model_name} (INPUT_DATA => SYSTEM$REFERENCE ('TABLE','{unique_trainingtable_name}'),TARGET_COLNAME => 'LABEL')"""
                                      
                             crtmodel=session.sql(crtmodelsql).collect()
                             st.session_state["model_created"] = True  # Store model creation success
                             st.session_state["created_model_name"] = model_name
                             st.success(f"✅ Model `{model_name}` created successfully.")
                             
                 
                             #modelcreationsucces =st.success(f"✅ Model `{model_name}` created successfully.")
                             
                             drptblrawpredictrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
                             drptblrawpredictref = session.sql(drptblrawpredictrefsql).collect()
                             drptblrawpredictcompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
                             drptblrawpredictcomp = session.sql(drptblrawpredictcompsql).collect()
                             drptbltfpredictrefsql =f"""DROP TABLE IF EXISTS {unique_reftftable_name}"""
                             drptbltfpredictref = session.sql(drptbltfpredictrefsql).collect()
                             drptbltfpredictcompsql =f"""DROP TABLE IF EXISTS {unique_comptftable_name}"""
                             drptbltfpredictcomp = session.sql(drptbltfpredictcompsql).collect()
                             drptbltrainingsql =f"""DROP TABLE IF EXISTS {unique_trainingtable_name}"""
                             drptbltraining = session.sql(drptbltrainingsql).collect()
                                 
                                     # Drop full dataset table
                             drptbltfpredictfullsql = f"""DROP TABLE IF EXISTS {unique_fulltftable_name}"""
                             drptbltfpredictfull = session.sql(drptbltfpredictfullsql).collect()
                                 
                         #  Show "Inspect Model" button **ONLY** if "Test Model" ran successfully
                             if st.session_state.get("model_created", False):
                                 
                                 # Inspect model evaluation metrics
                                 showevalmetricssql = f"""CALL {model_name}!SHOW_EVALUATION_METRICS()"""
                                 showevalmetrics = session.sql(showevalmetricssql).collect()
                                 dfevalmetrics = pd.DataFrame(showevalmetrics)
                                 st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Model Evaluation Metrics</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                                 st.dataframe(dfevalmetrics, use_container_width=True)
                                 # Inspect confusion matrix
                                 showconfmatrixsql = f"""CALL {model_name}!SHOW_CONFUSION_MATRIX()"""
                                 showconfmatrix = session.sql(showconfmatrixsql).collect()
                                 dfconfmatrix = pd.DataFrame(showconfmatrix)
                                 st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Confusion Matrix</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                                 st.dataframe(dfconfmatrix, use_container_width=True)
                                 # Inspect feature importance
                                 showfeatimportancesql = f"""CALL {model_name}!SHOW_FEATURE_IMPORTANCE()"""
                                 showfeatimportance = session.sql(showfeatimportancesql).collect()
                                 dffeatimportance = pd.DataFrame(showfeatimportance)
                                                                        
                                 st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Model Feature Importance</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
                                 st.dataframe(dffeatimportance, use_container_width=True)