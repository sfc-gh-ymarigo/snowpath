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
#ATTRIBUTION ANALYSIS
#--------------------------------------

st.sidebar.markdown("")

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
        ATTRIBUTION ANALYSIS
    </h5>
</div>
""", unsafe_allow_html=True) 
with st.expander("Input Parameters"):
        
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
            database = st.selectbox('Select Database', key='attribdb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
        if database:
            sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
            schemas = session.sql(sqlschemas).collect()
            schema0 = pd.DataFrame(schemas)
            
            with col2:
                schema = st.selectbox('Select Schema', key='attribsch', index=None, 
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
                tbl = st.selectbox('Select Event Table or View', key='attribtbl', index=None, 
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
                        sess1=f"{sess},"
                        groupby = f"group by {uid}, match_number,{sess} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
            elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        sess1=f"SESSION,"
                        groupby = f"group by {uid}, match_number,SESSION "
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
                
                with col1:
                    maxnbbevt = st.number_input("Max events preceding", value=5, min_value=1, placeholder="Type a number...",help="Select the maximum number of events preceding the 'conversion'.")
                with col2:
                    startdt_input = st.date_input('Start date', value=defstartdt1)
                with col3:
                    enddt_input = st.date_input('End date',help="Apply a time window to the data on a specific date range or over the entire lifespan of your data (default values)")
           
                
    #--------------------------------------
    #FILTERS
    #--------------------------------------

                #initialize sql_where_clause
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2 =st.columns([5,10])
                    with col1:
                # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0,placeholder="Select event(s)...",help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                          excl3 = "''"
                        else:
                         excl3= ', '.join([f"'{excl2}'" for excl2 in excl1])
                    with col2:
                        st.write("")
                
                #ADDITIONAL FILTERS
                col1, col2 = st.columns([5,10])
                with col1 :
                    checkfilters = st.toggle("Additional filters",help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                with col2:
                    st.write("")    
                    
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
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT']:
                                operator = st.selectbox(f"Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME','TIMESTAMP_NTZ']:
                                operator = st.selectbox(f"Operator", ['<=', '>=', '='], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox(f"Operator", ['=', '!=', 'IN'], key=operator_key)
                            return operator
                        
                        # Helper function to display value input based on column data type
                        def get_value_input(col_name, col_data_type, operator, filter_index):
                            """ Returns the value for filtering based on column type """
                            value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
                        
                            if operator == 'IN':
                                # For IN operator, allow multiple value selection
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.multiselect(f"Values for {col_name}", distinct_values, key=value_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME','TIMESTAMP_NTZ']:
                                # For date columns, let the user input a date value
                                value = st.date_input(f"Value for {col_name}", key=value_key)
                            else:
                                # For other operators, allow single value selection from distinct values
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key)
                        
                            return value
                        
                        # Initialize variables to store filters and logical conditions
                        filters = []
                        logic_operator = None
                        filter_index = 0  # A unique index to assign unique keys to each filter
                        
                        # Dynamic filter loop with one selectbox at a time
                        while True:
                            # Get the columns to select from 
                            available_columns = remaining_columns  # Allow columns to be reused for multiple filters
                                
                            if available_columns.empty:
                                st.write("No more columns available for filtering.")
                                break
                        
                            # Create 3 columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])  # Adjust width ratios for better layout
                        
                            with col1:
                                # Select a column to filter on
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns)
                        
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
                                #st.write(value)
                            # Append filter as a tuple (column, operator, value)
                            if operator and  (value is not None or value == 0):
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
                if all([uid, evt, tmstp]) and conv!= None and conv_value != 'Select...':
                 with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Modeling Technique</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2 =st.columns([5,10])
                    with col1:
                     model = st.selectbox(
                     "Rule Based, Markov Chain or both",
                         ["Rule Based", "Markov Chain","Rule Based & Markov Chain"],placeholder="Choose an option",index=None,key='modeltech'
            )
# SQL LOGIC
        
#partitionby = f"partition by {uid}"
sql_where_clause= ""

if all([uid, evt, tmstp]) and conv!= None and conv_value !="''":
    # Rule based models
    def rulebased ():
            if unitoftime ==None and timeout ==None:
                    
                attributionsql= f"""
                WITH ALLOTHERS AS (
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
                    from (select * from {database}.{schema}.{tbl} where {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                    match_recognize(
                        {partitionby}
                        order by {tmstp}
                        measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl
                        all rows per match
                        pattern(A{{0,{maxnbbevt}}} B)
                        define A as true, B AS {conv}='{conv_value}'
                    )
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
                        SELECT {uid}, {sess1} msq, MAX(MSQ) OVER ({partitionby}) AS max_msq
                        FROM (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        match_recognize(
                            {partitionby}
                            order by {tmstp}
                            measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl
                            all rows per match
                            pattern(A{{0,{maxnbbevt}}} B)
                            define A as true, B AS {conv}='{conv_value}'
                        )
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
                
                attributionsql= f"""
                WITH ALLOTHERS AS (
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
                        pattern(A{{0,{maxnbbevt}}} B)
                        define A as true, B AS {conv}='{conv_value}'
                    )
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
                        SELECT {uid}, {sess1} msq, MAX(MSQ) OVER ({partitionby}) AS max_msq
                        FROM (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                                {tmstp}) AS TIMEWINDOW FROM {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                                SELECT *FROM sessions) 
                        match_recognize(
                            {partitionby}
                            order by {tmstp}
                            measures match_number() as "MATCH_NUMBER", match_sequence_number() as msq, classifier() as cl
                            all rows per match
                            pattern(A{{0,{maxnbbevt}}} B)
                            define A as true, B AS {conv}='{conv_value}'
                        )
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
             
             crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
             select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
             from  (select * from {database}.{schema}.{tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                 match_recognize(
                 {partitionby} 
                 order by {tmstp} 
                 measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                 all rows per match
                 pattern(A{{0,{maxnbbevt}}} B)
                 define A as true, B AS {conv}='{conv_value}'
             ) WHERE cl != 'B' {groupby}) """
             
         elif unitoftime != None and timeout !=None :
    
            crttblrawsmceventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
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
                 pattern(A{{0,{maxnbbevt}}} B)
                 define A as true, B AS {conv}='{conv_value}'
             ) WHERE cl != 'B' {groupby}) """
             
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
                 measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                 all rows per match
                 pattern(A{{0,{maxnbbevt}}})
                 define A AS {conv} !='{conv_value}'
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
                 measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                 all rows per match
                 pattern(A{{0,{maxnbbevt}}})
                 define A AS {conv} !='{conv_value}'
             )  {groupby}) """
         # Run the SQL
         crttblrawsmceventscomp = session.sql(crttblrawsmceventscompsql).collect()
         markovinputsql = f"""select {uid}, 'start > ' || REPLACE(path, ',', ' > ') || ' > conv' AS paths from {unique_reftable_name}
         UNION All
         select {uid}, 'start > ' || REPLACE(path, ',', ' > ') || ' > null' AS paths from {unique_comptable_name}
         """
         markovinput = session.sql(markovinputsql).collect()
         markov=pd.DataFrame(markovinput)
        
         drptblrawmceventsrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
         drptblrawmceventsref = session.sql(drptblrawmceventsrefsql).collect()

         drptblrawmceventscompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
         drptblrawmceventscomp = session.sql(drptblrawmceventscompsql).collect() 
         def run_model(paths):
            #regex = re.compile('[^a-zA-Z>_ -]')
            regex = re.compile('[^a-zA-Z0-9>_ -]')
            paths.rename(columns={paths.columns[0]: "Paths"}, inplace=True)
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
            paths = np.array(paths).tolist()
            sublist = []
            total_paths = 0
            for path in paths:
                for touchpoint in path:
                    userpath = touchpoint.split(' > ')
                    sublist.append(userpath)
                total_paths += 1
            paths = sublist
        
            unique_touch_list = set(x for element in paths for x in element)
            # get total last touch conversion counts
            conv_dict = {}
            total_conversions = 0
            for item in unique_touch_list:
                conv_dict[item] = 0
            for path in paths:
                if 'conv' in path:
                    total_conversions += 1
                    conv_dict[path[-2]] += 1
        
            transitionStates = {}
            base_cvr = total_conversions / total_paths
            for x in unique_touch_list:
                for y in unique_touch_list:
                    transitionStates[x + ">" + y] = 0
        
            for possible_state in unique_touch_list:
                if possible_state != "null" and possible_state != "conv":
                    # print(possible_state)
                    for user_path in paths:
        
                        if possible_state in user_path:
                            indices = [i for i, s in enumerate(user_path) if possible_state == s]
                            for col in indices:
                                transitionStates[user_path[col] + ">" + user_path[col + 1]] += 1
        
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
             
        
         mc_paths_df = pd.DataFrame(markov['PATHS'], columns=['PATHS'])
         mcmodel = run_model(paths=mc_paths_df)
         return mcmodel
        
    if model== 'Rule Based & Markov Chain':
        # Function to compute models (only run when not cached)
        def compute_models():
            if 'dfattrib' not in st.session_state:
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
                # Cache results in session state
                st.session_state['dfattrib'] = dfattrib

                    
        # Track last-used table and conversion column
        last_tbl = st.session_state.get('last_tbl')
        last_conv = st.session_state.get('last_conv')
        last_excl3 = st.session_state.get('last_excl3')
        
        # If table or conversion column changed, invalidate cached dfattrib
        if (tbl != last_tbl) or (conv != last_conv) or (excl3 != last_excl3):
            if 'dfattrib' in st.session_state:
                del st.session_state['dfattrib']
        
        # Update session state with current selections
        st.session_state['last_tbl'] = tbl
        st.session_state['last_conv'] = conv
        st.session_state['last_excl3'] = excl3
        # Call computation only if not already cached
        compute_models()
        # Load results from session state
        dfattrib = st.session_state['dfattrib']
        #st.write (dfattrib)
        
        if st.toggle("Show me!", help="Detailed Scores Table, Attribution Models Summary Bar Chart and Models Bubble Charts"):

            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            st.write("")
            st.dataframe(dfattrib, use_container_width=True)

            if st.toggle("💾 Writeback the detailed scores table to Snowflake"):
            
                # Fetch DBs
                databases = session.sql("SHOW DATABASES").collect()
                db0 = pd.DataFrame(databases)
            
                # Row with DB / Schema / Table Name
                db_col, schema_col, tbl_col, btn_col = st.columns([1.5, 1.5, 1.5, 0.8])
            
                with db_col:
                    database = st.selectbox("Select Database", db0["name"].unique(), index=None, key="wb_db", placeholder="Choose...")
            
                schema = None
                if database:
                    schemas = session.sql(f"SHOW SCHEMAS IN DATABASE {database}").collect()
                    schema0 = pd.DataFrame(schemas)
            
                    with schema_col:
                        schema = st.selectbox("Select Schema", schema0["name"].unique(), index=None, key="wb_schema", placeholder="Choose...")
            
                table_name = None
                if database and schema:
                    with tbl_col:
                        table_name = st.text_input("Enter Table Name", key="wb_tbl", placeholder="e.g. scores_table")
            
                # Show button inline only if valid
                success = False
                if database and schema and table_name:
                    with btn_col:
                        if st.button("✅ Write Table"):
                            cleaned_df = dfattrib.copy()
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
                                st.error(f"❌ Write failed: {e}")

                # Show success below full row
                if success:
                    st.success(f"✅ Successfully wrote to `{database}.{schema}.{table_name}`")

    
            
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
                fig2.add_trace(go.Bar(
                    x=dfattrib[evt],  # Events on x-axis
                    y=dfattrib[model],  # Model values on y-axis
                    name=model,
                    marker_color=model_colors[model],  # Use consistent color by model
                    text=dfattrib[model].round(2),  # Display value on top of the bar
                    textposition='outside',  # Place text above the bar
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
                    'text': "",
                    'font': {
                        'size': 16,
                        'color': 'black',
                        'family': 'Arial'
                    },
                    'x': 0.5  # Center the title
                },
                xaxis=dict(
                    title="Event",
                    tickmode="array",
                    tickvals=list(range(len(dfattrib[evt]))),
                    ticktext=dfattrib[evt]
                ),
                yaxis=dict(
                    title="Attribution Score",
                    showgrid=True,
                    zeroline=True
                ),
                barmode='group',  # Grouped bars
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                height=500,
                legend=dict(
                    title="Model",
                    orientation="h",
                    yanchor="top",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                )
            )
            # Display chart below the table
            with st.container():
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Models Summary</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
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
            
            unique_events = dfattrib[evt].unique()
            
            # Reset color map if table, conversion column, or unique events change
            if (
                'events_color_map' not in st.session_state
                or st.session_state.get('events_color_map_evt') != evt
                or st.session_state.get('events_color_map_tbl') != tbl
                or st.session_state.get('events_color_map_conv') != conv
                or set(st.session_state['events_color_map'].keys()) != set(unique_events)
            ):
                st.session_state['events_color_map'] = {event: random_color() for event in unique_events}
                st.session_state['events_color_map_evt'] = evt
                st.session_state['events_color_map_tbl'] = tbl
                st.session_state['events_color_map_conv'] = conv
            
            dfattrib['color'] = dfattrib[evt].map(st.session_state['events_color_map'])
            # Define y-axis column dynamically based on selected time unit
            y_col = f"Time To Conversion ({timetoconversionunit})"
            # ✅ Function to create bubble chart (DRY principle)
            def create_bubble_chart(x_col, y_col, title):
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[evt] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': title,
                        'font': {'size': 12, 'color': 'black', 'family': 'Arial'}
                    }
                )
                return fig
            
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: -5px;'>
                """, unsafe_allow_html=True)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)  
                    
                with col2:
                    st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)      
                    st.plotly_chart(create_bubble_chart('Markov Conversions', y_col, "Markov Conversions"), use_container_width=True)
        
        if st.toggle ("Expl**AI**n Me!",help="Explain Attribution and derive insights by leveraging Cortex AI and the interpretive and generative power of LLMs"):

            def attrib_row_to_text(row):
        
                return (
                f"{row[conv]}: Count={row['Count']}, "
                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                f"Markov Conversions={row.get('Markov Conversions', 'N/A')}, "
                f"Time To Conversion (Day)={row['Time To Conversion (Day)']:.2f}",
                f"Time To Conversion (Hour)={row['Time To Conversion (Hour)']:.2f}",
                f"Time To Conversion (Min)={row['Time To Conversion (Min)']:.2f}",
                f"Time To Conversion (Sec)={row['Time To Conversion (Sec)']:.2f}"
            )

            dfattrib['ATTRIB_TEXT'] = dfattrib.apply(attrib_row_to_text, axis=1)
            attrib_text_df = dfattrib[['ATTRIB_TEXT']]
            
            from snowflake.snowpark.functions import col, lit, call_function

            prompt = st.text_input("Prompt your question about attribution", key="aiattribprompt")
            if prompt and prompt != "None":
                try:
                    # Step 1: Create a Snowpark DataFrame from the attribution text
                    attrib_snowpark_df = session.create_dataframe(attrib_text_df)
            
                    # Step 2: Apply the AI aggregation directly
                    ai_result_df = attrib_snowpark_df.select(
                        call_function("AI_AGG", col("ATTRIB_TEXT"), lit(prompt)).alias("AI_RESPONSE")
                    )
            
                    # Step 3: Collect and display result
                    explain = ai_result_df.to_pandas()
            
                    message = st.chat_message("assistant")
                    if not explain.empty:
                        formatted_output = explain.iloc[0, 0]
                        message.markdown(f"**AI Response:**\n\n{formatted_output}")
                    else:
                        message.markdown("No results found for the given prompt.")
            
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif model== 'Markov Chain':
        mcmodel = markovchain()
        markov_df = pd.DataFrame.from_dict(mcmodel['markov_conversions'], orient='index', columns=['Markov Conversions'])
        markov_df.index.name = conv  # Set index name to match the key in dfattrib
        markov_df.reset_index(inplace=True)  # Reset index to make conversion column a column
        
        st.dataframe(markov_df, use_container_width=True)
        
        
    elif model== 'Rule Based':
        # Function to compute rule-based model and store in session state
        def compute_rulebased():
            if 'dfattrib_rulebased' not in st.session_state:
                attribution = rulebased()
                dfattrib = pd.DataFrame(attribution)
                st.session_state['dfattrib_rulebased'] = dfattrib
        # Track last-used table and conversion column
        last_tbl = st.session_state.get('last_tbl')
        last_conv = st.session_state.get('last_conv')
        
        # If table or conversion column changed, invalidate cached dfattrib
        if (tbl != last_tbl) or (conv != last_conv):
            if 'dfattrib' in st.session_state:
                del st.session_state['dfattrib']
        
        # Update session state with current selections
        st.session_state['last_tbl'] = tbl
        st.session_state['last_conv'] = conv
        # Call computation only if not already cached
        compute_rulebased()
        # Load cached results
        dfattrib = st.session_state['dfattrib_rulebased']
        #dfattrib = dfattrib.drop(columns=['color'])
        if st.toggle("Show me!", help="Detailed Scores Table, Attribution Models Summary Bar Chart and Models Bubble Charts"):
            # Display Scores Table
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Detailed Scores Table</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
            st.write("")
            st.dataframe(dfattrib, use_container_width=True)
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
                unique_events = dfattrib[evt].unique()
                st.session_state['events_color_map_rulebased'] = {event: random_color() for event in unique_events}
            # Map colors
            dfattrib['color'] = dfattrib[evt].map(st.session_state['events_color_map_rulebased'])
            # Define y-axis column dynamically based on selected time unit
            y_col = f"Time To Conversion ({timetoconversionunit})"
            # ✅ Function to create bubble chart (DRY principle)
            def create_bubble_chart(x_col, y_col, title):
                fig = go.Figure(data=[go.Scatter(
                    x=dfattrib[x_col],
                    y=dfattrib[y_col],
                    text=dfattrib[evt] + "<br>Count: " + dfattrib['Count'].astype(str),
                    mode='markers',
                    marker=dict(
                        size=dfattrib['Count'],
                        color=dfattrib['color'],
                        sizemode='area',
                        sizeref=1.*max(dfattrib['Count'])/(40.**2)
                    )
                )])
                fig.update_layout(
                    xaxis_title="Attribution Score",
                    yaxis_title="Time To Conversion",
                    title={
                        'text': title,
                        'font': {'size': 14, 'color': 'black', 'family': 'Arial'}
                    }
                )
                return fig
            
            unique_events = dfattrib[evt].unique()
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
                fig.add_trace(go.Bar(
                    x=dfattrib[evt],  # Events on x-axis
                    y=dfattrib[model],  # Model values on y-axis
                    name=model,
                    marker_color=model_colors[model],  # Use consistent color by model
                    text=dfattrib[model].round(2),  # Display value on top of the bar
                    textposition='outside',  # Place text above the bar
                ))
            # Add vertical separators between events
            for i in range(len(dfattrib[evt]) - 1):
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
                    title="Event",
                    tickmode="array",
                    tickvals=list(range(len(dfattrib[evt]))),
                    ticktext=dfattrib[evt]
                ),
                yaxis=dict(
                    title="Attribution Score",
                    showgrid=True,
                    zeroline=True
                ),
                barmode='group',  # Grouped bars
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                height=500,
                legend=dict(
                    title="Model",
                    orientation="h",
                    yanchor="top",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                )
            )
            # Display chart below the table
            with st.container():
                st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Attribution Models Summary</h2>
                <hr style='margin-top: -8px;margin-bottom: 10px;'>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Bubble Charts</h2>
                <hr style='margin-top: -8px;margin-bottom: -5px;'>
                """, unsafe_allow_html=True)
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_bubble_chart('Uniform', y_col, "Uniform Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('Exponential Decay', y_col, "Exponential Decay Model"), use_container_width=True)
                with col2:
                    st.plotly_chart(create_bubble_chart('First Click', y_col, "First Click Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('Last Click', y_col, "Last Click Model"), use_container_width=True)
                    st.plotly_chart(create_bubble_chart('U Shape', y_col, "U Shape Model"), use_container_width=True)
        
        if st.toggle ("Expl**AI**n Me!",help="Explain Attribution and derive insights by leveraging Cortex AI and the interpretive and generative power of LLMs"):

            def attrib_row_to_text(row):
        
                return (
                f"{row[conv]}: Count={row['Count']}, "
                f"Last Click={row['Last Click']}, First Click={row['First Click']}, "
                f"Uniform={row['Uniform']}, Exponential Decay={row['Exponential Decay']}, "
                f"Time To Conversion (Day)={row['Time To Conversion (Day)']:.2f}",
                f"Time To Conversion (Hour)={row['Time To Conversion (Hour)']:.2f}",
                f"Time To Conversion (Min)={row['Time To Conversion (Min)']:.2f}",
                f"Time To Conversion (Sec)={row['Time To Conversion (Sec)']:.2f}"
            )

            dfattrib['ATTRIB_TEXT'] = dfattrib.apply(attrib_row_to_text, axis=1)
            attrib_text_df = dfattrib[['ATTRIB_TEXT']]
            
            from snowflake.snowpark.functions import col, lit, call_function

            prompt = st.text_input("Prompt your question about attribution", key="aiattribprompt")
            if prompt and prompt != "None":
                try:
                    # Step 1: Create a Snowpark DataFrame from the attribution text
                    attrib_snowpark_df = session.create_dataframe(attrib_text_df)
            
                    # Step 2: Apply the AI aggregation directly
                    ai_result_df = attrib_snowpark_df.select(
                        call_function("AI_AGG", col("ATTRIB_TEXT"), lit(prompt)).alias("AI_RESPONSE")
                    )
            
                    # Step 3: Collect and display result
                    explain = ai_result_df.to_pandas()
            
                    message = st.chat_message("assistant")
                    if not explain.empty:
                        formatted_output = explain.iloc[0, 0]
                        message.markdown(f"**AI Response:**\n\n{formatted_output}")
                    else:
                        message.markdown("No results found for the given prompt.")
            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
    else:
        st.markdown("""
            <style>
            .custom-container-1 {
                padding: 10px 10px 10px 10px;
                border-radius: 10px;
                background-color: #f0f2f6;  /* Light blue background f0f8ff */
                border: none;  /* Blue border */
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200 ; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
                    Please select one or more modeling technique
                </h5>
            </div>
            """, unsafe_allow_html=True)
        #st.warning('Please select one or more modeling technique')
        
else:   
        st.markdown("""
            <style>
            .custom-container-1 {
                padding: 10px 10px 10px 10px;
                border-radius: 10px;
                background-color: #f0f2f6;  /* Light blue background f0f8ff */
                border: none;  /* Blue border */
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200 ; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
                    Please ensure all required inputs are selected before running the app.
                </h5>
            </div>
            """, unsafe_allow_html=True)
        #st.warning("Please ensure all required inputs are selected before running the app.")

        
 

