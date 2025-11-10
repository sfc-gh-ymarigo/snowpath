/*
 * ===========================================================================
 * Snowpath™ - Attribution Analysis Stored Procedures Setup
 * ===========================================================================
 * 
 * Purpose: Pre-create stored procedures to eliminate 90-second overhead
 *          on first run of attribution analysis
 * 
 * Execution Time: ~2-3 minutes (one-time setup)
 * Time Saved: ~90 seconds per session
 * 
 * Instructions:
 * 1. Run this script ONCE in your Snowflake worksheet
 * 2. Procedures will be available for all future Snowpath sessions
 * 3. Re-run only when procedure logic needs updating
 * 
 * ===========================================================================
 */

-- Set context (adjust as needed for your environment)
USE ROLE SYSADMIN;  -- Or your preferred role
USE WAREHOUSE COMPUTE_WH;  -- Or your preferred warehouse

SELECT 'Starting Snowpath stored procedures setup...' AS status;

-- ===========================================================================
-- PROCEDURE 1: MARKOV CHAIN ATTRIBUTION
-- ===========================================================================
-- Calculates Markov Chain attribution using transition matrices
-- Runtime: ~5-15 seconds for typical datasets
-- Memory: Moderate (scales with number of unique touchpoints)

CREATE OR REPLACE PROCEDURE MARKOV_ATTRIBUTION_SP(
    paths_table STRING,
    path_column STRING,
    frequency_column STRING
)
RETURNS TABLE(channel STRING, attribution_pct FLOAT, removal_effect FLOAT, conversions FLOAT)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run_markov_attribution'
COMMENT = 'Snowpath™ - Markov Chain Attribution (optimized server-side computation)'
AS
$$
import numpy as np
import pandas as pd
from collections import defaultdict
import re

def run_markov_attribution(session, paths_table, path_column, frequency_column):
    # Load paths from table
    query = f"SELECT {path_column}, {frequency_column} FROM {paths_table}"
    df = session.sql(query).to_pandas()
    df.columns = ['path', 'frequency']
    
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
    
    channels = [tp for tp in all_touchpoints if tp not in ['start', 'conv', 'null']]
    
    if not channels:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        empty_result = pd.DataFrame([{'CHANNEL': 'No channels', 'ATTRIBUTION_PCT': 0.0, 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0}])
        return session.create_dataframe(empty_result, schema=schema)
    
    def build_transition_matrix(paths, frequencies, exclude_channel=None):
        unique_touch_list = set()
        for path in paths:
            unique_touch_list.update(path)
        
        filtered_paths = []
        for path in paths:
            if exclude_channel:
                filtered_path = [tp for tp in path if tp != exclude_channel]
                filtered_paths.append(filtered_path)
            else:
                filtered_paths.append(path)
        
        if exclude_channel and exclude_channel in unique_touch_list:
            unique_touch_list.remove(exclude_channel)
        
        transitionStates = {}
        for x in unique_touch_list:
            for y in unique_touch_list:
                transitionStates[x + ">" + y] = 0
        
        for possible_state in unique_touch_list:
            if possible_state != "null" and possible_state != "conv":
                for i, user_path in enumerate(filtered_paths):
                    freq = frequencies[i]
                    if possible_state in user_path:
                        indices = [j for j, s in enumerate(user_path) if possible_state == s]
                        for col in indices:
                            if col + 1 < len(user_path):
                                transitionStates[user_path[col] + ">" + user_path[col + 1]] += freq
        
        actual_paths = []
        for state in unique_touch_list:
            if state != "null" and state != "conv":
                counter = 0
                state_transitions = {k: v for k, v in transitionStates.items() if k.startswith(state + '>')}
                counter = sum(state_transitions.values())
                
                if counter > 0:
                    for trans, count in state_transitions.items():
                        if count > 0:
                            state_prob = float(count) / float(counter)
                            actual_paths.append({trans: state_prob})
        
        transState = []
        transMatrix = []
        for item in actual_paths:
            for key in item:
                transState.append(key)
                transMatrix.append(item[key])
        
        if not transState:
            return None, None, None
            
        tmatrix_df = pd.DataFrame({'paths': transState, 'prob': transMatrix})
        tmatrix_split = tmatrix_df['paths'].str.split('>', expand=True)
        tmatrix_df['channel0'] = tmatrix_split[0]
        tmatrix_df['channel1'] = tmatrix_split[1]
        
        test_df = pd.DataFrame(0.0, index=list(unique_touch_list), columns=list(unique_touch_list))
        
        for _, v in tmatrix_df.iterrows():
            x = v['channel0']
            y = v['channel1']
            val = v['prob']
            test_df.loc[x, y] = val
        
        test_df.loc['conv', 'conv'] = 1.0
        test_df.loc['null', 'null'] = 1.0
        
        return test_df, unique_touch_list, None
    
    def calculate_conversion_rate(test_df):
        R = test_df[['null', 'conv']]
        R = R.drop(['null', 'conv'], axis=0)
        Q = test_df.drop(['null', 'conv'], axis=1)
        Q = Q.drop(['null', 'conv'], axis=0)
        
        t = len(Q.columns)
        if t == 0:
            return 0.0
        
        try:
            N = np.linalg.inv(np.identity(t) - np.asarray(Q))
            M = np.dot(N, np.asarray(R))
            base_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
            return base_cvr
        except:
            return 0.0
    
    def calculate_removals(df, base_cvr):
        removal_effect_list = dict()
        channels_to_remove = [col for col in df.columns if col not in ['conv', 'null', 'start']]
        
        for channel in channels_to_remove:
            removal_df = df.drop(channel, axis=1)
            removal_df = removal_df.drop(channel, axis=0)
            
            for col in removal_df.columns:
                if col not in ['null', 'conv']:
                    one = float(1)
                    row_sum = np.sum(list(removal_df.loc[col]))
                    null_percent = one - row_sum
                    if null_percent != 0:
                        removal_df.loc[col, 'null'] = null_percent
            
            removal_df.loc['null', 'null'] = 1.0
            
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
    
    test_df, unique_touch_list, _ = build_transition_matrix(all_paths, all_frequencies)
    
    if test_df is None:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        equal_share = 100.0 / len(channels)
        equal_result = pd.DataFrame([{'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(equal_share), 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0} for ch in channels])
        return session.create_dataframe(equal_result, schema=schema)
    
    base_cvr = calculate_conversion_rate(test_df)
    
    if base_cvr == 0:
        from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
        schema = StructType([
            StructField("CHANNEL", StringType()),
            StructField("ATTRIBUTION_PCT", FloatType()),
            StructField("REMOVAL_EFFECT", FloatType()),
            StructField("CONVERSIONS", FloatType())
        ])
        equal_share = 100.0 / len(channels)
        equal_result = pd.DataFrame([{'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(equal_share), 'REMOVAL_EFFECT': 0.0, 'CONVERSIONS': 0.0} for ch in channels])
        return session.create_dataframe(equal_result, schema=schema)
    
    removal_effects = calculate_removals(test_df, base_cvr)
    denominator = np.sum(list(removal_effects.values()))
    total_conversions = sum(all_frequencies)
    
    if denominator > 0:
        attribution_pcts = {ch: (removal_effects[ch] / denominator) * 100 for ch in channels}
        conversions = {ch: (removal_effects[ch] / denominator) * total_conversions for ch in channels}
    else:
        equal_share = 100.0 / len(channels)
        attribution_pcts = {ch: equal_share for ch in channels}
        conversions = {ch: (equal_share / 100) * total_conversions for ch in channels}
    
    result = pd.DataFrame([
        {'CHANNEL': str(ch), 'ATTRIBUTION_PCT': float(attribution_pcts[ch]), 'REMOVAL_EFFECT': float(removal_effects.get(ch, 0)), 'CONVERSIONS': float(conversions[ch])}
        for ch in channels
    ])
    result_sorted = result.sort_values('ATTRIBUTION_PCT', ascending=False)
    
    from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
    schema = StructType([
        StructField("CHANNEL", StringType()),
        StructField("ATTRIBUTION_PCT", FloatType()),
        StructField("REMOVAL_EFFECT", FloatType()),
        StructField("CONVERSIONS", FloatType())
    ])
    return session.create_dataframe(result_sorted, schema=schema)
$$;

SELECT '✓ Markov Chain Attribution procedure created successfully' AS status;

-- ===========================================================================
-- PROCEDURE 2: SHAPLEY VALUE ATTRIBUTION
-- ===========================================================================
-- Calculates Shapley Value attribution with Monte Carlo sampling
-- Runtime: ~10-30 seconds depending on n_samples parameter
-- Memory: Moderate (scales with number of unique touchpoints and samples)

CREATE OR REPLACE PROCEDURE SHAPLEY_ATTRIBUTION_SP(
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
COMMENT = 'Snowpath™ - Shapley Value Attribution (frequency-weighted Monte Carlo sampling)'
AS
$$
import numpy as np
import pandas as pd

def run_shapley_attribution(session, paths_table, path_column, frequency_column, n_samples):
    # Load paths from table
    query = f"SELECT {path_column}, {frequency_column} FROM {paths_table}"
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
        empty_result = pd.DataFrame([{'CHANNEL': 'No channels', 'SHAPLEY_VALUE': 0.0, 'ATTRIBUTION_PCT': 0.0, 'CONVERSIONS': 0.0}])
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
    shapley_values = {ch: 0.0 for ch in channels}
    for _ in range(n_samples):
        permutation = np.random.permutation(channels)
        for i, channel in enumerate(permutation):
            coalition_without = set(permutation[:i])
            coalition_with = coalition_without | {channel}
            value_without = coalition_value(coalition_without, path_data)
            value_with = coalition_value(coalition_with, path_data)
            marginal = value_with - value_without
            shapley_values[channel] += marginal
    
    # Average across samples
    for ch in shapley_values:
        shapley_values[ch] /= n_samples
    
    # Normalize to percentages
    abs_values = {ch: abs(val) for ch, val in shapley_values.items()}
    total_abs = sum(abs_values.values())
    if total_abs > 0:
        attribution_pcts = {ch: (abs_values[ch] / total_abs) * 100 for ch in channels}
    else:
        equal_share = 100.0 / len(channels)
        attribution_pcts = {ch: equal_share for ch in channels}
    
    # Calculate conversions
    total_conversions = sum(freq for _, freq in path_data)
    conversions = {ch: (pct / 100) * total_conversions for ch, pct in attribution_pcts.items()}
    
    # Create result DataFrame
    result = pd.DataFrame([
        {'CHANNEL': str(ch), 'SHAPLEY_VALUE': float(shapley_values[ch]), 'ATTRIBUTION_PCT': float(attribution_pcts[ch]), 'CONVERSIONS': float(conversions[ch])}
        for ch in channels
    ])
    result_sorted = result.sort_values('ATTRIBUTION_PCT', ascending=False)
    
    from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
    schema = StructType([
        StructField("CHANNEL", StringType()),
        StructField("SHAPLEY_VALUE", FloatType()),
        StructField("ATTRIBUTION_PCT", FloatType()),
        StructField("CONVERSIONS", FloatType())
    ])
    return session.create_dataframe(result_sorted, schema=schema)
$$;

SELECT '✓ Shapley Value Attribution procedure created successfully' AS status;

-- ===========================================================================
-- VERIFICATION
-- ===========================================================================

-- Verify both procedures exist
SHOW PROCEDURES LIKE 'MARKOV_ATTRIBUTION_SP';
SHOW PROCEDURES LIKE 'SHAPLEY_ATTRIBUTION_SP';

-- Display completion message
SELECT '══════════════════════════════════════════════════════════' AS message
UNION ALL
SELECT '  Snowpath™ Stored Procedures Setup Complete!' AS message
UNION ALL
SELECT '══════════════════════════════════════════════════════════' AS message
UNION ALL
SELECT '' AS message
UNION ALL
SELECT '✓ MARKOV_ATTRIBUTION_SP   - Ready for use' AS message
UNION ALL
SELECT '✓ SHAPLEY_ATTRIBUTION_SP  - Ready for use' AS message
UNION ALL
SELECT '' AS message
UNION ALL
SELECT 'Time saved per session: ~90 seconds' AS message
UNION ALL
SELECT 'Procedures are now cached and ready for immediate use' AS message
UNION ALL
SELECT '' AS message
UNION ALL
SELECT '══════════════════════════════════════════════════════════' AS message;
