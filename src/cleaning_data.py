import pandas as pd
import duckdb

def clean_data(df_input):
    orig_count = duckdb.sql("SELECT COUNT(*) FROM df_input").fetchone()[0]
    print(f"Original shape: ({orig_count}, {df_input.shape[1]})")
    
    # Build SQL query for each column
    select_exprs = []
    for col in df_input.columns:
        if col == 'transactionid':
            # Equivalent to df['transactionid'].fillna(0)
            select_exprs.append(f"COALESCE({col}, 0) AS {col}")
            
        elif 'timestamp' in col.lower():
            # Equivalent to pd.to_datetime(df[col], unit='ms')
            select_exprs.append(f"epoch_ms({col}) AS {col}") 
            
        else:
            # Equivalent to df.fillna(-1) but safe for data types
            if pd.api.types.is_numeric_dtype(df_input[col]):
                select_exprs.append(f"COALESCE({col}, -1) AS {col}") # If column is numeric, fill with -1
            else:
                select_exprs.append(f"COALESCE({col}, '-1') AS {col}") # If column is string, fill with '-1'
                
    select_clause = ",\n        ".join(select_exprs)
    

    # The DISTINCT keyword works exactly like df.drop_duplicates(), 
    # it removes rows that have the same values.
    query = f"""
    SELECT DISTINCT
        {select_clause}
    FROM df_input
    """
    
    # Execute query
    cleaned_rel = duckdb.sql(query)
    cleaned_count = cleaned_rel.count('*').fetchone()[0]
    
    print(f"Cleaned shape: ({cleaned_count}, {df_input.shape[1]})")
    print("="*30)
    
    return cleaned_rel.df()