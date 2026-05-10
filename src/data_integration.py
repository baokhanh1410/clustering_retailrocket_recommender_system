import duckdb

def load_data(path):
    return duckdb.read_csv(path)

def merge_data(data):
    events = data[0]
    item_properties_1 = data[1]
    item_properties_2 = data[2]
    category_tree = data[3]

    has_categoryid = 'categoryid' in category_tree.columns

    if has_categoryid:
        query = """
            WITH item_categories AS (
                SELECT 
                    CAST(itemid AS VARCHAR) AS itemid, 
                    CAST(value AS VARCHAR) AS categoryid
                FROM (
                    SELECT * FROM item_properties_1
                    UNION ALL 
                    SELECT * FROM item_properties_2
                )
                WHERE property = 'categoryid'
                QUALIFY ROW_NUMBER() OVER(PARTITION BY itemid ORDER BY timestamp DESC) = 1
            )
            SELECT 
                e.*,
                c.categoryid,
                t.* EXCLUDE (categoryid) -- Tránh cảnh báo trùng lặp cột khi join
            FROM events e
            LEFT JOIN item_categories c 
                ON CAST(e.itemid AS VARCHAR) = c.itemid
            LEFT JOIN category_tree t 
                ON c.categoryid = CAST(t.categoryid AS VARCHAR)
        """
    else:
        query = """
            WITH item_categories AS (
                SELECT 
                    CAST(itemid AS VARCHAR) AS itemid, 
                    CAST(value AS VARCHAR) AS categoryid
                FROM (
                    SELECT * FROM item_properties_1
                    UNION ALL 
                    SELECT * FROM item_properties_2
                )
                WHERE property = 'categoryid'
                QUALIFY ROW_NUMBER() OVER(PARTITION BY itemid ORDER BY timestamp DESC) = 1
            )
            SELECT 
                e.*,
                c.categoryid
            FROM events e
            LEFT JOIN item_categories c 
                ON CAST(e.itemid AS VARCHAR) = c.itemid
        """
    return duckdb.sql(query).df()