import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def merge_data(data):
    events = data[0]
    item_properties_1 = data[1]
    item_properties_2 = data[2]
    category_tree = data[3]

    item_properties = pd.concat([item_properties_1, item_properties_2], ignore_index=True)
    del item_properties_1, item_properties_2

    item_categories = item_properties[item_properties['property'] == 'categoryid'].copy()
    del item_properties

    item_categories = item_categories.sort_values('timestamp', ascending=False).drop_duplicates('itemid')
    item_categories.rename(columns={'value': 'categoryid'}, inplace=True)
    
    events['itemid'] = events['itemid'].astype(str)
    item_categories['itemid'] = item_categories['itemid'].astype(str)
    
    merged = pd.merge(events, item_categories[['itemid', 'categoryid']], on='itemid', how='left')
    del item_categories

    if 'categoryid' in category_tree.columns:
        category_tree['categoryid'] = category_tree['categoryid'].astype(str)
        merged['categoryid'] = merged['categoryid'].astype(str)
        
        final_merged = pd.merge(merged, category_tree, on='categoryid', how='left')
    else:
        final_merged = merged

    return final_merged