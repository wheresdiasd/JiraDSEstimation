#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[2]:


# Database & File IO
from pymongo import MongoClient
import json5 as json

# Standard Data Manipulation
from collections import defaultdict
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)  # We want to see all data
from statistics import mean, median

# Tracking Time
from time import time


# ### Load Data Sources

# In[ ]:


# Connect to the database
client = MongoClient("mongodb://admin:password@localhost:27017/")
db = client['JiraRepos']

# Load the Jira Data Sources JSON
with open('../0. DataDefinition/jira_data_sources.json') as f:
    jira_data_sources = json.load(f)

# Load the Jira Issue Types Information (Downloaded using the DataDownload script)
with open('../0. DataDefinition/jira_issuetype_information.json') as f:
    jira_issuetype_information = json.load(f)

# Load the Jira Issue Link Types Information (Downloaded using the DataDownload script)
with open('../0. DataDefinition/jira_issuelinktype_information.json') as f:
    jira_issuelinktype_information = json.load(f)

# Load the Jira Thematic Analysis JSON
# with open('./jira_issuetype_thematic_analysis.json') as f:
#     issuetype_themes_codes = json.load(f)


# ### Define Helpful Globals

# In[4]:


ALL_JIRAS = [jira_name for jira_name in jira_data_sources.keys()]


# ### Define Data Structures

# In[5]:


# These are the global dataframes that we will perform our analysis on.
df_jiras = pd.DataFrame(
    np.nan,
    columns=['Born', 'Issues', 'DIT', 'UIT', 'Links', 'DLT', 'ULT', 'Changes', 'Ch/I', 'UP', 'Comments', 'Co/I'],
    index=ALL_JIRAS + ['Sum', 'Median', 'Std Dev']
)


# ### Query Data for Stats

# 

# In[1]:


import json
import dtale
import pandas as pd
import random
import numpy as np
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model (you can choose a smaller one if needed)
desc_model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_date_str(x):
    """
    Parse a date string using dateparser. If the string is "Missing", empty, or cannot be parsed, return pd.NaT.
    """
    if pd.isnull(x):
        return pd.NaT
    s = str(x).strip()
    if s.lower() == "missing" or s == "":
        return pd.NaT
    
    return x

def convert_date_columns_dateparser(df, date_columns):
    """
    Convert the specified date columns from string to datetime using our custom parse_date_str.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame containing date strings.
      date_columns (list): List of column names to convert.
    
    Returns:
      pd.DataFrame: The DataFrame with specified columns converted to datetime objects.
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_date_str)
    return df

def drop_invalid_dates(df, date_columns):
    """
    Drop rows where any of the specified date columns are NaT.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      date_columns (list): List of date column names to check.
    
    Returns:
      pd.DataFrame: DataFrame with rows dropped if any of the specified date columns are NaT.
    """
    return df.dropna(subset=date_columns)


def fix_data_types(df, numeric_threshold=0.9):
    """
    Convert DataFrame columns (stored as strings) to appropriate data types,
    excluding any date formatting.

    For each column that is not list-like:
      - If at least `numeric_threshold` fraction of values can be converted to numeric,
        the column is converted to a numeric dtype.
      - Otherwise, the column is cast to 'category' dtype.
    (Date-like strings remain as strings.)
    """
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            continue
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        if numeric_series.notnull().mean() >= numeric_threshold:
            df[col] = numeric_series
        else:
            df[col] = df[col].astype('category')
    return df


def flatten_histories(histories):
    """
    Flatten a list of changelog history entries into a DataFrame.
    Each row represents a single change item.
    """
    rows = []
    for history in histories:
        history_id = history.get("id")
        author = history.get("author", {}).get("name")
        created = history.get("created")
        items = history.get("items", [])
        for item in items:
            rows.append({
                "changelog.history_id": history_id,
                "changelog.author": author,
                "changelog.created": created,
                "changelog.field": item.get("field"),
                "changelog.fieldtype": item.get("fieldtype"),
                "changelog.from": item.get("from"),
                "changelog.fromString": item.get("fromString"),
                "changelog.to": item.get("to"),
                "changelog.toString": item.get("toString")
            })
    return pd.DataFrame(rows)


def process_issue_histories(issue):
    """
    Process a single issue's changelog histories:
      - Flatten the histories.
      - Apply type conversion.
      - Add an 'issue_key' (using 'key' if available, else 'id') for merging.
    """
    if "changelog" in issue and "histories" in issue["changelog"]:
        histories = issue["changelog"]["histories"]
        df_history = flatten_histories(histories)
        df_history = fix_data_types(df_history)
        df_history["issue_key"] = issue.get("key", issue.get("id"))
        return df_history
    return None


def extract_and_flatten_histories(issues):
    """
    Extract and flatten changelog histories from a list of issues using parallel processing.
    """
    flattened_histories = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_issue_histories, issue): issue for issue in issues}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                flattened_histories.append(result)
    if flattened_histories:
        return pd.concat(flattened_histories, ignore_index=True)
    return pd.DataFrame()


def summarize_changelog_histories(df_histories):
    """
    Summarize flattened changelog histories by counting the number of changes per field.
    Returns a DataFrame with one row per issue (keyed by 'issue_key').
    """
    summary = df_histories.groupby('issue_key')['changelog.field'].value_counts().unstack(fill_value=0).reset_index()
    summary = summary.rename(columns=lambda x: f'changelog_count_{x}' if x != 'issue_key' else x)
    return summary


def drop_zero_dominated_columns(df, prefix='changelog_count_', zero_threshold=0.8):
    """
    Drop numeric columns with names starting with `prefix` if more than `zero_threshold`
    fraction of their values are zeros.
    """
    cols_to_drop = []
    for col in df.columns:
        if col.startswith(prefix) and df[col].dtype.kind in 'biufc':
            frac_zeros = (df[col] == 0).mean()
            if frac_zeros > zero_threshold:
                cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def process_issue_links(issuelinks):
    """
    Process the 'fields.issuelinks' JSON array and extract features:
      - Total number of links.
      - Count and binary flag for each link type.
    """
    features = {"issuelinks_total": 0}
    link_types = {}
    if isinstance(issuelinks, list):
        features["issuelinks_total"] = len(issuelinks)
        for link in issuelinks:
            lt = link.get("type", {}).get("name", "Unknown")
            link_types[lt] = link_types.get(lt, 0) + 1
    else:
        features["issuelinks_total"] = 0
    for lt, count in link_types.items():
        features[f"issuelinks_{lt.lower()}_count"] = count
        features[f"has_issuelinks_{lt.lower()}"] = 1 if count > 0 else 0
    return features


def process_comments(comments):
    """
    Process the 'fields.comments' JSON array and extract summary features:
      - Total number of comments.
      - Average and maximum comment length.
      - Number of unique authors.
    """
    features = {
        "comment_count": 0,
        "avg_comment_length": 0,
        "max_comment_length": 0,
        "unique_authors_count": 0
    }
    if not isinstance(comments, list) or len(comments) == 0:
        return features
    comment_bodies = [c.get('body', '') for c in comments if isinstance(c, dict)]
    authors = [c.get('author', {}).get('name') for c in comments if isinstance(c, dict)]
    features["comment_count"] = len(comment_bodies)
    lengths = [len(body) for body in comment_bodies]
    if lengths:
        features["avg_comment_length"] = sum(lengths) / len(lengths)
        features["max_comment_length"] = max(lengths)
    unique_authors = {a for a in authors if a is not None}
    features["unique_authors_count"] = len(unique_authors)
    return features


def process_description_field(descriptions):
    """
    Process the 'fields.description' field by generating dense embeddings
    using a pre-trained Sentence Transformer. The resulting embedding vector
    is expanded into multiple columns (one per dimension).
    """
    descriptions = descriptions.fillna("").astype(str)
    embeddings = descriptions.apply(lambda x: desc_model.encode(x, show_progress_bar=False))
    emb_array = np.vstack(embeddings.values)
    emb_df = pd.DataFrame(emb_array, index=descriptions.index,
                          columns=[f"desc_emb_{i}" for i in range(emb_array.shape[1])])
    return emb_df


# def process_repo(jira_name, db, sample_ratio, max_records=3000, batch_size=500):
#     """
#     Process a single Jira repository using batch processing to prevent connection timeouts:
#       - Load issues from MongoDB in batches.
#       - Apply sampling only after all batches are collected.
#       - Handle large datasets more efficiently with memory considerations.
#       - Apply the same processing pipeline as before.
#     """
#     print(f"\nProcessing repository: {jira_name} ...")
    
#     # Get total count first (fast operation)
#     total_issues = db[jira_name].count_documents({})
#     if total_issues == 0:
#         print(f"⚠️ No documents found for '{jira_name}', skipping.")
#         return None
    
#     print(f"Found {total_issues} total issues in '{jira_name}'. Processing in batches of {batch_size}...")
    
#     # --- 1) Calculate sample size ---
#     desired_sample_size = min(max(1, int(total_issues * sample_ratio)), max_records or float('inf'))
    
#     # --- 2) Determine if we need to sample during fetching or after ---
#     if desired_sample_size < total_issues / 10:
#         # If we want a small sample, use random skip to efficiently get samples
#         # This avoids loading all documents when we only need a small fraction
#         sample_indices = sorted(random.sample(range(total_issues), desired_sample_size))
#         sampled_issues = []
        
#         # Fetch documents by their indices using skip/limit
#         for idx in sample_indices:
#             doc = db[jira_name].find().skip(idx).limit(1)
#             sampled_issues.extend(list(doc))
#     else:
#         # For larger samples, process in batches
#         sampled_issues = []
#         cursor = db[jira_name].find()
        
#         # Process in batches to avoid loading everything at once
#         batch_count = 0
#         while True:
#             batch = list(cursor.limit(batch_size).skip(batch_count * batch_size))
#             if not batch:
#                 break
                
#             batch_count += 1
#             print(f"  - Processed batch {batch_count} ({len(batch)} issues)")
#             sampled_issues.extend(batch)
            
#         # Apply sampling after all batches are collected
#         if len(sampled_issues) > desired_sample_size:
#             print(f"  - Sampling {desired_sample_size} issues from {len(sampled_issues)} collected issues")
#             sampled_issues = random.sample(sampled_issues, desired_sample_size)
    
#     print(f"Final sample for '{jira_name}': {len(sampled_issues)} issues (out of {total_issues} total).")
    
#     # --- 3) Process the sample as before ---
#     if not sampled_issues:
#         return None
        
#     # Convert to DataFrame and apply the pipeline
#     df_main = pd.json_normalize(sampled_issues, sep='.')
#     df_main = fix_data_types(df_main)
    
#     # Process changelog histories in batches too
#     df_histories = extract_and_flatten_histories_batched(sampled_issues, batch_size=100)
    
#     if not df_histories.empty:
#         changelog_summary = summarize_changelog_histories(df_histories)
#         if "key" not in df_main.columns:
#             df_main["key"] = df_main["id"]
#         df_main = pd.merge(df_main, changelog_summary, how="left", left_on="key", right_on="issue_key")
#         df_main.drop(columns=["issue_key"], inplace=True, errors='ignore')
    
#     return df_main

def process_repo(jira_name, db, sample_ratio, batch_size=500):
    """
    Process a single Jira repository using batch processing to prevent connection timeouts.
    Uses a fixed maximum of 500 records per repository and queries only necessary fields.
    
    Parameters:
        jira_name (str): Name of the Jira repository
        db: MongoDB database connection
        sample_ratio (float): Original sample ratio parameter (kept for compatibility)
        batch_size (int): Size of batches for processing
    
    Returns:
        pd.DataFrame: Processed dataframe with sampled issues
    """
    print(f"\nProcessing repository: {jira_name} ...")
    
    # Get total count first (fast operation)
    total_issues = db[jira_name].count_documents({})
    if total_issues == 0:
        print(f"⚠️ No documents found for '{jira_name}', skipping.")
        return None
    
    # Define only the fields we actually need
    needed_fields = {
        # Essential identification fields
        "_id": 1,
        "id": 1,
        "key": 1,
        "changelog": 1,  # Needed for changelog histories
        
        # Issue metadata
        "fields.summary": 1,
        "fields.description": 1,
        "fields.created": 1,
        "fields.updated": 1,
        "fields.resolutiondate": 1,
        
        # Classification fields
        "fields.issuetype.name": 1,
        "fields.priority.name": 1,
        "fields.status.name": 1,
        
        # People fields
        "fields.assignee.key": 1,
        "fields.assignee.name": 1,
        "fields.reporter.key": 1, 
        "fields.reporter.name": 1,
        "fields.creator.key": 1,
        "fields.creator.name": 1,
        
        # Project context
        "fields.project.id": 1,
        "fields.project.key": 1, 
        "fields.project.name": 1,
        
        # Relationships
        "fields.issuelinks": 1,
        "fields.customfield_10557": 1,  # Sprint field
        
        # Components and labels
        "fields.components": 1,
        "fields.labels": 1,
        "fields.fixVersions": 1,
        
        # Comments
        "fields.comments": 1
    }
    
    print(f"Found {total_issues} total issues in '{jira_name}'. Processing in batches of {batch_size}...")
    
    # --- 1) Calculate sample size with fixed maximum ---
    MAX_RECORDS = 100  # Fixed maximum number of records
    desired_sample_size = min(MAX_RECORDS, total_issues)
    print(f"Using fixed maximum of {MAX_RECORDS} records. Will retrieve {desired_sample_size} issues.")
    
    # --- 2) Determine if we need to sample during fetching or after ---
    if desired_sample_size < total_issues / 10:
        # If we want a small sample, use random skip to efficiently get samples
        # This avoids loading all documents when we only need a small fraction
        sample_indices = sorted(random.sample(range(total_issues), desired_sample_size))
        sampled_issues = []
        
        # Fetch documents by their indices using skip/limit, but only retrieve needed fields
        for idx in sample_indices:
            doc = db[jira_name].find({}, needed_fields).skip(idx).limit(1)
            sampled_issues.extend(list(doc))
    else:
        # For larger samples, process in batches
        sampled_issues = []
        cursor = db[jira_name].find({}, needed_fields)  # Only retrieve needed fields
        
        # Process in batches to avoid loading everything at once
        batch_count = 0
        while True:
            batch = list(cursor.limit(batch_size).skip(batch_count * batch_size))
            if not batch:
                break
                
            batch_count += 1
            print(f"  - Processed batch {batch_count} ({len(batch)} issues)")
            sampled_issues.extend(batch)
            
        # Apply sampling after all batches are collected
        if len(sampled_issues) > desired_sample_size:
            print(f"  - Sampling {desired_sample_size} issues from {len(sampled_issues)} collected issues")
            sampled_issues = random.sample(sampled_issues, desired_sample_size)
    
    print(f"Final sample for '{jira_name}': {len(sampled_issues)} issues (out of {total_issues} total).")
    
    # --- 3) Process the sample as before ---
    if not sampled_issues:
        return None
        
    # Convert to DataFrame and apply the pipeline
    df_main = pd.json_normalize(sampled_issues, sep='.')
    df_main = fix_data_types(df_main)
    
    # Process changelog histories in batches too
    df_histories = extract_and_flatten_histories_batched(sampled_issues, batch_size=100)
    
    if not df_histories.empty:
        changelog_summary = summarize_changelog_histories(df_histories)
        if "key" not in df_main.columns:
            df_main["key"] = df_main["id"]
        df_main = pd.merge(df_main, changelog_summary, how="left", left_on="key", right_on="issue_key")
        df_main.drop(columns=["issue_key"], inplace=True, errors='ignore')
    
    # Add repository name for traceability
    df_main['repository'] = jira_name
    
    return df_main

def extract_and_flatten_histories_batched(issues, batch_size=100):
    """
    Extract and flatten changelog histories from a list of issues using batched parallel processing.
    """
    flattened_histories = []
    
    # Process in batches
    for i in range(0, len(issues), batch_size):
        batch = issues[i:i+batch_size]
        print(f"  - Processing changelog history batch {i//batch_size + 1}/{(len(issues)-1)//batch_size + 1}")
        
        # Process each batch in parallel
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_issue_histories, issue): issue for issue in batch}
            for future in as_completed(futures):
                result = future.result()
                if result is not None and not result.empty:
                    flattened_histories.append(result)
    
    if flattened_histories:
        return pd.concat(flattened_histories, ignore_index=True)
    return pd.DataFrame()

def drop_high_missing_columns(df, threshold=0.3):
    """
    Drop columns from the DataFrame where the fraction of missing values exceeds the threshold.
    """
    return df.loc[:, df.isnull().mean() <= threshold]


def impute_missing_values(df, numeric_strategy='median', categorical_strategy='constant', fill_value='Missing'):
    """
    Impute missing values using scikit-learn's SimpleImputer.
      - Numeric columns: impute with the specified strategy (default: median).
      - Categorical columns: impute with a constant value (default: "Missing").
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value=fill_value)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    return df


def explore_all_fields_in_dtale(selected_jiras=None, sample_ratio=0.2, missing_threshold=0.3,
                                zero_threshold=0.8, open_dtale=True):
    """
    Connect to the MongoDB 'JiraRepos' database, sample issues from selected repositories,
    process and flatten changelog histories (summarizing them without from/to transitions),
    process JSON array fields (issuelinks, comments) into engineered features,
    process the 'fields.description' field into dense embedding features,
    drop columns with excessive missing data, impute missing values,
    and drop changelog summary columns dominated by zeros.
    
    If open_dtale is True, launch a D-Tale session for interactive visualization;
    otherwise, simply return the final DataFrame.
    """
    # Connect to MongoDB
    client = MongoClient("mongodb://admin:password@localhost:27017/")
    db = client["JiraRepos"]

    # Load Jira data sources configuration
    with open("../0. DataDefinition/jira_data_sources.json") as f:
        jira_data_sources = json.load(f)

    all_jiras = list(jira_data_sources.keys())
    if selected_jiras and len(selected_jiras) > 0:
        all_jiras = [j for j in all_jiras if j in selected_jiras]
        if not all_jiras:
            print(f"⚠️ No valid Jira repositories found for {selected_jiras}.")
            return

    # Process each repository in parallel
    merged_dfs = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_repo, jira_name, db, sample_ratio): jira_name for jira_name in all_jiras}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                merged_dfs.append(result)

    if merged_dfs:
        final_df = pd.concat(merged_dfs, ignore_index=True)
    else:
        print("No data to display.")
        return
    

    # Define columns to drop – these include system URLs, avatar URLs, raw MongoDB identifiers, etc.
    cols_to_drop = [
        "_id", "expand",
        # System URLs and redundant identifiers
        "self",
        # FixVersions and versions (if not used for estimation)
        "fields.fixVersions", "fields.versions",
        # Avatar URLs and similar, since they are user-specific noise
        "fields.assignee.avatarUrls.48x48", "fields.assignee.avatarUrls.24x24",
        "fields.assignee.avatarUrls.16x16", "fields.assignee.avatarUrls.32x32",
        "fields.reporter.avatarUrls.48x48", "fields.reporter.avatarUrls.24x24",
        "fields.reporter.avatarUrls.16x16", "fields.reporter.avatarUrls.32x32",
        "fields.creator.avatarUrls.48x48", "fields.creator.avatarUrls.24x24",
        "fields.creator.avatarUrls.16x16", "fields.creator.avatarUrls.32x32",
        # Some of the raw MongoDB fields
        "changelog.startAt", "changelog.maxResults", "changelog.total",
        # Votes and similar rarely useful fields
        "fields.votes.self", "fields.votes.hasVoted"
    ]

    # Only drop columns that exist in the DataFrame.
    existing_cols = [col for col in cols_to_drop if col in final_df.columns]
    final_df = final_df.drop(columns=existing_cols, errors='ignore')

    # Drop columns with high missing ratios
    final_df = drop_high_missing_columns(final_df, threshold=missing_threshold)

    # Process JSON array field for issuelinks
    if "fields.issuelinks" in final_df.columns:
        issuelinks_features = final_df["fields.issuelinks"].apply(process_issue_links)
        issuelinks_df = pd.json_normalize(issuelinks_features)
        final_df = pd.concat([final_df.drop(columns=["fields.issuelinks"]), issuelinks_df], axis=1)

    # Process JSON array field for comments
    if "fields.comments" in final_df.columns:
        comments_features = final_df["fields.comments"].apply(process_comments)
        comments_df = pd.json_normalize(comments_features)
        final_df = pd.concat([final_df.drop(columns=["fields.comments"]), comments_df], axis=1)

    # # Process the 'fields.description' field to generate dense embeddings
    # if "fields.description" in final_df.columns:
    #     desc_embeddings = process_description_field(final_df["fields.description"])
    #     final_df = pd.concat([final_df.drop(columns=["fields.description"]), desc_embeddings], axis=1)

    # Impute missing values
    final_df = impute_missing_values(final_df)

    # Drop changelog summary columns dominated by zeros
    final_df = drop_zero_dominated_columns(final_df, prefix='changelog_count_', zero_threshold=zero_threshold)

    date_cols = ["fields.created", "fields.updated", "fields.resolutiondate"]

    final_df = convert_date_columns_dateparser(final_df, date_cols)
    final_df = drop_invalid_dates(final_df, date_cols)
    final_df['fields.created'] = pd.to_datetime(final_df['fields.created'], errors="coerce", utc=True)
    final_df['fields.updated'] = pd.to_datetime(final_df['fields.updated'], errors="coerce", utc=True)
    final_df['fields.resolutiondate'] = pd.to_datetime(final_df['fields.resolutiondate'], errors="coerce", utc=True)

    if open_dtale:
        print("Data processed. Launching D-Tale session...")
        d = dtale.show(final_df, ignore_duplicate=True, allow_cell_edits=False)
        d.open_browser()
        print("✅ D-Tale session launched successfully.")

    return final_df


def export_clean_df():
    """
    Run the full OverviewAnalysis pipeline and return the final DataFrame with all engineered features.
    This version does not launch D-Tale (open_dtale=False) so that it can be used directly as the training dataset.
    
    Returns:
        pd.DataFrame: The final processed DataFrame ready for training.
    """
    final_df = explore_all_fields_in_dtale(
        selected_jiras=["MongoDB"],
        sample_ratio=0.01,
        missing_threshold=0.3,
        zero_threshold=0.8,
        open_dtale=True
    )
    return final_df


# For testing purposes, you can run export_clean_df() if executing this module directly.
if __name__ == "__main__":
    df_for_training = export_clean_df()

