import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Function to impute missing values based on recommended strategies
def impute_missing_values(df, grouping_cols=None):
    """
    Impute missing values using recommended strategies
    
    Args:
        df: DataFrame to process
        grouping_cols: Columns to group by for group-based imputation
    
    Returns:
        DataFrame with imputed values
    """
    # Make a copy to avoid modifying the original
    imputed_df = df.copy()

    # Use default grouping columns if none provided
    if grouping_cols is None:
        # Check if these columns exist in the dataframe
        possible_groups = ['fields.issuetype.name', 'fields.priority.name', 'fields.project.key']
        grouping_cols = [col for col in possible_groups if col in df.columns]

    # If no grouping columns are available, use median/mode without grouping
    has_groups = len(grouping_cols) > 0

    # 1. Drop columns with too many missing values
    cols_to_drop = ['fields.creator', 'type_fug', 'type_simple_sub_task', 'type_rfe', 'type_access', 'type_sub_requirement_', 'type_docs_sub_task', 'type_dev_sub_task', 'type_simple_task', 'type_choose_from_below_...', 'priority_important', 'type_business_requirement', 'type_tck_challenge', 'priority_undefined', 'estimation_ratio', 'type_investigation', 'type_rtc', 'type_market_problem', 'type_workload', 'type_suitable_name_search', 'type_okr', 'type_dev_task', 'type_problem_ticket', 'priority_must_have', 'type_risk', 'type_spike', 'priority_p0', 'priority_p1', 'type_design_request', 'type_doc_api', 'type_doc_removal', 'type_doc_ui', 'type_backport_sub_task', 'type_typo', 'type_tracking', 'type_feedback', 'priority_should_have', 'type_comment', 'priority_p3', 'type_svn_>git_mirroring', 'type_pending_review', 'type_new_bugzilla_project', 'type_new_confluence_wiki', 'type_new_tlp___common_tasks', 'type_blogs___access_to_existing_blog', 'priority_p4', 'type_svn_>git_migration', 'priority_p2', 'type_backport', 'type_gitbox_request', 'type_dependency', 'type_ticket', 'type_release_tracker', 'type_analysis', 'priority_unprioritized', 'type_test_task', 'type_issue', 'type_blogs___new_blog_user_account_request', 'type_new_tlp_', 'type_outage', 'type_qe_sub_task', 'type_build_failure', 'type_feature', 'type_new_git_repo', 'priority_lowest', 'priority_not_a_priority', 'priority_normal', 'type_temp', 'type_proposal', 'type_technical_requirement', 'type_docs_task', 'type_incident', 'type_spec_change', 'type_qe_task', 'priority_unknown', 'type_tracker', 'type_library_upgrade', 'type_initiative', 'priority_urgent', 'fields.timeoriginalestimate', 'type_cts_challenge', 'type_requirement', 'type_technical_debt', 'type_it_help', 'type_github_integration', 'type_clarification', 'type_project', 'type_blog___new_blog_request', 'type_new_jira_project', 'type_planned_work', 'fields.timespent', 'type_component_upgrade_subtask', 'type_support_patch', 'type_request', 'fields.timeestimate', 'type_umbrella', 'priority_blocker___p1', 'priority_trivial___p5', 'priority_minor___p4', 'priority_major___p3', 'priority_critical___p2', 'type_public_security_vulnerability', 'type_release', 'type_quality_risk', 'type_patch', 'type_brainstorming', 'type_technical_task', 'type_support_request', 'type_component_upgrade', 'type_dependency_upgrade', 'type_suggestion', 'type_feature_request', 'priority_highest', 'type_enhancement', 'type_question', 'priority_optional', 'priority_medium', 'type_documentation', 'priority_high', 'priority_low', 'type_story', 'type_epic', 'type_test', 'type_wish', 'type_new_feature', 'type_improvement', 'priority_trivial', 'priority_minor', 'priority_major', 'priority_critical', 'priority_blocker']
    print(f"Dropping {len(cols_to_drop)} columns with >35% missing values")
    imputed_df = imputed_df.drop(columns=[col for col in cols_to_drop if col in imputed_df.columns])

    # 2. Simple median imputation for numeric columns
    median_cols = ['type_bug', 'age_days', 'fields.issuetype.id', 'fields.project.id', 'issue_type_id', 'id']
    existing_median_cols = [col for col in median_cols if col in imputed_df.columns]
    if existing_median_cols:
        print(f"Applying median imputation to {len(existing_median_cols)} columns")
        imputer = SimpleImputer(strategy='median')
        imputed_df[existing_median_cols] = imputer.fit_transform(imputed_df[existing_median_cols])

    # 3. Simple mode imputation for categorical columns
    mode_cols = ['fields.creator.displayName', 'fields.creator.key', 'fields.creator.name', 'fields.creator.avatarUrls.32x32', 'fields.creator.active', 'fields.creator.avatarUrls.48x48', 'fields.creator.timeZone', 'fields.creator.avatarUrls.16x16', 'fields.creator.avatarUrls.24x24', 'fields.creator.self', 'inward_count', 'fields.issuetype.name', 'outward_count', 'is_resolved', 'fields.project.key', 'fields.project.name', 'source_file', 'is_completed', 'status', 'issue_type', 'repository', 'fields.status.name', 'fields.issuelinks']
    existing_mode_cols = [col for col in mode_cols if col in imputed_df.columns]
    if existing_mode_cols:
        print(f"Applying mode imputation to {len(existing_mode_cols)} columns")
        for col in existing_mode_cols:
            mode_val = imputed_df[col].mode()[0] if not imputed_df[col].mode().empty else None
            imputed_df[col] = imputed_df[col].fillna(mode_val)

    # 4. Grouped median imputation for numeric columns
    grouped_median_cols = ['type_sub_task', 'type_task', 'fields.priority.id', 'priority_id', 'resolution_time_days']
    existing_grouped_median_cols = [col for col in grouped_median_cols if col in imputed_df.columns]
    if existing_grouped_median_cols and has_groups:
        print(f"Applying grouped median imputation to {len(existing_grouped_median_cols)} columns")
        for col in existing_grouped_median_cols:
            # Calculate medians by group
            group_medians = imputed_df.groupby(grouping_cols)[col].median()
            # For each combination of grouping values, fill with the group median
            for group_values, median_value in group_medians.items():
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                if pd.notna(median_value):
                    # Create a mask for this group
                    mask = pd.Series(True, index=imputed_df.index)
                    for i, group_col in enumerate(grouping_cols):
                        mask = mask & (imputed_df[group_col] == group_values[i])
                    # Apply the group median to missing values in this group
                    mask = mask & imputed_df[col].isna()
                    imputed_df.loc[mask, col] = median_value
            # For any remaining NaNs, use overall median
            overall_median = imputed_df[col].median()
            imputed_df[col] = imputed_df[col].fillna(overall_median)
    elif existing_grouped_median_cols:
        # Fall back to simple median if no grouping columns
        imputer = SimpleImputer(strategy='median')
        imputed_df[existing_grouped_median_cols] = imputer.fit_transform(imputed_df[existing_grouped_median_cols])

    # 6. New category imputation for categorical columns
    new_category_cols = ['fields.priority.name', 'priority_name']
    existing_new_cat_cols = [col for col in new_category_cols if col in imputed_df.columns]
    if existing_new_cat_cols:
        print(f"Applying new category imputation to {len(existing_new_cat_cols)} columns")
        for col in existing_new_cat_cols:
            # Fill missing with a new category 'Unknown'
            imputed_df[col] = imputed_df[col].fillna('Unknown')

    # 7. Finally, drop rows with remaining NaNs in essential columns
    essential_columns = ['fields.issuetype.name', 'fields.created', 'key']
    existing_essential = [col for col in essential_columns if col in imputed_df.columns]
    if existing_essential:
        before_rows = len(imputed_df)
        imputed_df = imputed_df.dropna(subset=existing_essential)
        dropped_rows = before_rows - len(imputed_df)
        print(f"Dropped {dropped_rows} rows with missing values in essential columns")

    return imputed_df

# Example usage:
# df = pd.read_csv('your_file.csv')
# imputed_df = impute_missing_values(df)
# imputed_df.to_csv('imputed_data.csv', index=False)