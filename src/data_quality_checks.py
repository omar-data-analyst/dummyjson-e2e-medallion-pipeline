import os
import sys
from google.cloud import bigquery


def run_quality_checks(snapshot_date: str, project_id: str, dataset: str = "gold"):
    client = bigquery.Client(project=project_id)
    view_fqn = f"`{project_id}.{dataset}.vw_fact_orders_enriched`"

    print(f"--- Starting BigQuery Data Quality Checks for: {snapshot_date} ---")

    # 1. Check for null values ​​in user_id and order_id for the snapshot day
    null_check_sql = f"""
        SELECT
            COUNTIF(user_id IS NULL) AS null_user_ids,
            COUNTIF(order_id IS NULL) AS null_order_ids,
            COUNT(*) AS current_rows
        FROM {view_fqn}
        WHERE DATE(order_processed_at) = @snapshot_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("snapshot_date", "DATE", snapshot_date)
        ]
    )

    res = list(client.query(null_check_sql, job_config=job_config).result())[0]

    if res.null_user_ids > 0 or res.null_order_ids > 0:
        raise ValueError(
            f"❌ DATA QUALITY ERROR on {snapshot_date}: "
            f"null_user_ids={res.null_user_ids}, null_order_ids={res.null_order_ids}"
        )

    # 2. Check for anomalous spikes/drops compared to the average of the last 7 days
    history_sql = f"""
        WITH daily_counts AS (
            SELECT
                DATE(order_processed_at) AS dt,
                COUNT(*) AS cnt
            FROM {view_fqn}
            WHERE DATE(order_processed_at) < @snapshot_date
              AND DATE(order_processed_at) >= DATE_SUB(@snapshot_date, INTERVAL 7 DAY)
            GROUP BY dt
        )
        SELECT AVG(cnt) AS avg_7d_count
        FROM daily_counts
    """

    hist_res = list(client.query(history_sql, job_config=job_config).result())[0]
    avg_7d = hist_res.avg_7d_count

    if avg_7d is not None and avg_7d > 0:
        if res.current_rows < (avg_7d * 0.5):
            raise ValueError(
                f"❌ VOLUME ANOMALY on {snapshot_date}: "
                f"today rows ({res.current_rows}) < 50% of 7-day avg ({avg_7d:.1f})"
            )

    print(
        f"✅ Data Quality PASSED for {snapshot_date} "
        f"(rows={res.current_rows}, 7d_avg={avg_7d if avg_7d is not None else 'N/A'})"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python scripts/data_quality_checks.py YYYY-MM-DD")

    snapshot_dt = sys.argv[1]
    bq_project = os.environ.get("GCP_PROJECT_ID") or os.environ.get("BIGQUERY_PROJECT")
    bq_dataset = os.environ.get("BQ_GOLD_DATASET", "gold")

    if not bq_project:
        raise ValueError("Missing GCP_PROJECT_ID or BIGQUERY_PROJECT environment variable.")

    run_quality_checks(snapshot_dt, bq_project, bq_dataset)
