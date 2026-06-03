from src.eda.visualizations.generate_charts import generate_all_charts
from src.eda.reports.generate_reports import generate_reports

def run_all():
    generate_all_charts()
    generate_reports()

if __name__ == "__main__":
    run_all()
