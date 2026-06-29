"""Demo: Register predictions, resolve them, check calibration."""
import tempfile
from datetime import datetime, timedelta

db_file = tempfile.mktemp(suffix=".db")
from forecasting_tracker.db.database import create_db_engine, init_db, get_session_factory
from forecasting_tracker.tracker import ForecastingTracker

engine = create_db_engine(f"sqlite:///{db_file}")
init_db(engine)
Session = get_session_factory(engine)
past = datetime.utcnow() - timedelta(days=1)

print("=== Forecasting Tracker Demo ===\n")
predictions = [
    ("Python will remain top 3 language in 2026", 0.90, "tech", True),
    ("GPT-5 will be released in 2025", 0.70, "ai", True),
    ("Bitcoin will exceed $100k in 2025", 0.55, "finance", True),
    ("Remote work will decline by 20% in 2025", 0.40, "work", False),
    ("EVs will exceed 30% market share by 2026", 0.60, "energy", False),
]
with Session() as session:
    tracker = ForecastingTracker(session)
    ids = []
    for statement, conf, domain, _ in predictions:
        p = tracker.add_prediction(statement, conf, domain=domain, deadline=past)
        ids.append(p.id)
        print(f"  Registered [{conf:.0%}]: {statement[:55]}...")
    print("\nResolving...")
    for i, (_, _, _, outcome) in enumerate(predictions):
        tracker.resolve_prediction(ids[i], outcome)
        print(f"  {'TRUE ' if outcome else 'FALSE'}")
    stats = tracker.summary_stats()
print(f"\nTotal: {stats['total']}  Resolved: {stats['resolved']}")
print(f"Brier: {stats['mean_brier']:.4f}  ECE: {stats['ece']:.4f}")
print("\nDemo complete.")
