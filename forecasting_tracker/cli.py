"""Command-line interface for the forecasting tracker.

Usage:
    python -m forecasting_tracker add "X will happen by 2026" --confidence 0.7 --domain tech
    python -m forecasting_tracker list
    python -m forecasting_tracker resolve 1 --outcome true
    python -m forecasting_tracker stats
    python -m forecasting_tracker export --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from forecasting_tracker.db.database import create_db_engine, get_session_factory, init_db
from forecasting_tracker.db.models import PredictionStatus
from forecasting_tracker.tracker import (
    DeadlineNotPassed,
    ForecastingError,
    ForecastingTracker,
    PredictionAlreadyResolved,
    PredictionNotFound,
)


def _get_tracker() -> tuple[ForecastingTracker, object]:
    engine = create_db_engine()
    init_db(engine)
    session = get_session_factory(engine)()
    return ForecastingTracker(session), session


def cmd_add(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        deadline = datetime.fromisoformat(args.deadline)
        pred = tracker.add_prediction(
            statement=args.statement,
            confidence=args.confidence,
            deadline=deadline,
            domain=args.domain,
            notes=args.notes,
        )
        session.commit()
        print(f"Registered prediction #{pred.id}: {pred.statement!r}")
        print(f"  confidence={pred.confidence}  domain={pred.domain}  deadline={pred.deadline.date()}")
        return 0
    except ForecastingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()


def cmd_list(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        status_enum = PredictionStatus(args.status) if args.status else None
        preds = tracker.list_predictions(domain=args.domain, status=status_enum, limit=args.limit)
        if not preds:
            print("No predictions found.")
            return 0
        fmt = "{:>4}  {:<50}  {:>6}  {:>12}  {:>10}  {:>8}"
        print(fmt.format("ID", "Statement", "Conf", "Domain", "Status", "Brier"))
        print("-" * 100)
        for p in preds:
            stmt = (p.statement[:47] + "...") if len(p.statement) > 50 else p.statement
            brier = f"{p.brier_score:.4f}" if p.brier_score is not None else "-"
            print(fmt.format(p.id, stmt, f"{p.confidence:.2f}", p.domain, p.status.value, brier))
        return 0
    finally:
        session.close()


def cmd_resolve(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        outcome = args.outcome.lower() in ("true", "yes", "1", "t")
        pred = tracker.resolve_prediction(args.id, outcome)
        session.commit()
        print(f"Resolved prediction #{pred.id} as {'TRUE' if pred.outcome else 'FALSE'}")
        print(f"  Brier score: {pred.brier_score:.4f}")
        return 0
    except (PredictionNotFound, PredictionAlreadyResolved, DeadlineNotPassed) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()


def cmd_show(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        pred = tracker.get_prediction(args.id)
        data = {
            "id": pred.id,
            "statement": pred.statement,
            "confidence": pred.confidence,
            "deadline": pred.deadline.isoformat(),
            "domain": pred.domain,
            "status": pred.status.value,
            "outcome": pred.outcome,
            "brier_score": pred.brier_score,
            "created_at": pred.created_at.isoformat() if pred.created_at else None,
            "resolved_at": pred.resolved_at.isoformat() if pred.resolved_at else None,
            "notes": pred.notes,
        }
        print(json.dumps(data, indent=2))
        return 0
    except PredictionNotFound as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()


def cmd_delete(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        tracker.delete_prediction(args.id)
        session.commit()
        print(f"Deleted prediction #{args.id}")
        return 0
    except PredictionNotFound as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()


def cmd_stats(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        stats = tracker.summary_stats(domain=args.domain)
        print(f"Summary statistics{' (domain=' + args.domain + ')' if args.domain else ''}:")
        print(f"  Total predictions : {stats['total']}")
        print(f"  Open              : {stats['open']}")
        print(f"  Resolved          : {stats['resolved']}")
        if stats["mean_brier"] is not None:
            print(f"  Mean Brier score  : {stats['mean_brier']:.4f}")
        if stats["ece"] is not None:
            print(f"  ECE               : {stats['ece']:.4f}")
        return 0
    finally:
        session.close()


def cmd_export(args: argparse.Namespace) -> int:
    tracker, session = _get_tracker()
    try:
        csv_content = tracker.export_csv(domain=args.domain)
        if args.output:
            with open(args.output, "w", newline="") as f:
                f.write(csv_content)
            print(f"Exported to {args.output}")
        else:
            print(csv_content, end="")
        return 0
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forecasting_tracker",
        description="Personal forecasting tracker with Brier scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Register a new prediction")
    p_add.add_argument("statement", help="Prediction statement")
    p_add.add_argument("--confidence", type=float, required=True, help="Probability 0-1")
    p_add.add_argument(
        "--deadline",
        required=True,
        help="Deadline ISO datetime, e.g. 2026-12-31T23:59:59",
    )
    p_add.add_argument("--domain", default="general", help="Domain tag")
    p_add.add_argument("--notes", default=None, help="Optional notes")

    # list
    p_list = sub.add_parser("list", help="List predictions")
    p_list.add_argument("--domain", default=None, help="Filter by domain")
    p_list.add_argument("--status", default=None, help="Filter by status")
    p_list.add_argument("--limit", type=int, default=50)

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve a prediction")
    p_resolve.add_argument("id", type=int, help="Prediction ID")
    p_resolve.add_argument("--outcome", required=True, help="true/false")

    # show
    p_show = sub.add_parser("show", help="Show a prediction as JSON")
    p_show.add_argument("id", type=int, help="Prediction ID")

    # delete
    p_del = sub.add_parser("delete", help="Delete a prediction")
    p_del.add_argument("id", type=int, help="Prediction ID")

    # stats
    p_stats = sub.add_parser("stats", help="Show calibration statistics")
    p_stats.add_argument("--domain", default=None)

    # export
    p_export = sub.add_parser("export", help="Export predictions to CSV")
    p_export.add_argument("--domain", default=None)
    p_export.add_argument("--output", "-o", default=None, help="Output file (stdout if omitted)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "add": cmd_add,
        "list": cmd_list,
        "resolve": cmd_resolve,
        "show": cmd_show,
        "delete": cmd_delete,
        "stats": cmd_stats,
        "export": cmd_export,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
