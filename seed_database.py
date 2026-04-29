"""Seed weather_forecast.db with 2 years of data for all 6 cities.

Usage:
    python seed_database.py              # Full seed (2 years × 6 cities)
    python seed_database.py --days 365   # 1 year only
    python seed_database.py --city hanoi # Single city
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config.cities import CITIES
from src.config.db import init_db
from src.config.settings import HISTORICAL_DAYS
from src.data_pipeline.store_data import (
    seed_all_cities,
    store_current,
    store_forecast,
    store_historical,
)


def main():
    parser = argparse.ArgumentParser(description="Seed weather database")
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS, help="Historical days (default: 730)")
    parser.add_argument("--city", type=str, default=None, help="Single city ID (default: all)")
    parser.add_argument("--skip-current", action="store_true", help="Skip OWM current fetch")
    args = parser.parse_args()

    init_db()

    if args.city:
        if args.city not in CITIES:
            print(f"❌ Unknown city: {args.city}")
            print(f"   Available: {', '.join(CITIES.keys())}")
            sys.exit(1)

        print(f"\n🌤 Seeding single city: {args.city}")
        store_historical(args.city, days=args.days)
        store_forecast(args.city)
        if not args.skip_current:
            store_current(args.city)
    else:
        seed_all_cities(days=args.days)


if __name__ == "__main__":
    main()
