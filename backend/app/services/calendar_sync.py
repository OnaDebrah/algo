"""
Economic calendar sync service.

Generates a realistic calendar of recurring economic events (FOMC, CPI, NFP, GDP, etc.)
and seeds them into the database. Uses known schedules for US macro events.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.economic_event import EconomicEvent

logger = logging.getLogger(__name__)

# ── Well-known recurring US economic events ──────────────────────────────
# Each entry: (name, category, impact, typical_day_of_month_or_rule, country)
RECURRING_EVENTS: List[Dict] = [
    # Employment
    {"name": "Non-Farm Payrolls", "category": "Employment", "impact": "high", "day": "first_friday"},
    {"name": "Unemployment Rate", "category": "Employment", "impact": "high", "day": "first_friday"},
    {"name": "ADP Employment Change", "category": "Employment", "impact": "medium", "day_offset": -2, "anchor": "first_friday"},
    {"name": "Initial Jobless Claims", "category": "Employment", "impact": "medium", "weekday": 3},
    # weekly Thursday
    # Inflation
    {"name": "CPI (MoM)", "category": "Inflation", "impact": "high", "day": 12},
    {"name": "CPI (YoY)", "category": "Inflation", "impact": "high", "day": 12},
    {"name": "Core CPI (MoM)", "category": "Inflation", "impact": "high", "day": 12},
    {"name": "PPI (MoM)", "category": "Inflation", "impact": "medium", "day": 14},
    # Fed
    {"name": "FOMC Interest Rate Decision", "category": "Fed", "impact": "high", "months": [1, 3, 5, 6, 7, 9, 11, 12], "day": 15},
    {"name": "FOMC Meeting Minutes", "category": "Fed", "impact": "high", "months": [1, 2, 4, 5, 7, 8, 10, 11], "day": 20},
    {"name": "Fed Chair Press Conference", "category": "Fed", "impact": "high", "months": [1, 3, 5, 6, 7, 9, 11, 12], "day": 15},
    # GDP
    {"name": "GDP (QoQ) - Advance", "category": "GDP", "impact": "high", "months": [1, 4, 7, 10], "day": 26},
    {"name": "GDP (QoQ) - Second Estimate", "category": "GDP", "impact": "medium", "months": [2, 5, 8, 11], "day": 27},
    # Consumer
    {"name": "Retail Sales (MoM)", "category": "Consumer", "impact": "high", "day": 15},
    {"name": "Consumer Confidence", "category": "Consumer", "impact": "medium", "day": 28},
    {"name": "Michigan Consumer Sentiment", "category": "Consumer", "impact": "medium", "day": 16},
    {"name": "Personal Spending (MoM)", "category": "Consumer", "impact": "medium", "day": 28},
    # Housing
    {"name": "Existing Home Sales", "category": "Housing", "impact": "medium", "day": 21},
    {"name": "New Home Sales", "category": "Housing", "impact": "medium", "day": 24},
    {"name": "Housing Starts", "category": "Housing", "impact": "medium", "day": 17},
    # Manufacturing / Business
    {"name": "ISM Manufacturing PMI", "category": "Manufacturing", "impact": "high", "day": 1},
    {"name": "ISM Services PMI", "category": "Manufacturing", "impact": "high", "day": 3},
    {"name": "Industrial Production (MoM)", "category": "Manufacturing", "impact": "medium", "day": 16},
    {"name": "Durable Goods Orders (MoM)", "category": "Manufacturing", "impact": "medium", "day": 25},
    # Trade
    {"name": "Trade Balance", "category": "Trade", "impact": "medium", "day": 5},
    # Treasury
    {"name": "10-Year Note Auction", "category": "Treasury", "impact": "medium", "day": 10},
    {"name": "30-Year Bond Auction", "category": "Treasury", "impact": "medium", "day": 11},
]


def _first_friday(year: int, month: int) -> int:
    """Return day-of-month for the first Friday."""
    d = datetime(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    days_ahead = 4 - d.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return d.day + days_ahead


def _clamp_day(year: int, month: int, day: int) -> int:
    """Clamp day to valid range for the given month."""
    import calendar

    _, max_day = calendar.monthrange(year, month)
    return min(day, max_day)


def _generate_events(months_ahead: int = 6) -> List[Dict]:
    """Generate upcoming economic events for the next N months."""
    now = datetime.now(timezone.utc)
    events = []

    for m_offset in range(-1, months_ahead + 1):
        month = ((now.month - 1 + m_offset) % 12) + 1
        year = now.year + ((now.month - 1 + m_offset) // 12)

        for ev in RECURRING_EVENTS:
            # Skip if event only occurs in certain months
            if "months" in ev and month not in ev["months"]:
                continue

            # Determine day
            if ev.get("day") == "first_friday":
                day = _first_friday(year, month)
            elif "anchor" in ev and ev["anchor"] == "first_friday":
                day = _first_friday(year, month) + ev.get("day_offset", 0)
                if day < 1:
                    continue
            elif "weekday" in ev:
                # Weekly event — generate all occurrences in the month
                import calendar

                _, max_day = calendar.monthrange(year, month)
                for d in range(1, max_day + 1):
                    if datetime(year, month, d).weekday() == ev["weekday"]:
                        event_dt = datetime(year, month, d, 13, 30, tzinfo=timezone.utc)
                        events.append(
                            {
                                "event_name": ev["name"],
                                "country": "US",
                                "event_date": event_dt,
                                "impact": ev["impact"],
                                "category": ev.get("category", "Other"),
                                "source": "generated",
                            }
                        )
                continue
            else:
                day = _clamp_day(year, month, ev["day"])

            # Choose a realistic release time (8:30 ET = 13:30 UTC for most,
            # FOMC at 14:00 UTC, Consumer Confidence at 15:00 UTC)
            hour = 13
            minute = 30
            if "FOMC" in ev["name"] or "Fed Chair" in ev["name"]:
                hour, minute = 19, 0
            elif "Consumer Confidence" in ev["name"] or "Michigan" in ev["name"]:
                hour, minute = 15, 0

            try:
                event_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                continue

            events.append(
                {
                    "event_name": ev["name"],
                    "country": "US",
                    "event_date": event_dt,
                    "impact": ev["impact"],
                    "category": ev.get("category", "Other"),
                    "source": "generated",
                }
            )

    return events


async def sync_calendar(db: AsyncSession, months_ahead: int = 6) -> int:
    """
    Seed / refresh the economic_events table with generated events.
    Deletes old generated events and inserts fresh ones.
    Returns the number of events inserted.
    """
    # Remove previously generated events (keep manual ones)
    await db.execute(delete(EconomicEvent).where(EconomicEvent.source == "generated"))

    events_data = _generate_events(months_ahead)
    count = 0
    for ev in events_data:
        db.add(EconomicEvent(**ev))
        count += 1

    await db.commit()
    logger.info("Synced %d economic calendar events", count)
    return count
