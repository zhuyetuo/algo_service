from sqlalchemy.ext.asyncio import AsyncSession

from db.client import AsyncSessionLocal


async def run_baseline_update() -> None:
    """Recalculate user baselines from the last N days of data."""
    async with AsyncSessionLocal() as db:
        # TODO: query historical records, compute stats, upsert baseline table
        pass
