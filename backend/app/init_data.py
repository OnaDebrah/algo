"""Initialize default data"""

import logging

from sqlalchemy import select

from .database import DatabaseSession
from .models.user import User
from .utils.security import get_password_hash

logger = logging.getLogger(__name__)


async def init_default_data():
    """Create default admin user"""
    async with DatabaseSession() as db:
        try:
            # Check if admin exists
            result = await db.execute(select(User).filter(User.email == "admin@example.com"))
            user = result.scalar_one_or_none()

            if not user:
                user = User(
                    email="admin@example.com",
                    hashed_password=get_password_hash("admin123"),
                    username="Admin User",
                    is_active=True,
                    is_superuser=True,
                    tier="ENTERPRISE",
                )
                db.add(user)
                await db.commit()
                logger.info("✓ Created admin user: admin@example.com / admin123")
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            await db.rollback()
