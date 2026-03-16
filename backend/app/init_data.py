"""Initialize default data"""

import logging

from sqlalchemy import select

from .config import ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_USERNAME
from .database import DatabaseSession
from .models.user import User
from .models.user_settings import UserSettings
from .utils.security import get_password_hash

logger = logging.getLogger(__name__)


async def init_default_data():
    """Create default admin user"""
    async with DatabaseSession() as db:
        try:
            result = await db.execute(select(User).filter(User.email == ADMIN_EMAIL))
            user = result.scalar_one_or_none()

            if not user:
                user = User(
                    email=ADMIN_EMAIL,
                    hashed_password=get_password_hash(ADMIN_PASSWORD),
                    username=ADMIN_USERNAME,
                    is_active=True,
                    is_superuser=True,
                    tier="ENTERPRISE",
                )

                db.add(user)
                await db.flush()

                user_settings = UserSettings(user_id=user.id)
                db.add(user_settings)

                await db.commit()

                logger.info("✓ Created admin user")
            else:
                logger.debug("Admin user already exists.")

        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            await db.rollback()
