import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SMSProvider:
    """
    SMS notification provider

    Supports Twilio, AWS SNS, etc.
    """

    def __init__(
        self, provider: str = "twilio", account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_number: Optional[str] = None
    ):
        self.provider = provider
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.client = None

        if provider == "twilio":
            self._init_twilio()

    def _init_twilio(self):
        """Initialize Twilio client"""
        try:
            from twilio.rest import Client

            self.client = Client(self.account_sid, self.auth_token)
            logger.info("Twilio SMS provider initialized")
        except ImportError:
            logger.error("Twilio library not installed. Run: pip install twilio")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")

    async def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send SMS message

        Args:
            to_number: Recipient phone number (E.164 format: +1234567890)
            message: SMS message (max 160 chars recommended)

        Returns:
            bool: Success status
        """
        if not self.client:
            logger.error("SMS client not initialized")
            return False

        try:
            # Truncate message if too long
            if len(message) > 160:
                message = message[:157] + "..."

            # Send via Twilio
            sms = self.client.messages.create(body=message, from_=self.from_number, to=to_number)

            logger.info(f"SMS sent to {to_number}: {sms.sid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
