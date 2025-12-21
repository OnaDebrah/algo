"""
Alert management system for email and SMS notifications
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage trading alerts via email/SMS"""

    def __init__(self, email_config: Dict = None, sms_config: Dict = None):
        """
        Initialize alert manager

        Args:
            email_config: Email configuration dictionary
            sms_config: SMS configuration dictionary
        """
        self.email_config = email_config or {}
        self.sms_config = sms_config or {}

    def send_email_alert(self, subject: str, message: str):
        """
        Send email alert

        Args:
            subject: Email subject
            message: Email message body
        """
        if not self.email_config.get("enabled"):
            logger.info(f"Email Alert (disabled): {subject} - {message}")
            return

        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = self.email_config["to_email"]
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            )
            server.starttls()
            server.login(self.email_config["from_email"], self.email_config["password"])

            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent: {subject}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def send_sms_alert(self, message: str):
        """
        Send SMS alert via Twilio

        Args:
            message: SMS message body
        """
        if not self.sms_config.get("enabled"):
            logger.info(f"SMS Alert (disabled): {message}")
            return

        try:
            from twilio.rest import Client

            client = Client(
                self.sms_config["account_sid"], self.sms_config["auth_token"]
            )

            message_obj = client.messages.create(
                body=message,
                from_=self.sms_config["from_number"],
                to=self.sms_config["to_number"],
            )

            logger.info(f"SMS sent: {message_obj.sid}")

        except ImportError:
            logger.error(
                "Twilio package not installed. Install with: pip install twilio"
            )
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")

    def alert_trade_executed(self, trade_data: Dict):
        """
        Send alert when trade is executed

        Args:
            trade_data: Trade information dictionary
        """
        message = (
            f"Trade Executed: {trade_data['order_type']} "
            f"{trade_data['quantity']} {trade_data['symbol']} "
            f"@ ${trade_data['price']:.2f}"
        )

        self.send_email_alert("Trade Executed", message)
        self.send_sms_alert(message)
        logger.info(message)

    def alert_position_closed(self, trade_data: Dict):
        """
        Send alert when position is closed

        Args:
            trade_data: Trade information with profit/loss
        """
        profit = trade_data.get("profit", 0)
        profit_pct = trade_data.get("profit_pct", 0)

        message = (
            f"Position Closed: {trade_data['symbol']} "
            f"P&L: ${profit:.2f} ({profit_pct:.2f}%)"
        )

        self.send_email_alert("Position Closed", message)
        self.send_sms_alert(message)
        logger.info(message)

    def alert_risk_event(self, event_type: str, details: str):
        """
        Send alert for risk management events

        Args:
            event_type: Type of risk event
            details: Event details
        """
        message = f"Risk Alert - {event_type}: {details}"

        self.send_email_alert(f"Risk Alert: {event_type}", message)
        self.send_sms_alert(message)
        logger.warning(message)

    def alert_system_error(self, error_msg: str):
        """
        Send alert for system errors

        Args:
            error_msg: Error message
        """
        message = f"System Error: {error_msg}"

        self.send_email_alert("System Error", message)
        self.send_sms_alert(message)
        logger.error(message)
