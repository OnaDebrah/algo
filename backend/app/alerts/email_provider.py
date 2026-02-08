import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class EmailProvider:
    """
    Email notification provider

    Supports SMTP (Gmail, SendGrid, AWS SES, etc.)
    """

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, from_email: str, from_name: str = "Trading Platform"):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.from_name = from_name

    async def send_email(self, to_email: str, subject: str, body: str, html: bool = False) -> bool:
        """
        Send email via SMTP

        Args:
            to_email: Recipient email
            subject: Email subject
            body: Email body (text or HTML)
            html: Whether body is HTML

        Returns:
            bool: Success status
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            # Add body
            if html:
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_email}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
