import logging
import os
import smtplib
from email.message import EmailMessage


logger = logging.getLogger(__name__)


def send_otp_email(recipient: str, code: str, purpose: str) -> None:
    """Send a short-lived verification code using Gmail SMTP STARTTLS."""
    smtp_email = os.getenv("SMTP_EMAIL", "").strip()
    smtp_password = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "").strip()
    if not smtp_email or not smtp_password:
        logger.error(
            "SMTP configuration incomplete (SMTP_EMAIL_present=%s, SMTP_APP_PASSWORD_present=%s)",
            bool(smtp_email),
            bool(smtp_password),
        )
        raise RuntimeError("SMTP_EMAIL and SMTP_APP_PASSWORD must be configured.")

    host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    sender_name = os.getenv("SMTP_FROM_NAME", "FinMind Security").strip()
    action = "complete your FinMind registration" if purpose == "registration" else "sign in to FinMind"

    message = EmailMessage()
    message["Subject"] = "Your FinMind verification code"
    message["From"] = f"{sender_name} <{smtp_email}>"
    message["To"] = recipient
    message.set_content(
        f"Your FinMind verification code is: {code}\n\n"
        f"Use this code to {action}. It expires in 5 minutes and can only be used once. "
        "If you did not request this code, you can ignore this email.\n"
    )

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_email, smtp_password)
            server.send_message(message)
    except Exception as exc:
        # Log only non-secret diagnostics; never log credentials, message body, or recipient.
        logger.error(
            "Gmail SMTP delivery failed (error_type=%s, smtp_code=%s, errno=%s)",
            type(exc).__name__,
            getattr(exc, "smtp_code", None),
            getattr(exc, "errno", None),
        )
        raise
