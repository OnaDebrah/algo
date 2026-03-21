"""
Welcome email service with LLM-powered personalization.

Uses Anthropic Haiku to generate dynamic, personalized welcome emails
that highlight platform features and top marketplace strategies.
Falls back to a static template if the API is unavailable.
"""

import logging

from anthropic.types import MessageParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..alerts.email_provider import EmailProvider
from ..config import settings
from ..models.marketplace import MarketplaceStrategy

logger = logging.getLogger(__name__)

# ── Branded HTML wrapper ─────────────────────────────────────────────
EMAIL_WRAPPER = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px; background: #0f172a;">
    <div style="background: linear-gradient(135deg, #7c3aed, #c026d3); padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 28px; letter-spacing: 2px;">ORACULUM</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0; font-size: 14px;">AI-Powered Algorithmic Trading Platform</p>
    </div>
    <div style="background: #1e1b4b; padding: 30px; border-radius: 0 0 16px 16px; color: #e2e8f0;">
        {content}
    </div>
    <div style="text-align: center; padding: 20px; color: #64748b; font-size: 12px;">
        <p>&copy; 2026 Oraculum AI. All rights reserved.</p>
        <p>You received this email because you signed up for an Oraculum account.</p>
    </div>
</div>
"""

# ── Static fallback template ─────────────────────────────────────────
STATIC_WELCOME = """
<h2 style="color: #c4b5fd; margin-top: 0;">Welcome to Oraculum, {username}! 🎉</h2>
<p>We're excited to have you on board. Oraculum is your all-in-one AI-powered algorithmic trading platform.</p>

<h3 style="color: #a78bfa;">Here's what you can do:</h3>
<ul style="line-height: 1.8; color: #cbd5e1;">
    <li><strong style="color: #c4b5fd;">Strategy Builder</strong> — Build and backtest custom trading strategies with AI-powered code generation</li>
    <li><strong style="color: #c4b5fd;">20+ Built-in Strategies</strong> — Technical, KAMA, Parabolic SAR, ML, Deep Learning, RL, and more</li>
    <li><strong style="color: #c4b5fd;">Market Analysis</strong> — In depth market analysis - AI Analyst, Crash Analysis and more </li>
    <li><strong style="color: #c4b5fd;">Options Desk</strong> — Price options, analyze Greeks, and detect arbitrage opportunities</li>
    <li><strong style="color: #c4b5fd;">Walk-Forward Analysis</strong> — Test your strategies for robustness across multiple time windows</li>
    <li><strong style="color: #c4b5fd;">AI Analyst</strong> — Get comprehensive investment thesis reports powered by AI</li>
    <li><strong style="color: #c4b5fd;">ML Studio</strong> — Train machine learning and reinforcement learning models</li>
    <li><strong style="color: #c4b5fd;">Marketplace</strong> — Discover, share, and deploy proven strategies from the community</li>
</ul>

{strategies_section}

<div style="text-align: center; margin: 30px 0;">
    <p style="color: #94a3b8;">Ready to start building?</p>
    <a href="#" style="background: linear-gradient(135deg, #7c3aed, #c026d3); color: white; padding: 14px 32px; border-radius: 10px; text-decoration: none; font-weight: 600; display: inline-block;">Open Oraculum</a>
</div>

<p style="color: #94a3b8; font-size: 14px;">Happy trading!<br/>The Oraculum Team</p>
"""

STRATEGIES_SECTION = """
<h3 style="color: #a78bfa;">🔥 Top Strategies on the Marketplace:</h3>
<div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 16px; margin: 16px 0;">
{items}
</div>
"""

STRATEGY_ITEM = """<div style="padding: 8px 0; border-bottom: 1px solid rgba(139, 92, 246, 0.15);">
    <strong style="color: #c4b5fd;">{name}</strong>
    <span style="color: #94a3b8; font-size: 13px;"> — Return: {return_pct}% · Sharpe: {sharpe}</span>
</div>"""


class WelcomeEmailService:
    """Generates and sends personalized welcome emails on signup."""

    async def _fetch_top_strategies(self, db: AsyncSession, limit: int = 5) -> list[dict]:
        """Fetch top approved marketplace strategies for the email."""
        try:
            result = await db.execute(
                select(MarketplaceStrategy)
                .where(MarketplaceStrategy.status == "approved")
                .order_by(MarketplaceStrategy.rating.desc(), MarketplaceStrategy.downloads.desc())
                .limit(limit)
            )
            strategies = result.scalars().all()
            return [
                {
                    "name": s.name,
                    "total_return": round(s.total_return or 0, 1),
                    "sharpe_ratio": round(s.sharpe_ratio or 0, 2),
                }
                for s in strategies
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch marketplace strategies for welcome email: {e}")
            return []

    async def _generate_with_llm(self, username: str, investor_type: str, risk_profile: str, strategies: list[dict]) -> str | None:
        """Use Anthropic Haiku to generate personalized welcome email content."""
        if not settings.ANTHROPIC_API_KEY:
            logger.info("No Anthropic API key — using static welcome template")
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            strategy_list = ""
            if strategies:
                strategy_list = "\n".join(f"- {s['name']}: {s['total_return']}% return, Sharpe {s['sharpe_ratio']}" for s in strategies)
            else:
                strategy_list = "No strategies published yet — be the first!"

            prompt = f"""You are writing a welcome email body for a new user of Oraculum, an AI-powered algorithmic trading platform.

User details:
- Username: {username}
- Investor Type: {investor_type or 'Not specified'}
- Risk Profile: {risk_profile or 'Not specified'}

Platform features to highlight (pick the most relevant based on their profile):
- AI-powered Strategy Builder with code generation
- 20+ built-in strategies (Technical, KAMA, Parabolic SAR, ML, Deep Learning, RL, etc.)
- Options Desk with pricing, Greeks analysis, and arbitrage detection
- Walk-Forward Analysis for strategy robustness testing
- AI Analyst for comprehensive investment thesis reports
- ML Studio for training machine learning and reinforcement learning models
- Marketplace for sharing and discovering proven strategies
- Real-time market data and sector scanning

Top strategies on the marketplace right now:
{strategy_list}

Write a warm, concise HTML email body (just the inner content — no <html>, <head>, <body> tags).
Use inline CSS styles. Use colors: #c4b5fd for headings, #e2e8f0 for text, #94a3b8 for secondary text.
Keep it under 250 words. Be professional but approachable.
Personalize based on the user's investor type and risk profile.
Include a brief mention of 2-3 top marketplace strategies if available.
End with an encouraging call to action."""

            messages: list[MessageParam] = [{"role": "user", "content": prompt}]
            response = client.messages.create(
                model=settings.ANTHROPIC_MODEL_HAIKU_3,
                max_tokens=1024,
                messages=messages,
            )

            content = response.content[0].text if response.content else None
            if content:
                logger.info(f"✅ LLM-generated welcome email for {username}")
                return content

        except Exception as e:
            logger.warning(f"Anthropic API call failed for welcome email: {e}")

        return None

    def _build_static_email(self, username: str, strategies: list[dict]) -> str:
        """Build static welcome email as fallback."""
        strategies_section = ""
        if strategies:
            items = "\n".join(
                STRATEGY_ITEM.format(
                    name=s["name"],
                    return_pct=s["total_return"],
                    sharpe=s["sharpe_ratio"],
                )
                for s in strategies
            )
            strategies_section = STRATEGIES_SECTION.format(items=items)

        return STATIC_WELCOME.format(username=username, strategies_section=strategies_section)

    async def send_welcome_email(self, user, db: AsyncSession):
        """Generate and send welcome email to a newly registered user."""
        username = user.username
        email = user.email
        investor_type = getattr(user, "investor_type", None)
        risk_profile = getattr(user, "risk_profile", None)

        strategies = await self._fetch_top_strategies(db)

        llm_content = await self._generate_with_llm(username, investor_type, risk_profile, strategies)
        body_content = llm_content or self._build_static_email(username, strategies)

        # Wrap in branded template
        html_body = EMAIL_WRAPPER.format(content=body_content)

        if not settings.EMAIL_ENABLED:
            logger.info(f"📧 [EMAIL_DISABLED] Welcome email generated for {email} (not sent)")
            logger.debug(f"Welcome email HTML:\n{html_body[:500]}...")
            return

        # Send via SMTP
        email_provider = EmailProvider(
            smtp_host=settings.SMTP_SERVER,
            smtp_port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            from_email=settings.FROM_EMAIL,
            from_name="Oraculum AI",
        )

        await email_provider.send_email(
            to_email=email,
            subject=f"Welcome to Oraculum, {username}! 🚀",
            body=html_body,
            html=True,
        )

        logger.info(f"✅ Welcome email sent to {email}")
