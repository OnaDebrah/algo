"""
Stripe payment endpoints for marketplace strategy purchases.
"""

import logging

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_user, get_db
from ...config import settings
from ...models.marketplace import MarketplaceStrategy, StrategyPurchase
from ...models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payments"])


class CheckoutRequest(BaseModel):
    strategy_id: int


class CheckoutResponse(BaseModel):
    checkout_url: str


class PurchaseCheck(BaseModel):
    purchased: bool


# ── Create Stripe Checkout Session ──────────────────────────────────
@router.post("/create-checkout-session", response_model=CheckoutResponse)
async def create_checkout_session(
    body: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a Stripe Checkout Session for a marketplace strategy."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    stripe.api_key = settings.STRIPE_SECRET_KEY

    # Look up strategy
    result = await db.execute(
        select(MarketplaceStrategy).where(MarketplaceStrategy.id == body.strategy_id)
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    if strategy.status != "approved":
        raise HTTPException(status_code=400, detail="Strategy is not available for purchase")
    if not strategy.price or strategy.price <= 0:
        raise HTTPException(status_code=400, detail="Strategy is free — no purchase needed")

    # Check if already purchased
    existing = await db.execute(
        select(StrategyPurchase).where(
            StrategyPurchase.user_id == current_user.id,
            StrategyPurchase.strategy_id == body.strategy_id,
            StrategyPurchase.status == "completed",
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="You already own this strategy")

    # Determine success/cancel URLs
    frontend_origin = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": strategy.name,
                            "description": f"Marketplace strategy: {strategy.name}",
                        },
                        "unit_amount": int(strategy.price * 100),  # cents
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=f"{frontend_origin}?payment=success&strategy_id={body.strategy_id}",
            cancel_url=f"{frontend_origin}?payment=cancelled",
            metadata={
                "strategy_id": str(body.strategy_id),
                "user_id": str(current_user.id),
            },
        )
    except stripe.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=502, detail="Payment provider error")

    return CheckoutResponse(checkout_url=session.url)


# ── Stripe Webhook ──────────────────────────────────────────────────
@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle Stripe webhook events (no auth — verified by signature)."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    stripe.api_key = settings.STRIPE_SECRET_KEY
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        if settings.STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
        else:
            # Dev mode — no signature verification
            import json
            event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    except (ValueError, stripe.SignatureVerificationError) as e:
        logger.warning(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        strategy_id = int(session["metadata"]["strategy_id"])
        user_id = int(session["metadata"]["user_id"])
        amount = session.get("amount_total", 0) / 100  # cents → dollars

        # Check for duplicate
        existing = await db.execute(
            select(StrategyPurchase).where(
                StrategyPurchase.user_id == user_id,
                StrategyPurchase.strategy_id == strategy_id,
                StrategyPurchase.status == "completed",
            )
        )
        if not existing.scalar_one_or_none():
            purchase = StrategyPurchase(
                user_id=user_id,
                strategy_id=strategy_id,
                stripe_checkout_session_id=session.get("id"),
                amount_paid=amount,
                currency=session.get("currency", "usd"),
                status="completed",
            )
            db.add(purchase)
            await db.commit()
            logger.info(f"Purchase recorded: user={user_id} strategy={strategy_id} amount=${amount}")

    return {"status": "ok"}


# ── User's purchases ───────────────────────────────────────────────
@router.get("/purchases")
async def get_purchases(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return list of strategy IDs the current user has purchased."""
    result = await db.execute(
        select(StrategyPurchase.strategy_id).where(
            StrategyPurchase.user_id == current_user.id,
            StrategyPurchase.status == "completed",
        )
    )
    return [row[0] for row in result.all()]


# ── Check single purchase ──────────────────────────────────────────
@router.get("/check/{strategy_id}", response_model=PurchaseCheck)
async def check_purchase(
    strategy_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check if current user has purchased a specific strategy."""
    result = await db.execute(
        select(StrategyPurchase).where(
            StrategyPurchase.user_id == current_user.id,
            StrategyPurchase.strategy_id == strategy_id,
            StrategyPurchase.status == "completed",
        )
    )
    return PurchaseCheck(purchased=result.scalar_one_or_none() is not None)
