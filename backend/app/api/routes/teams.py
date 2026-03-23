"""Team collaboration routes."""

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.team import Team, TeamComment, TeamMember

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teams", tags=["Teams"])


class TeamCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class CommentCreate(BaseModel):
    target_type: str  # backtest, strategy, paper_portfolio
    target_id: int
    content: str = Field(..., min_length=1)
    parent_comment_id: Optional[int] = None


class MemberRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(admin|member|viewer)$")


@router.get("/")
async def list_teams(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List teams the current user belongs to."""
    result = await db.execute(
        select(Team, TeamMember.role)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .where(TeamMember.user_id == current_user.id)
        .order_by(Team.created_at.desc())
    )
    rows = result.all()
    return [
        {
            "id": team.id,
            "name": team.name,
            "description": team.description,
            "invite_code": team.invite_code if role == "owner" else None,
            "role": role,
            "created_at": team.created_at.isoformat() if team.created_at else None,
        }
        for team, role in rows
    ]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_team(
    request: TeamCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new team."""
    team = Team(
        name=request.name,
        description=request.description,
        owner_id=current_user.id,
        invite_code=secrets.token_urlsafe(12),
    )
    db.add(team)
    await db.flush()

    # Add owner as member
    member = TeamMember(team_id=team.id, user_id=current_user.id, role="owner")
    db.add(member)
    await db.commit()
    await db.refresh(team)

    return {"id": team.id, "name": team.name, "invite_code": team.invite_code}


@router.post("/{team_id}/join")
async def join_team(
    team_id: int,
    invite_code: str = Query(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Join a team via invite code."""
    result = await db.execute(select(Team).where(Team.id == team_id, Team.invite_code == invite_code))
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail="Invalid team or invite code")

    # Check if already member
    existing = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Already a member of this team")

    member = TeamMember(team_id=team_id, user_id=current_user.id, role="member")
    db.add(member)
    await db.commit()
    return {"status": "joined", "team_name": team.name}


@router.get("/{team_id}/members")
async def get_members(
    team_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List team members."""
    # Verify membership
    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a member of this team")

    result = await db.execute(
        select(TeamMember, User.username, User.email)
        .join(User, User.id == TeamMember.user_id)
        .where(TeamMember.team_id == team_id)
        .order_by(TeamMember.joined_at)
    )
    rows = result.all()
    return [
        {
            "user_id": member.user_id,
            "username": username,
            "email": email,
            "role": member.role,
            "joined_at": member.joined_at.isoformat() if member.joined_at else None,
        }
        for member, username, email in rows
    ]


@router.patch("/{team_id}/members/{user_id}")
async def update_member_role(
    team_id: int,
    user_id: int,
    request: MemberRoleUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a team member's role (owner/admin only)."""
    # Check caller is owner or admin
    caller_result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == current_user.id,
            TeamMember.role.in_(["owner", "admin"]),
        )
    )
    if not caller_result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Only owners and admins can change roles")

    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == user_id))
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    if member.role == "owner":
        raise HTTPException(status_code=400, detail="Cannot change owner role")

    member.role = request.role
    await db.commit()
    return {"status": "updated"}


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: int,
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a member from team (owner/admin or self-leave)."""
    is_self = user_id == current_user.id
    if not is_self:
        caller_result = await db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id,
                TeamMember.user_id == current_user.id,
                TeamMember.role.in_(["owner", "admin"]),
            )
        )
        if not caller_result.scalar_one_or_none():
            raise HTTPException(status_code=403, detail="Permission denied")

    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == user_id))
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    if member.role == "owner":
        raise HTTPException(status_code=400, detail="Owner cannot leave. Transfer ownership first.")

    await db.delete(member)
    await db.commit()
    return {"status": "removed"}


@router.post("/{team_id}/comments")
async def add_comment(
    team_id: int,
    request: CommentCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a comment to a resource within a team."""
    # Verify membership
    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a member of this team")

    comment = TeamComment(
        team_id=team_id,
        user_id=current_user.id,
        username=current_user.username or "Unknown",
        target_type=request.target_type,
        target_id=request.target_id,
        content=request.content,
        parent_comment_id=request.parent_comment_id,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    return {"id": comment.id, "message": "Comment added"}


@router.get("/{team_id}/comments")
async def get_comments(
    team_id: int,
    target_type: Optional[str] = Query(None),
    target_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get comments for a team, optionally filtered by target."""
    # Verify membership
    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == current_user.id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a member of this team")

    query = select(TeamComment).where(TeamComment.team_id == team_id)
    if target_type:
        query = query.where(TeamComment.target_type == target_type)
    if target_id:
        query = query.where(TeamComment.target_id == target_id)

    result = await db.execute(query.order_by(TeamComment.created_at))
    comments = result.scalars().all()

    return [
        {
            "id": c.id,
            "user_id": c.user_id,
            "username": c.username,
            "target_type": c.target_type,
            "target_id": c.target_id,
            "content": c.content,
            "parent_comment_id": c.parent_comment_id,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in comments
    ]
