from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user

router = APIRouter(prefix="/help-desk", tags=["help-desk"])

class TicketResponse(pydantic.BaseModel):
    id: int
    title: str
    status: str
    priority: str
    category: str
    
    class Config:
        orm_mode = True

class CreateTicketRequest(pydantic.BaseModel):
    title: str
    description: str
    priority: str
    category: str

@router.get("", response_model=List[TicketResponse])
@router.get("/", response_model=List[TicketResponse], include_in_schema=False)
def get_tickets(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    tickets = db.query(models.Ticket).filter(
        models.Ticket.business_id == current_user.business_id
    ).order_by(models.Ticket.created_at.desc()).all()
    return tickets

@router.post("", response_model=TicketResponse)
@router.post("/", response_model=TicketResponse, include_in_schema=False)
def create_ticket(
    request: CreateTicketRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    ticket = models.Ticket(
        business_id=current_user.business_id,
        title=request.title,
        description=request.description,
        priority=request.priority,
        category=request.category,
        status="open"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket

@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(
    ticket_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    ticket = db.query(models.Ticket).filter(
        models.Ticket.id == ticket_id,
        models.Ticket.business_id == current_user.business_id
    ).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket
