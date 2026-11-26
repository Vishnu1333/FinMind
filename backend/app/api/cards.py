from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user

router = APIRouter(prefix="/cards", tags=["cards"])

class CardResponse(pydantic.BaseModel):
    id: int
    last_four: str
    status: str
    
    class Config:
        from_attributes = True

@router.get("", response_model=List[CardResponse])
@router.get("/", response_model=List[CardResponse], include_in_schema=False)
def get_cards(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """List the business's virtual cards.

    Cards are issued at business registration by the provisioning service.
    An empty list is a genuine empty state – no demo data is fabricated here.
    """
    cards = db.query(models.Card).filter(models.Card.business_id == current_user.business_id).all()
    return cards

@router.post("/{card_id}/block")
def block_card(
    card_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    card = db.query(models.Card).filter(
        models.Card.id == card_id, 
        models.Card.business_id == current_user.business_id
    ).first()
    
    if not card:
        raise HTTPException(status_code=404, detail="Card not found or unauthorized")
        
    card.status = "blocked"
    db.commit()
    return {"status": "blocked", "message": f"Card {card.last_four} has been securely blocked."}
