"""用户反馈 API。"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import json
import os

router = APIRouter(prefix="/api/v1", tags=["feedback"])

FEEDBACK_FILE = "./data/feedback.jsonl"


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str  # assistant 消息的 ID
    rating: str  # "like" | "dislike"
    comment: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(fb: FeedbackRequest):
    """记录用户反馈。"""
    record = {
        "conversation_id": fb.conversation_id,
        "message_id": fb.message_id,
        "rating": fb.rating,
        "comment": fb.comment,
    }

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 点踩的消息进入 hard-negative 集
    if fb.rating == "dislike":
        _add_to_hard_negative(fb)

    return {"status": "recorded"}


def _add_to_hard_negative(fb: FeedbackRequest):
    """将点踩消息加入 hard-negative 评估集。"""
    from app.conversation.manager import conv_manager

    history = conv_manager.get_history(fb.conversation_id, last_n=2)
    question = None
    answer = None
    for m in history:
        if m["role"] == "user":
            question = m["content"]
        elif m["role"] == "assistant":
            answer = m["content"]

    if question and answer:
        negative_file = "./data/hard_negatives.jsonl"
        with open(negative_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "question": question,
                "answer": answer,
                "reason": fb.comment or "用户点踩",
            }, ensure_ascii=False) + "\n")
        print(f"  📝 已记录 hard-negative: {question[:50]}...")


@router.get("/feedback/stats")
async def feedback_stats():
    """反馈统计。"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "likes": 0, "dislikes": 0}

    likes = 0
    dislikes = 0
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["rating"] == "like":
                likes += 1
            elif record["rating"] == "dislike":
                dislikes += 1

    return {"total": likes + dislikes, "likes": likes, "dislikes": dislikes}