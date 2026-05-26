"""问答 API。"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.graph.workflow import rag_app
from app.conversation.manager import conv_manager
from app.core.auth import verify_api_key
import uuid
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    kb_id: str = "docmind"
    conversation_id: Optional[str] = None
    message: str
    stream: bool = False


class Citation(BaseModel):
    index: int
    source: str
    chunk_index: int
    preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[Citation] = []
    confidence_score: float = 0.0


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """单轮或多轮对话。"""
    try:
        conv_id = request.conversation_id or conv_manager.create_conversation()

        # Step 1: 指代消解
        rewritten = conv_manager.rewrite_with_history(conv_id, request.message)

        # Step 2: 话题切换检测
        if conv_manager.detect_topic_switch(conv_id, request.message):
            # 话题切换时，用压缩后的上下文而不是完整历史
            compressed = conv_manager.build_compressed_context(conv_id)
        else:
            compressed = conv_manager.build_compressed_context(conv_id)

        # Step 3: 保存用户消息
        conv_manager.add_message(conv_id, "user", request.message)

        # Step 4: RAG
        result = rag_app.invoke({
            "question": rewritten,
            "retry_count": 0,
            "confidence_score": 0.0,
            "sub_queries": [],
            "conversation_history": conv_manager.get_history(conv_id, last_n=5),
        })

        # Step 5: 保存助手消息
        conv_manager.add_message(conv_id, "assistant", result["answer"])

        return ChatResponse(
            conversation_id=conv_id,
            answer=result["answer"],
            citations=[Citation(**c) for c in result.get("citations", [])],
            confidence_score=result.get("confidence_score", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """流式问答接口（Server-Sent Events）。"""
    conv_id = request.conversation_id or str(uuid.uuid4())

    # 重写 Query
    rewritten = conv_manager.rewrite_with_history(conv_id, request.message)

    async def event_generator():
        try:
            # 1. 发送"开始"事件
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv_id})}\n\n"

            # 2. 运行 RAG（非流式部分）
            result = rag_app.invoke({
                "question": rewritten,
                "retry_count": 0,
                "confidence_score": 0.0,
                "sub_queries": [],
                "conversation_history": conv_manager.get_history(conv_id, last_n=5),
            })

            # 3. 用 LLM streaming 重新生成答案（真正的逐字输出）
            from app.llm.chat_model import get_llm
            llm = get_llm()

            context_parts = []
            for i, doc in enumerate(result.get("final_context", result.get("retrieved_docs", []))):
                source = doc.get("source", "未知")
                context_parts.append(f"[资料{i+1}] {source}\n{doc['content']}")
            context = "\n\n---\n\n".join(context_parts)

            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=f"""你是技术文档助手。请按以下规则回答：

1. 只使用【参考资料】中明确提供的信息，禁止编造、推测或补充
2. 用清晰的结构组织答案：先给结论，再分点详述
3. 涉及步骤、命令、配置时，逐条列出并标注来源
4. 如果资料不足以回答问题，直接说"当前知识库中没有相关信息"
5. 回答简洁，不要展开与问题无关的内容

【参考资料】
{context}"""),
                HumanMessage(content=request.message),
            ]

            full_answer = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # 让出控制权

            # 4. 发送引用和元数据
            yield f"data: {json.dumps({'type': 'citations', 'citations': result.get('citations', []), 'confidence_score': result.get('confidence_score', 0.0)}, ensure_ascii=False)}\n\n"

            # 5. 结束
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

            # 保存对话
            conv_manager.add_message(conv_id, "user", request.message)
            conv_manager.add_message(conv_id, "assistant", full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )