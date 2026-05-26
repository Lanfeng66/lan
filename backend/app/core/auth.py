"""API Key 认证。"""
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import os

load_dotenv()

security = HTTPBearer(auto_error=False)

VALID_API_KEYS = os.getenv("API_KEYS", "dev-key-docmind-2025").split(",")


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """验证 API Key。支持 Authorization: Bearer <key> 方式。开发模式跳过认证。"""
    if os.getenv("ENV", "dev") == "dev" and not credentials:
        return "anonymous"

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="请提供 API Key：Authorization: Bearer <your-key>",
        )

    token = credentials.credentials
    if token not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="无效的 API Key")

    return token


async def optional_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """可选认证：有 Key 就验证，没有也可以访问。"""
    if not credentials:
        return "anonymous"
    token = credentials.credentials
    return token if token in VALID_API_KEYS else "anonymous"
