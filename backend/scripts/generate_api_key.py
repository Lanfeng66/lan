"""生成 API Key。"""
import secrets


def generate_key() -> str:
    raw = secrets.token_hex(24)
    return f"dm-{raw}"  # dm = docmind 前缀


if __name__ == "__main__":
    key = generate_key()
    print(f"新的 API Key: {key}")
    print(f"请将以下行添加到 .env 文件：")
    print(f'API_KEYS=dev-key-docmind-2025,{key}')
