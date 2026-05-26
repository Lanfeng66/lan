from fastapi import FastAPI

app = FastAPI()

# 模拟数据库
fake_db = [
    {"item_id": 1, "name": "Foo"},
    {"item_id": 2, "name": "Bar"},
]

@app.get("/items/{item_id}")  # 路径参数 item_id
def get_item(item_id: int, q: str = None):  # 查询参数 q，并指定默认值
    # 根据路径参数查找
    result = next((item for item in fake_db if item["item_id"] == item_id), None)
    if result:
        # 如果提供了查询参数，也把它加入到返回结果中
        if q:
            result.update({"query": q})
        return result
    return {"error": "Item not found"}

# 访问 http://127.0.0.1:8000/items/1?q=test 试试看