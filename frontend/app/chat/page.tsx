'use client'

import { useState, useRef, useEffect } from "react";

interface Citation {
  index: number;
  source: string;
  chunk_index: number;
  preview: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export default function ChatPage() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const msg = query.trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setQuery("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, conversation_id: conversationId || "test" }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error("无法读取响应流");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === "token") {
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              copy[copy.length - 1] = { ...last, content: last.content + data.content };
              return copy;
            });
          } else if (data.type === "citations") {
            setMessages((prev) => {
              const copy = [...prev];
              copy[copy.length - 1] = { ...copy[copy.length - 1], citations: data.citations };
              return copy;
            });
            if (data.conversation_id) setConversationId(data.conversation_id);
          } else if (data.type === "start" && data.conversation_id) {
            setConversationId(data.conversation_id);
          } else if (data.type === "error") {
            throw new Error(data.message || "服务端错误");
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...copy[copy.length - 1], content: "请求失败: " + (err.message || "未知错误") };
        return copy;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      <header className="border-b py-4 px-6">
        <h1 className="text-xl font-bold">DocMind</h1>
        <p className="text-sm text-gray-500">技术文档智能助手</p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">你好，我是 DocMind</p>
            <p className="text-sm mt-2">试试问我：Redis集群怎么扩容？Docker如何优化镜像大小？</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex mb-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className="max-w-[80%] rounded-lg px-4 py-3 whitespace-pre-wrap"
              style={{
                backgroundColor: msg.role === "user" ? "#3b82f6" : "#f3f4f6",
                color: msg.role === "user" ? "#ffffff" : "#111827",
              }}
            >
              {msg.content}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3" style={{ borderTop: "1px solid #d1d5db" }}>
                  <p className="text-xs font-semibold mb-1" style={{ color: "#6b7280" }}>参考来源</p>
                  {msg.citations.map((c) => (
                    <div key={c.index} className="text-xs mt-1" style={{ color: "#6b7280" }}>
                      <span className="font-medium">[{c.index}]</span> {c.source} 第{c.chunk_index}段
                      <p className="truncate" style={{ color: "#9ca3af" }}>{c.preview}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-lg px-4 py-3 text-gray-500">思考中...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="输入你的问题..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            type="button"
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
