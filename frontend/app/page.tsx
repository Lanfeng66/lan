'use client'

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface Citation {
  index: number;
  source: string;
  chunk_index: number;
  preview: string;
}

export default function Home() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isComposing = useRef(false);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 原生 DOM 事件监听，确保中文输入法触发状态更新
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    const sync = () => setInput(el.value);
    el.addEventListener("input", sync);
    return () => el.removeEventListener("input", sync);
  }, []);

  const sendMessage = async () => {
    const text = (inputRef.current?.value ?? input).trim();
    if (!text || loading) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    if (inputRef.current) inputRef.current.value = "";
    setInput("");
    setLoading(true);

    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId || "test",
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }

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
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));

            if (data.type === "token") {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + data.content,
                };
                return updated;
              });
            } else if (data.type === "citations") {
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  citations: data.citations,
                };
                return updated;
              });
              if (data.conversation_id) {
                setConversationId(data.conversation_id);
              }
            } else if (data.type === "start" && data.conversation_id) {
              setConversationId(data.conversation_id);
            } else if (data.type === "error") {
              throw new Error(data.message || "服务端错误");
            }
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: "请求失败: " + (err.message || "未知错误"),
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (rating: "like" | "dislike") => {
    await fetch("http://localhost:8000/api/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        message_id: Date.now().toString(),
        rating,
      }),
    });
  };

  const handleKeyDown = () => {
    if (!isComposing.current) {
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      <header className="border-b py-4 px-6">
        <h1 className="text-xl font-bold">DocMind</h1>
        <p className="text-sm text-gray-500">技术文档智能助手</p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">你好，我是 DocMind</p>
            <p className="text-sm mt-2">
              试试问我：Redis集群怎么扩容？Docker如何优化镜像大小？
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>

              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <p className="text-xs font-semibold text-gray-500 mb-1">
                    参考来源
                  </p>
                  {msg.citations.map((c) => (
                    <div key={c.index} className="text-xs text-gray-500 mt-1">
                      <span className="font-medium">[{c.index}]</span>{" "}
                      {c.source} 第{c.chunk_index}段
                      <p className="text-gray-400 truncate">{c.preview}</p>
                    </div>
                  ))}
                </div>
              )}

              {msg.role === "assistant" && msg.content && (
                <div className="mt-2 flex gap-2 justify-end">
                  <button
                    onClick={() => handleFeedback("like")}
                    className="text-xs px-2 py-1 rounded hover:bg-gray-200"
                    title="有用"
                  >
                    👍
                  </button>
                  <button
                    onClick={() => handleFeedback("dislike")}
                    className="text-xs px-2 py-1 rounded hover:bg-gray-200"
                    title="无用"
                  >
                    👎
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-3 text-gray-500">
              思考中...
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onCompositionStart={() => { isComposing.current = true; }}
            onCompositionEnd={(e) => {
              isComposing.current = false;
              setInput(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleKeyDown();
              }
            }}
            placeholder="输入你的问题..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
