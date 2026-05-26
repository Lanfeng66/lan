'use client'

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">DocMind</h1>
      <p className="mb-4">技术文档智能问答系统</p>
      <a
        href="/chat"
        className="inline-block bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600"
        style={{ position: "relative", zIndex: 9999 }}
      >
        进入聊天
      </a>
    </div>
  );
}

