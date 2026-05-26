"use client";
import { useState } from "react";

interface Step {
  step: string;
  count?: number;
  results?: any[];
  elapsed_ms: number;
  input?: string;
  output?: string;
}

const STEP_LABELS: Record<string, string> = {
  query_rewrite: "Query 改写",
  vector_search: "向量检索",
  bm25_search: "BM25 检索",
  rrf_fusion: "RRF 融合",
};

export default function DebugPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const search = async () => {
    setLoading(true);
    const res = await fetch(
      `http://localhost:8000/api/v1/debug/retrieve?query=${encodeURIComponent(query)}`
    );
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">检索调试面板</h1>

      <div className="flex gap-2 mb-8">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入查询..."
          className="flex-1 border rounded-lg px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <button
          onClick={search}
          disabled={loading}
          className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "检索中..." : "调试"}
        </button>
      </div>

      {result && (
        <div>
          <div className="mb-4 text-sm text-gray-500">
            原始 Query: "{result.original_query}" →
            改写: "{result.rewritten_query}" · 总耗时: {result.total_elapsed_ms}ms
          </div>

          <div className="grid grid-cols-1 gap-6">
            {result.steps?.map((step: Step, i: number) => (
              <div key={i} className="border rounded-lg p-4">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-semibold text-lg">
                    Step {i + 1}: {STEP_LABELS[step.step] || step.step}
                  </h3>
                  <span className="text-sm text-gray-400">{step.elapsed_ms}ms</span>
                </div>

                {step.step === "query_rewrite" && (
                  <div>
                    <p className="text-gray-500">输入: {step.input}</p>
                    <p className="text-green-600 font-medium">输出: {step.output}</p>
                  </div>
                )}

                <div className="text-sm text-gray-500 mb-2">
                  共 {step.count} 条结果
                </div>

                <div className="space-y-2">
                  {step.results?.slice(0, 5).map((r: any, j: number) => (
                    <div key={j} className="bg-gray-50 rounded p-3 text-sm">
                      <div className="flex justify-between text-gray-400 mb-1">
                        <span>{r.source}</span>
                        <span>#{r.chunk_index}</span>
                      </div>
                      <p className="text-gray-700 line-clamp-3">{r.content_preview}</p>
                      {(r.rrf_score || r.bm25_score) && (
                        <div className="text-xs text-blue-500 mt-1">
                          {r.rrf_score && `RRF: ${r.rrf_score.toFixed(4)} `}
                          {r.bm25_score && `BM25: ${r.bm25_score.toFixed(2)}`}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
