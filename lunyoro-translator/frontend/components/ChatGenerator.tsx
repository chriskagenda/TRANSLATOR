"use client";
import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  prompt?: string;
  prompt_translated?: string;
  generated: string;
  related?: { english: string; lunyoro: string; score: number }[];
}

export default function ChatGenerator() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const prompt = input.trim();
    if (!prompt || loading) return;
    setInput("");
    setError("");
    setLoading(true);

    // Optimistically add user bubble
    setMessages((prev) => [...prev, { role: "user", generated: prompt }]);

    try {
      const res = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          prompt: data.prompt,
          prompt_translated: data.prompt_translated,
          generated: data.generated,
          related: data.related,
        },
      ]);
    } catch {
      setError("Could not reach the backend. Make sure it is running.");
      // remove optimistic user bubble on failure
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[600px]">
      {/* Chat history */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 mb-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 text-sm mt-16">
            <p className="text-2xl mb-2">💬</p>
            <p>Ask anything in English and get a response in Lunyoro / Rutooro.</p>
            <p className="mt-1 text-xs">e.g. "How are you?" · "What is your name?" · "Tell me about the weather."</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "user" ? (
              <div className="bg-blue-600 text-white rounded-2xl rounded-br-sm px-4 py-2.5 max-w-[75%] text-sm leading-relaxed">
                {msg.generated}
              </div>
            ) : (
              <div className="max-w-[80%] space-y-1.5">
                {/* Lunyoro response */}
                <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                  <p className="text-xs text-gray-400 mb-1 font-medium uppercase tracking-wide">Lunyoro / Rutooro</p>
                  <p className="text-gray-800 text-sm leading-relaxed">{msg.generated}</p>
                </div>

                {/* Prompt translation hint */}
                {msg.prompt_translated && msg.prompt_translated !== msg.generated && (
                  <p className="text-xs text-gray-400 px-1">
                    Your prompt translated: <span className="italic">{msg.prompt_translated}</span>
                  </p>
                )}

                {/* Related sentences toggle */}
                {msg.related && msg.related.length > 0 && (
                  <details className="bg-gray-50 border border-gray-100 rounded-xl px-3 py-2 text-xs">
                    <summary className="cursor-pointer text-gray-500 font-medium">
                      {msg.related.length} related phrase{msg.related.length > 1 ? "s" : ""}
                    </summary>
                    <ul className="mt-2 space-y-2">
                      {msg.related.map((r, j) => (
                        <li key={j} className="border-t border-gray-100 pt-1.5">
                          <p className="text-gray-400">{r.english}</p>
                          <p className="text-gray-700 font-medium">{r.lunyoro}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
              <span className="flex gap-1 items-center text-gray-400 text-sm">
                <span className="animate-bounce [animation-delay:0ms]">●</span>
                <span className="animate-bounce [animation-delay:150ms]">●</span>
                <span className="animate-bounce [animation-delay:300ms]">●</span>
              </span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-2.5 text-sm mb-3">{error}</div>
      )}

      {/* Input bar */}
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
          placeholder="Type in English..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl px-4 py-2.5 text-sm font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
