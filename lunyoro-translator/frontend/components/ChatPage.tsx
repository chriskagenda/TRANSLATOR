"use client";

import { useState, useEffect, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
};

export default function ChatPage() {
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatItem[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Quick Action Suggestions
  const suggestions = [
    { label: "Translate to Runyoro", prompt: "How do I say 'Welcome to our home' in Runyoro?" },
    { label: "Grammar Help", prompt: "Explain the difference between 'Kwebembera' and 'Kutandika'." },
    { label: "Culture", prompt: "Tell me about the importance of Empaako in Bunyoro culture." },
    { label: "Short Story", prompt: "Tell me a very short story in Rutooro." }
  ];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, loading]);

  async function sendMessage(overrideMessage?: string) {
    const textToSend = overrideMessage || message;
    if (!textToSend.trim() || loading) return;

    const userMessage: ChatItem = { role: "user", content: textToSend };
    const newHistory = [...history, userMessage];
    
    setHistory(newHistory);
    setMessage("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: textToSend,
          history: history,
        }),
      });

      const data = await res.json();
      const botMessage: ChatItem = {
        role: "assistant",
        content: data.reply || "No response returned.",
      };

      setHistory([...newHistory, botMessage]);
    } catch (error) {
      setHistory([
        ...newHistory,
        { role: "assistant", content: "Error talking to backend. Make sure the server is running." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col transition-all">
      {/* Header */}
      <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
        <div>
          <h2 className="font-bold text-gray-800 text-lg">AI Language Assistant</h2>
          <p className="text-xs text-gray-500">Ask questions in English or Runyoro-Rutooro</p>
        </div>
        {history.length > 0 && (
          <button 
            onClick={() => setHistory([])}
            className="text-xs text-red-500 hover:text-red-700 font-medium px-2 py-1 rounded-md hover:bg-red-50 transition-colors"
          >
            Clear Chat
          </button>
        )}
      </div>

      {/* Chat Window */}
      <div 
        ref={scrollRef}
        className="h-112.5 overflow-y-auto p-4 space-y-4 bg-white scroll-smooth"
      >
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
            <div className="space-y-2">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-2xl mx-auto shadow-inner">💬</div>
              <h3 className="font-semibold text-gray-700">Oraire otya?</h3>
              <p className="text-gray-400 text-sm max-w-62.5">I can help you translate, explain grammar, or chat about culture.</p>
            </div>
            
            {/* Suggestion Chips */}
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s.prompt)}
                  className="text-xs bg-white border border-blue-200 text-blue-600 px-3 py-2 rounded-full hover:bg-blue-600 hover:text-white hover:border-blue-600 transition-all shadow-sm"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          history.map((item, index) => (
            <div
              key={index}
              className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                  item.role === "user"
                    ? "bg-blue-600 text-white rounded-tr-none"
                    : "bg-gray-100 text-gray-800 rounded-tl-none border border-gray-200"
                }`}
              >
                {item.content}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="flex justify-start items-center space-x-2">
            <div className="bg-gray-100 px-4 py-3 rounded-2xl rounded-tl-none border border-gray-200">
               <div className="flex space-x-1">
                 <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                 <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                 <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
               </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-gray-50 border-t border-gray-100">
        <div className="flex gap-2">
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Type your question here..."
            className="flex-1 px-4 py-3 bg-white border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-sm text-gray-800 shadow-sm transition-all"
          />
          <button
            onClick={() => sendMessage()}
            disabled={loading || !message.trim()}
            className="px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl font-bold text-sm transition-all shadow-md flex items-center justify-center active:scale-95"
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
