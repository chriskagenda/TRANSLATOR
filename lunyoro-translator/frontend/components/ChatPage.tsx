"use client";

import { useState, useEffect, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
};

const SECTORS = [
  { code: "ALL", label: "All Sectors",           icon: "🌐", prompt: "Give me a mix of Runyoro-Rutooro vocabulary across all topics." },
  { code: "DLY", label: "Daily Life",             icon: "🏠", prompt: "What are common Runyoro-Rutooro words used in everyday daily life?" },
  { code: "NAR", label: "Storytelling",           icon: "📖", prompt: "Tell me a short story or proverb in Runyoro-Rutooro." },
  { code: "SPR", label: "Spirituality",           icon: "🙏", prompt: "Tell me about spiritual and religious terms in Runyoro-Rutooro." },
  { code: "AGR", label: "Agriculture",            icon: "🌾", prompt: "What are common Runyoro-Rutooro words used in farming and agriculture?" },
  { code: "EDU", label: "Education",              icon: "📚", prompt: "What are common Runyoro-Rutooro words used in education and schools?" },
  { code: "ENV", label: "Environment & Nature",   icon: "🌿", prompt: "What are Runyoro-Rutooro words related to the environment and nature?" },
  { code: "ART", label: "Arts & Music",           icon: "🎵", prompt: "Tell me about traditional Runyoro-Rutooro arts and music." },
  { code: "CUL", label: "Culture & Traditions",   icon: "🏛️", prompt: "Tell me about Runyoro-Rutooro culture and traditions." },
  { code: "GOV", label: "Governance",             icon: "⚖️", prompt: "What are Runyoro-Rutooro words related to governance and leadership?" },
  { code: "ECO", label: "Economy & Trade",        icon: "💰", prompt: "What are Runyoro-Rutooro words related to trade and the economy?" },
  { code: "HIS", label: "History",                icon: "🏺", prompt: "Tell me about historical terms and events in Runyoro-Rutooro." },
  { code: "HLT", label: "Health",                 icon: "🏥", prompt: "What are Runyoro-Rutooro words related to health and medicine?" },
  { code: "POL", label: "Politics",               icon: "🗳️", prompt: "What are Runyoro-Rutooro words related to politics?" },
];

export default function ChatPage() {
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [sectorOpen, setSectorOpen] = useState(false);
  const [selectedSector, setSelectedSector] = useState<typeof SECTORS[0] | null>(null);
  const [conversationMode, setConversationMode] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    { label: "Translate to Runyoro", prompt: "How do I say 'Welcome to our home' in Runyoro?" },
    { label: "Grammar Help",         prompt: "Explain the difference between 'Kwebembera' and 'Kutandika'." },
    { label: "Conversation",         prompt: "CONVERSATION_MODE" },
  ];

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setSectorOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

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
          sector: selectedSector?.code || null,
          conversation_mode: conversationMode,
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
            onClick={() => { setHistory([]); setConversationMode(false); }}
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

            {/* Suggestion Chips + Culture Dropdown */}
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => {
                    if (s.prompt === "CONVERSATION_MODE") {
                      setConversationMode(true);
                    } else {
                      sendMessage(s.prompt);
                    }
                  }}
                  className="text-xs bg-white border border-blue-200 text-blue-600 px-3 py-2 rounded-full hover:bg-blue-600 hover:text-white hover:border-blue-600 transition-all shadow-sm"
                >
                  {s.label}
                </button>
              ))}

              {/* Culture sector dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setSectorOpen((o) => !o)}
                  className="text-xs bg-white border border-blue-200 text-blue-600 px-3 py-2 rounded-full hover:bg-blue-600 hover:text-white hover:border-blue-600 transition-all shadow-sm flex items-center gap-1"
                >
                  {selectedSector ? `${selectedSector.icon} ${selectedSector.label}` : "🏛️ Culture & Sectors"}
                  <span className="ml-1">▾</span>
                </button>

                {sectorOpen && (
                  <div className="absolute left-0 top-full mt-1 w-56 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
                    {SECTORS.map((s) => (
                      <button
                        key={s.code}
                        onClick={() => {
                          setSelectedSector(s);
                          setSectorOpen(false);
                        }}
                        className="w-full text-left px-4 py-2.5 text-sm text-gray-700 hover:bg-blue-50 hover:text-blue-700 flex items-center gap-2 transition-colors"
                      >
                        <span>{s.icon}</span>
                        <span>{s.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
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

      {/* Sector badge + Input Area */}
      <div className="p-4 bg-gray-50 border-t border-gray-100 space-y-2">
        {conversationMode && (
          <div className="flex items-center gap-2 text-xs text-green-700 bg-green-50 border border-green-100 rounded-lg px-3 py-1.5">
            <span>💬</span>
            <span>Conversation mode — type in Runyoro-Rutooro, model replies in Runyoro-Rutooro</span>
            <button onClick={() => setConversationMode(false)} className="ml-auto text-green-400 hover:text-green-700">✕</button>
          </div>
        )}
        {selectedSector && (
          <div className="flex items-center gap-2 text-xs text-blue-700 bg-blue-50 border border-blue-100 rounded-lg px-3 py-1.5">
            <span>{selectedSector.icon}</span>
            <span>Sector: <strong>{selectedSector.label}</strong> — translations will focus on this domain</span>
            <button
              onClick={() => setSelectedSector(null)}
              className="ml-auto text-blue-400 hover:text-blue-700"
            >✕</button>
          </div>
        )}
        <div className="flex gap-2">
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder={conversationMode ? "Ngamba omu Runyoro-Rutooro..." : selectedSector ? `Ask about ${selectedSector.label} in Runyoro-Rutooro...` : "Type your question here..."}
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
