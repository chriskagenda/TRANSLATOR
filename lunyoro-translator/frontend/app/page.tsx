"use client";
import { useState } from "react";
import Translator from "@/components/Translator";
import Dictionary from "@/components/Dictionary";
import History from "@/components/History";
import PdfTranslator from "@/components/PdfTranslator";
import VoiceTranslator from "@/components/VoiceTranslator";

type Tab = "translate" | "voice" | "pdf" | "dictionary" | "history";

export default function Home() {
  const [tab, setTab] = useState<Tab>("translate");

  return (
    <main className="max-w-3xl mx-auto px-4 py-10">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">Lunyoro / Rutooro Translator</h1>
        <p className="text-gray-500 mt-1">Translate English to Lunyoro and Rutooro</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-200">
        {(["translate", "voice", "pdf", "dictionary", "history"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize transition-colors ${
              tab === t
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {t === "pdf" ? "PDF" : t === "voice" ? "🎙 Voice" : t}
          </button>
        ))}
      </div>

      {tab === "translate" && <Translator />}
      {tab === "voice" && <VoiceTranslator />}
      {tab === "pdf" && <PdfTranslator />}
      {tab === "dictionary" && <Dictionary />}
      {tab === "history" && <History />}
    </main>
  );
}
