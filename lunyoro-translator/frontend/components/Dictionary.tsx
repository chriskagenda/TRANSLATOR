"use client";
import { useState, useMemo, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DictDirection = "en→lun" | "lun→en";
type PosFilter = "ALL" | "N" | "V" | "ADJ" | "OTHER";

const POS_LABELS: Record<string, { label: string; color: string }> = {
  N:    { label: "Noun",      color: "bg-blue-100 text-blue-700" },
  V:    { label: "Verb",      color: "bg-green-100 text-green-700" },
  ADJ:  { label: "Adjective", color: "bg-orange-100 text-orange-700" },
  PART: { label: "Particle",  color: "bg-yellow-100 text-yellow-700" },
  PRON: { label: "Pronoun",   color: "bg-pink-100 text-pink-700" },
};

interface DictEntry {
  word: string;
  definitionEnglish: string;
  definitionNative: string;
  exampleSentence1: string;
  exampleSentence1English: string;
  dialect: string;
  pos: string;
  source?: "neural_mt" | "dictionary" | "corpus";
  confidence?: number;
  pos_matched?: boolean;
}

export default function Dictionary() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<DictEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [direction, setDirection] = useState<DictDirection>("en→lun");
  const [posFilter, setPosFilter] = useState<PosFilter>("ALL");
  const [interjections, setInterjections] = useState<Record<string, string>>({});
  const [idioms, setIdioms] = useState<Record<string, string>>({});
  const [ruleHint, setRuleHint] = useState<string | null>(null);

  // Load interjections and idioms once
  useEffect(() => {
    fetch(`${API}/language-rules/interjections`)
      .then(r => r.json()).then(d => setInterjections(d.interjections || {})).catch(() => {});
    fetch(`${API}/language-rules/idioms`)
      .then(r => r.json()).then(d => setIdioms(d.idioms || {})).catch(() => {});
  }, []);

  const placeholder = direction === "en→lun"
    ? "Search an English word..."
    : "Search a Lunyoro / Rutooro word...";

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    setPosFilter("ALL");
    setRuleHint(null);

    // Check interjections and idioms first
    const q = query.toLowerCase().trim();
    if (interjections[q]) {
      setRuleHint(`Interjection: "${query}" — ${interjections[q]}`);
    } else if (idioms[q]) {
      setRuleHint(`Idiom: "${query}" — ${idioms[q]}`);
    }

    // R/L rule hint for lun→en searches
    if (direction === "lun→en" && /[lL]/.test(query)) {
      setRuleHint(prev => (prev ? prev + "\n" : "") +
        "R/L Rule: L is only used before/after 'e' or 'i' vowels. All other positions use R.");
    }

    try {
      const res = await fetch(`${API}/lookup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: query, direction }),
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleDirectionChange(d: DictDirection) {
    setDirection(d);
    setQuery("");
    setResults([]);
    setSearched(false);
    setPosFilter("ALL");
    setRuleHint(null);
  }

  const filtered = useMemo(() => {
    if (posFilter === "ALL") return results;
    if (posFilter === "OTHER") return results.filter(r => !r.pos || !["N","V","ADJ"].includes(r.pos.toUpperCase()));
    return results.filter(r => (r.pos || "").toUpperCase() === posFilter);
  }, [results, posFilter]);

  return (
    <div className="space-y-4">
      {/* Direction toggle */}
      <div className="flex rounded-lg border border-gray-200 overflow-hidden text-sm font-medium">
        <button
          onClick={() => handleDirectionChange("en→lun")}
          className={`flex-1 py-2 transition-colors ${direction === "en→lun" ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}
        >
          English → Lunyoro / Rutooro
        </button>
        <button
          onClick={() => handleDirectionChange("lun→en")}
          className={`flex-1 py-2 transition-colors ${direction === "lun→en" ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}
        >
          Lunyoro / Rutooro → English
        </button>
      </div>

      {/* Search bar */}
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "..." : "Search"}
        </button>
      </div>

      {/* Language rule hint */}
      {ruleHint && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800 whitespace-pre-line">
          📖 {ruleHint}
        </div>
      )}

      {/* POS filter tabs — only shown when results exist */}
      {results.length > 0 && (        <div className="flex gap-1.5 flex-wrap">
          {(["ALL", "N", "V", "ADJ"] as PosFilter[]).map((p) => {
            const info = p === "ALL" ? null : POS_LABELS[p];
            const count = p === "ALL"
              ? results.length
              : results.filter(r => (r.pos || "").toUpperCase() === p).length;
            if (p !== "ALL" && count === 0) return null;
            return (
              <button
                key={p}
                onClick={() => setPosFilter(p)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  posFilter === p
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {p === "ALL" ? "All" : info?.label} {count > 0 && <span className="opacity-70">({count})</span>}
              </button>
            );
          })}
        </div>
      )}

      {searched && results.length === 0 && !loading && (
        <p className="text-gray-500 text-sm text-center py-4">
          No results found for &quot;{query}&quot;
        </p>
      )}

      <div className="space-y-3">
        {results.length > 0 && (
          <p className="text-xs text-gray-400 font-medium">Best match</p>
        )}
        {filtered.map((entry, i) => {
          const posKey = (entry.pos || "").toUpperCase();
          const posInfo = POS_LABELS[posKey];
          return (
            <div key={i} className={`bg-white border rounded-lg p-4 ${entry.pos_matched ? "border-blue-200" : "border-gray-200"}`}>
              <div className="flex justify-between items-start mb-1">
                {direction === "en→lun" ? (
                  <div>
                    <span className="text-xs text-gray-400 uppercase tracking-wide">Lunyoro / Rutooro</span>
                    <p className="text-lg font-semibold text-gray-800">{entry.word}</p>
                  </div>
                ) : (
                  <div>
                    <span className="text-xs text-gray-400 uppercase tracking-wide">English</span>
                    <p className="text-lg font-semibold text-gray-800">{entry.definitionEnglish || entry.word}</p>
                  </div>
                )}

                <div className="flex gap-1 mt-1 flex-wrap justify-end items-center">
                  {entry.source === "neural_mt" && (
                    <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">AI</span>
                  )}
                  {entry.source === "corpus" && (
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">corpus</span>
                  )}
                  {posInfo ? (
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${posInfo.color}`}>
                      {posInfo.label}
                    </span>
                  ) : entry.pos ? (
                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">{entry.pos}</span>
                  ) : null}
                  {entry.dialect && (
                    <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded">{entry.dialect}</span>
                  )}
                  {entry.confidence !== undefined && entry.confidence < 1 && (
                    <span className="text-xs text-gray-400">{Math.round(entry.confidence * 100)}%</span>
                  )}
                </div>
              </div>

              {/* Definitions */}
              {direction === "en→lun" ? (
                <>
                  {entry.definitionEnglish && (
                    <p className="text-sm text-gray-600">{entry.definitionEnglish}</p>
                  )}
                  {entry.definitionNative && (
                    <p className="text-sm text-gray-500 italic mt-0.5">{entry.definitionNative}</p>
                  )}
                </>
              ) : (
                <>
                  {entry.word && (
                    <p className="text-sm text-gray-600">
                      <span className="font-medium text-gray-700">Lunyoro: </span>{entry.word}
                    </p>
                  )}
                  {entry.definitionNative && (
                    <p className="text-sm text-gray-500 italic mt-0.5">{entry.definitionNative}</p>
                  )}
                </>
              )}

              {/* Example sentences */}
              {(entry.exampleSentence1 || entry.exampleSentence1English) && (
                <div className="mt-2 text-xs text-gray-500 border-t border-gray-100 pt-2 space-y-0.5">
                  {entry.exampleSentence1 && (
                    <p className="font-medium text-gray-600">{entry.exampleSentence1}</p>
                  )}
                  {entry.exampleSentence1English && (
                    <p>{entry.exampleSentence1English}</p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
