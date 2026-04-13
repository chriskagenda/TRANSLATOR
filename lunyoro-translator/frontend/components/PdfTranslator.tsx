"use client";
import { useState, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SummaryResult {
  filename: string;
  total_pages: number;
  total_sentences: number;
  language_detected: string;
  summary: string;
  summary_lunyoro: string;
  summary_lunyoro_marian?: string;
  summary_lunyoro_nllb?: string;
  sentences_used: number;
}

export default function PdfTranslator() {
  const [summary, setSummary] = useState<SummaryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file: File) {
    const supported = [".pdf", ".docx", ".doc", ".txt"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!supported.includes(ext)) {
      setError("Supported formats: PDF, DOCX, DOC, TXT");
      return;
    }
    setFileName(file.name);
    setLoading(true);
    setError("");
    setSummary(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/summarize-pdf`, { method: "POST", body: form });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || "Could not process the document.");
        return;
      }
      setSummary(await res.json());
    } catch {
      setError("Could not connect to the backend. Make sure it is running.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  }

  return (
    <div className="space-y-4">
      {/* Upload area */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          dragOver ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        }`}
      >
        <input ref={inputRef} type="file" accept=".pdf,.docx,.doc,.txt" className="hidden" onChange={handleFileChange} />
        <div className="text-4xl mb-3">📝</div>
        {loading ? (
          <p className="text-sm text-blue-600 font-medium">Summarizing {fileName}...</p>
        ) : fileName && summary ? (
          <p className="text-sm text-green-600 font-medium">{fileName} — done</p>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-700">Drop a document here or click to upload</p>
            <p className="text-xs text-gray-400 mt-1">PDF, DOCX, DOC, TXT → English summary</p>
          </>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">{error}</div>
      )}

      {summary && (
        <div className="space-y-3">
          <div className="flex justify-between items-center text-sm text-gray-500">
            <span>
              {summary.total_pages} page{summary.total_pages > 1 ? "s" : ""} · {summary.total_sentences} sentences · {summary.sentences_used} key sentences extracted
            </span>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded capitalize">
              {summary.language_detected} detected
            </span>
          </div>
          <details className="bg-white border border-blue-200 rounded-lg">
            <summary className="px-5 py-3 text-xs text-blue-600 font-medium uppercase tracking-wide cursor-pointer select-none">
              English Summary
            </summary>
            <p className="px-5 pb-4 text-sm text-gray-800 leading-relaxed">{summary.summary}</p>
          </details>
          {summary.summary_lunyoro && (
            <div className="bg-white border border-green-200 rounded-lg p-5 space-y-3">
              <p className="text-xs text-green-600 font-medium uppercase tracking-wide">Lunyoro / Rutooro Summary</p>
              <p className="text-sm text-gray-800 leading-relaxed">{summary.summary_lunyoro}</p>
              {summary.summary_lunyoro_marian && summary.summary_lunyoro_nllb && (
                <div className="pt-3 border-t border-gray-100 space-y-2">
                  <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Model comparison</p>
                  <div className="space-y-1">
                    <p className="text-xs text-gray-700"><span className="font-semibold text-blue-600">MarianMT:</span> {summary.summary_lunyoro_marian}</p>
                    <p className="text-xs text-gray-700"><span className="font-semibold text-purple-600">NLLB-200:</span> {summary.summary_lunyoro_nllb}</p>
                  </div>
                </div>
              )}
            </div>
          )}
          <button
            onClick={() => {
              const content = `English Summary:\n${summary.summary}\n\nLunyoro / Rutooro Summary:\n${summary.summary_lunyoro}`;
              const blob = new Blob([content], { type: "text/plain" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = summary.filename.replace(/\.[^.]+$/, "_summary.txt");
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="text-sm bg-green-600 text-white px-3 py-1.5 rounded-lg hover:bg-green-700 transition-colors"
          >
            ⬇ Download summary
          </button>
        </div>
      )}
    </div>
  );
}
