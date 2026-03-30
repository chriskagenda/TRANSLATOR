"use client";
import { useState, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Sentence {
  original: string;
  translation: string;
  confidence?: number;
  method?: string;
}

interface Page {
  page: number;
  sentences: Sentence[];
}

interface PdfResult {
  filename: string;
  total_pages: number;
  pages: Page[];
}

export default function PdfTranslator() {
  const [result, setResult] = useState<PdfResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file: File) {
    if (!file.name.endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setFileName(file.name);
    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/translate-pdf`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error("Translation failed");
      setResult(await res.json());
    } catch {
      setError("Could not translate the PDF. Make sure the backend is running.");
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

  function downloadResult() {
    if (!result) return;
    const lines: string[] = [`PDF Translation: ${result.filename}\n`];
    for (const page of result.pages) {
      lines.push(`\n--- Page ${page.page} ---\n`);
      for (const s of page.sentences) {
        lines.push(`EN: ${s.original}`);
        lines.push(`LUN: ${s.translation}\n`);
      }
    }
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = result.filename.replace(".pdf", "_lunyoro.txt");
    a.click();
    URL.revokeObjectURL(url);
  }

  const totalSentences = result?.pages.reduce((acc, p) => acc + p.sentences.length, 0) ?? 0;

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
        <input ref={inputRef} type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
        <div className="text-4xl mb-3">📄</div>
        {loading ? (
          <p className="text-sm text-blue-600 font-medium">Translating {fileName}...</p>
        ) : fileName && result ? (
          <p className="text-sm text-green-600 font-medium">{fileName} — done</p>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-700">Drop a PDF here or click to upload</p>
            <p className="text-xs text-gray-400 mt-1">English PDF → Lunyoro / Rutooro translation</p>
          </>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">{error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">
              {result.total_pages} page{result.total_pages > 1 ? "s" : ""} · {totalSentences} sentences translated
            </p>
            <button
              onClick={downloadResult}
              className="text-sm bg-green-600 text-white px-3 py-1.5 rounded-lg hover:bg-green-700 transition-colors"
            >
              ⬇ Download .txt
            </button>
          </div>

          {result.pages.map((page) => (
            <details key={page.page} open={page.page === 1} className="bg-white border border-gray-200 rounded-lg">
              <summary className="px-4 py-3 text-sm font-medium text-gray-700 cursor-pointer select-none">
                Page {page.page} — {page.sentences.length} sentences
              </summary>
              <div className="divide-y divide-gray-100">
                {page.sentences.map((s, i) => (
                  <div key={i} className="px-4 py-3 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-gray-400 mb-0.5 uppercase tracking-wide">English</p>
                      <p className="text-sm text-gray-700">{s.original}</p>
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-0.5">
                        <p className="text-xs text-gray-400 uppercase tracking-wide">Lunyoro / Rutooro</p>
                        {s.confidence !== undefined && (
                          <span className={`text-xs ${s.confidence > 0.8 ? "text-green-500" : s.confidence > 0.5 ? "text-yellow-500" : "text-red-400"}`}>
                            {Math.round(s.confidence * 100)}%
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-800">{s.translation}</p>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
