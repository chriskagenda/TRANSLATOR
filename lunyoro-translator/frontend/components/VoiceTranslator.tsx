"use client";
import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Lang = "English" | "Lunyoro" | "Rutooro";

const LANGS: Lang[] = ["English", "Lunyoro", "Rutooro"];

// Web Speech API types
declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly length: number;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
}

export default function VoiceTranslator() {
  const [lang, setLang] = useState<Lang>("English");
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [translation, setTranslation] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [pulse, setPulse] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    if (pulse) {
      const t = setTimeout(() => setPulse(false), 600);
      return () => clearTimeout(t);
    }
  }, [pulse]);

  function getSpeechLang(l: Lang): string {
    if (l === "English") return "en-US";
    // Lunyoro/Rutooro — use Uganda English as closest available
    return "en-UG";
  }

  function startRecording() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setError("Speech recognition is not supported in this browser. Try Chrome.");
      return;
    }
    setError("");
    setTranscript("");
    setTranslation("");

    const rec = new SR();
    rec.lang = getSpeechLang(lang);
    rec.continuous = false;
    rec.interimResults = true;

    rec.onresult = (e: SpeechRecognitionEvent) => {
      const text = Array.from({ length: e.results.length }, (_, i) => e.results[i][0].transcript).join("");
      setTranscript(text);
      setPulse(true);
    };

    rec.onerror = (e: SpeechRecognitionErrorEvent) => {
      setError(`Microphone error: ${e.error}`);
      setRecording(false);
    };

    rec.onend = () => {
      setRecording(false);
      const finalText = transcript;
      if (finalText.trim()) translateVoice(finalText);
    };

    recognitionRef.current = rec;
    rec.start();
    setRecording(true);
  }

  function stopRecording() {
    recognitionRef.current?.stop();
    setRecording(false);
  }

  async function translateVoice(text: string) {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const endpoint = lang === "English" ? "/translate" : "/translate-reverse";
      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setTranslation(data.translation || "No translation found.");

      // speak the translation back
      if (data.translation) {
        const utter = new SpeechSynthesisUtterance(data.translation);
        utter.lang = lang === "English" ? "en-UG" : "en-US";
        window.speechSynthesis.speak(utter);
      }
    } catch {
      setError("Could not connect to the translation server.");
    } finally {
      setLoading(false);
    }
  }

  function handleMicClick() {
    if (recording) stopRecording();
    else startRecording();
  }

  return (
    <div className="flex flex-col items-center space-y-8 py-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800">Voice Translation</h2>
        <p className="text-gray-500 mt-2 max-w-sm text-sm leading-relaxed">
          Speak in your chosen language and get an instant translation to Lunyoro / Rutooro or English.
        </p>
      </div>

      {/* Mic button */}
      <div className="relative flex items-center justify-center">
        {recording && (
          <>
            <span className="absolute w-44 h-44 rounded-full bg-blue-200 opacity-40 animate-ping" />
            <span className="absolute w-36 h-36 rounded-full bg-blue-100 opacity-60 animate-ping" style={{ animationDelay: "0.15s" }} />
          </>
        )}
        <button
          onClick={handleMicClick}
          className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-lg transition-all duration-200 ${
            recording ? "bg-blue-600 scale-105" : "bg-gray-100 hover:bg-gray-200"
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
            stroke={recording ? "white" : "#6b7280"}
            strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-12 h-12">
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="19" x2="12" y2="22" />
            <line x1="8" y1="22" x2="16" y2="22" />
          </svg>
        </button>
      </div>

      {/* Start / Stop button */}
      <button
        onClick={handleMicClick}
        className={`flex items-center gap-2 px-8 py-2.5 rounded-lg text-white font-medium text-sm transition-colors ${
          recording ? "bg-red-500 hover:bg-red-600" : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="19" x2="12" y2="22" />
          <line x1="8" y1="22" x2="16" y2="22" />
        </svg>
        {recording ? "Stop Recording" : "Start Speaking"}
      </button>

      {/* Language pills */}
      <div className="flex gap-2 flex-wrap justify-center">
        {LANGS.map((l) => (
          <button
            key={l}
            onClick={() => { setLang(l); setTranscript(""); setTranslation(""); }}
            className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
              lang === l
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"
            }`}
          >
            {l}
          </button>
        ))}
      </div>

      {/* Transcript & translation */}
      {(transcript || loading || translation) && (
        <div className="w-full max-w-md space-y-3">
          {transcript && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-xs text-gray-400 mb-1 uppercase tracking-wide">You said</p>
              <p className={`text-sm text-gray-800 transition-opacity ${pulse ? "opacity-60" : "opacity-100"}`}>{transcript}</p>
            </div>
          )}
          {loading && (
            <div className="text-center text-sm text-blue-500 animate-pulse">Translating...</div>
          )}
          {translation && (
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <p className="text-xs text-blue-400 mb-1 uppercase tracking-wide">
                {lang === "English" ? "Lunyoro / Rutooro" : "English"}
              </p>
              <p className="text-sm text-gray-800">{translation}</p>
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="text-sm text-red-500 text-center">{error}</p>
      )}
    </div>
  );
}
