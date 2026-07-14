"use client";

import {
  Bot,
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileSearch,
  FileText,
  Loader2,
  LogOut,
  MessageSquareText,
  Search,
  Send,
  ShieldCheck,
  Upload,
  XCircle
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AskResponse,
  AuthUser,
  DocumentIndexingResponse,
  DocumentUploadResponse,
  QualityMode,
  SearchMode,
  askQuestion,
  currentUser,
  logout,
  uploadAndIndexDocument
} from "../lib/api";

const EXAMPLE_PROMPTS = [
  "What does IC.1 require?",
  "Summarize infection prevention standards.",
  "Which actions apply to sterile processing?"
];

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  response?: AskResponse;
};

export default function Home() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [question, setQuestion] = useState("");
  const [searchMode, setSearchMode] = useState<SearchMode>("auto");
  const [qualityMode, setQualityMode] = useState<QualityMode>("fast");
  const [topK, setTopK] = useState(3);
  const [rerank, setRerank] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<DocumentUploadResponse | null>(null);
  const [indexing, setIndexing] = useState<DocumentIndexingResponse | null>(null);
  const [documentBusy, setDocumentBusy] = useState<"upload" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void currentUser()
      .then((payload) => setUser(payload.user))
      .catch(() => {
        router.replace("/login");
      });
  }, [router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatBusy]);

  async function handleLogout() {
    await logout();
    router.replace("/login");
    router.refresh();
  }

  async function runAsk(userText: string) {
    if (!userText.trim() || chatBusy) return;
    setError(null);
    setChatBusy(true);
    setQuestion("");
    setMessages((current) => [...current, { role: "user", text: userText }]);
    try {
      const response = await askQuestion(userText, {
        searchMode,
        qualityMode,
        topK,
        rerank
      });
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: response.answer?.summary?.text ?? "No grounded answer returned.",
          response
        }
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Question failed.");
    } finally {
      setChatBusy(false);
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runAsk(question.trim());
  }

  async function submitUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setError(null);
    setDocumentBusy("upload");
    setIndexing(null);
    try {
      const result = await uploadAndIndexDocument(file);
      setUploadResult({ ...result.document, status: result.status });
      setIndexing(result.indexing);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload and indexing failed.");
    } finally {
      setDocumentBusy(null);
    }
  }

  return (
    <main className="app-shell">
      <header className="brand-bar">
        <div className="brand-mark">
          <h1 className="brand-name">
            Cite<em>Med</em>
          </h1>
          <p className="brand-tag">Grounded clinical answers from your source documents.</p>
        </div>
        <div className="brand-actions">
          <div className="connection-chip">
            <span className="dot" aria-hidden />
            {user ? user.name : "Loading…"}
          </div>
          <button className="ghost" onClick={() => void handleLogout()} type="button">
            <LogOut size={16} />
            Log out
          </button>
        </div>
      </header>

      {error && (
        <div className="alert" role="alert">
          <XCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div className="workspace">
        <section className="rail" aria-labelledby="documents-title">
          <div className="rail-header">
            <div>
              <h2 id="documents-title">Documents</h2>
              <p>Upload once. We extract, chunk, embed, and index automatically.</p>
            </div>
            <div className="step-badge">
              <Database size={14} />
              Pipeline
            </div>
          </div>

          <form className="upload-zone" onSubmit={submitUpload}>
            <label
              className="drop-target"
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
            >
              <Upload size={22} color="#0d6b5c" />
              <strong>{file ? "Ready to upload" : "Drop a source file"}</strong>
              <span>PDF, CSV, or JSON. Upload makes it searchable.</span>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.csv,.json,application/pdf,text/csv,application/json"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {file && (
              <div className="file-chip">
                <FileText size={15} />
                {file.name}
              </div>
            )}
            <button disabled={!file || documentBusy === "upload"} type="submit">
              {documentBusy === "upload" ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
              {documentBusy === "upload" ? "Processing" : "Upload"}
            </button>
          </form>

          {uploadResult && (
            <div className="pipeline">
              <div className="pipeline-title">
                <CheckCircle2 size={18} color="#0d6b5c" />
                <span>{uploadResult.filename}</span>
              </div>
              <dl className="meta-grid">
                <div>
                  <dt>ID</dt>
                  <dd>{uploadResult.document_id}</dd>
                </div>
                <div>
                  <dt>Status</dt>
                  <dd>{uploadResult.status}</dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{formatBytes(uploadResult.size_bytes)}</dd>
                </div>
              </dl>
            </div>
          )}

          {indexing && (
            <div className="pipeline">
              <div className="index-success">
                <div className="pipeline-title">
                  <CheckCircle2 size={18} color="#0d6b5c" />
                  <span>Indexed into MongoDB</span>
                </div>
                <dl className="meta-grid">
                  <div>
                    <dt>Chunks</dt>
                    <dd>{indexing.chunk_count}</dd>
                  </div>
                  <div>
                    <dt>Upserted</dt>
                    <dd>{indexing.upserted_count}</dd>
                  </div>
                  <div>
                    <dt>Model</dt>
                    <dd>{indexing.model_name}</dd>
                  </div>
                </dl>
              </div>
            </div>
          )}
        </section>

        <section className="chat-stage" aria-labelledby="chat-title">
          <div className="chat-header">
            <div>
              <h2 id="chat-title">Ask</h2>
              <p>Cited answers from indexed clinical sources.</p>
            </div>
            <div className="step-badge">
              <MessageSquareText size={14} />
              Grounded
            </div>
          </div>

          <div className="controls-row">
            <label>
              Search
              <select value={searchMode} onChange={(event) => setSearchMode(event.target.value as SearchMode)}>
                <option value="auto">Auto</option>
                <option value="hybrid">Hybrid</option>
                <option value="semantic">Semantic</option>
                <option value="keyword">Keyword</option>
              </select>
            </label>
            <label>
              Quality
              <select value={qualityMode} onChange={(event) => setQualityMode(event.target.value as QualityMode)}>
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="strict">Strict</option>
              </select>
            </label>
            <label>
              Top K
              <input
                min={1}
                max={10}
                type="number"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
            <label className="toggle-row">
              <input checked={rerank} onChange={(event) => setRerank(event.target.checked)} type="checkbox" />
              Rerank
            </label>
          </div>

          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-orb" aria-hidden>
                  <Bot size={30} />
                </div>
                <h3>Ask from evidence</h3>
                <p>Index a document, then ask a clinical or accreditation question with citations.</p>
                <div className="prompt-chips">
                  {EXAMPLE_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" onClick={() => void runAsk(prompt)} disabled={chatBusy}>
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <div className="message-label">{message.role === "user" ? "You" : "CiteMed"}</div>
                {message.response ? (
                  <AnswerDetails response={message.response} />
                ) : (
                  <p className="bubble">{message.text}</p>
                )}
              </article>
            ))}
            {chatBusy && (
              <article className="message assistant">
                <div className="message-label">CiteMed</div>
                <div className="clinical-answer" style={{ padding: "14px 16px" }}>
                  <div className="section-kicker">
                    <Loader2 className="spin" size={16} />
                    Retrieving grounded answer…
                  </div>
                </div>
              </article>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="ask-form" onSubmit={submitQuestion}>
            <div className="question-input">
              <Search size={18} />
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void runAsk(question.trim());
                  }
                }}
                placeholder="Ask a grounded clinical question…"
                rows={2}
              />
            </div>
            <button disabled={chatBusy || !question.trim()} type="submit">
              {chatBusy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              Ask
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}

function AnswerDetails({ response }: { response: AskResponse }) {
  const answer = response.answer;
  return (
    <div className="clinical-answer">
      <section className="answer-summary">
        <div className="section-kicker">
          <ShieldCheck size={16} />
          Grounded Summary
        </div>
        <p>
          {answer?.summary?.text ?? "No grounded answer returned."}{" "}
          <CitationList citations={answer?.summary?.citations ?? []} />
        </p>
      </section>

      {answer?.key_requirements?.length ? (
        <AnswerSection title="Key Requirements" tone="requirements" items={answer.key_requirements} />
      ) : null}

      {answer?.clinical_actions?.length ? (
        <AnswerSection title="Clinical Actions" tone="actions" items={answer.clinical_actions} />
      ) : null}

      {answer?.limitations ? (
        <section className="limitations">
          <div className="section-kicker">Limitations</div>
          <p>{answer.limitations}</p>
        </section>
      ) : null}

      <SourcePanel response={response} />

      <div className="timing-row">
        <span>{response.search_mode} search</span>
        <span>{formatSeconds(response.timings.total_ms)} total</span>
        <span>{formatSeconds(response.timings.retrieval_ms)} retrieval</span>
        <span>{formatSeconds(response.timings.answer_generation_ms)} answer</span>
        <span>
          {response.validation.enabled
            ? `Strands validation ${response.validation.verified ? "passed" : "reviewed"}`
            : "validation off"}
        </span>
        <span>{response.evidence_sufficient ? "evidence sufficient" : "low evidence"}</span>
      </div>
    </div>
  );
}

function AnswerSection({
  title,
  tone,
  items
}: {
  title: string;
  tone: "requirements" | "actions";
  items: { text: string; citations: number[] }[];
}) {
  return (
    <section>
      <div className="answer-section-header">
        <div className="section-kicker">
          <ClipboardCheck size={16} />
          {title}
        </div>
        <span className="count-pill">{items.length}</span>
      </div>
      <ol className={`answer-list ${tone}`}>
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>
            <span className="answer-number">{index + 1}</span>
            <p>
              {item.text} <CitationList citations={item.citations} />
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function SourcePanel({ response }: { response: AskResponse }) {
  return (
    <section>
      <div className="answer-section-header">
        <div className="section-kicker">
          <FileSearch size={16} />
          Sources
        </div>
        <span className="count-pill">{response.sources.length}</span>
      </div>
      <div className="source-grid">
        {response.sources.map((source) => (
          <details key={source.chunk_id} className="source-card">
            <summary>
              <span className="source-rank">[{source.rank}]</span>
              <span>{source.section_title}</span>
              <span className="source-pages">
                pages {source.page_number}-{source.end_page_number}
              </span>
            </summary>
            <div className="source-meta">
              <span>retrieval {source.retrieval_score.toFixed(4)}</span>
              {source.vector_score !== null && source.vector_score !== undefined ? (
                <span>vector {source.vector_score.toFixed(4)}</span>
              ) : null}
              {source.keyword_score !== null && source.keyword_score !== undefined ? (
                <span>keyword {source.keyword_score.toFixed(2)}</span>
              ) : null}
            </div>
            <p>{source.text}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

function CitationList({ citations }: { citations: number[] }) {
  if (!citations.length) return null;
  return (
    <span className="citations">
      {citations.map((citation) => (
        <span key={citation}>[{citation}]</span>
      ))}
    </span>
  );
}

function formatSeconds(ms: number) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
