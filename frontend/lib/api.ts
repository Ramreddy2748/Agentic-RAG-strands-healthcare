export type QualityMode = "fast" | "balanced" | "strict";
export type SearchMode = "auto" | "semantic" | "keyword" | "hybrid";

export type CitedStatement = {
  text: string;
  citations: number[];
};

export type ClinicalAnswer = {
  summary?: CitedStatement | null;
  key_requirements?: CitedStatement[];
  clinical_actions?: CitedStatement[];
  limitations?: string | null;
};

export type Source = {
  rank: number;
  chunk_id: string;
  source_id: string;
  source_path: string;
  chapter_title: string;
  section_title: string;
  page_number: number;
  end_page_number: number;
  text: string;
  vector_score?: number | null;
  keyword_score?: number | null;
  retrieval_score: number;
  rerank_score?: number | null;
};

export type ValidationMetadata = {
  enabled: boolean;
  verified: boolean;
  confidence: number;
  checked_claims: number;
  supported_claims: number;
  removed_claims: number;
  unclear_claims: number;
  reason: string;
  unsupported_claims: string[];
};

export type AskResponse = {
  request_id: string;
  question: string;
  quality_mode: string;
  search_mode: string;
  routing_reason: string;
  answer: ClinicalAnswer | null;
  verification: ValidationMetadata;
  validation: ValidationMetadata;
  sources: Source[];
  evidence_sufficient: boolean;
  evidence_score: number;
  evidence_reason: string;
  timings: {
    routing_ms: number;
    retrieval_ms: number;
    reranking_ms: number;
    answer_generation_ms: number;
    verification_ms: number;
    total_ms: number;
  };
};

export type DocumentUploadResponse = {
  document_id: string;
  filename: string;
  stored_filename: string;
  content_type: string;
  file_extension: string;
  size_bytes: number;
  status: string;
  created_at: string;
};

export type DocumentElement = {
  content_type: string;
  text: string;
  page_number?: number | null;
  row_number?: number | null;
  json_path?: string | null;
  metadata?: Record<string, string> | null;
};

export type DocumentIngestionResponse = {
  document_id: string;
  filename: string;
  file_extension: string;
  element_count: number;
  elements: DocumentElement[];
};

export type DocumentIndexingResponse = {
  document_id: string;
  filename: string;
  file_extension: string;
  element_count: number;
  chunk_count: number;
  upserted_count: number;
  model_name: string;
};

export type DocumentUploadAndIndexResponse = {
  document: DocumentUploadResponse;
  indexing: DocumentIndexingResponse;
  status: string;
};

export type AuthUser = {
  email: string;
  name: string;
  createdAt?: string;
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...options.headers
    }
  });
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const payload = await response.json();
      message = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch {
      message = await response.text();
    }
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function askQuestion(
  question: string,
  options: {
    searchMode: SearchMode;
    qualityMode: QualityMode;
    topK: number;
    rerank: boolean;
  }
) {
  return request<AskResponse>("/api/ask", {
    method: "POST",
    body: JSON.stringify({
      question,
      search_mode: options.searchMode,
      quality_mode: options.qualityMode,
      top_k: options.topK,
      rerank: options.rerank,
      generate_answer: true
    })
  });
}

export function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request<DocumentUploadResponse>("/api/documents/upload", {
    method: "POST",
    body: formData
  });
}

export function uploadAndIndexDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request<DocumentUploadAndIndexResponse>("/api/documents/upload-and-index", {
    method: "POST",
    body: formData
  });
}

export function ingestDocument(documentId: string, show = 5) {
  return request<DocumentIngestionResponse>(`/api/documents/${documentId}/ingest?show=${show}`, {
    method: "POST"
  });
}

export function indexDocument(documentId: string) {
  return request<DocumentIndexingResponse>(`/api/documents/${documentId}/index`, {
    method: "POST"
  });
}

export function login(email: string, password: string) {
  return request<{ user: AuthUser }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export function signup(name: string, email: string, password: string) {
  return request<{ user: AuthUser }>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ name, email, password })
  });
}

export function logout() {
  return request<{ ok: boolean }>("/api/auth/logout", { method: "POST" });
}

export function currentUser() {
  return request<{ user: AuthUser }>("/api/auth/me");
}
