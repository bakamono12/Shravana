const BASE = "/api";

export interface Job {
  id: string;
  filename: string;
  status: string;
  total_chunks: number;
  completed_chunks: number;
  detected_language: string | null;
  language_hint: string | null;
  error_message: string | null;
  waiting_for_model: string | null;
  // translation
  translate: boolean;
  target_language: string | null;
  translator_mode: string | null;
  enable_refinement: boolean;
  translation_status: string | null;
  executor: string | null;
  // live progress fields (merged from WS events)
  subphase?: string | null;
  percent?: number | null;
  stage_index?: number | null;
  stage_total?: number | null;
  phase?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LangInfo {
  name: string;
  seamless: boolean;
  vlm: boolean;
}

export interface ContextBundle {
  domain: string;
  format: string;
  source_language: string;
  target_language: string;
  named_entities: string[];
  idioms_detected: string[];
  scene_description: string | null;
  notes: string | null;
}

export interface Chunk {
  id: string;
  sequence: number;
  start_time: number;
  end_time: number;
  duration: number;
  status: string;
  detected_language: string | null;
  assigned_model: string | null;
}

export interface JobDetail extends Job {
  chunks: Chunk[];
}

export interface ModelStatus {
  id: string;
  name: string;
  repo_id: string;
  status: string;
  bytes_downloaded: number;
  bytes_total: number;
  error_message: string | null;
  updated_at: string;
  // transient WS-only signals (not persisted in DB)
  attempt?: number;
  note?: string;
}

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listJobs: () => req<Job[]>("/jobs"),
  getJob: (id: string) => req<JobDetail>(`/jobs/${id}`),
  retryJob: (id: string) => req<{ job_id: string; status: string }>(`/jobs/${id}/retry`, { method: "POST" }),
  deleteJob: (id: string) => req<{ deleted: string }>(`/jobs/${id}`, { method: "DELETE" }),
  listModels: () => req<ModelStatus[]>("/models"),
  listLanguages: () => req<Record<string, LangInfo>>("/translation/languages"),
  getJobContext: (id: string) => req<ContextBundle>(`/jobs/${id}/context`),

  translateJob: (
    id: string,
    targetLanguage: string,
    opts?: { translatorMode?: string; enableRefinement?: boolean; glossary?: string },
  ) =>
    req<{ job_id: string; status: string; target_language: string }>(
      `/jobs/${id}/translate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_language: targetLanguage,
          translator_mode: opts?.translatorMode ?? null,
          enable_refinement: opts?.enableRefinement ?? true,
          glossary: opts?.glossary ?? null,
        }),
      },
    ),

  uploadFile: (
    file: File,
    languageHint?: string,
    translationOpts?: {
      targetLanguage?: string;
      translatorMode?: string;
      enableRefinement?: boolean;
      glossary?: string;
    },
  ): { promise: Promise<{ job_id: string }>; abort: () => void } => {
    const controller = new AbortController();
    const form = new FormData();
    form.append("file", file);
    if (languageHint) form.append("language_hint", languageHint);
    if (translationOpts?.targetLanguage) {
      form.append("target_language", translationOpts.targetLanguage);
      if (translationOpts.translatorMode) form.append("translator_mode", translationOpts.translatorMode);
      form.append("enable_refinement", String(translationOpts.enableRefinement ?? true));
      if (translationOpts.glossary) form.append("glossary", translationOpts.glossary);
    }
    const promise = req<{ job_id: string; filename: string; status: string }>("/upload", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    return { promise, abort: () => controller.abort() };
  },
};
