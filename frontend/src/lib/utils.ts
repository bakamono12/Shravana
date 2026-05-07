import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
}

export function fmtEta(seconds: number | null | undefined): string {
  if (!seconds) return "";
  if (seconds < 60) return `~${seconds}s`;
  return `~${Math.ceil(seconds / 60)}min`;
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString();
}
