import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Uploader } from "../components/Uploader";
import { JobCard } from "../components/JobCard";
import { ModelStatusPanel } from "../components/ModelStatusPanel";
import { api, type Job } from "../lib/api";

export function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    api.listJobs().then(setJobs).catch(() => {});
    const interval = setInterval(() => {
      api.listJobs().then(setJobs).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleJobCreated = (jobId: string) => {
    api.listJobs().then(setJobs).catch(() => {});
    navigate(`/jobs/${jobId}`);
  };

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Subtitle Pipeline</h1>
        <p className="text-muted-foreground">
          Upload a video or audio file to get AI-generated subtitles in SRT / VTT format.
          Supports English, Hindi, and Marathi.
        </p>
      </div>

      <Uploader onJobCreated={handleJobCreated} />

      <ModelStatusPanel />

      {jobs.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-3">Recent jobs</h2>
          <div className="space-y-3">
            {jobs.map((j) => (
              <JobCard
                key={j.id}
                job={j}
                onDelete={(id) => setJobs((prev) => prev.filter((x) => x.id !== id))}
              />
            ))}
          </div>
        </div>
      )}

      {jobs.length === 0 && (
        <div className="text-center text-muted-foreground py-8">
          No jobs yet. Upload a file to get started.
        </div>
      )}
    </div>
  );
}
