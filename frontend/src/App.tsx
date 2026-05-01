import { Routes, Route } from "react-router-dom";
import { ThemeToggle } from "./components/ThemeToggle";
import { Home } from "./pages/Home";
import { JobDetail } from "./pages/JobDetail";
import { Headphones } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border sticky top-0 z-10 bg-background/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2 font-semibold text-lg hover:text-primary transition-colors">
            <Headphones size={22} className="text-primary" />
            Shravana
          </a>
          <ThemeToggle />
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/jobs/:jobId" element={<JobDetail />} />
        </Routes>
      </main>
    </div>
  );
}
