import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme } from "../hooks/useTheme";
import { cn } from "../lib/utils";

const themes = ["light", "dark", "system"] as const;
const icons = { light: Sun, dark: Moon, system: Monitor };
const labels = { light: "Light", dark: "Dark", system: "System" };

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  function cycle() {
    const idx = themes.indexOf(theme);
    setTheme(themes[(idx + 1) % themes.length]);
  }

  const Icon = icons[theme];
  return (
    <button
      onClick={cycle}
      title={`Theme: ${labels[theme]}`}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium",
        "bg-secondary text-secondary-foreground hover:bg-muted transition-colors"
      )}
    >
      <Icon size={15} />
      <span className="hidden sm:inline">{labels[theme]}</span>
    </button>
  );
}
