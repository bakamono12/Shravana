import { useState, useEffect } from "react";

type Theme = "light" | "dark" | "system";

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem("shravana-theme") as Theme) || "system";
  });

  useEffect(() => {
    const root = document.documentElement;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");

    function apply(t: Theme) {
      const isDark = t === "dark" || (t === "system" && mq.matches);
      root.classList.toggle("dark", isDark);
    }

    apply(theme);
    localStorage.setItem("shravana-theme", theme);

    const listener = () => { if (theme === "system") apply("system"); };
    mq.addEventListener("change", listener);
    return () => mq.removeEventListener("change", listener);
  }, [theme]);

  return { theme, setTheme };
}
