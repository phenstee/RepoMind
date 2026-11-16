"use client";

import { useSyncExternalStore } from "react";

import { THEME_EVENT, THEME_STORAGE_KEY, type Theme } from "../lib/theme";
import { Icon } from "./icons";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function currentTheme(): Theme {
  const explicit = document.documentElement.dataset.theme;
  return explicit === "light" || explicit === "dark" ? explicit : systemTheme();
}

function subscribe(onChange: () => void): () => void {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", onChange);
  window.addEventListener(THEME_EVENT, onChange);
  return () => {
    media.removeEventListener("change", onChange);
    window.removeEventListener(THEME_EVENT, onChange);
  };
}

export function ThemeToggle() {
  const theme = useSyncExternalStore<Theme>(subscribe, currentTheme, () => "dark");
  const next: Theme = theme === "dark" ? "light" : "dark";

  function toggle() {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage can be unavailable; the choice then lasts for this page only.
    }
    window.dispatchEvent(new Event(THEME_EVENT));
  }

  return (
    <button type="button" className="btn btnGhost btnIcon" onClick={toggle} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      <Icon name={theme === "dark" ? "sun" : "moon"} />
    </button>
  );
}
