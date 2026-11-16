export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "repomind.theme";
export const THEME_EVENT = "repomind-theme";

/**
 * Inlined in <head> so a stored explicit choice applies before first paint (no flash).
 * Without one, globals.css follows prefers-color-scheme.
 */
export const themeBootScript = `(function(){try{var t=localStorage.getItem("${THEME_STORAGE_KEY}");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}})()`;
