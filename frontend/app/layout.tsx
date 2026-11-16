import type { Metadata, Viewport } from "next";

import { themeBootScript } from "../lib/theme";
import "./globals.css";

export const metadata: Metadata = {
  title: "RepoMind",
  description: "Trusted local repository intelligence workspace",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0d12" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // The boot script may set data-theme before hydration; the DOM value must win.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
