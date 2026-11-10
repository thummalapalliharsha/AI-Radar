import type { Metadata } from "next";
import { Inter, Oxanium } from "next/font/google";
import "./globals.css";
import { SiteBackground } from "./SiteBackground";
import { SectionNavigation } from "./SectionNavigation";
import { CinematicIntro } from "./CinematicIntro";

const oxanium = Oxanium({ subsets: ["latin"], variable: "--font-heading", display: "swap" });
const inter = Inter({ subsets: ["latin"], variable: "--font-ui", display: "swap" });

export const metadata: Metadata = {
  title: "AI Radar",
  description: "AI Radar — local-first AI intelligence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${oxanium.variable} ${inter.variable}`}>
      <body>
        <SiteBackground />
        <CinematicIntro />
        <SectionNavigation />
        {children}
      </body>
    </html>
  );
}
