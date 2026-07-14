import type { Metadata } from "next";
import { Literata, Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap"
});

const literata = Literata({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap"
});

export const metadata: Metadata = {
  title: "CiteMed — Healthcare RAG",
  description: "Upload, index, and ask grounded questions against clinical source documents."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} ${literata.variable}`}>
      <body>{children}</body>
    </html>
  );
}
