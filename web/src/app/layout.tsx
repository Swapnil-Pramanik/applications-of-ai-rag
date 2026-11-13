import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Grounded Notebook",
  description:
    "Ask questions of your own documents: hybrid retrieval, reranking, an abstention gate and cited LLM answers.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // suppressHydrationWarning: browser extensions add attributes to <html> (e.g. webcrx="")
    // before React loads. It only silences attribute mismatches on this one element.
    <html lang="en" className={`${inter.variable} h-full antialiased`} suppressHydrationWarning>
      <body className="h-full font-sans text-[15px]">{children}</body>
    </html>
  );
}
