import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Loan Data Analysis & Modeling",
  description: "Statistical analysis of loan data using R",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
