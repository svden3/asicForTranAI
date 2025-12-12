import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gazillioner - AI Wealth Advisor",
  description: "Improve your Financial Quotient with personalized AI coaching",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
