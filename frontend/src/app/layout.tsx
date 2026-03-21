'use client'

import { useEffect } from 'react';
import { live } from '@/utils/api';

import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";
import { ThemeProvider, useTheme } from "@/contexts/ThemeContext";

const geistSans = Geist({
    variable: "--font-geist-sans",
    subsets: ["latin"],
});

const geistMono = Geist_Mono({
    variable: "--font-geist-mono",
    subsets: ["latin"],
});

// Small wrapper so Toaster reads the current theme from context
function ThemedToaster() {
    const { theme } = useTheme();
    return <Toaster theme={theme} position="bottom-right" />;
}

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    useEffect(() => {
        const autoConnect = async () => {
            try {
                const result = await live.autoConnect();
                if (result.status === 'connected') {
                    console.log(`Auto-connected to ${result.broker}`);
                }
            } catch {
                // Not authenticated or broker unavailable — silently skip
            }
        };

        autoConnect();
    }, []);
    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                {/* Prevent flash of wrong theme — runs before React hydration */}
                <script dangerouslySetInnerHTML={{ __html: `
                    (function() {
                        try {
                            var theme = localStorage.getItem('oraculum_theme');
                            if (theme !== 'light') {
                                document.documentElement.classList.add('dark');
                            }
                        } catch(e) {
                            document.documentElement.classList.add('dark');
                        }
                    })();
                `}} />
            </head>
            <body
                className={`${geistSans.variable} ${geistMono.variable} antialiased`}
            >
                <ThemeProvider>
                    {children}
                    <ThemedToaster />
                </ThemeProvider>
            </body>
        </html>
    );
}
