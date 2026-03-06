'use client'

import { useEffect } from 'react';
import { live } from '@/utils/api';

import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const geistSans = Geist({
    variable: "--font-geist-sans",
    subsets: ["latin"],
});

const geistMono = Geist_Mono({
    variable: "--font-geist-mono",
    subsets: ["latin"],
});

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    useEffect(() => {
        const autoConnect = async () => {
            // Auto-connect relies on the httpOnly cookie being present.
            // If the user isn't authenticated the API call will 401 and we skip.
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
            <body
                className={`${geistSans.variable} ${geistMono.variable} antialiased`}
            >
                {children}
                <Toaster theme="dark" position="bottom-right" />
            </body>
        </html>
    );
}
