'use client'

import {useEffect} from 'react';
import {live} from '@/utils/api';

import {Geist, Geist_Mono} from "next/font/google";
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
            try {
                const result = await live.autoConnect();
                if (result.status === 'connected') {
                    console.log(`Auto-connected to ${result.broker}`);
                }
            } catch (error) {
                console.error('Auto-connect failed:', error);
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
        </body>
        </html>
    );
}
