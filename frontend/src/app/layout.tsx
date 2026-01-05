import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/lib/providers/Providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata {
  title: 'Trading Platform',
  description: 'Professional algorithmic trading platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
        {children}
  )
}