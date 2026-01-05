'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { useUIStore } from '@/lib/store/uiStore'
import {
  LayoutDashboard,
  TrendingUp,
  Bot,
  Compass,
  Store,
  Settings,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Target,
  Activity,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  {
    name: 'Backtest',
    icon: BarChart3,
    children: [
      { name: 'Single Asset', href: '/backtest/single' },
      { name: 'Multi-Asset', href: '/backtest/multi' },
      { name: 'Options', href: '/backtest/options' },
    ],
  },
  { name: 'AI Advisor', href: '/advisor', icon: Bot },
  { name: 'Market Regime', href: '/regime', icon: Activity },
  { name: 'Marketplace', href: '/marketplace', icon: Store },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()
  const { sidebarCollapsed, toggleSidebar } = useUIStore()

  return (
    <div
      className={cn(
        'fixed left-0 top-0 z-40 h-screen border-r border-border bg-card transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b border-border px-4">
        {!sidebarCollapsed && (
          <Link href="/dashboard" className="flex items-center space-x-2">
            <Target className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold">Trading AI</span>
          </Link>
        )}
        <button
          onClick={toggleSidebar}
          className="rounded-lg p-2 hover:bg-accent"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <ChevronLeft className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => (
          <div key={item.name}>
            {item.children ? (
              <div className="space-y-1">
                <div
                  className={cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium',
                    'text-muted-foreground hover:bg-accent hover:text-foreground'
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  {!sidebarCollapsed && <span>{item.name}</span>}
                </div>
                {!sidebarCollapsed && (
                  <div className="ml-8 space-y-1">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        className={cn(
                          'block rounded-lg px-3 py-2 text-sm',
                          pathname === child.href
                            ? 'bg-primary text-primary-foreground'
                            : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                        )}
                      >
                        {child.name}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <Link
                href={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  pathname === item.href
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                )}
              >
                <item.icon className="h-5 w-5" />
                {!sidebarCollapsed && <span>{item.name}</span>}
              </Link>
            )}
          </div>
        ))}
      </nav>

      {/* Footer */}
      {!sidebarCollapsed && (
        <div className="border-t border-border p-4">
          <div className="text-xs text-muted-foreground">
            <div>Version 1.0.0</div>
            <div className="mt-1">© 2024 Trading Platform</div>
          </div>
        </div>
      )}
    </div>
  )
}
