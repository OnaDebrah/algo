'use client'

import { useAuth } from '@/lib/hooks/useAuth'
import { usePortfolioStore } from '@/lib/store/portfolioStore'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Bell, User, LogOut, Settings } from 'lucide-react'
import { formatCurrency } from '@/lib/utils/formatters'
import {cn} from "@/lib/utils";

export function Header() {
  const { user, logout } = useAuth()
  const { metrics } = usePortfolioStore()

  const initials = user?.username
    ?.split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase() || 'U'

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-card px-6">
      {/* Portfolio Quick Stats */}
      <div className="flex items-center gap-6">
        <div>
          <div className="text-xs text-muted-foreground">Portfolio Value</div>
          <div className="text-lg font-semibold">
            {metrics ? formatCurrency(metrics.totalValue) : '--'}
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Daily P&L</div>
          <div
            className={cn(
              'text-lg font-semibold',
              metrics && metrics.dailyReturn >= 0
                ? 'text-profit'
                : 'text-loss'
            )}
          >
            {metrics ? (
              <>
                {formatCurrency(metrics.dailyReturn)}{' '}
                <span className="text-sm">
                  ({metrics.dailyReturnPct.toFixed(2)}%)
                </span>
              </>
            ) : (
              '--'
            )}
          </div>
        </div>
      </div>

      {/* User Menu */}
      <div className="flex items-center gap-4">
        {/* Notifications */}
        <button className="relative rounded-lg p-2 hover:bg-accent">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-primary" />
        </button>

        {/* User Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger className="focus:outline-none">
            <Avatar>
              <AvatarFallback>{initials}</AvatarFallback>
            </Avatar>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <div className="flex flex-col space-y-1">
                <p className="text-sm font-medium">{user?.username}</p>
                <p className="text-xs text-muted-foreground">{user?.email}</p>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              <User className="mr-2 h-4 w-4" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Settings className="mr-2 h-4 w-4" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={logout} className="text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
