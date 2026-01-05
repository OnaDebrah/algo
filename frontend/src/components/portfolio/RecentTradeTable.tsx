import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatCurrency, formatPercent, formatDateTime } from '@/lib/utils/formatters'
import type { Trade } from '@/types'

export function RecentTradesTable({ trades }: { trades: Trade[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Symbol</TableHead>
          <TableHead>Type</TableHead>
          <TableHead className="text-right">Quantity</TableHead>
          <TableHead className="text-right">Price</TableHead>
          <TableHead className="text-right">P&L</TableHead>
          <TableHead>Strategy</TableHead>
          <TableHead>Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {trades.map((trade) => (
          <TableRow key={trade.id}>
            <TableCell className="font-medium">{trade.symbol}</TableCell>
            <TableCell>
              <span className={`px-2 py-1 rounded text-xs ${trade.orderType === 'BUY' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
                {trade.orderType}
              </span>
            </TableCell>
            <TableCell className="text-right">{trade.quantity}</TableCell>
            <TableCell className="text-right">{formatCurrency(trade.price)}</TableCell>
            <TableCell className={`text-right ${(trade.profit || 0) >= 0 ? 'text-profit' : 'text-loss'}`}>
              {trade.profit !== undefined ? (
                <>
                  <div>{formatCurrency(trade.profit)}</div>
                  <div className="text-xs">{formatPercent(trade.profitPct || 0)}</div>
                </>
              ) : '--'}
            </TableCell>
            <TableCell className="text-sm text-muted-foreground">{trade.strategy}</TableCell>
            <TableCell>{formatDateTime(trade.timestamp)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
