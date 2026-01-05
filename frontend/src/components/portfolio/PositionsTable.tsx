import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatCurrency, formatPercent, formatDateTime } from '@/lib/utils/formatters'
import type { Position } from '@/types'

export function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Symbol</TableHead>
          <TableHead>Side</TableHead>
          <TableHead className="text-right">Quantity</TableHead>
          <TableHead className="text-right">Entry Price</TableHead>
          <TableHead className="text-right">Current Price</TableHead>
          <TableHead className="text-right">P&L</TableHead>
          <TableHead>Entry Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {positions.map((position) => (
          <TableRow key={position.id}>
            <TableCell className="font-medium">{position.symbol}</TableCell>
            <TableCell>
              <span className={`px-2 py-1 rounded text-xs ${position.side === 'long' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
                {position.side.toUpperCase()}
              </span>
            </TableCell>
            <TableCell className="text-right">{position.quantity}</TableCell>
            <TableCell className="text-right">{formatCurrency(position.entryPrice)}</TableCell>
            <TableCell className="text-right">{formatCurrency(position.currentPrice)}</TableCell>
            <TableCell className={`text-right ${position.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss'}`}>
              <div>{formatCurrency(position.unrealizedPnl)}</div>
              <div className="text-xs">{formatPercent(position.unrealizedPnlPct)}</div>
            </TableCell>
            <TableCell>{formatDateTime(position.entryTime)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
