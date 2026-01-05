import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow} from '@/components/ui/table'
import {Button} from '@/components/ui/button'
import {Download} from 'lucide-react'
import {formatCurrency, formatDateTime, formatPercent} from '@/lib/utils/formatters'
import type {Trade} from '@/types'

export function TradeLogTable({ trades }: { trades: Trade[] }) {
  const downloadCSV = () => {
    const headers = ['Symbol', 'Type', 'Quantity', 'Price', 'Profit', 'Profit %', 'Strategy', 'Timestamp']
    const rows = trades.map(t => [
      t.symbol,
      t.orderType,
      t.quantity,
      t.price,
      t.profit || '',
      t.profitPct || '',
      t.strategy,
      t.timestamp
    ])

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `trades_${Date.now()}.csv`
    a.click()
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={downloadCSV}>
          <Download className="mr-2 h-4 w-4" />
          Export CSV
        </Button>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-card">
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Type</TableHead>
              <TableHead className="text-right">Quantity</TableHead>
              <TableHead className="text-right">Price</TableHead>
              <TableHead className="text-right">P&L</TableHead>
              <TableHead>Strategy</TableHead>
              <TableHead>Timestamp</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trades.map((trade, idx) => (
              <TableRow key={idx}>
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
                <TableCell className="text-sm">{trade.strategy}</TableCell>
                <TableCell className="text-sm text-muted-foreground">{formatDateTime(trade.timestamp)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
