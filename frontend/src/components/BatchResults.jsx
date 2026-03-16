import { useState, useMemo } from 'react';
import { Download, Search, ChevronLeft, ChevronRight, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import * as XLSX from 'xlsx';
import './BatchResults.css';

export default function BatchResults({ results, originalFile }) {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');
  const rowsPerPage = 20;

  // Filter & Search
  const filteredData = useMemo(() => {
    return results.filter(row => {
      const label = row.sentiment || row.label;
      const matchLabel = filter === 'All' || label === filter;
      const matchSearch = row.comment.toLowerCase().includes(search.toLowerCase());
      return matchLabel && matchSearch;
    });
  }, [results, search, filter]);

  // Pagination
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);
  const paginatedData = useMemo(() => {
    const start = (page - 1) * rowsPerPage;
    return filteredData.slice(start, start + rowsPerPage);
  }, [filteredData, page]);

  // Summary Stats
  const stats = useMemo(() => {
    const counts = { Positive: 0, Neutral: 0, Negative: 0 };
    results.forEach(r => { 
      const label = r.sentiment || r.label;
      if (counts[label] !== undefined) counts[label]++; 
    });
    const total = results.length || 1;
    return {
      Positive: ((counts.Positive / total) * 100).toFixed(1),
      Neutral: ((counts.Neutral / total) * 100).toFixed(1),
      Negative: ((counts.Negative / total) * 100).toFixed(1),
      raw: counts
    };
  }, [results]);

  const chartData = [
    { name: 'Positive', value: stats.raw.Positive, color: 'var(--positive)' },
    { name: 'Neutral',  value: stats.raw.Neutral,  color: 'var(--neutral)' },
    { name: 'Negative', value: stats.raw.Negative, color: 'var(--negative)' },
  ];

  const handleExport = () => {
    const ws = XLSX.utils.json_to_sheet(results.map((r, i) => {
      const isNew = r.sentiment !== undefined;
      return {
        ID: i + 1,
        Comment: r.comment,
        Sentiment: isNew ? r.sentiment : r.label,
        Comment_Type: isNew ? r.comment_type : '-',
        Is_Toxic: isNew ? Boolean(r.is_toxic) : '-',
        Confidence_Positive: isNew ? r.conf_pos : r.confidence_positive,
        Confidence_Neutral: isNew ? r.conf_neu : r.confidence_neutral,
        Confidence_Negative: isNew ? r.conf_neg : r.confidence_negative,
        Toxicity_Score: isNew ? r.toxicity : '-'
      };
    }));
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Classified Results");
    
    // YYYYMMDD format
    const d = new Date();
    const dateStr = `${d.getFullYear()}${(d.getMonth()+1).toString().padStart(2,'0')}${d.getDate().toString().padStart(2,'0')}`;
    XLSX.writeFile(wb, `classified_results_${dateStr}.csv`, { bookType: 'csv' });
  };

  const getLabelIcon = (label) => {
    if (label === 'Positive') return <TrendingUp size={14} />;
    if (label === 'Negative') return <TrendingDown size={14} />;
    return <Minus size={14} />;
  };

  return (
    <div className="batch-results glass-card animate-slide-in">
      <div className="batch-header">
        <div>
          <h2 className="panel-title">Batch Results</h2>
          <p className="panel-desc">Classified {results.length.toLocaleString()} rows from {originalFile?.name}</p>
        </div>
        <button className="btn-primary export-btn" onClick={handleExport}>
          <Download size={16} /> Export CSV
        </button>
      </div>

      <div className="summary-dashboard">
        <div className="summary-chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                innerRadius={30}
                outerRadius={45}
                paddingAngle={5}
                dataKey="value"
                stroke="none"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip 
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-default)', borderRadius: '8px', color: 'var(--text-primary)' }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="summary-stats">
          <div className="stat-item"><span className="stat-dot positive"></span>Positive: {stats.Positive}%</div>
          <div className="stat-item"><span className="stat-dot neutral"></span>Neutral: {stats.Neutral}%</div>
          <div className="stat-item"><span className="stat-dot negative"></span>Negative: {stats.Negative}%</div>
        </div>
      </div>

      <div className="table-controls">
        <div className="search-box">
          <Search size={16} className="search-icon" />
          <input 
            type="text" 
            placeholder="Search comments..." 
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            className="search-input"
          />
        </div>
        <select 
          className="filter-select"
          value={filter}
          onChange={(e) => { setFilter(e.target.value); setPage(1); }}
        >
          <option value="All">All Labels</option>
          <option value="Positive">Positive</option>
          <option value="Neutral">Neutral</option>
          <option value="Negative">Negative</option>
        </select>
      </div>

      <div className="table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th width="5%">#</th>
              <th width="45%">Comment</th>
              <th width="20%">Sentiment</th>
              <th width="15%">Type</th>
              <th width="15%">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.length > 0 ? (
              paginatedData.map((row, idx) => {
                const globalIdx = (page - 1) * rowsPerPage + idx + 1;
                
                // Backwards compatibility for old format vs ModernBERT format
                const isNewFormat = row.sentiment !== undefined;
                const displayLabel = isNewFormat ? row.sentiment : row.label;
                const labelClass = displayLabel.toLowerCase();
                
                // Find max confidence
                let maxConf = 0;
                if (displayLabel === 'Positive') maxConf = isNewFormat ? row.conf_pos : row.confidence_positive;
                else if (displayLabel === 'Negative') maxConf = isNewFormat ? row.conf_neg : row.confidence_negative;
                else maxConf = isNewFormat ? row.conf_neu : row.confidence_neutral;

                return (
                  <tr key={globalIdx} className="animate-fade-in" style={{ animationDelay: `${idx * 20}ms` }}>
                    <td className="col-id">{globalIdx}</td>
                    <td className="col-comment" title={row.comment}>
                      {row.comment.length > 80 ? row.comment.substring(0, 80) + '...' : row.comment}
                      {isNewFormat && row.is_toxic && (
                        <span className="label-pill toxic-pill" style={{ display: 'inline-flex', marginLeft: '8px', padding: '2px 6px', fontSize: '0.7em', background: 'rgba(255,68,68,0.1)', color: '#ff4444', border: '1px solid rgba(255,68,68,0.2)' }}>
                          Toxic
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`label-pill ${labelClass}`}>
                        {getLabelIcon(displayLabel)} {displayLabel}
                      </span>
                    </td>
                    <td>
                      {isNewFormat ? (
                         <span className="label-pill" style={{ background: 'var(--glass-bg)', color: 'var(--text-secondary)', border: '1px solid var(--border-default)' }}>
                           {row.comment_type}
                         </span>
                      ) : (
                         <span className="text-muted">-</span>
                      )}
                    </td>
                    <td className="col-conf">{(maxConf * 100).toFixed(1)}%</td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan="5" className="empty-state">No matching results found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button 
            className="page-btn" 
            disabled={page === 1} 
            onClick={() => setPage(p => p - 1)}
          >
            <ChevronLeft size={16} /> Prev
          </button>
          <span className="page-info">Page {page} of {totalPages}</span>
          <button 
            className="page-btn" 
            disabled={page === totalPages} 
            onClick={() => setPage(p => p + 1)}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
