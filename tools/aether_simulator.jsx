import { useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine, Cell, CartesianGrid
} from "recharts";

// ══════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════
const S0 = 50, SIGMA = 2.51, R = 0;
const STEPS_PER_DAY = 4, DAYS_PER_YEAR = 252;
const DT = 1 / (STEPS_PER_DAY * DAYS_PER_YEAR);
const T1 = 40, T2 = 60, CS = 3000;

// ══════════════════════════════════════════════════════
// MATH UTILITIES
// ══════════════════════════════════════════════════════
function ncdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.821256 + t * 1.3302744))));
  return x > 0 ? 1 - p : p;
}

function bs(S, K, T, σ, type) {
  if (T <= 1e-9) return type === "c" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const d1 = (Math.log(S / K) + σ * σ / 2 * T) / (σ * Math.sqrt(T));
  const d2 = d1 - σ * Math.sqrt(T);
  return type === "c"
    ? S * ncdf(d1) - K * ncdf(d2)
    : K * ncdf(-d2) - S * ncdf(-d1);
}

function randn() {
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

// ══════════════════════════════════════════════════════
// INSTRUMENTS
// ══════════════════════════════════════════════════════
const INSTS = [
  { id: "AC",       label: "AC",        type: "spot",    K: null, bar: null, t1: false, bid: 49.975, ask: 50.025, maxV: 200 },
  { id: "AC_50_P",  label: "AC_50_P",   type: "put",     K: 50,   bar: null, t1: false, bid: 12,     ask: 12.05,  maxV: 50  },
  { id: "AC_50_C",  label: "AC_50_C",   type: "call",    K: 50,   bar: null, t1: false, bid: 12,     ask: 12.05,  maxV: 50  },
  { id: "AC_35_P",  label: "AC_35_P",   type: "put",     K: 35,   bar: null, t1: false, bid: 4.33,   ask: 4.35,   maxV: 50  },
  { id: "AC_40_P",  label: "AC_40_P",   type: "put",     K: 40,   bar: null, t1: false, bid: 6.5,    ask: 6.55,   maxV: 50  },
  { id: "AC_45_P",  label: "AC_45_P",   type: "put",     K: 45,   bar: null, t1: false, bid: 9.05,   ask: 9.1,    maxV: 50  },
  { id: "AC_60_C",  label: "AC_60_C",   type: "call",    K: 60,   bar: null, t1: false, bid: 8.8,    ask: 8.85,   maxV: 50  },
  { id: "AC_50_P2", label: "AC_50_P_2", type: "put",     K: 50,   bar: null, t1: true,  bid: 9.7,    ask: 9.75,   maxV: 50  },
  { id: "AC_50_C2", label: "AC_50_C_2", type: "call",    K: 50,   bar: null, t1: true,  bid: 9.7,    ask: 9.75,   maxV: 50  },
  { id: "AC_50_CO", label: "AC_50_CO",  type: "chooser", K: 50,   bar: null, t1: false, bid: 22.2,   ask: 22.3,   maxV: 50  },
  { id: "AC_40_BP", label: "AC_40_BP",  type: "binput",  K: 40,   bar: null, t1: false, bid: 5,      ask: 5.1,    maxV: 50  },
  { id: "AC_45_KO", label: "AC_45_KO",  type: "koput",   K: 45,   bar: 45,   t1: false, bid: 0.15,   ask: 0.175,  maxV: 500 },
];

// ══════════════════════════════════════════════════════
// PAYOFF
// ══════════════════════════════════════════════════════
function calcPayoff(inst, path, minS, binPay) {
  const ST1 = path[T1], ST2 = path[T2];
  const S = inst.t1 ? ST1 : ST2;
  switch (inst.type) {
    case "spot":    return ST2 - S0;
    case "put":     return Math.max(inst.K - S, 0);
    case "call":    return Math.max(S - inst.K, 0);
    case "chooser": {
      const Tr = (T2 - T1) * DT;
      const cv = bs(ST1, inst.K, Tr, SIGMA, "c");
      const pv = bs(ST1, inst.K, Tr, SIGMA, "p");
      return cv >= pv ? Math.max(ST2 - inst.K, 0) : Math.max(inst.K - ST2, 0);
    }
    case "binput":  return ST2 < inst.K ? binPay : 0;
    case "koput":   return minS > inst.bar ? Math.max(inst.K - ST2, 0) : 0;
    default:        return 0;
  }
}

// ══════════════════════════════════════════════════════
// SIMULATION CORE
// ══════════════════════════════════════════════════════
function simulate(posMap, N, binPay) {
  const active = INSTS.filter(x => {
    const p = posMap[x.id];
    return p && Number(p.vol) > 0 && p.side !== "none";
  });
  if (active.length === 0) return null;

  const pnls = new Array(N);
  const scatRaw = [];
  const sampledPaths = [];
  const pathStep = Math.max(1, Math.floor(N / 30));

  const sqrtDT = Math.sqrt(DT);
  const drift = (R - SIGMA * SIGMA / 2) * DT;

  for (let i = 0; i < N; i++) {
    let S = S0, minS = S0;
    const path = new Array(T2 + 1);
    path[0] = S0;
    for (let t = 1; t <= T2; t++) {
      S = S * Math.exp(drift + SIGMA * sqrtDT * randn());
      if (S < minS) minS = S;
      path[t] = S;
    }

    let totalPnl = 0;
    for (const inst of active) {
      const pos = posMap[inst.id];
      const sign = pos.side === "buy" ? 1 : -1;
      const prem = pos.side === "buy" ? inst.ask : inst.bid;
      const pay = calcPayoff(inst, path, minS, binPay);
      totalPnl += sign * (pay - prem) * Number(pos.vol) * CS;
    }

    pnls[i] = totalPnl;
    if (i % 4 === 0) scatRaw.push({ x: parseFloat(S.toFixed(2)), y: totalPnl });
    if (i % pathStep === 0) sampledPaths.push([...path]);
  }

  const sorted = [...pnls].sort((a, b) => a - b);
  const mean = pnls.reduce((s, x) => s + x, 0) / N;
  const std = Math.sqrt(pnls.reduce((s, x) => s + (x - mean) ** 2, 0) / N);
  const pctProfit = pnls.filter(x => x > 0).length / N * 100;

  const hMin = sorted[0], hMax = sorted[N - 1];
  const nBins = 55;
  const bw = (hMax - hMin) / nBins || 1;
  const histBins = Array.from({ length: nBins }, (_, k) => ({
    mid: hMin + (k + 0.5) * bw,
    count: 0,
    pos: hMin + (k + 0.5) * bw >= 0,
  }));
  for (const p of sorted) {
    const idx = Math.min(Math.floor((p - hMin) / bw), nBins - 1);
    histBins[idx].count++;
  }

  // Subsample scatter for perf
  const scatStep = Math.max(1, Math.floor(scatRaw.length / 600));
  const scatterData = scatRaw.filter((_, i) => i % scatStep === 0);

  return {
    stats: {
      mean, std,
      sharpe: std > 0 ? mean / std : 0,
      pctProfit,
      p5:  sorted[Math.floor(N * 0.05)],
      p25: sorted[Math.floor(N * 0.25)],
      p75: sorted[Math.floor(N * 0.75)],
      p95: sorted[Math.floor(N * 0.95)],
      min: sorted[0], max: sorted[N - 1],
    },
    histBins, scatterData, sampledPaths,
  };
}

// ══════════════════════════════════════════════════════
// PRESETS
// ══════════════════════════════════════════════════════
const PRESETS = [
  { id: "straddle",     name: "ATM Straddle",   color: "#00d4ff",
    desc: "Buy 25 × 50P + 25 × 50C (3w)",
    pos: { AC_50_P: { side: "buy", vol: 25 }, AC_50_C: { side: "buy", vol: 25 } }},
  { id: "put_spread",   name: "Put Spread",      color: "#ff9f00",
    desc: "Buy 45P×30, Sell 35P×30",
    pos: { AC_45_P: { side: "buy", vol: 30 }, AC_35_P: { side: "sell", vol: 30 } }},
  { id: "chooser",      name: "Chooser Hedge",   color: "#b44bff",
    desc: "Chooser×20 + short spot×60",
    pos: { AC_50_CO: { side: "buy", vol: 20 }, AC: { side: "sell", vol: 60 } }},
  { id: "exotic",       name: "Exotic Blast",    color: "#ff4b6e",
    desc: "Binary×40 + KO×200",
    pos: { AC_40_BP: { side: "buy", vol: 40 }, AC_45_KO: { side: "buy", vol: 200 } }},
  { id: "teammate_a",   name: "Teammate A",      color: "#39ff14",
    desc: "50P×13, sell 50C×12, 2w P×8, chooser×22, binary×7, KO×6",
    pos: { AC_50_P: { side: "buy", vol: 13 }, AC_50_C: { side: "sell", vol: 12 },
           AC_50_P2: { side: "buy", vol: 8 }, AC_50_CO: { side: "buy", vol: 22 },
           AC_40_BP: { side: "buy", vol: 7 }, AC_45_KO: { side: "buy", vol: 6 } }},
  { id: "teammate_b",   name: "Teammate B",      color: "#ffef00",
    desc: "50P×8, 50C×15, chooser×5, binary×50, KO×1",
    pos: { AC_50_P: { side: "buy", vol: 8 }, AC_50_C: { side: "buy", vol: 15 },
           AC_50_CO: { side: "buy", vol: 5 }, AC_40_BP: { side: "buy", vol: 50 },
           AC_45_KO: { side: "buy", vol: 1 } }},
];

// ══════════════════════════════════════════════════════
// FORMATTERS
// ══════════════════════════════════════════════════════
const fmtK = v => {
  if (!isFinite(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(0);
};
const fmtPct  = v => `${v.toFixed(1)}%`;
const fmtSharp = v => v.toFixed(3);
const pnlColor = v => (v >= 0 ? "#00ff87" : "#ff4b6e");

// ══════════════════════════════════════════════════════
// TYPE META
// ══════════════════════════════════════════════════════
const TYPE_LABEL = { spot: "SPOT", put: "PUT", call: "CALL", chooser: "CHOOSER", binput: "BIN-PUT", koput: "KO-PUT" };
const TYPE_COLOR = { spot: "#00d4ff", put: "#ff7b7b", call: "#7bff9e", chooser: "#b44bff", binput: "#ff9f00", koput: "#ff4b6e" };

// ══════════════════════════════════════════════════════
// STYLES
// ══════════════════════════════════════════════════════
const S = {
  root: { fontFamily: "'Courier New', 'Lucida Console', monospace", background: "#07090e", color: "#b8c8d8", minHeight: "100vh", fontSize: 12 },
  hdr:  { background: "#0b0f16", borderBottom: "1px solid #162030", padding: "14px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" },
  hdrTitle: { color: "#00d4ff", fontSize: 16, letterSpacing: 5, fontWeight: "bold" },
  hdrSub:   { color: "#2d4a60", fontSize: 10, marginTop: 2, letterSpacing: 1 },
  tabBtn: (active) => ({ padding: "5px 14px", fontSize: 10, letterSpacing: 2, border: `1px solid ${active ? "#00d4ff" : "#162030"}`, background: active ? "#00d4ff12" : "transparent", color: active ? "#00d4ff" : "#2d4a60", cursor: "pointer" }),
  body:   { padding: "16px 20px" },
  label:  { fontSize: 9, letterSpacing: 3, color: "#2d4a60", marginBottom: 6, textTransform: "uppercase" },
  card:   { background: "#0b0f16", border: "1px solid #162030", padding: "12px 16px" },
  select: (hi) => ({ background: "#0b0f16", border: `1px solid ${hi || "#162030"}`, color: hi ? hi : "#b8c8d8", padding: "4px 8px", fontSize: 11 }),
  input:  { background: "#0b0f16", border: "1px solid #162030", color: "#b8c8d8", padding: "4px 8px", fontSize: 11 },
  chip:   (color) => ({ padding: "2px 6px", background: `${color}18`, border: `1px solid ${color}40`, color, fontSize: 9, letterSpacing: 1 }),
  runBtn: (active) => ({ padding: "8px 22px", background: active ? "#00ff8710" : "#0b1820", border: `1px solid ${active ? "#00ff87" : "#162030"}`, color: active ? "#00ff87" : "#2d4a60", fontSize: 11, letterSpacing: 2, cursor: active ? "pointer" : "default" }),
};

// ══════════════════════════════════════════════════════
// PATHS CHART
// ══════════════════════════════════════════════════════
function PathsChart({ paths }) {
  if (!paths || paths.length === 0) return null;
  const W = 580, H = 280, PL = 44, PR = 8, PT = 10, PB = 24;
  const iW = W - PL - PR, iH = H - PT - PB;

  let yMin = Infinity, yMax = -Infinity;
  for (const p of paths) for (const v of p) {
    if (v < yMin) yMin = v;
    if (v > yMax) yMax = v;
  }
  yMin = Math.min(yMin, S0 * 0.3);
  yMax = Math.max(yMax, S0 * 2.2);

  const xS = t => PL + (t / T2) * iW;
  const yS = s => PT + iH - ((s - yMin) / (yMax - yMin)) * iH;

  const yTicks = [yMin, yMin + (yMax-yMin)*0.25, yMin + (yMax-yMin)*0.5, yMin + (yMax-yMin)*0.75, yMax];

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}>
      {yTicks.map((y, i) => (
        <g key={i}>
          <line x1={PL} x2={W - PR} y1={yS(y)} y2={yS(y)} stroke="#162030" strokeDasharray="4 4" />
          <text x={PL - 4} y={yS(y) + 3} fill="#2d4a60" fontSize={8} textAnchor="end">{y.toFixed(0)}</text>
        </g>
      ))}
      <line x1={PL} x2={W - PR} y1={yS(S0)} y2={yS(S0)} stroke="#2d4a60" strokeDasharray="6 4" strokeWidth={1.5} />
      <line x1={xS(T1)} x2={xS(T1)} y1={PT} y2={H - PB} stroke="#ffef00" strokeDasharray="4 4" strokeWidth={1} strokeOpacity={0.6} />
      <line x1={xS(T2)} x2={xS(T2)} y1={PT} y2={H - PB} stroke="#00d4ff" strokeDasharray="4 4" strokeWidth={1} strokeOpacity={0.4} />
      <text x={xS(T1) + 3} y={PT + 10} fill="#ffef00" fontSize={8} opacity={0.8}>T1=40</text>
      <text x={xS(T2) - 24} y={PT + 10} fill="#00d4ff" fontSize={8} opacity={0.6}>T2=60</text>
      {paths.map((path, pi) => {
        const pts = path.map((s, t) => `${xS(t).toFixed(1)},${yS(s).toFixed(1)}`).join(" ");
        return <polyline key={pi} points={pts} fill="none" stroke="#00d4ff" strokeWidth={0.7} strokeOpacity={0.22} />;
      })}
      <text x={PL - 4} y={yS(S0) - 4} fill="#4a6a80" fontSize={8} textAnchor="end">S₀=50</text>
      {[0, T1, T2].map(t => (
        <text key={t} x={xS(t)} y={H - 6} fill="#2d4a60" fontSize={8} textAnchor="middle">
          {t === 0 ? "0" : t === T1 ? "40" : "60"}
        </text>
      ))}
      <text x={(PL + W - PR) / 2} y={H - 6} fill="#162030" fontSize={8} textAnchor="middle">steps →</text>
    </svg>
  );
}

// ══════════════════════════════════════════════════════
// COMPARE TAB
// ══════════════════════════════════════════════════════
function CompareTab({ compareRes, running, onCompare }) {
  if (running) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 300, color: "#2d4a60", fontSize: 11, letterSpacing: 4 }}>
      ▶▶ RUNNING SIMULATIONS...
    </div>
  );
  if (!compareRes) return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 320, gap: 16 }}>
      <div style={{ color: "#2d4a60", fontSize: 11, letterSpacing: 3 }}>NO COMPARISON DATA</div>
      <button onClick={onCompare} style={{ padding: "10px 28px", border: "1px solid #00d4ff", background: "#00d4ff10", color: "#00d4ff", fontSize: 11, letterSpacing: 2, cursor: "pointer" }}>
        ⚡ RUN ALL STRATEGIES
      </button>
    </div>
  );

  const barData = compareRes.map(r => ({ name: r.preset.name, mean: Math.round(r.stats.mean), color: r.preset.color }));

  return (
    <div style={S.body}>
      <div style={S.label}>STRATEGY RANKING — E[PnL] ACROSS ALL PRESETS</div>
      <div style={{ ...S.card, height: 230, marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={barData} margin={{ top: 8, right: 12, left: 20, bottom: 36 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#162030" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: "#2d4a60", fontSize: 9 }} angle={-12} textAnchor="end" interval={0} />
            <YAxis tickFormatter={v => fmtK(v)} tick={{ fill: "#2d4a60", fontSize: 9 }} />
            <Tooltip formatter={v => [fmtK(v), "E[PnL]"]} contentStyle={{ background: "#0b0f16", border: "1px solid #162030", color: "#b8c8d8", fontSize: 10 }} />
            <ReferenceLine y={0} stroke="#2d4a60" strokeDasharray="4 4" />
            <Bar dataKey="mean" isAnimationActive={false} radius={[2, 2, 0, 0]}>
              {barData.map((b, i) => <Cell key={i} fill={b.color} fillOpacity={0.75} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ border: "1px solid #162030" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: "#0b0f16", borderBottom: "1px solid #162030" }}>
              {["#", "STRATEGY", "DESC", "E[PnL]", "STD", "SHARPE", "% WIN", "P5", "P95", "MIN", "MAX"].map(h => (
                <th key={h} style={{ padding: "7px 10px", textAlign: "left", color: "#2d4a60", fontSize: 9, letterSpacing: 2, fontWeight: "normal" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {compareRes.map(({ preset, stats }, i) => (
              <tr key={preset.id} style={{ borderBottom: "1px solid #16203010" }}>
                <td style={{ padding: "7px 10px", color: "#2d4a60" }}>#{i + 1}</td>
                <td style={{ padding: "7px 10px", color: preset.color, fontWeight: "bold", whiteSpace: "nowrap" }}>{preset.name}</td>
                <td style={{ padding: "7px 10px", color: "#2d4a60", fontSize: 9, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{preset.desc}</td>
                <td style={{ padding: "7px 10px", color: pnlColor(stats.mean), fontWeight: "bold" }}>{fmtK(stats.mean)}</td>
                <td style={{ padding: "7px 10px", color: "#00d4ff" }}>{fmtK(stats.std)}</td>
                <td style={{ padding: "7px 10px", color: pnlColor(stats.sharpe) }}>{fmtSharp(stats.sharpe)}</td>
                <td style={{ padding: "7px 10px", color: "#ffef00" }}>{fmtPct(stats.pctProfit)}</td>
                <td style={{ padding: "7px 10px", color: "#ff7b7b" }}>{fmtK(stats.p5)}</td>
                <td style={{ padding: "7px 10px", color: "#7bff9e" }}>{fmtK(stats.p95)}</td>
                <td style={{ padding: "7px 10px", color: "#ff4b6e", fontSize: 10 }}>{fmtK(stats.min)}</td>
                <td style={{ padding: "7px 10px", color: "#00ff87", fontSize: 10 }}>{fmtK(stats.max)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 12, padding: "10px 14px", background: "#0b0f16", border: "1px solid #162030", fontSize: 10, color: "#2d4a60", lineHeight: 1.6 }}>
        <span style={{ color: "#00d4ff" }}>MODEL:</span> GBM · S₀={S0} · σ={SIGMA*100}% · r=0 · dt=1/{STEPS_PER_DAY*DAYS_PER_YEAR} · T1={T1} steps · T2={T2} steps · CS={CS}
        {" · "}<span style={{ color: "#ffef00" }}>Prices are per-unit option premiums. PnL = Σ sign·(payoff−premium)·vol·3000</span>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════
// STAT CARD
// ══════════════════════════════════════════════════════
function StatCard({ label, value, color, big }) {
  return (
    <div style={{ background: "#0b0f16", border: "1px solid #162030", padding: "10px 14px" }}>
      <div style={{ fontSize: 8, letterSpacing: 3, color: "#2d4a60", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: big ? 22 : 16, color, fontWeight: big ? "bold" : "normal" }}>{value}</div>
    </div>
  );
}

// ══════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════
export default function AetherSim() {
  const [pos, setPos] = useState({});
  const [simN, setSimN] = useState(2000);
  const [binPay, setBinPay] = useState(10);
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState("build");
  const [compareRes, setCompareRes] = useState(null);
  const [activePreset, setActivePreset] = useState(null);
  const [chartView, setChartView] = useState("hist");

  const getP = id => pos[id] || { side: "none", vol: 0 };
  const setField = (id, key, val) => setPos(prev => ({ ...prev, [id]: { ...getP(id), [key]: val } }));

  const applyPreset = p => {
    setActivePreset(p.id);
    setPos({ ...p.pos });
    setResults(null);
    setTab("build");
  };

  const handleRun = useCallback(() => {
    setRunning(true);
    setResults(null);
    setTimeout(() => {
      const res = simulate(pos, Number(simN), Number(binPay));
      setResults(res);
      setRunning(false);
    }, 20);
  }, [pos, simN, binPay]);

  const handleCompare = useCallback(() => {
    setRunning(true);
    setCompareRes(null);
    setTab("compare");
    setTimeout(() => {
      const cres = PRESETS
        .map(p => ({ preset: p, stats: simulate(p.pos, Number(simN), Number(binPay))?.stats }))
        .filter(x => x.stats)
        .sort((a, b) => b.stats.mean - a.stats.mean);
      setCompareRes(cres);
      setRunning(false);
    }, 20);
  }, [simN, binPay]);

  const handleClear = () => { setPos({}); setResults(null); setActivePreset(null); };

  const totalLong = INSTS.reduce((s, inst) => {
    const p = getP(inst.id);
    return p.side === "buy" ? s + inst.ask * Number(p.vol || 0) : s;
  }, 0);

  const posCount = INSTS.filter(x => {
    const p = getP(x.id);
    return p.vol > 0 && p.side !== "none";
  }).length;

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.hdr}>
        <div>
          <div style={S.hdrTitle}>AETHER_CRYSTAL // MONTE CARLO ENGINE</div>
          <div style={S.hdrSub}>
            GBM · σ=251% · S₀=50 · r=0 · {STEPS_PER_DAY} steps/day · T1=40 · T2=60 · CONTRACT_SIZE=3000
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {[["build", "STRATEGY BUILDER"], ["compare", "COMPARE ALL"]].map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} style={S.tabBtn(tab === id)}>{label}</button>
          ))}
        </div>
      </div>

      {/* ── BUILD TAB ── */}
      {tab === "build" && (
        <div style={S.body}>
          {/* Presets */}
          <div style={{ marginBottom: 14 }}>
            <div style={S.label}>STRATEGY PRESETS</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {PRESETS.map(p => (
                <button key={p.id} onClick={() => applyPreset(p)}
                  title={p.desc}
                  style={{ padding: "5px 12px", fontSize: 10, letterSpacing: 1, border: `1px solid ${activePreset === p.id ? p.color : "#162030"}`, background: activePreset === p.id ? `${p.color}14` : "transparent", color: activePreset === p.id ? p.color : "#4a6a80", cursor: "pointer" }}>
                  {p.name}
                </button>
              ))}
            </div>
            {activePreset && (
              <div style={{ marginTop: 6, fontSize: 10, color: "#2d4a60" }}>
                → {PRESETS.find(p => p.id === activePreset)?.desc}
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: "flex", gap: 12, marginBottom: 14, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div>
              <div style={S.label}>SIMULATIONS</div>
              <select value={simN} onChange={e => setSimN(e.target.value)} style={S.select()}>
                {[500, 1000, 2000, 5000, 10000].map(n => <option key={n} value={n}>{n.toLocaleString()}</option>)}
              </select>
            </div>
            <div>
              <div style={S.label}>BINARY PAYOFF / UNIT</div>
              <input type="number" min="0.01" step="0.5" value={binPay} onChange={e => setBinPay(e.target.value)}
                style={{ ...S.input, width: 72 }} />
            </div>
            <button onClick={handleRun} disabled={running || posCount === 0}
              style={S.runBtn(!running && posCount > 0)}>
              {running ? "▶ RUNNING..." : "▶ RUN SIMULATION"}
            </button>
            <button onClick={handleCompare} disabled={running}
              style={{ padding: "8px 18px", background: "#00d4ff08", border: "1px solid #00d4ff30", color: "#00d4ff", fontSize: 10, letterSpacing: 2, cursor: "pointer" }}>
              ⚡ COMPARE ALL
            </button>
            <button onClick={handleClear}
              style={{ padding: "8px 14px", background: "transparent", border: "1px solid #162030", color: "#2d4a60", fontSize: 10, letterSpacing: 1, cursor: "pointer" }}>
              ✕ CLEAR
            </button>
            <div style={{ marginLeft: "auto", textAlign: "right" }}>
              <div style={{ fontSize: 9, color: "#2d4a60", letterSpacing: 2 }}>LONG PREMIUM COST</div>
              <div style={{ color: "#ffef00", fontSize: 15 }}>{totalLong.toFixed(2)}</div>
              <div style={{ fontSize: 9, color: "#2d4a60" }}>{posCount} position{posCount !== 1 ? "s" : ""} active</div>
            </div>
          </div>

          {/* Position table */}
          <div style={{ border: "1px solid #162030", marginBottom: 18, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#0b0f16", borderBottom: "1px solid #162030" }}>
                  {["OPTION", "TYPE", "EXPIRY", "BID", "ASK", "MAX VOL", "SIDE", "VOLUME", "FAIR VALUE*"].map(h => (
                    <th key={h} style={{ padding: "7px 10px", textAlign: "left", color: "#2d4a60", fontSize: 9, letterSpacing: 2, fontWeight: "normal", whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {INSTS.map((inst, i) => {
                  const p = getP(inst.id);
                  const isExotic = ["chooser", "binput", "koput"].includes(inst.type);
                  const sideColor = p.side === "buy" ? "#00ff87" : p.side === "sell" ? "#ff4b6e" : undefined;
                  // Rough BS fair value for display
                  let fv = "—";
                  if (inst.type === "put")  fv = bs(S0, inst.K, T2 * DT, SIGMA, "p").toFixed(2);
                  if (inst.type === "call") fv = bs(S0, inst.K, T2 * DT, SIGMA, "c").toFixed(2);
                  if (inst.type === "spot") fv = "50.00";
                  return (
                    <tr key={inst.id} style={{ borderBottom: "1px solid #16203015", background: i % 2 === 0 ? "transparent" : "#0b0f1608" }}>
                      <td style={{ padding: "6px 10px", color: isExotic ? "#b44bff" : "#b8c8d8", fontWeight: "bold" }}>{inst.label}</td>
                      <td style={{ padding: "6px 10px" }}>
                        <span style={S.chip(TYPE_COLOR[inst.type])}>{TYPE_LABEL[inst.type]}</span>
                      </td>
                      <td style={{ padding: "6px 10px", color: "#2d4a60" }}>
                        {inst.type === "spot" ? "N/A" : inst.t1 ? "T+14" : "T+21"}
                      </td>
                      <td style={{ padding: "6px 10px", color: "#39ff14" }}>{inst.bid}</td>
                      <td style={{ padding: "6px 10px", color: "#ff4b6e" }}>{inst.ask}</td>
                      <td style={{ padding: "6px 10px", color: "#2d4a60" }}>{inst.maxV}</td>
                      <td style={{ padding: "6px 10px" }}>
                        <select value={p.side} onChange={e => setField(inst.id, "side", e.target.value)}
                          style={S.select(sideColor)}>
                          <option value="none">—</option>
                          <option value="buy">BUY</option>
                          <option value="sell">SELL</option>
                        </select>
                      </td>
                      <td style={{ padding: "6px 10px" }}>
                        <input type="number" min="0" max={inst.maxV} value={p.vol || ""}
                          placeholder="0"
                          onChange={e => setField(inst.id, "vol", Math.min(Number(e.target.value), inst.maxV))}
                          style={{ ...S.input, width: 68 }} />
                      </td>
                      <td style={{ padding: "6px 10px", color: "#4a6a80", fontSize: 10 }}>{fv}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ padding: "6px 10px", fontSize: 9, color: "#2d4a60", borderTop: "1px solid #162030" }}>
              * BS fair value at T=0 (vanilla only). Binary payoff = {binPay} / unit. KO barrier = 45. Chooser chooses optimally at T1 via BS.
            </div>
          </div>

          {/* Results */}
          {running && !results && (
            <div style={{ textAlign: "center", padding: 40, color: "#2d4a60", letterSpacing: 4, fontSize: 11 }}>
              ▶ SIMULATING {Number(simN).toLocaleString()} PATHS...
            </div>
          )}

          {results && (
            <div>
              {/* Stats */}
              <div style={{ marginBottom: 14 }}>
                <div style={S.label}>SIMULATION RESULTS · {Number(simN).toLocaleString()} PATHS</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 2, marginBottom: 2 }}>
                  <StatCard label="E[PnL]"   value={fmtK(results.stats.mean)}         color={pnlColor(results.stats.mean)} big />
                  <StatCard label="STD DEV"  value={fmtK(results.stats.std)}          color="#00d4ff" big />
                  <StatCard label="SHARPE"   value={fmtSharp(results.stats.sharpe)}   color={pnlColor(results.stats.sharpe)} big />
                  <StatCard label="% PROFIT" value={fmtPct(results.stats.pctProfit)}  color="#ffef00" big />
                  <StatCard label="MIN"      value={fmtK(results.stats.min)}          color="#ff4b6e" big />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 2 }}>
                  <StatCard label="MAX" value={fmtK(results.stats.max)}  color="#00ff87" />
                  <StatCard label="P5"  value={fmtK(results.stats.p5)}   color="#ff7b7b" />
                  <StatCard label="P25" value={fmtK(results.stats.p25)}  color="#ffb07b" />
                  <StatCard label="P75" value={fmtK(results.stats.p75)}  color="#7bff9e" />
                  <StatCard label="P95" value={fmtK(results.stats.p95)}  color="#00ff87" />
                </div>
              </div>

              {/* Chart Tab Selector */}
              <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
                {[["hist", "PnL DISTRIBUTION"], ["scatter", "PnL vs S_T₂"], ["paths", "SAMPLE PATHS"]].map(([id, label]) => (
                  <button key={id} onClick={() => setChartView(id)}
                    style={{ padding: "5px 14px", fontSize: 9, letterSpacing: 2, border: `1px solid ${chartView === id ? "#00d4ff" : "#162030"}`, background: chartView === id ? "#00d4ff12" : "transparent", color: chartView === id ? "#00d4ff" : "#2d4a60", cursor: "pointer" }}>
                    {label}
                  </button>
                ))}
              </div>

              {/* Charts */}
              <div style={{ ...S.card, height: 310, position: "relative" }}>
                {chartView === "hist" && (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={results.histBins} margin={{ top: 8, right: 8, left: 16, bottom: 8 }} barCategoryGap={1}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#162030" vertical={false} />
                      <XAxis dataKey="mid" tickFormatter={v => fmtK(v)} tick={{ fill: "#2d4a60", fontSize: 8 }} tickLine={false} />
                      <YAxis tick={{ fill: "#2d4a60", fontSize: 8 }} tickLine={false} />
                      <Tooltip formatter={v => [v, "paths"]} labelFormatter={v => `PnL ≈ ${fmtK(v)}`}
                        contentStyle={{ background: "#0b0f16", border: "1px solid #162030", color: "#b8c8d8", fontSize: 10 }} />
                      <ReferenceLine x={0} stroke="#2d4a60" strokeDasharray="4 4" label={{ value: "0", fill: "#4a6a80", fontSize: 9, position: "top" }} />
                      <Bar dataKey="count" isAnimationActive={false} radius={[1, 1, 0, 0]}>
                        {results.histBins.map((b, i) => (
                          <Cell key={i} fill={b.pos ? "#00ff87" : "#ff4b6e"} fillOpacity={0.72} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
                {chartView === "scatter" && (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 8, right: 8, left: 16, bottom: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#162030" />
                      <XAxis dataKey="x" type="number" name="S_T₂" tickFormatter={v => v.toFixed(0)}
                        tick={{ fill: "#2d4a60", fontSize: 8 }}
                        label={{ value: "Final Price S_T₂", position: "insideBottom", offset: -12, fill: "#2d4a60", fontSize: 9 }} />
                      <YAxis dataKey="y" type="number" name="PnL" tickFormatter={v => fmtK(v)}
                        tick={{ fill: "#2d4a60", fontSize: 8 }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }}
                        contentStyle={{ background: "#0b0f16", border: "1px solid #162030", color: "#b8c8d8", fontSize: 10 }}
                        formatter={(v, name) => [name === "PnL" ? fmtK(v) : v.toFixed(2), name]} />
                      <ReferenceLine y={0} stroke="#2d4a60" strokeDasharray="4 4" />
                      <ReferenceLine x={S0} stroke="#ffef0040" strokeDasharray="6 4" label={{ value: "S₀=50", fill: "#ffef0060", fontSize: 8 }} />
                      <Scatter data={results.scatterData} isAnimationActive={false}>
                        {results.scatterData.map((d, i) => (
                          <Cell key={i} fill={d.y >= 0 ? "#00ff87" : "#ff4b6e"} opacity={0.45} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
                {chartView === "paths" && (
                  <PathsChart paths={results.sampledPaths} />
                )}
              </div>

              <div style={{ marginTop: 10, padding: "8px 12px", background: "#0b0f16", border: "1px solid #162030", fontSize: 9, color: "#2d4a60" }}>
                <span style={{ color: "#00d4ff" }}>NOTE:</span> Binary payoff = {binPay}/unit (configurable). KO put knocked out if min(S_t) ≤ 45 on any discrete step. Chooser selects call/put at T1 using BS value for remaining week. Prices are entry costs (ask for buys, bid for sells).
              </div>
            </div>
          )}

          {!results && !running && posCount === 0 && (
            <div style={{ border: "1px dashed #162030", padding: "30px 20px", textAlign: "center", color: "#2d4a60", fontSize: 11, letterSpacing: 2 }}>
              SELECT A PRESET OR BUILD A POSITION ABOVE, THEN RUN SIMULATION
            </div>
          )}
        </div>
      )}

      {/* ── COMPARE TAB ── */}
      {tab === "compare" && (
        <CompareTab compareRes={compareRes} running={running} onCompare={handleCompare} />
      )}
    </div>
  );
}
