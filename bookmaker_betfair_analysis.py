"""
Bookmaker → Betfair Value Betting Strategy  —  Serie A

Premessa:
  Ogni bookmaker ha dati e algoritmi proprietari. Invece di costruire un nostro
  modello, troviamo quale bookmaker ha il miglior potere predittivo storico
  e usiamo le sue quote (ripulite dal margine) come segnale.

Strategia:
  Per ogni partita, se la probabilità implicita "fair" del bookmaker segnale
  è maggiore della probabilità implicita di Betfair (proxy = Max odds),
  c'è valore → scommetti su Betfair.

Metriche:
  - Brier score / log-loss per misurare la qualità predittiva dei bookmaker
  - ROI, equity curve, sensitivity all'edge minimo
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
df = pd.read_csv('serie-A.csv', encoding='latin-1', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['FTR', 'Date']).copy()
df = df.sort_values('Date').reset_index(drop=True)
print(f"Loaded {len(df)} matches  ({df['Date'].min().date()} → {df['Date'].max().date()})")

# ──────────────────────────────────────────────
# 2. BOOKMAKERS & BETFAIR PROXY
# ──────────────────────────────────────────────
BOOKMAKERS = {
    'Bet365':   ('B365H', 'B365D', 'B365A'),
    'BetWay':   ('BWH',   'BWD',   'BWA'),
    'GamBet':   ('GBH',   'GBD',   'GBA'),
    'IWin':     ('IWH',   'IWD',   'IWA'),
    'LadBrk':   ('LBH',   'LBD',   'LBA'),
    'SportBet': ('SBH',   'SBD',   'SBA'),
    'WillHill': ('WHH',   'WHD',   'WHA'),
    'Pinnacle': ('PSH',   'PSD',   'PSA'),
    'VCBet':    ('VCH',   'VCD',   'VCA'),
    'StanJam':  ('SJH',   'SJD',   'SJA'),
    'BlueSq':   ('BSH',   'BSD',   'BSA'),
}

# Proxy per le quote Betfair = quota massima disponibile tra tutti i bookmaker
# Betfair exchange offre tipicamente quote pari o superiori al massimo dei bookmaker.
# Applichiamo la commissione reale Betfair del 5% sulle vincite.
BF_COLS       = ('MaxH', 'MaxD', 'MaxA')
BF_COMMISSION = 0.05

# ──────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────
def normalize_probs(h_odds, d_odds, a_odds):
    """Rimuove il margine del bookmaker → probabilità fair."""
    imp_h = 1.0 / h_odds
    imp_d = 1.0 / d_odds
    imp_a = 1.0 / a_odds
    total = imp_h + imp_d + imp_a
    return imp_h / total, imp_d / total, imp_a / total


def prepare(df, bk_cols):
    h, d, a = bk_cols
    bfh, bfd, bfa = BF_COLS
    needed = [h, d, a, bfh, bfd, bfa, 'FTR', 'Date']
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    sub['p_h'], sub['p_d'], sub['p_a'] = normalize_probs(sub[h], sub[d], sub[a])
    sub['overround'] = 1 / sub[h] + 1 / sub[d] + 1 / sub[a]
    sub['actual_h']  = (sub['FTR'] == 'H').astype(int)
    sub['actual_d']  = (sub['FTR'] == 'D').astype(int)
    sub['actual_a']  = (sub['FTR'] == 'A').astype(int)
    return sub


# ──────────────────────────────────────────────
# 4. ACCURATEZZA PREDITTIVA DI OGNI BOOKMAKER
# ──────────────────────────────────────────────
acc_rows = []
for bk_name, bk_cols in BOOKMAKERS.items():
    if not all(c in df.columns for c in bk_cols):
        continue
    sub = prepare(df, bk_cols)
    if len(sub) < 100:
        continue

    y_true = np.concatenate([sub['actual_h'], sub['actual_d'], sub['actual_a']])
    y_pred = np.clip(
        np.concatenate([sub['p_h'], sub['p_d'], sub['p_a']]),
        1e-6, 1 - 1e-6
    )
    bs = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    acc_rows.append({
        'bookmaker':     bk_name,
        'n_matches':     len(sub),
        'brier_score':   round(bs, 6),
        'log_loss':      round(ll, 6),
        'avg_overround': round(sub['overround'].mean(), 4),
    })

acc_df = pd.DataFrame(acc_rows).sort_values('brier_score')
print("\n──── Accuratezza predittiva (Brier score più basso = predittore migliore) ────")
print(acc_df.to_string(index=False))


# ──────────────────────────────────────────────
# 5. BACKTEST: segnale bookmaker vs Betfair
# ──────────────────────────────────────────────
def backtest(df, bk_cols, min_edge=0.0):
    """
    Per ogni esito di ogni partita:
      edge = prob_fair_bookie - (1 / betfair_odds)
      se edge > min_edge → scommetti a quota Betfair (- 5% comm. sulle vincite)

    Ritorna DataFrame con colonne: edge, pnl, bf_odds, date.
    Stake = 1 unità per scommessa.
    """
    bfh, bfd, bfa = BF_COLS
    sub = prepare(df, bk_cols)
    records = []
    for _, row in sub.iterrows():
        for prob, bf_col, won in [
            (row['p_h'], bfh, row['actual_h']),
            (row['p_d'], bfd, row['actual_d']),
            (row['p_a'], bfa, row['actual_a']),
        ]:
            bf_odds = row[bf_col]
            edge    = prob - (1.0 / bf_odds)
            if edge > min_edge:
                pnl = (bf_odds - 1) * (1 - BF_COMMISSION) * won - (1 - won)
                records.append({
                    'edge':    edge,
                    'pnl':     pnl,
                    'bf_odds': bf_odds,
                    'date':    row['Date'],
                })
    return pd.DataFrame(records)


# ── Backtest su tutti i bookmaker (edge=0) ──
strat_rows = []
for bk_name, bk_cols in BOOKMAKERS.items():
    if not all(c in df.columns for c in bk_cols):
        continue
    bets = backtest(df, bk_cols, min_edge=0.0)
    if len(bets) < 50:
        continue
    strat_rows.append({
        'bookmaker':    bk_name,
        'n_bets':       len(bets),
        'roi_%':        round(bets['pnl'].mean() * 100, 2),
        'total_profit': round(bets['pnl'].sum(), 2),
        'win_rate_%':   round((bets['pnl'] > 0).mean() * 100, 1),
    })

strat_df = pd.DataFrame(strat_rows).sort_values('roi_%', ascending=False)
print("\n──── Backtest strategia (min_edge=0, commissione Betfair 5%) ────")
print(strat_df.to_string(index=False))


# ──────────────────────────────────────────────
# 6. BOOKMAKER MIGLIORE: sensitivity all'edge
# ──────────────────────────────────────────────
best_bk      = strat_df.iloc[0]['bookmaker']
best_bk_cols = BOOKMAKERS[best_bk]
print(f"\n→ Bookmaker con il segnale migliore: {best_bk}")

edge_rows = []
for min_edge in np.arange(0.00, 0.16, 0.01):
    bets = backtest(df, best_bk_cols, min_edge=round(min_edge, 2))
    if len(bets) < 20:
        break
    edge_rows.append({
        'min_edge':     round(min_edge, 2),
        'n_bets':       len(bets),
        'roi_%':        round(bets['pnl'].mean() * 100, 2),
        'total_profit': round(bets['pnl'].sum(), 2),
    })

edge_df = pd.DataFrame(edge_rows)
print("\n──── ROI al variare della soglia di edge minimo ────")
print(edge_df.to_string(index=False))

best_edge_row = edge_df.loc[edge_df['roi_%'].idxmax()]
print(f"\n→ Edge ottimale: {best_edge_row['min_edge']}  "
      f"ROI={best_edge_row['roi_%']}%  N={int(best_edge_row['n_bets'])}")


# ──────────────────────────────────────────────
# 7. PLOT
# ──────────────────────────────────────────────
bets_base = backtest(df, best_bk_cols, min_edge=0.0)
opt_edge  = best_edge_row['min_edge']
bets_opt  = backtest(df, best_bk_cols, min_edge=opt_edge)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f'Bookmaker → Betfair Value Strategy  |  Segnale: {best_bk}',
    fontsize=13
)

# Equity curves
axes[0, 0].plot(bets_base['pnl'].cumsum().values, label='edge ≥ 0', alpha=0.85)
axes[0, 0].plot(bets_opt['pnl'].cumsum().values,  label=f'edge ≥ {opt_edge}', alpha=0.85)
axes[0, 0].axhline(0, color='red', lw=0.8, ls='--')
axes[0, 0].set_title('Equity Curve (P&L cumulativo, stake=1 unità)')
axes[0, 0].set_xlabel('Scommessa #')
axes[0, 0].set_ylabel('Profitto (unità)')
axes[0, 0].legend()

# ROI vs edge threshold
ax_roi = axes[0, 1]
ax_n   = ax_roi.twinx()
ax_roi.plot(edge_df['min_edge'], edge_df['roi_%'], marker='o', color='green')
ax_roi.axhline(0, color='red', lw=0.8, ls='--')
ax_n.bar(edge_df['min_edge'], edge_df['n_bets'], alpha=0.2, color='blue', width=0.008)
ax_roi.set_title(f'ROI vs soglia edge minimo  ({best_bk})')
ax_roi.set_xlabel('Edge minimo')
ax_roi.set_ylabel('ROI %', color='green')
ax_n.set_ylabel('N scommesse', color='blue')

# ROI per bookmaker
axes[1, 0].barh(
    strat_df['bookmaker'], strat_df['roi_%'],
    color=['#2ecc71' if x > 0 else '#e74c3c' for x in strat_df['roi_%']]
)
axes[1, 0].axvline(0, color='black', lw=0.8)
axes[1, 0].set_title('ROI% per bookmaker segnale (edge ≥ 0)')
axes[1, 0].set_xlabel('ROI %')

# Brier score
axes[1, 1].barh(acc_df['bookmaker'], acc_df['brier_score'], color='steelblue')
axes[1, 1].set_title('Brier Score per bookmaker\n(più basso = predittore più preciso)')
axes[1, 1].set_xlabel('Brier Score')

plt.tight_layout()
plt.savefig('bookmaker_betfair_analysis.png', dpi=150, bbox_inches='tight')
print("\nGrafico salvato → bookmaker_betfair_analysis.png")


# ──────────────────────────────────────────────
# 8. RIEPILOGO FINALE
# ──────────────────────────────────────────────
base_row = strat_df.iloc[0]
brier_best = acc_df[acc_df['bookmaker'] == best_bk]['brier_score'].values[0]
print(f"""
══════════════════════════════════════════════════════
 RIEPILOGO FINALE
══════════════════════════════════════════════════════
 Dataset         : {len(df)} partite di Serie A
 Proxy Betfair   : Max odds (MaxH/MaxD/MaxA) − 5% commissione

 Bookmaker con il miglior potere predittivo : {best_bk}
   Brier score = {brier_best}  (riferimento: Pinnacle è il più sharp)

 Strategia senza filtro (edge ≥ 0):
   Scommesse  = {int(base_row['n_bets'])}
   ROI        = {base_row['roi_%']}%
   P&L totale = {base_row['total_profit']} unità

 Soglia edge ottimale: {opt_edge}
   Scommesse  = {int(best_edge_row['n_bets'])}
   ROI        = {best_edge_row['roi_%']}%
   P&L totale = {best_edge_row['total_profit']} unità
══════════════════════════════════════════════════════
""")
