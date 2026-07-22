import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Ensure tippingpoint is in path
sys.path.insert(0, os.path.abspath('../../src'))
from tippingpoint import MarketingReturnCurve
from tippingpoint.math import geometric_adstock

out_dir = 'images'
os.makedirs(out_dir, exist_ok=True)

# Generate Mock Data
spends = np.array([1000, 5000, 15000, 25000, 40000, 60000])
returns = np.array([200, 1500, 12000, 22000, 28000, 32000])

# 1. Generic Hill Fit with Scatter
print("Generating Hill Fit Image...")
model1 = MarketingReturnCurve.from_historical_data(spends, returns, channel_name="Paid Social", epochs=100)
fig1 = model1.plot_response_curve(target_mroas=1.0, show=False, scatter=(spends, returns))
fig1.savefig(os.path.join(out_dir, 'hill_fit.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# 2. Ad Stocking Plot
print("Generating Adstock Image...")
model2 = MarketingReturnCurve.from_historical_data(spends, returns, channel_name="Television", adstock_type="fixed", adstock_fixed_days=7.0, epochs=100)
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='white')
ax2.set_facecolor('white')
timeline = np.arange(len(spends))
adstocked = model2.adstock_spend(spends)

ax2.bar(timeline, spends, color='#EA4335', alpha=0.4, label='Raw Daily Spend')
ax2.plot(timeline, adstocked, color='#4285F4', linewidth=3, label='Effective Adstocked Spend')
ax2.fill_between(timeline, adstocked, color='#4285F4', alpha=0.1)

ax2.set_title('Adstock Carryover Effect (Half-Life: 7 Days)', fontsize=14, fontweight='bold', color='#202124', pad=15)
ax2.set_xlabel('Timeline (Days)', fontsize=11, color='#5F6368')
ax2.set_ylabel('Spend Weight', fontsize=11, color='#5F6368')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#F8F9FA')
ax2.spines['bottom'].set_color('#F8F9FA')
ax2.grid(True, linestyle='-', alpha=0.1, color='#5F6368')
ax2.legend(frameon=True, facecolor='white', framealpha=1.0)
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'adstock.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)

# 3. Example Output (Console Summary text rendered as image for the PDF)
print("Generating Module Output Image...")
fig3, ax3 = plt.subplots(figsize=(8, 4), facecolor='#202124')
ax3.axis('off')

summary_text = """
> tipp evaluate --channel "Paid Search" --spend 45000

--- Budget Evaluation: Paid Search ---
Current Spend: $45,000.00
Current mROAS: 0.85

Status: OVER-SATURATED (Unprofitable Marginal Growth)
Recommendation: Scale back spend to $32,450.00 to maintain
                target unit economics.

Optimal Scaling Zone: $8,500.00 -> $32,450.00
"""

ax3.text(0.05, 0.5, summary_text, fontsize=13, family='monospace', color='#34A853', va='center')
fig3.savefig(os.path.join(out_dir, 'module_output.png'), dpi=300, bbox_inches='tight')
plt.close(fig3)

print("Images generated successfully.")
