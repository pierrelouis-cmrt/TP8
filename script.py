import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os

# =============================================================================
# PARAMÈTRES GLOBAUX
# =============================================================================
ROUGE = '#B22133'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (9, 6)

# Créer dossier figures
os.makedirs('figures', exist_ok=True)

# Constante tachymètre
K_TACHY = 0.06  # V/(tr/min)
DELTA_K_REL = 0.01  # 1% incertitude sur K

# =============================================================================
# FONCTIONS D'INCERTITUDE
# =============================================================================
def incertitude_tension(U):
    """Incertitude wattmètre mode voltmètre: ±(1%L + 2 digits), rés 0.1V"""
    return np.sqrt((0.01 * U)**2 + (2 * 0.1)**2)

def incertitude_puissance_pince(P):
    """Incertitude pince F205 puissance DC: ±(2%L + 10 digits), rés 1W"""
    return np.sqrt((0.02 * np.abs(P))**2 + 10**2)

def incertitude_puissance_wattmetre(P):
    """Incertitude wattmètre 10W-1kW: ±(1%L + 2 digits), rés 1W"""
    return np.sqrt((0.01 * np.abs(P))**2 + 2**2)

def convert_to_omega_rad(U_tachy):
    """Convertit U_tachy (V) en vitesse angulaire (rad/s)"""
    omega_rpm = U_tachy / K_TACHY
    return omega_rpm * 2 * np.pi / 60

def incertitude_omega(U_tachy):
    """Incertitude sur omega par propagation"""
    delta_U = incertitude_tension(U_tachy)
    omega = convert_to_omega_rad(U_tachy)
    delta_omega_rel = np.sqrt((delta_U / U_tachy)**2 + DELTA_K_REL**2)
    return omega * delta_omega_rel

# =============================================================================
# DONNÉES - PARTIE 1: E(Ω)
# =============================================================================
data_fem = {
    1.91: [(4.6, 29.3), (5.6, 35.8), (6.1, 39.3), (6.8, 43.9), (7.9, 51.2), (9.5, 62.2)],
    3.68: [(2.1, 12.8), (5.6, 34.3), (6.4, 39.6), (7.4, 45.8), (8.3, 52.3), (9.0, 56.4)],
    5.24: [(3.1, 17.9), (6.4, 37.7), (7.2, 42.7), (8.2, 49.1), (9.3, 56.1), (10.4, 62.8)],
    7.60: [(6.0, 33.7), (6.8, 37.9), (8.3, 47.0), (9.5, 54.4), (10.8, 61.4), (12.2, 70.4)],
    9.53: [(5.5, 29.0), (7.2, 38.7), (8.6, 46.2), (10.6, 57.5), (12.3, 66.9), (13.2, 72.8)]
}

# =============================================================================
# FIGURE 1: E(Ω)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))
colors = plt.cm.Reds(np.linspace(0.35, 0.95, 5))
markers = ['o', 's', '^', 'D', 'v']
results_regression = {}

for idx, (I_ind, measurements) in enumerate(data_fem.items()):
    E_vals = np.array([m[0] for m in measurements])
    U_tachy_vals = np.array([m[1] for m in measurements])
    
    omega_vals = convert_to_omega_rad(U_tachy_vals)
    delta_omega = incertitude_omega(U_tachy_vals)
    delta_E = incertitude_tension(E_vals)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(omega_vals, E_vals)
    
    results_regression[I_ind] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'std_err': std_err
    }
    
    ax1.errorbar(omega_vals, E_vals, xerr=delta_omega, yerr=delta_E,
                 fmt=markers[idx], color=colors[idx], capsize=3, markersize=6,
                 label=f'$I_{{ind}}$ = {I_ind} A')
    
    omega_fit = np.linspace(0, max(omega_vals)*1.05, 100)
    E_fit = slope * omega_fit + intercept
    ax1.plot(omega_fit, E_fit, '--', color=colors[idx], alpha=0.7, linewidth=1.5)

ax1.set_xlabel(r'$\Omega$ (rad/s)', fontsize=12)
ax1.set_ylabel(r'$E$ (V)', fontsize=12)
ax1.set_title('Force électromotrice induite en fonction de la vitesse de rotation', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 140)
ax1.set_ylim(0, 15)
plt.tight_layout()
plt.savefig('figures/figure1_E_omega.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure1_E_omega.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("RÉSULTATS DES RÉGRESSIONS LINÉAIRES E = a·Ω + b")
print("="*70)
print(f"{'I_ind (A)':<12} {'a (V·s/rad)':<14} {'Δa':<12} {'b (V)':<10} {'R²':<10}")
print("-"*70)
for I_ind, res in results_regression.items():
    print(f"{I_ind:<12.2f} {res['slope']:<14.4f} {res['std_err']:<12.4f} "
          f"{res['intercept']:<10.2f} {res['r_squared']:<10.5f}")

# =============================================================================
# FIGURE 2: k(I_inducteur)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

I_inducteurs = np.array(list(results_regression.keys()))
slopes = np.array([results_regression[I]['slope'] for I in I_inducteurs])
slopes_err = np.array([results_regression[I]['std_err'] for I in I_inducteurs])

ax2.errorbar(I_inducteurs, slopes, yerr=slopes_err, 
             fmt='o', color=ROUGE, capsize=5, markersize=8, linewidth=2,
             label='Données expérimentales')

slope_k, intercept_k, r_k, p_k, std_err_k = stats.linregress(I_inducteurs, slopes)
I_fit = np.linspace(0, max(I_inducteurs)*1.1, 100)
k_fit = slope_k * I_fit + intercept_k
ax2.plot(I_fit, k_fit, '--', color=ROUGE, alpha=0.7, linewidth=2,
         label=f'Régression: $k = {slope_k:.5f} I + {intercept_k:.4f}$\n$R^2 = {r_k**2:.4f}$')

ax2.set_xlabel(r'$i_{\mathrm{inducteur}}$ (A)', fontsize=12)
ax2.set_ylabel(r'$k = \partial E / \partial \Omega$ (V·s/rad)', fontsize=12)
ax2.set_title("Coefficient directeur en fonction du courant d'inducteur", fontsize=12)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 11)
ax2.set_ylim(0.08, 0.11)
plt.tight_layout()
plt.savefig('figures/figure2_k_I.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure2_k_I.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("RÉGRESSION k(I_inducteur)")
print(f"k = {slope_k:.5f} × I + {intercept_k:.5f}")
print(f"R² = {r_k**2:.5f}")
print(f"{'='*70}")

# =============================================================================
# DONNÉES - PARTIE 2: RENDEMENT
# =============================================================================
R_pct = np.array([5, 10, 15, 25, 30, 35, 40, 50, 75, 100])
P_abs = np.array([245.3, 302.3, 328, 335, 326, 318, 312, 303, 280, 266])
P_induc = 16.0
P_charge = np.array([80.4, 126, 141, 136, 125, 115, 107, 95, 69, 54])

eta = P_charge / (P_abs + P_induc) * 100

delta_P_abs = incertitude_puissance_pince(P_abs)
delta_P_charge = incertitude_puissance_wattmetre(P_charge)
delta_P_induc = incertitude_puissance_wattmetre(P_induc)

P_tot = P_abs + P_induc
delta_P_tot = np.sqrt(delta_P_abs**2 + delta_P_induc**2)
delta_eta_rel = np.sqrt((delta_P_charge / P_charge)**2 + (delta_P_tot / P_tot)**2)
delta_eta = eta * delta_eta_rel

# =============================================================================
# FIGURE 3: Rendement - MODÈLE POLYNOMIAL EMPIRIQUE
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3.errorbar(P_charge, eta, xerr=delta_P_charge, yerr=delta_eta,
             fmt='o', color=ROUGE, capsize=4, markersize=8, linewidth=2,
             label='Données expérimentales')

# Ajustement polynomial quadratique: η(P) = c0 + c1*P + c2*P²
# Justification: modèle empirique permettant de capturer le maximum de rendement
# sans hypothèse forte sur la forme des pertes
coeffs = np.polyfit(P_charge, eta, 2, w=1/delta_eta)
c2, c1, c0 = coeffs

# Incertitude sur les coefficients par bootstrap simplifié
n_boot = 1000
coeffs_boot = np.zeros((n_boot, 3))
for i in range(n_boot):
    eta_boot = eta + np.random.normal(0, delta_eta)
    coeffs_boot[i] = np.polyfit(P_charge, eta_boot, 2)
coeffs_std = np.std(coeffs_boot, axis=0)

P_fit = np.linspace(40, 160, 200)
eta_fit = c0 + c1*P_fit + c2*P_fit**2

# Maximum du polynôme: dη/dP = c1 + 2*c2*P = 0 => P_opt = -c1/(2*c2)
P_opt = -c1 / (2 * c2)
eta_max_theo = c0 + c1*P_opt + c2*P_opt**2

ax3.plot(P_fit, eta_fit, '--', color=ROUGE, alpha=0.8, linewidth=2,
         label=f'Ajustement: $\\eta = {c0:.1f} + {c1:.3f}P + {c2:.5f}P^2$')

ax3.axvline(x=P_opt, color='gray', linestyle=':', alpha=0.7)
ax3.annotate(f'$P_{{opt}} = {P_opt:.0f}$ W\n$\\eta_{{max}} = {eta_max_theo:.1f}$%', 
             xy=(P_opt+5, 25), fontsize=10, color='gray')

ax3.set_xlabel(r'$P_{\mathrm{charge}}$ (W)', fontsize=12)
ax3.set_ylabel(r'$\eta$ (%)', fontsize=12)
ax3.set_title('Rendement du banc MCC + MS en fonction de la puissance de charge', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(40, 160)
ax3.set_ylim(15, 50)
plt.tight_layout()
plt.savefig('figures/figure3_rendement.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_rendement.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("MODÈLE DE RENDEMENT POLYNOMIAL: η = c₀ + c₁P + c₂P²")
print(f"c₀ = {c0:.2f} ± {coeffs_std[2]:.2f} %")
print(f"c₁ = {c1:.4f} ± {coeffs_std[1]:.4f} %/W")
print(f"c₂ = {c2:.6f} ± {coeffs_std[0]:.6f} %/W²")
print(f"P_optimal = -c₁/(2c₂) = {P_opt:.0f} W")
print(f"η_max (modèle) = {eta_max_theo:.1f}%")
print(f"η_max (expérimental) = {max(eta):.1f}% à P = {P_charge[np.argmax(eta)]:.0f} W")
print(f"{'='*70}")

# Tableau des rendements
print("\nTableau des rendements:")
print(f"{'R(%)':<6} {'P_abs':<10} {'δP_abs':<8} {'P_ch':<8} {'δP_ch':<7} {'η(%)':<8} {'δη(%)':<7}")
print("-"*60)
for i in range(len(R_pct)):
    print(f"{R_pct[i]:<6} {P_abs[i]:<10.1f} {delta_P_abs[i]:<8.1f} "
          f"{P_charge[i]:<8.1f} {delta_P_charge[i]:<7.1f} {eta[i]:<8.1f} {delta_eta[i]:<7.1f}")

# =============================================================================
# DONNÉES BONUS: Rémanence
# =============================================================================
U_tachy_rem = np.array([48, 60, 73])
E_rem = np.array([56, 61, 65])

omega_rem = convert_to_omega_rad(U_tachy_rem)
delta_omega_rem = incertitude_omega(U_tachy_rem)
delta_E_rem = incertitude_tension(E_rem)

# =============================================================================
# FIGURE 4: Rémanence
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(8, 6))

ax4.errorbar(omega_rem, E_rem, xerr=delta_omega_rem, yerr=delta_E_rem,
             fmt='o', color=ROUGE, capsize=5, markersize=10, linewidth=2,
             label='Données ($i_{inducteur} = 0$)')

slope_rem, intercept_rem, r_rem, _, std_rem = stats.linregress(omega_rem, E_rem)
omega_fit_rem = np.linspace(70, 140, 100)
E_fit_rem = slope_rem * omega_fit_rem + intercept_rem
ax4.plot(omega_fit_rem, E_fit_rem, '--', color=ROUGE, alpha=0.7, linewidth=2,
         label=f'Régression: $E = {slope_rem:.3f}\\Omega + {intercept_rem:.1f}$\n$R^2 = {r_rem**2:.4f}$')

ax4.set_xlabel(r'$\Omega$ (rad/s)', fontsize=12)
ax4.set_ylabel(r'$E_{\mathrm{rémanent}}$ (V)', fontsize=12)
ax4.set_title("FEM induite par magnétisme rémanent ($i_{inducteur} = 0$)", fontsize=12)
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure4_remanence.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure4_remanence.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("BONUS: MAGNÉTISME RÉMANENT")
print(f"Pente k_rémanent = {slope_rem:.4f} ± {std_rem:.4f} V·s/rad")
print(f"Ordonnée = {intercept_rem:.1f} V")
print(f"R² = {r_rem**2:.4f}")
print(f"{'='*70}")