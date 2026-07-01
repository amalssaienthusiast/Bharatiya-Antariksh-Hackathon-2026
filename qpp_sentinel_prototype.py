# ============================================================
# CELL 1: Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pywt
# Install: pip install sunpy astropy celerite

# ============================================================
# CELL 2: Synthetic Data Generation (Proxy for SoLEXS + HEL1OS)
# ============================================================
def generate_flare_lightcurve(duration=300, dt_sxr=1.0, dt_hxr=0.01, 
                               has_qpp=True, qpp_period=20.0, qpp_amp=0.3):
    """
    Generate synthetic SXR + HXR light curves mimicking Aditya-L1 data
    
    Parameters:
    - duration: flare duration in seconds
    - dt_sxr: SoLEXS cadence (1 second)
    - dt_hxr: HEL1OS cadence (10 ms = 0.01 s)
    - has_qpp: whether to include QPP modulation
    - qpp_period: QPP period in seconds
    - qpp_amp: QPP amplitude relative to background
    """
    # Time arrays
    t_sxr = np.arange(0, duration, dt_sxr)
    t_hxr = np.arange(0, duration, dt_hxr)
    
    # SXR: slow thermal rise and decay (GOES-like)
    sxr_background = 1e-6  # baseline flux
    sxr_rise = np.exp((t_sxr - 100) / 30) * (t_sxr < 100) + \
               (1 + 0.5 * np.exp(-(t_sxr - 100) / 50)) * (t_sxr >= 100)
    sxr_flux = sxr_background * sxr_rise + np.random.normal(0, 1e-7, len(t_sxr))
    
    # HXR: impulsive spike + QPP modulation
    hxr_background = 1e-3
    hxr_impulsive = 10 * np.exp(-((t_hxr - 80) / 20) ** 2)  # Gaussian spike
    
    if has_qpp:
        # QPP: damped oscillation in HXR
        qpp = qpp_amp * np.sin(2 * np.pi * t_hxr / qpp_period) * \
              np.exp(-(t_hxr - 80) / 100) * (t_hxr > 60) * (t_hxr < 200)
        hxr_flux = hxr_background + hxr_impulsive + qpp + np.random.normal(0, 1e-4, len(t_hxr))
    else:
        hxr_flux = hxr_background + hxr_impulsive + np.random.normal(0, 1e-4, len(t_hxr))
    
    return {
        't_sxr': t_sxr, 'sxr_flux': sxr_flux,
        't_hxr': t_hxr, 'hxr_flux': hxr_flux,
        'has_qpp': has_qpp, 'qpp_period': qpp_period if has_qpp else None
    }

# Generate sample data
data = generate_flare_lightcurve(has_qpp=True, qpp_period=20.0)

# ============================================================
# CELL 3: Visualize Combined SXR + HXR Data
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# SXR (SoLEXS proxy)
axes[0].plot(data['t_sxr'], data['sxr_flux'] * 1e6, 'b-', linewidth=1.5, label='SoLEXS SXR (1-30 keV)')
axes[0].set_ylabel('SXR Flux (×10⁻⁶ W/m²)')
axes[0].set_title('QPP-Sentinel: Combined SXR + HXR Light Curves')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# HXR (HEL1OS proxy)
axes[1].plot(data['t_hxr'], data['hxr_flux'], 'r-', linewidth=0.8, alpha=0.7, label='HEL1OS HXR (10-150 keV)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('HXR Flux (counts/s)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('combined_lightcurves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 4: GOES-Class Proxy from SXR
# ============================================================
def estimate_goes_class(sxr_flux_w_m2):
    """
    Convert SXR flux (W/m²) to GOES class proxy
    """
    flux = np.max(sxr_flux_w_m2)
    if flux < 1e-7: return 'A', flux / 1e-8
    elif flux < 1e-6: return 'B', flux / 1e-7
    elif flux < 1e-5: return 'C', flux / 1e-6
    elif flux < 1e-4: return 'M', flux / 1e-5
    else: return 'X', flux / 1e-4

goes_class, multiplier = estimate_goes_class(data['sxr_flux'])
print(f"GOES Class Estimate: {goes_class}{multiplier:.1f}")

# ============================================================
# CELL 5: Neupert Effect Correlation
# ============================================================
def compute_neupert_correlation(t_sxr, sxr_flux, t_hxr, hxr_flux):
    """
    Compute Neupert correlation: cumulative HXR vs dSXR/dt
    """
    # Compute dSXR/dt
    dsxr_dt = np.gradient(sxr_flux, t_sxr)
    
    # Interpolate HXR to SXR time grid
    hxr_interp = np.interp(t_sxr, t_hxr, hxr_flux)
    
    # Cumulative HXR fluence
    hxr_cumulative = np.cumsum(hxr_interp) * np.mean(np.diff(t_sxr))
    
    # Correlation
    corr = np.corrcoef(hxr_cumulative, dsxr_dt)[0, 1]
    
    return corr, hxr_cumulative, dsxr_dt

corr, hxr_cum, dsxr = compute_neupert_correlation(
    data['t_sxr'], data['sxr_flux'], data['t_hxr'], data['hxr_flux']
)
print(f"Neupert Correlation: {corr:.3f}")

# Plot Neupert correlation
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(hxr_cum, dsxr, c='purple', alpha=0.6, s=20)
ax.set_xlabel('Cumulative HXR Fluence')
ax.set_ylabel('dSXR/dt')
ax.set_title(f'Neupert Effect Correlation: r = {corr:.3f}')
ax.grid(True, alpha=0.3)
plt.savefig('neupert_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 6: QPP Detection using Wavelet Analysis
# ============================================================
def detect_qpp_wavelet(t_hxr, hxr_flux, min_period=4, max_period=100):
    """
    Detect QPPs using Continuous Wavelet Transform (Morlet)
    """
    # Detrend
    detrended = hxr_flux - gaussian_filter1d(hxr_flux, sigma=50)
    
    # CWT with Morlet wavelet
    widths = np.arange(1, 256)
    cwtmatr, freqs = pywt.cwt(detrended, widths, 'cmor1.5-1.0', 
                               sampling_period=np.mean(np.diff(t_hxr)))
    
    # Convert frequencies to periods
    periods = 1.0 / freqs
    
    # Find peak power in valid period range
    valid_idx = (periods >= min_period) & (periods <= max_period)
    if not np.any(valid_idx):
        return None, None, None
    
    power = np.abs(cwtmatr) ** 2
    valid_power = power[valid_idx, :]
    valid_periods = periods[valid_idx]
    
    # Find global maximum power
    max_idx = np.unravel_index(np.argmax(valid_power), valid_power.shape)
    detected_period = valid_periods[max_idx[0]]
    max_power = valid_power[max_idx]
    
    # Significance: compare to red noise (simple threshold)
    significance = max_power / np.mean(valid_power)
    
    return detected_period, significance, cwtmatr

period, sig, cwt = detect_qpp_wavelet(data['t_hxr'], data['hxr_flux'])
if period:
    print(f"Detected QPP Period: {period:.1f} seconds")
    print(f"Significance: {sig:.2f}")
    print(f"Ground Truth: {data['qpp_period']:.1f} seconds")
else:
    print("No QPP detected")

# ============================================================
# CELL 7: 3-Tier Alert Logic
# ============================================================
def generate_alert(goes_class, neupert_corr, qpp_detected, qpp_confidence):
    """
    Generate 3-tier alert based on combined indicators
    """
    if goes_class in ['X'] or (qpp_detected and qpp_confidence > 0.7):
        return 'ALERT', 0.85, 'High confidence flare escalation. Initiate protective protocols.'
    elif goes_class in ['M'] or (qpp_detected and qpp_confidence > 0.5) or neupert_corr > 0.7:
        return 'ADVISORY', 0.6, 'Flare detected or QPP precursor identified. Notify duty officer.'
    else:
        return 'WATCH', 0.2, 'Baseline monitoring. No significant activity.'

alert_tier, confidence, action = generate_alert(
    goes_class, corr, period is not None, sig if sig else 0
)
print(f"\\nAlert Tier: {alert_tier}")
print(f"Confidence: {confidence:.2f}")
print(f"Action: {action}")

# ============================================================
# CELL 8: Summary Dashboard
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SXR + HXR combined
ax = axes[0, 0]
ax.plot(data['t_sxr'], data['sxr_flux'] * 1e6, 'b-', label='SoLEXS SXR')
ax2 = ax.twinx()
ax2.plot(data['t_hxr'], data['hxr_flux'], 'r-', alpha=0.6, label='HEL1OS HXR')
ax.set_xlabel('Time (s)'); ax.set_ylabel('SXR Flux (×10⁻⁶)', color='b')
ax2.set_ylabel('HXR Flux', color='r')
ax.set_title('Combined SXR + HXR Light Curves')
ax.grid(True, alpha=0.3)

# QPP Wavelet Power
ax = axes[0, 1]
if cwt is not None:
    im = ax.imshow(np.abs(cwt)**2, aspect='auto', cmap='hot',
                   extent=[data['t_hxr'][0], data['t_hxr'][-1], 
                           1.0/pywt.scale2frequency('cmor1.5-1.0', 1)*0.01, 
                           1.0/pywt.scale2frequency('cmor1.5-1.0', 256)*0.01])
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Wavelet Power Spectrum (QPP Detection)')
    plt.colorbar(im, ax=ax)

# Neupert Correlation
ax = axes[1, 0]
ax.scatter(hxr_cum, dsxr, c='purple', alpha=0.6, s=15)
ax.set_xlabel('Cumulative HXR Fluence')
ax.set_ylabel('dSXR/dt')
ax.set_title(f'Neupert Correlation: r = {corr:.3f}')
ax.grid(True, alpha=0.3)

# Alert Status
ax = axes[1, 1]
ax.axis('off')
colors = {'WATCH': 'green', 'ADVISORY': 'orange', 'ALERT': 'red'}
circle = plt.Circle((0.5, 0.7), 0.15, color=colors[alert_tier], alpha=0.8)
ax.add_patch(circle)
ax.text(0.5, 0.7, alert_tier, ha='center', va='center', fontsize=16, 
        fontweight='bold', color='white')
ax.text(0.5, 0.4, f'Confidence: {confidence:.2f}', ha='center', fontsize=14)
ax.text(0.5, 0.25, f'GOES Class: {goes_class}{multiplier:.1f}', ha='center', fontsize=12)
ax.text(0.5, 0.1, action, ha='center', fontsize=10, wrap=True)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('Alert Status')

plt.tight_layout()
plt.savefig('qpp_sentinel_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n QPP-Sentinel Prototype Complete!")
print(\"Files generated: combined_lightcurves.png, neupert_correlation.png, qpp_sentinel_dashboard.png")
