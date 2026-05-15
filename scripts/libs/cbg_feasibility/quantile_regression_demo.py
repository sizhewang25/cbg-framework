"""
Quantile Regression for RTT Lower Bound Estimation: Theory and Demo

This script demonstrates how quantile regression can solve the lower-bound RTT
problem for Constraint-Based Geolocation (CBG).

Theory References:
- Koenker & Bassett (1978): "Regression Quantiles" - Econometrica
- Koenker & Hallock (2001): "Quantile Regression" - Journal of Economic Perspectives
  https://www.aeaweb.org/articles?id=10.1257/jep.15.4.143

Author: Research demonstration for CBG feasibility analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check for optional dependencies
try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    from sklearn.linear_model import QuantileRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# PART 1: THEORY OF QUANTILE REGRESSION
# =============================================================================

"""
## What is Quantile Regression?

### The Problem with Ordinary Least Squares (OLS)

OLS regression minimizes the sum of SQUARED errors:

    minimize Σ (y_i - (β₀ + β₁x_i))²

This estimates the CONDITIONAL MEAN: E[Y | X = x]

For RTT data, this is problematic because:
1. RTT noise is ASYMMETRIC (mostly positive outliers from congestion)
2. We want the LOWER BOUND, not the mean
3. Outliers have outsized influence (squared penalty)


### Quantile Regression Solution

Instead of estimating the mean, quantile regression estimates any
CONDITIONAL QUANTILE: Q_τ[Y | X = x]

For τ = 0.05 (5th percentile), we get the lower bound!


### The Loss Function (Check Function / Pinball Loss)

For quantile τ ∈ (0, 1), the loss function is:

    ρ_τ(u) = u × (τ - I(u < 0))

           = { τ × u        if u ≥ 0  (under-prediction)
             { (τ - 1) × u  if u < 0  (over-prediction)

Where u = y - ŷ (residual)

For τ = 0.05:
    - Under-prediction (y > ŷ): penalty = 0.05 × |error|
    - Over-prediction (y < ŷ):  penalty = 0.95 × |error|

This ASYMMETRIC penalty pushes the fit toward the lower quantile!


### Mathematical Formulation

    minimize Σ ρ_τ(y_i - (β₀ + β₁x_i))

    = minimize Σ [ τ × max(0, y_i - ŷ_i) + (1-τ) × max(0, ŷ_i - y_i) ]


### Why This Works for RTT Lower Bounds

1. τ = 0.05 means 95% of points should be ABOVE the line
2. Over-estimating (line above data) is penalized 19× more than under-estimating
3. The regression naturally finds the 5th percentile envelope
4. No binning required - works on raw data
5. Robust to high RTT outliers (they don't pull the line up)


### Comparison: OLS vs Median vs Quantile

| Method          | Estimates        | Loss Function      | Outlier Sensitivity |
|-----------------|------------------|--------------------|--------------------|
| OLS             | Conditional Mean | Squared errors     | HIGH (squared)     |
| Median (τ=0.5)  | Conditional Median | Absolute errors  | LOW                |
| Quantile (τ=0.05)| 5th Percentile  | Asymmetric absolute | LOW + lower bound |

"""


def check_function(u: np.ndarray, tau: float) -> np.ndarray:
    """
    The quantile regression loss function (check function / pinball loss).

    ρ_τ(u) = u × (τ - I(u < 0))

    Args:
        u: Residuals (y - y_hat)
        tau: Quantile (0 < tau < 1)

    Returns:
        Loss values
    """
    return u * (tau - (u < 0).astype(float))


def visualize_check_function():
    """Visualize the asymmetric check function for different quantiles."""
    u = np.linspace(-3, 3, 1000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    taus = [0.05, 0.50, 0.95]
    titles = ['τ=0.05 (Lower Bound)', 'τ=0.50 (Median)', 'τ=0.95 (Upper Bound)']

    for ax, tau, title in zip(axes, taus, titles):
        loss = check_function(u, tau)
        ax.plot(u, loss, 'b-', linewidth=2)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Residual (y - ŷ)', fontsize=11)
        ax.set_ylabel('Loss ρ_τ(u)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate slopes
        ax.annotate(f'slope = τ = {tau}', xy=(1.5, check_function(np.array([1.5]), tau)[0]),
                   fontsize=10, color='blue')
        ax.annotate(f'slope = τ-1 = {tau-1:.2f}', xy=(-2.5, check_function(np.array([-1.5]), tau)[0]),
                   fontsize=10, color='blue')

    plt.suptitle('Quantile Regression Check Function (Pinball Loss)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# PART 2: APPLICATION TO RTT DATA
# =============================================================================

def generate_synthetic_rtt_data(n_points: int = 500, seed: int = 42) -> tuple:
    """
    Generate synthetic RTT-distance data that mimics real network behavior.

    RTT characteristics:
    1. Linear relationship with distance (physics: d = speed × t)
    2. Positive intercept (processing/queuing baseline)
    3. Asymmetric noise (mostly positive outliers from congestion)
    4. Heteroscedastic (variance increases with distance)

    True model: RTT = 0.012 × distance + 8 + noise
    (slope ~0.012 ms/km corresponds to ~2/3 speed of light)
    """
    np.random.seed(seed)

    # Distance range: 0 to 3000 km (continental US scale)
    distances = np.random.uniform(50, 3000, n_points)

    # True lower bound parameters
    true_slope = 0.012  # ms/km (slightly above theoretical 0.01)
    true_intercept = 8.0  # ms (processing delays)

    # Base RTT (the "true" minimum achievable)
    base_rtt = true_slope * distances + true_intercept

    # Add realistic noise:
    # 1. Small symmetric noise (measurement variance)
    symmetric_noise = np.random.normal(0, 1, n_points)

    # 2. Positive skewed noise (congestion, queuing) - exponential
    congestion_noise = np.random.exponential(scale=3 + 0.005 * distances, size=n_points)

    # 3. Occasional large outliers (route changes, server issues)
    outlier_mask = np.random.random(n_points) < 0.05  # 5% outliers
    outliers = np.where(outlier_mask, np.random.uniform(20, 80, n_points), 0)

    # 4. Rare "too fast" points (mislabeled coordinates)
    mislocated_mask = np.random.random(n_points) < 0.02  # 2% mislocated
    mislocated_offset = np.where(mislocated_mask, -np.random.uniform(5, 15, n_points), 0)

    # Combine
    rtts = base_rtt + symmetric_noise + congestion_noise + outliers + mislocated_offset
    rtts = np.maximum(rtts, 1)  # RTT must be positive

    return distances, rtts, true_slope, true_intercept


def fit_quantile_regression(distances: np.ndarray, rtts: np.ndarray,
                            quantile: float = 0.05) -> dict:
    """
    Fit quantile regression to RTT-distance data.

    Args:
        distances: Array of distances (km)
        rtts: Array of RTT values (ms)
        quantile: Target quantile (default 0.05 for lower bound)

    Returns:
        Dictionary with slope, intercept, and fit statistics
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for quantile regression")

    # Add constant for intercept
    X = sm.add_constant(distances)

    # Fit quantile regression
    model = QuantReg(rtts, X)
    result = model.fit(q=quantile)

    return {
        'slope': result.params[1],
        'intercept': result.params[0],
        'quantile': quantile,
        'n_points': len(distances),
        'pseudo_r_squared': result.prsquared,
        'params': result.params,
        'pvalues': result.pvalues,
        'conf_int': result.conf_int(),
        'result_object': result
    }


def fit_ols_regression(distances: np.ndarray, rtts: np.ndarray) -> dict:
    """Fit ordinary least squares for comparison."""
    # Simple linear regression using numpy
    coeffs = np.polyfit(distances, rtts, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # Calculate R²
    predicted = slope * distances + intercept
    ss_res = np.sum((rtts - predicted) ** 2)
    ss_tot = np.sum((rtts - np.mean(rtts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'n_points': len(distances)
    }


def compare_methods_visualization(distances: np.ndarray, rtts: np.ndarray,
                                  true_slope: float, true_intercept: float,
                                  output_path: Path = None):
    """
    Compare OLS, median regression, and quantile regression on RTT data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Fit all methods
    ols_result = fit_ols_regression(distances, rtts)

    if STATSMODELS_AVAILABLE:
        q05_result = fit_quantile_regression(distances, rtts, quantile=0.05)
        q50_result = fit_quantile_regression(distances, rtts, quantile=0.50)
        q95_result = fit_quantile_regression(distances, rtts, quantile=0.95)
    else:
        print("statsmodels not available, skipping quantile regression")
        return fig

    # Distance range for plotting lines
    dist_range = np.linspace(distances.min(), distances.max(), 100)

    # === LEFT PLOT: Scatter with all fits ===
    ax1 = axes[0]

    # Scatter plot
    ax1.scatter(distances, rtts, alpha=0.3, s=15, c='gray',
                label=f'Data (n={len(distances)})', edgecolors='none')

    # True lower bound
    true_line = true_slope * dist_range + true_intercept
    ax1.plot(dist_range, true_line, 'k-', linewidth=2,
             label=f'True Lower Bound: {true_slope:.4f}x + {true_intercept:.1f}')

    # OLS fit
    ols_line = ols_result['slope'] * dist_range + ols_result['intercept']
    ax1.plot(dist_range, ols_line, 'r--', linewidth=2,
             label=f'OLS (Mean): {ols_result["slope"]:.4f}x + {ols_result["intercept"]:.1f}')

    # Median regression
    q50_line = q50_result['slope'] * dist_range + q50_result['intercept']
    ax1.plot(dist_range, q50_line, 'orange', linestyle='-.', linewidth=2,
             label=f'Median (τ=0.5): {q50_result["slope"]:.4f}x + {q50_result["intercept"]:.1f}')

    # Quantile regression (5th percentile)
    q05_line = q05_result['slope'] * dist_range + q05_result['intercept']
    ax1.plot(dist_range, q05_line, 'g-', linewidth=2.5,
             label=f'Quantile (τ=0.05): {q05_result["slope"]:.4f}x + {q05_result["intercept"]:.1f}')

    # 95th percentile for reference
    q95_line = q95_result['slope'] * dist_range + q95_result['intercept']
    ax1.plot(dist_range, q95_line, 'b:', linewidth=1.5, alpha=0.7,
             label=f'Quantile (τ=0.95): {q95_result["slope"]:.4f}x + {q95_result["intercept"]:.1f}')

    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('RTT (ms)', fontsize=12)
    ax1.set_title('Comparison: OLS vs Quantile Regression', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, distances.max() * 1.05)
    ax1.set_ylim(0, np.percentile(rtts, 98))

    # === RIGHT PLOT: Error comparison ===
    ax2 = axes[1]

    methods = ['True\nLower Bound', 'Quantile\n(τ=0.05)', 'Median\n(τ=0.5)', 'OLS\n(Mean)']
    slopes = [true_slope, q05_result['slope'], q50_result['slope'], ols_result['slope']]
    intercepts = [true_intercept, q05_result['intercept'], q50_result['intercept'], ols_result['intercept']]

    # Calculate how many points are BELOW each line (should be ~5% for τ=0.05)
    pct_below = []
    for slope, intercept in zip(slopes, intercepts):
        predicted = slope * distances + intercept
        below = np.mean(rtts < predicted) * 100
        pct_below.append(below)

    colors = ['black', 'green', 'orange', 'red']
    bars = ax2.bar(methods, pct_below, color=colors, alpha=0.7, edgecolor='black')

    # Add percentage labels on bars
    for bar, pct in zip(bars, pct_below):
        ax2.annotate(f'{pct:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add target line at 5%
    ax2.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Target (5%)')

    ax2.set_ylabel('% of Points Below Line', fontsize=12)
    ax2.set_title('Lower Bound Accuracy:\n% of Data Points Below Fitted Line',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, max(pct_below) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def demonstrate_quantile_regression():
    """
    Main demonstration: show how quantile regression solves the RTT lower bound problem.
    """
    print("=" * 70)
    print("QUANTILE REGRESSION FOR RTT LOWER BOUND ESTIMATION")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic RTT-distance data...")
    distances, rtts, true_slope, true_intercept = generate_synthetic_rtt_data(n_points=500)
    print(f"   - {len(distances)} data points")
    print(f"   - True lower bound: RTT = {true_slope:.4f} × distance + {true_intercept:.1f}")

    # Fit OLS
    print("\n2. Fitting Ordinary Least Squares (OLS)...")
    ols = fit_ols_regression(distances, rtts)
    print(f"   - OLS slope:     {ols['slope']:.4f} ms/km")
    print(f"   - OLS intercept: {ols['intercept']:.1f} ms")
    print(f"   - OLS R²:        {ols['r_squared']:.4f}")

    # Fit quantile regression
    if STATSMODELS_AVAILABLE:
        print("\n3. Fitting Quantile Regression (τ = 0.05)...")
        qr = fit_quantile_regression(distances, rtts, quantile=0.05)
        print(f"   - QR slope:     {qr['slope']:.4f} ms/km")
        print(f"   - QR intercept: {qr['intercept']:.1f} ms")
        print(f"   - Pseudo R²:    {qr['pseudo_r_squared']:.4f}")

        # Compare accuracy
        print("\n4. Comparing Lower Bound Accuracy...")

        # Count points below each line
        ols_predicted = ols['slope'] * distances + ols['intercept']
        qr_predicted = qr['slope'] * distances + qr['intercept']
        true_predicted = true_slope * distances + true_intercept

        ols_below = np.mean(rtts < ols_predicted) * 100
        qr_below = np.mean(rtts < qr_predicted) * 100
        true_below = np.mean(rtts < true_predicted) * 100

        print(f"   - Points below TRUE lower bound: {true_below:.1f}% (expected ~2-3%)")
        print(f"   - Points below QR (τ=0.05):      {qr_below:.1f}% (target: 5%)")
        print(f"   - Points below OLS (mean):       {ols_below:.1f}% (expected ~50%)")

        # Slope error comparison
        print("\n5. Slope Estimation Error:")
        ols_slope_error = abs(ols['slope'] - true_slope) / true_slope * 100
        qr_slope_error = abs(qr['slope'] - true_slope) / true_slope * 100
        print(f"   - OLS slope error: {ols_slope_error:.1f}%")
        print(f"   - QR slope error:  {qr_slope_error:.1f}%")

    else:
        print("\n[!] statsmodels not available - install for quantile regression")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Quantile regression (τ=0.05) directly estimates the")
    print("5th percentile - exactly what we need for CBG lower bound!")
    print("=" * 70)

    return distances, rtts, true_slope, true_intercept


# =============================================================================
# PART 3: COMPARISON WITH CURRENT LP APPROACH
# =============================================================================

"""
## Quantile Regression vs Current LP + 5-Stage Filter

| Aspect              | Current (LP + Filter)          | Quantile Regression         |
|---------------------|--------------------------------|-----------------------------|
| **Data prep**       | 5-stage filtering required     | Works on raw data           |
| **Binning**         | Required (100km bins)          | Not required                |
| **Parameters**      | 6+ tuning parameters           | Just τ (quantile level)     |
| **Constraint**      | "Below ALL points" (strict)    | "Below τ% of points" (soft) |
| **Outlier handling**| Explicit filtering stages      | Inherent robustness         |
| **Interpretation**  | LP optimization                | Statistical regression      |
| **Confidence ints** | Not standard                   | Built-in                    |
| **Dependencies**    | scipy (linprog)                | statsmodels                 |

### When to Use Each:

**Use LP + Filter when:**
- You need STRICT lower bound (line below ALL valid points)
- Data has known mislabeled coordinates (need explicit filtering)
- Reproducibility with existing CBG papers is important

**Use Quantile Regression when:**
- You want statistical rigor with confidence intervals
- Data is relatively clean (few gross errors)
- You want to avoid tuning filter parameters
- You need interpretable uncertainty estimates
"""


if __name__ == "__main__":
    # Run demonstration
    distances, rtts, true_slope, true_intercept = demonstrate_quantile_regression()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Check function visualization
    fig1 = visualize_check_function()
    plt.savefig('outputs/quantile_check_function.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/quantile_check_function.png")

    # Method comparison
    if STATSMODELS_AVAILABLE:
        fig2 = compare_methods_visualization(
            distances, rtts, true_slope, true_intercept,
            output_path=Path('outputs/quantile_vs_ols_comparison.png')
        )

    plt.show()
