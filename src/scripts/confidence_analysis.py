"""
Confidence and Uncertainty Analysis for Model Predictions.

Analyzes prediction confidence distributions and entropy patterns
to understand model certainty and identify potential failure modes.
"""

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from scipy import stats


def collect_prediction_stats(model, test_loader, device):
    """
    Collect confidence, entropy, and correctness for all predictions.
    
    Args:
        model: PyTorch model (in eval mode)
        test_loader: DataLoader for test data
        device: torch device
    
    Returns:
        dict with 'confidences', 'entropies', 'correct' arrays
    """
    model.eval()
    all_confidences = []
    all_entropies = []
    all_correct = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # Get confidence (max probability)
            confidences = probs.max(dim=1)[0]
            
            # Compute entropy (measure of uncertainty)
            entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            
            # Check correctness
            correct = (preds == labels.to(device))
            
            all_confidences.extend(confidences.cpu().numpy())
            all_entropies.extend(entropies.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
    
    return {
        'confidences': np.array(all_confidences),
        'entropies': np.array(all_entropies),
        'correct': np.array(all_correct)
    }


def plot_confidence_distribution(confidences, correct):
    """
    Create confidence distribution histogram for correct vs incorrect predictions.
    
    Args:
        confidences: Array of prediction confidences
        correct: Boolean array indicating correctness
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Correct predictions
    fig.add_trace(go.Histogram(
        x=confidences[correct],
        name='Correct Predictions',
        marker=dict(color='green', opacity=0.7),
        nbinsx=30,
        hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Incorrect predictions
    fig.add_trace(go.Histogram(
        x=confidences[~correct],
        name='Incorrect Predictions',
        marker=dict(color='red', opacity=0.7),
        nbinsx=30,
        hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Confidence Distribution: Correct vs Incorrect Predictions</b><br>" +
                 "<sub>Higher confidence for correct predictions indicates model certainty</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title="<b>Prediction Confidence</b>", range=[0, 1]),
        yaxis=dict(title="<b>Frequency</b>"),
        barmode='overlay',
        height=450,
        margin=dict(t=100, b=80),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def plot_entropy_boxplot(entropies, correct):
    """
    Create entropy box plot comparing correct vs incorrect predictions.
    
    Args:
        entropies: Array of prediction entropies
        correct: Boolean array indicating correctness
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=entropies[correct],
        name='Correct',
        marker=dict(color='green'),
        boxmean='sd',
        hovertemplate='Entropy: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Box(
        y=entropies[~correct],
        name='Incorrect',
        marker=dict(color='red'),
        boxmean='sd',
        hovertemplate='Entropy: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Prediction Entropy: Measure of Uncertainty</b><br>" +
                 "<sub>Lower entropy indicates higher model certainty (better discrimination)</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        yaxis=dict(title="<b>Entropy (bits)</b>"),
        height=450,
        margin=dict(t=100, b=80)
    )
    
    return fig


def plot_confidence_vs_entropy(confidences, entropies, correct):
    """
    Create scatter plot of confidence vs entropy.
    
    Args:
        confidences: Array of prediction confidences
        entropies: Array of prediction entropies
        correct: Boolean array indicating correctness
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Correct predictions
    fig.add_trace(go.Scatter(
        x=confidences[correct],
        y=entropies[correct],
        mode='markers',
        name='Correct',
        marker=dict(color='green', size=6, opacity=0.6),
        hovertemplate='Confidence: %{x:.2%}<br>Entropy: %{y:.3f}<extra></extra>'
    ))
    
    # Incorrect predictions
    fig.add_trace(go.Scatter(
        x=confidences[~correct],
        y=entropies[~correct],
        mode='markers',
        name='Incorrect',
        marker=dict(color='red', size=8, opacity=0.8),
        hovertemplate='Confidence: %{x:.2%}<br>Entropy: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Confidence vs Uncertainty Analysis</b><br>" +
                 "<sub>Ideal predictions: high confidence + low entropy (bottom-right)</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title="<b>Prediction Confidence</b>", range=[0, 1]),
        yaxis=dict(title="<b>Entropy (Uncertainty)</b>"),
        height=500,
        margin=dict(t=100, b=80),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def print_confidence_summary(confidences, entropies, correct):
    """
    Print summary statistics for confidence and entropy analysis.
    
    Args:
        confidences: Array of prediction confidences
        entropies: Array of prediction entropies
        correct: Boolean array indicating correctness
    """
    print(f"\n{'='*70}")
    print("PREDICTION CONFIDENCE & UNCERTAINTY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📊 Correct Predictions ({correct.sum()} samples):")
    print(f"   Mean confidence: {confidences[correct].mean():.2%}")
    print(f"   Std confidence:  {confidences[correct].std():.2%}")
    print(f"   Mean entropy:    {entropies[correct].mean():.3f} bits")
    
    print(f"\n❌ Incorrect Predictions ({(~correct).sum()} samples):")
    print(f"   Mean confidence: {confidences[~correct].mean():.2%}")
    print(f"   Std confidence:  {confidences[~correct].std():.2%}")
    print(f"   Mean entropy:    {entropies[~correct].mean():.3f} bits")
    
    # Statistical test
    conf_ttest = stats.ttest_ind(confidences[correct], confidences[~correct])
    entropy_ttest = stats.ttest_ind(entropies[correct], entropies[~correct])
    
    def significance_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return 'ns'
    
    print(f"\n📈 Statistical Significance (t-test):")
    print(f"   Confidence difference: p-value = {conf_ttest.pvalue:.4f} {significance_stars(conf_ttest.pvalue)}")
    print(f"   Entropy difference:    p-value = {entropy_ttest.pvalue:.4f} {significance_stars(entropy_ttest.pvalue)}")
    print(f"{'='*70}")


def analyze_confidence_patterns(model, test_loader, device):
    """
    Complete confidence and uncertainty analysis.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: torch device
    
    Returns:
        dict with stats and figures
    """
    print("🔍 Analyzing model confidence and prediction patterns...")
    
    # Collect stats
    stats_dict = collect_prediction_stats(model, test_loader, device)
    confidences = stats_dict['confidences']
    entropies = stats_dict['entropies']
    correct = stats_dict['correct']
    
    # Create visualizations
    fig_conf = plot_confidence_distribution(confidences, correct)
    fig_conf.show()
    
    fig_entropy = plot_entropy_boxplot(entropies, correct)
    fig_entropy.show()
    
    fig_scatter = plot_confidence_vs_entropy(confidences, entropies, correct)
    fig_scatter.show()
    
    # Print summary
    print_confidence_summary(confidences, entropies, correct)
    
    print(f"\n✅ Advanced interpretability analysis complete!")
    print(f"📊 Generated 3 interactive Plotly visualizations")
    
    return {
        'confidences': confidences,
        'entropies': entropies,
        'correct': correct,
        'figures': {
            'confidence': fig_conf,
            'entropy': fig_entropy,
            'scatter': fig_scatter
        }
    }
