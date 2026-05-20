"""
Utility Functions - Helper functions for data analysis and processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from gnn_config import ARTIFACTS_DIR, DATA_DIR


def calculate_savings(predictions_df, consumption_kwh=1.0):
    """
    Calculate potential savings by using electricity at cheapest vs most expensive times.
    
    Args:
        predictions_df: DataFrame with predicted prices.
        consumption_kwh: Energy consumption in kWh.
    
    Returns:
        Dictionary with savings information.
    """
    if 'predicted_price' in predictions_df.columns:
        prices = predictions_df['predicted_price']
    elif 'predicted' in predictions_df.columns:
        prices = predictions_df['predicted']
    else:
        raise ValueError("No price column found")
    
    min_price = prices.min()
    max_price = prices.max()
    avg_price = prices.mean()
    
    cost_at_min = min_price * consumption_kwh
    cost_at_max = max_price * consumption_kwh
    cost_at_avg = avg_price * consumption_kwh
    
    savings_vs_max = cost_at_max - cost_at_min
    savings_vs_avg = cost_at_avg - cost_at_min
    
    return {
        'consumption_kwh': consumption_kwh,
        'min_price': min_price,
        'max_price': max_price,
        'avg_price': avg_price,
        'cost_at_min': cost_at_min,
        'cost_at_max': cost_at_max,
        'cost_at_avg': cost_at_avg,
        'savings_vs_max': savings_vs_max,
        'savings_vs_avg': savings_vs_avg,
        'savings_percentage_vs_max': (savings_vs_max / cost_at_max) * 100,
        'savings_percentage_vs_avg': (savings_vs_avg / cost_at_avg) * 100
    }


def find_optimal_charging_window(predictions_df, duration_hours=4, consumption_kwh=10.0):
    """
    Find the optimal time window to charge an EV or battery.
    
    Args:
        predictions_df: DataFrame with predicted prices.
        duration_hours: Duration of charging window in hours.
        consumption_kwh: Total energy to consume in kWh.
    
    Returns:
        Dictionary with optimal window information.
    """
    if 'predicted_price' in predictions_df.columns:
        price_col = 'predicted_price'
    elif 'predicted' in predictions_df.columns:
        price_col = 'predicted'
    else:
        raise ValueError("No price column found")
    
    if len(predictions_df) < duration_hours:
        raise ValueError(f"Not enough data for {duration_hours} hour window")
    
    # Calculate rolling sum of prices
    window_costs = []
    
    for i in range(len(predictions_df) - duration_hours + 1):
        window = predictions_df.iloc[i:i+duration_hours]
        avg_price = window[price_col].mean()
        total_cost = avg_price * consumption_kwh
        
        window_costs.append({
            'start_idx': i,
            'start_time': window.iloc[0]['timestamp'],
            'end_time': window.iloc[-1]['timestamp'],
            'avg_price': avg_price,
            'total_cost': total_cost
        })
    
    # Find minimum cost window
    optimal_window = min(window_costs, key=lambda x: x['total_cost'])
    worst_window = max(window_costs, key=lambda x: x['total_cost'])
    
    savings = worst_window['total_cost'] - optimal_window['total_cost']
    
    return {
        'optimal_window': {
            'start_time': optimal_window['start_time'],
            'end_time': optimal_window['end_time'],
            'avg_price': optimal_window['avg_price'],
            'total_cost': optimal_window['total_cost']
        },
        'worst_window': {
            'start_time': worst_window['start_time'],
            'end_time': worst_window['end_time'],
            'avg_price': worst_window['avg_price'],
            'total_cost': worst_window['total_cost']
        },
        'potential_savings': savings,
        'savings_percentage': (savings / worst_window['total_cost']) * 100,
        'duration_hours': duration_hours,
        'consumption_kwh': consumption_kwh
    }


def analyze_hourly_patterns(df):
    """
    Analyze price patterns by hour of day.
    
    Args:
        df: DataFrame with timestamp and price columns.
    
    Returns:
        DataFrame with hourly statistics.
    """
    df = df.copy()
    
    if 'predicted_price' in df.columns:
        price_col = 'predicted_price'
    elif 'predicted' in df.columns:
        price_col = 'predicted'
    elif 'actual' in df.columns:
        price_col = 'actual'
    elif 'price_dkk' in df.columns:
        price_col = 'price_dkk'
    else:
        raise ValueError("No price column found")
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    hourly_stats = df.groupby('hour')[price_col].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    return hourly_stats


def analyze_daily_patterns(df):
    """
    Analyze price patterns by day of week.
    
    Args:
        df: DataFrame with timestamp and price columns.
    
    Returns:
        DataFrame with daily statistics.
    """
    df = df.copy()
    
    if 'predicted_price' in df.columns:
        price_col = 'predicted_price'
    elif 'predicted' in df.columns:
        price_col = 'predicted'
    elif 'actual' in df.columns:
        price_col = 'actual'
    elif 'price_dkk' in df.columns:
        price_col = 'price_dkk'
    else:
        raise ValueError("No price column found")
    
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    daily_stats = df.groupby('day_of_week')[price_col].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats['day_of_week'] = pd.Categorical(
        daily_stats['day_of_week'], 
        categories=day_order, 
        ordered=True
    )
    daily_stats = daily_stats.sort_values('day_of_week').reset_index(drop=True)
    
    return daily_stats


def generate_recommendations(predictions_df, consumption_profile=None):
    """
    Generate smart electricity usage recommendations.
    
    Args:
        predictions_df: DataFrame with predicted prices.
        consumption_profile: Dict with consumption by activity type.
    
    Returns:
        Dictionary with recommendations.
    """
    if consumption_profile is None:
        # Default consumption profile
        consumption_profile = {
            'ev_charging': 10.0,  # kWh
            'dishwasher': 1.5,
            'washing_machine': 2.0,
            'dryer': 3.0,
            'water_heater': 4.0
        }
    
    if 'predicted_price' in predictions_df.columns:
        price_col = 'predicted_price'
    elif 'predicted' in predictions_df.columns:
        price_col = 'predicted'
    else:
        raise ValueError("No price column found")
    
    # Sort by price
    sorted_df = predictions_df.sort_values(price_col)
    
    recommendations = []
    
    for activity, kwh in consumption_profile.items():
        # Get cheapest hours for this activity
        if activity == 'ev_charging':
            # EV charging needs consecutive hours
            window = find_optimal_charging_window(
                predictions_df, 
                duration_hours=4, 
                consumption_kwh=kwh
            )
            
            recommendations.append({
                'activity': activity,
                'type': 'window',
                'recommendation': f"Charge between {window['optimal_window']['start_time']} and {window['optimal_window']['end_time']}",
                'estimated_cost': window['optimal_window']['total_cost'],
                'potential_savings': window['potential_savings'],
                'consumption_kwh': kwh
            })
        else:
            # Other appliances can run at any single hour
            cheapest_hour = sorted_df.iloc[0]
            cost = cheapest_hour[price_col] * kwh
            
            # Compare to average
            avg_price = predictions_df[price_col].mean()
            avg_cost = avg_price * kwh
            savings = avg_cost - cost
            
            recommendations.append({
                'activity': activity,
                'type': 'single_hour',
                'recommendation': f"Run at {cheapest_hour['timestamp']}",
                'estimated_cost': cost,
                'potential_savings': savings,
                'consumption_kwh': kwh
            })
    
    return recommendations


def export_predictions_to_csv(predictions_df, filename='predictions.csv'):
    """
    Export predictions to CSV file.
    
    Args:
        predictions_df: DataFrame with predictions.
        filename: Output filename.
    
    Returns:
        Path to saved file.
    """
    output_path = ARTIFACTS_DIR / filename
    predictions_df.to_csv(output_path, index=False)
    print(f"💾 Exported predictions to {output_path}")
    return output_path


def export_predictions_to_json(predictions_df, filename='predictions.json'):
    """
    Export predictions to JSON file.
    
    Args:
        predictions_df: DataFrame with predictions.
        filename: Output filename.
    
    Returns:
        Path to saved file.
    """
    output_path = ARTIFACTS_DIR / filename
    
    # Convert timestamps to strings
    df_copy = predictions_df.copy()
    if 'timestamp' in df_copy.columns:
        df_copy['timestamp'] = df_copy['timestamp'].astype(str)
    
    predictions_json = df_copy.to_dict('records')
    
    with open(output_path, 'w') as f:
        json.dump(predictions_json, f, indent=2)
    
    print(f"💾 Exported predictions to {output_path}")
    return output_path


def compare_with_baseline(predictions_df, baseline='persistence'):
    """
    Compare GNN predictions with baseline methods.
    
    Args:
        predictions_df: DataFrame with actual and predicted prices.
        baseline: Baseline method ('persistence', 'moving_average', 'naive').
    
    Returns:
        Dictionary with comparison metrics.
    """
    if 'actual' not in predictions_df.columns:
        raise ValueError("Need actual values for comparison")
    
    actual = predictions_df['actual'].values
    gnn_pred = predictions_df['predicted'].values
    
    if baseline == 'persistence':
        # Predict same as last value
        baseline_pred = np.roll(actual, 1)
        baseline_pred[0] = actual[0]
    
    elif baseline == 'moving_average':
        # 24-hour moving average
        baseline_pred = pd.Series(actual).rolling(24, min_periods=1).mean().values
    
    elif baseline == 'naive':
        # Predict mean of all values
        baseline_pred = np.full_like(actual, actual.mean())
    
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    # Calculate metrics for both
    gnn_mae = np.mean(np.abs(actual - gnn_pred))
    baseline_mae = np.mean(np.abs(actual - baseline_pred))
    
    gnn_rmse = np.sqrt(np.mean((actual - gnn_pred) ** 2))
    baseline_rmse = np.sqrt(np.mean((actual - baseline_pred) ** 2))
    
    improvement_mae = ((baseline_mae - gnn_mae) / baseline_mae) * 100
    improvement_rmse = ((baseline_rmse - gnn_rmse) / baseline_rmse) * 100
    
    return {
        'baseline_method': baseline,
        'gnn_mae': gnn_mae,
        'baseline_mae': baseline_mae,
        'improvement_mae_pct': improvement_mae,
        'gnn_rmse': gnn_rmse,
        'baseline_rmse': baseline_rmse,
        'improvement_rmse_pct': improvement_rmse
    }


def create_summary_report(predictions_df, save_path=None):
    """
    Create a comprehensive summary report.
    
    Args:
        predictions_df: DataFrame with predictions.
        save_path: Path to save the report.
    
    Returns:
        Dictionary with summary statistics.
    """
    if 'predicted_price' in predictions_df.columns:
        price_col = 'predicted_price'
    elif 'predicted' in predictions_df.columns:
        price_col = 'predicted'
    else:
        raise ValueError("No price column found")
    
    prices = predictions_df[price_col]
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'num_predictions': len(predictions_df),
        'time_range': {
            'start': predictions_df['timestamp'].min(),
            'end': predictions_df['timestamp'].max()
        },
        'price_statistics': {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'range': float(prices.max() - prices.min())
        },
        'cheapest_hours': [],
        'most_expensive_hours': []
    }
    
    # Add cheapest hours
    cheapest = predictions_df.nsmallest(3, price_col)
    for _, row in cheapest.iterrows():
        report['cheapest_hours'].append({
            'timestamp': str(row['timestamp']),
            'price': float(row[price_col])
        })
    
    # Add most expensive hours
    expensive = predictions_df.nlargest(3, price_col)
    for _, row in expensive.iterrows():
        report['most_expensive_hours'].append({
            'timestamp': str(row['timestamp']),
            'price': float(row[price_col])
        })
    
    # Calculate savings
    savings = calculate_savings(predictions_df, consumption_kwh=10.0)
    report['ev_charging_analysis'] = savings
    
    # Save report
    if save_path is None:
        save_path = ARTIFACTS_DIR / 'summary_report.json'
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"💾 Saved summary report to {save_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    from gnn_predict import GNNPredictor
    
    print("🔧 Testing utility functions...")
    
    predictor = GNNPredictor()
    future_df = predictor.predict_future(num_hours=24)
    
    # Test savings calculation
    print("\n💰 Savings Analysis (10 kWh consumption):")
    savings = calculate_savings(future_df, consumption_kwh=10.0)
    print(f"   Cost at min price: {savings['cost_at_min']:.2f} DKK")
    print(f"   Cost at avg price: {savings['cost_at_avg']:.2f} DKK")
    print(f"   Cost at max price: {savings['cost_at_max']:.2f} DKK")
    print(f"   Savings vs average: {savings['savings_vs_avg']:.2f} DKK ({savings['savings_percentage_vs_avg']:.1f}%)")
    
    # Test optimal charging window
    print("\n🔌 Optimal EV Charging Window (4 hours, 10 kWh):")
    window = find_optimal_charging_window(future_df, duration_hours=4, consumption_kwh=10.0)
    print(f"   Best time: {window['optimal_window']['start_time']} - {window['optimal_window']['end_time']}")
    print(f"   Total cost: {window['optimal_window']['total_cost']:.2f} DKK")
    print(f"   Potential savings: {window['potential_savings']:.2f} DKK ({window['savings_percentage']:.1f}%)")
    
    # Test recommendations
    print("\n💡 Smart Usage Recommendations:")
    recommendations = generate_recommendations(future_df)
    for rec in recommendations:
        print(f"   {rec['activity']}: {rec['recommendation']}")
        print(f"      Cost: {rec['estimated_cost']:.2f} DKK, Savings: {rec['potential_savings']:.2f} DKK")
    
    # Create summary report
    print("\n📊 Generating summary report...")
    report = create_summary_report(future_df)
    print(f"   Generated report with {report['num_predictions']} predictions")
    
    print("\n✅ Utility functions test complete!")