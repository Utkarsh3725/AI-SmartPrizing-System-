"""
==============================================================================
AI-Driven Smart Pricing System for Theme Parks
Data Generation Module
Team B | Utkarsh Arya | SRM University | Deloitte Digital Camp 2026
==============================================================================

Generates 2 years of realistic daily theme park data with:
- Indian holidays (Diwali, Holi, Independence Day, Republic Day, etc.)
- Seasonal weather patterns
- Demand-driven pricing dynamics
- Competitor price fluctuations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


def get_indian_holidays(year):
    """
    Returns a dictionary of major Indian holidays with approximate dates.
    Dates are approximate for lunar calendar holidays.
    """
    holidays = {
        # Fixed date holidays
        f"{year}-01-26": "Republic Day",
        f"{year}-08-15": "Independence Day",
        f"{year}-10-02": "Gandhi Jayanti",
        f"{year}-12-25": "Christmas",
        f"{year}-01-01": "New Year",
        f"{year}-11-14": "Children's Day",
        f"{year}-01-14": "Makar Sankranti",
        
        # Approximate dates for lunar holidays (varies by year)
        f"{year}-03-14": "Holi",
        f"{year}-03-15": "Holi (Day 2)",
        f"{year}-04-14": "Baisakhi",
        f"{year}-08-19": "Janmashtami",
        f"{year}-09-07": "Ganesh Chaturthi",
        f"{year}-10-02": "Gandhi Jayanti",
        f"{year}-10-12": "Dussehra",
        f"{year}-10-24": "Diwali Eve",
        f"{year}-10-25": "Diwali",
        f"{year}-10-26": "Diwali (Day 2)",
        f"{year}-10-27": "Govardhan Puja",
        f"{year}-10-28": "Bhai Dooj",
        f"{year}-11-01": "Chhath Puja",
        f"{year}-11-27": "Guru Nanak Jayanti",
        f"{year}-04-10": "Ram Navami",
        f"{year}-04-21": "Mahavir Jayanti",
        f"{year}-05-12": "Buddha Purnima",
        f"{year}-06-17": "Eid ul-Fitr",
        f"{year}-08-01": "Eid ul-Adha",
        f"{year}-09-27": "Milad un-Nabi",
    }
    
    # Add school vacation periods (summer + winter)
    summer_vacation = {f"{year}-05-{d:02d}": "Summer Vacation" for d in range(1, 32)}
    summer_vacation.update({f"{year}-06-{d:02d}": "Summer Vacation" for d in range(1, 16)})
    winter_vacation = {f"{year}-12-{d:02d}": "Winter Vacation" for d in range(20, 32)}
    
    holidays.update(summer_vacation)
    holidays.update(winter_vacation)
    
    return holidays


def get_weather(month, day, rng):
    """
    Generate realistic weather based on Indian seasonal patterns.
    Returns weather type and temperature.
    """
    # Indian seasons
    if month in [12, 1, 2]:  # Winter
        weather_probs = [0.60, 0.30, 0.10]  # sunny, cloudy, rainy
        temp_range = (15, 25)
    elif month in [3, 4, 5]:  # Summer
        weather_probs = [0.70, 0.20, 0.10]
        temp_range = (28, 40)
    elif month in [6, 7, 8, 9]:  # Monsoon
        weather_probs = [0.20, 0.35, 0.45]
        temp_range = (24, 35)
    else:  # Post-monsoon (Oct, Nov)
        weather_probs = [0.55, 0.30, 0.15]
        temp_range = (20, 32)
    
    weather = rng.choice(['sunny', 'cloudy', 'rainy'], p=weather_probs)
    temperature = rng.integers(temp_range[0], temp_range[1] + 1)
    
    # Adjust temperature based on weather
    if weather == 'rainy':
        temperature = max(15, temperature - rng.integers(2, 6))
    elif weather == 'cloudy':
        temperature = max(15, temperature - rng.integers(1, 3))
    
    return weather, int(temperature)


def calculate_visitors(row, rng):
    """
    Calculate realistic visitor count based on multiple factors.
    """
    base_visitors = 1500
    
    # Day of week effect
    day_multiplier = {
        0: 0.7,   # Monday
        1: 0.65,  # Tuesday
        2: 0.7,   # Wednesday
        3: 0.75,  # Thursday
        4: 0.85,  # Friday
        5: 1.4,   # Saturday
        6: 1.5,   # Sunday
    }
    visitors = base_visitors * day_multiplier.get(row['day_of_week'], 1.0)
    
    # Holiday boost
    if row['is_holiday']:
        visitors *= rng.uniform(1.8, 2.5)
    
    # Weather effect
    if row['weather'] == 'sunny':
        visitors *= rng.uniform(1.1, 1.3)
    elif row['weather'] == 'rainy':
        visitors *= rng.uniform(0.4, 0.65)
    elif row['weather'] == 'cloudy':
        visitors *= rng.uniform(0.85, 1.0)
    
    # Temperature effect (too hot or too cold reduces visitors)
    temp = row['temperature']
    if temp > 38:
        visitors *= 0.7
    elif temp > 35:
        visitors *= 0.85
    elif 22 <= temp <= 30:
        visitors *= 1.1
    elif temp < 18:
        visitors *= 0.8
    
    # Seasonal trends
    month = row['date'].month
    if month in [10, 11, 12]:  # Festival season
        visitors *= rng.uniform(1.1, 1.3)
    elif month in [6, 7, 8]:  # Monsoon - lower attendance
        visitors *= rng.uniform(0.7, 0.9)
    elif month in [5]:  # Summer vacation
        visitors *= rng.uniform(1.2, 1.4)
    
    # Add some random noise
    visitors *= rng.uniform(0.85, 1.15)
    
    return int(np.clip(visitors, 200, 5000))


def calculate_competitor_price(date, weather, is_holiday, rng):
    """
    Generate realistic competitor pricing.
    """
    base_price = 1000
    
    if is_holiday:
        base_price += rng.integers(200, 400)
    
    if date.weekday() >= 5:  # Weekend
        base_price += rng.integers(100, 250)
    
    if weather == 'rainy':
        base_price -= rng.integers(50, 150)
    elif weather == 'sunny':
        base_price += rng.integers(50, 100)
    
    # Seasonal adjustment
    month = date.month
    if month in [10, 11, 12]:
        base_price += rng.integers(100, 200)
    elif month in [6, 7, 8]:
        base_price -= rng.integers(50, 100)
    
    base_price += rng.integers(-50, 51)
    
    return int(np.clip(base_price, 800, 1500))


def calculate_optimal_price(row, rng):
    """
    Calculate the optimal recommended price using a demand-based formula.
    This serves as the target variable for ML training.
    """
    base_price = 1000
    
    # Demand-based pricing
    visitor_ratio = row['visitors'] / 3000  # Normalize visitors
    demand_premium = visitor_ratio * 400
    
    # Weather adjustments
    if row['weather'] == 'sunny':
        weather_adj = rng.integers(50, 120)
    elif row['weather'] == 'rainy':
        weather_adj = -rng.integers(80, 180)
    else:
        weather_adj = rng.integers(-20, 40)
    
    # Holiday premium
    holiday_adj = rng.integers(150, 350) if row['is_holiday'] else 0
    
    # Weekend premium
    weekend_adj = rng.integers(80, 180) if row['is_weekend'] else 0
    
    # Competitor-aware pricing (stay within 15% of competitor)
    comp_diff = row['competitor_price'] - base_price
    competitor_adj = comp_diff * rng.uniform(0.3, 0.6)
    
    # Temperature effect
    temp = row['temperature']
    if 22 <= temp <= 30:
        temp_adj = rng.integers(30, 80)
    elif temp > 36:
        temp_adj = -rng.integers(40, 100)
    else:
        temp_adj = 0
    
    optimal_price = base_price + demand_premium + weather_adj + holiday_adj + weekend_adj + competitor_adj + temp_adj
    
    # Add slight noise for realism
    optimal_price += rng.integers(-30, 31)
    
    return int(np.clip(optimal_price, 800, 1500))


def generate_theme_park_data():
    """
    Main function to generate the complete dataset.
    """
    print("=" * 60)
    print("  AI Smart Pricing System - Data Generator")
    print("  Team B | Utkarsh Arya | SRM University")
    print("  Deloitte Digital Camp 2026")
    print("=" * 60)
    print()
    
    rng = np.random.default_rng(seed=42)
    
    # Generate date range: 2 years
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"📅 Generating data from {start_date.date()} to {end_date.date()}")
    print(f"📊 Total days: {len(dates)}")
    print()
    
    # Collect all Indian holidays for both years
    all_holidays = {}
    for year in [2024, 2025]:
        all_holidays.update(get_indian_holidays(year))
    
    # Build the dataset
    data = []
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 1 if date_str in all_holidays else 0
        
        weather, temperature = get_weather(date.month, date.day, rng)
        competitor_price = calculate_competitor_price(date, weather, is_holiday, rng)
        
        row = {
            'date': date,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'weather': weather,
            'temperature': temperature,
            'competitor_price': competitor_price,
            'visitors': 0,  # placeholder
            'current_price': 0,  # placeholder
        }
        
        # Calculate visitors based on conditions
        row['visitors'] = calculate_visitors(row, rng)
        
        # Current price (what the park actually charged - slightly suboptimal)
        base_current = 1000 + rng.integers(-100, 101)
        if is_weekend:
            base_current += rng.integers(50, 150)
        if is_holiday:
            base_current += rng.integers(100, 200)
        row['current_price'] = int(np.clip(base_current, 800, 1500))
        
        # Revenue
        row['revenue'] = row['visitors'] * row['current_price']
        
        # Optimal recommended price (target for ML)
        row['recommended_price'] = calculate_optimal_price(row, rng)
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure correct data types
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['temperature'] = df['temperature'].astype(int)
    df['competitor_price'] = df['competitor_price'].astype(int)
    df['visitors'] = df['visitors'].astype(int)
    df['current_price'] = df['current_price'].astype(int)
    df['revenue'] = df['revenue'].astype(int)
    df['recommended_price'] = df['recommended_price'].astype(int)
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_dir, 'theme_park_data.csv')
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("✅ Data generation complete!")
    print(f"💾 Saved to: {output_path}")
    print()
    print("📊 Dataset Summary:")
    print(f"   • Total records: {len(df)}")
    print(f"   • Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   • Average visitors/day: {df['visitors'].mean():.0f}")
    print(f"   • Average ticket price: ₹{df['current_price'].mean():.0f}")
    print(f"   • Average recommended price: ₹{df['recommended_price'].mean():.0f}")
    print(f"   • Total revenue: ₹{df['revenue'].sum():,.0f}")
    print(f"   • Holiday days: {df['is_holiday'].sum()}")
    print(f"   • Weekend days: {df['is_weekend'].sum()}")
    print()
    print("🌦️  Weather Distribution:")
    weather_dist = df['weather'].value_counts()
    for w, count in weather_dist.items():
        pct = count / len(df) * 100
        print(f"   • {w.capitalize()}: {count} days ({pct:.1f}%)")
    print()
    print("📈 Revenue Statistics:")
    print(f"   • Min daily revenue:  ₹{df['revenue'].min():,.0f}")
    print(f"   • Max daily revenue:  ₹{df['revenue'].max():,.0f}")
    print(f"   • Mean daily revenue: ₹{df['revenue'].mean():,.0f}")
    print()
    
    return df


# Also save to SQLite for database integration
def save_to_sqlite(df):
    """Save the generated data to a SQLite database."""
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'theme_park.db')
    conn = sqlite3.connect(db_path)
    
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
    df_copy.to_sql('daily_data', conn, if_exists='replace', index=False)
    
    # Create summary tables
    # Monthly aggregation
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month').agg({
        'visitors': 'sum',
        'revenue': 'sum',
        'current_price': 'mean',
        'recommended_price': 'mean'
    }).reset_index()
    monthly.to_sql('monthly_summary', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"🗄️  SQLite database saved to: {db_path}")
    print()


if __name__ == "__main__":
    df = generate_theme_park_data()
    save_to_sqlite(df)
    print("🎉 All data generation tasks completed successfully!")
