"""
Redesigned Apponomics Tier Classification System
Based on Neutral Apps vs Discriminator Apps Framework

This system separates:
1. Neutral/Benchmark Apps (cross-tier) - enriched with behavioral features
2. Tier Discriminator Apps (strong signals for specific tiers)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class RedesignedAppTierClassifier:
    def __init__(self):
        """Initialize the redesigned classifier with neutral vs discriminator app framework."""
        
        # NEUTRAL/BENCHMARK APPS (Cross-tier, behavior matters)
        self.neutral_apps = {
            'food_delivery': ['zomato', 'swiggy', 'blinkit', 'zepto'],
            'transportation': ['ola', 'uber', 'rapido'],
            'payments': ['paytm', 'phonepe', 'google_pay', 'amazon_pay'],
            'ecommerce': ['flipkart', 'amazon', 'myntra'],
            'travel': ['irctc', 'makemytrip', 'goibibo'],
            'social_media': ['instagram', 'facebook', 'whatsapp', 'telegram'],
            'entertainment': ['youtube', 'netflix', 'spotify']
        }
        
        # TIER DISCRIMINATOR APPS (Strong signals)
        self.tier_discriminators = {
            # Tier A Indicators (Premium Signals)
            'tier_a_premium': [
                'cred', 'indmoney', 'zerodha', 'jupiter', 'groww',
                'airbnb', 'urban_company', 'nykaa_luxe', 'linkedin_premium',
                'apple_music', 'taj_hotels', 'binance', 'wazirx',
                'adobe_creative', 'notion', 'figma', 'slack'
            ],
            
            # Tier B Indicators (Value/Mainstream)
            'tier_b_mainstream': [
                'meesho', 'ajio', 'tata_neu', 'byju', 'unacademy',
                'sony_liv', 'magicbricks', 'apna', 'naukri',
                'bookmyshow', 'practo', 'policybazaar'
            ],
            
            # Tier C Indicators (Budget/Entry-level)
            'tier_c_budget': [
                'ludo_king', 'sharechat', 'moj', 'kreditbee', 'cashbean',
                'winzo', 'bharatpe_merchant', 'uc_browser', 'snack_video',
                'likee', 'mx_takatak', 'josh', 'chingari'
            ]
        }
        
        # Behavioral feature weights for neutral apps (reduced)
        self.neutral_behavior_weights = {
            'food_delivery': {
                'frequency_weight': 0.05,  # orders per month
                'aov_weight': 0.1,        # average order value
                'total_spend_weight': 0.08  # monthly spend
            },
            'transportation': {
                'frequency_weight': 0.05,
                'aov_weight': 0.08,
                'total_spend_weight': 0.06
            },
            'payments': {
                'frequency_weight': 0.02,
                'aov_weight': 0.15,        # transaction value is key
                'total_spend_weight': 0.12
            },
            'ecommerce': {
                'frequency_weight': 0.05,
                'aov_weight': 0.1,
                'total_spend_weight': 0.08
            }
        }
        
        # Discriminator app weights (more balanced)
        self.discriminator_weights = {
            'tier_a_premium': {'spending': 1.0, 'geographic': 0.8, 'lifestyle': 0.8},
            'tier_b_mainstream': {'spending': 0.3, 'geographic': 0.4, 'lifestyle': 0.5},
            'tier_c_budget': {'spending': -0.4, 'geographic': -0.5, 'lifestyle': -0.3}
        }
    
    def analyze_apps_with_behavior(self, installed_apps: List[str], 
                                  behavioral_data: Dict = None) -> Dict:
        """
        Analyze apps using the redesigned framework.
        
        Args:
            installed_apps: List of app names
            behavioral_data: Dict with usage patterns for neutral apps
                e.g., {'zomato_orders_per_month': 8, 'zomato_avg_order_value': 350}
        
        Returns:
            Dictionary with tier predictions
        """
        
        # Convert to lowercase
        apps_lower = [app.lower().replace(' ', '_') for app in installed_apps]
        
        # Analyze discriminator apps (strong signals)
        discriminator_scores = self._analyze_discriminators(apps_lower)
        
        # Analyze neutral apps with behavioral data
        neutral_scores = self._analyze_neutral_apps(apps_lower, behavioral_data)
        
        # Calculate final scores
        spending_score = discriminator_scores['spending'] + neutral_scores['spending']
        geographic_score = discriminator_scores['geographic'] + neutral_scores['geographic']
        lifestyle_score = discriminator_scores['lifestyle'] + neutral_scores['lifestyle']
        
        # Determine tiers
        spending_tier = self._get_spending_tier(spending_score)
        geographic_tier = self._get_geographic_tier(geographic_score)
        lifestyle_category = self._get_lifestyle_category(lifestyle_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(discriminator_scores, neutral_scores)
        
        return {
            'spending_tier': spending_tier,
            'geographic_tier': geographic_tier,
            'lifestyle_category': lifestyle_category,
            'spending_score': round(spending_score, 2),
            'geographic_score': round(geographic_score, 2),
            'lifestyle_score': round(lifestyle_score, 2),
            'confidence': round(confidence, 2),
            'discriminator_analysis': discriminator_scores,
            'neutral_analysis': neutral_scores,
            'recommendations': self._get_recommendations(spending_tier, geographic_tier, lifestyle_category)
        }
    
    def _analyze_discriminators(self, apps_lower: List[str]) -> Dict:
        """Analyze discriminator apps for strong tier signals."""
        scores = {'spending': 0, 'geographic': 0, 'lifestyle': 0}
        matched_apps = {'tier_a_premium': [], 'tier_b_mainstream': [], 'tier_c_budget': []}
        
        for category, apps in self.tier_discriminators.items():
            matches = [app for app in apps if app in apps_lower]
            matched_apps[category] = matches
            
            if matches:
                weights = self.discriminator_weights[category]
                score_multiplier = len(matches)  # More apps = stronger signal
                
                scores['spending'] += weights['spending'] * score_multiplier
                scores['geographic'] += weights['geographic'] * score_multiplier
                scores['lifestyle'] += weights['lifestyle'] * score_multiplier
        
        scores['matched_apps'] = matched_apps
        return scores
    
    def _analyze_neutral_apps(self, apps_lower: List[str], behavioral_data: Dict) -> Dict:
        """Analyze neutral apps with behavioral features."""
        scores = {'spending': 0, 'geographic': 0, 'lifestyle': 0}
        neutral_usage = {}
        
        # Default behavioral data if not provided
        if behavioral_data is None:
            behavioral_data = {}
        
        for category, apps in self.neutral_apps.items():
            matches = [app for app in apps if app in apps_lower]
            
            if matches and category in self.neutral_behavior_weights:
                weights = self.neutral_behavior_weights[category]
                
                # Extract behavioral data for this category
                freq_key = f"{matches[0]}_orders_per_month"  # Use first app as proxy
                aov_key = f"{matches[0]}_avg_order_value"
                spend_key = f"{matches[0]}_monthly_spend"
                
                frequency = behavioral_data.get(freq_key, 5)  # Default moderate usage
                aov = behavioral_data.get(aov_key, 200)      # Default moderate AOV
                monthly_spend = behavioral_data.get(spend_key, frequency * aov)
                
                # Calculate behavioral score
                behavioral_score = (
                    frequency * weights['frequency_weight'] +
                    aov * weights['aov_weight'] / 100 +  # Normalize AOV
                    monthly_spend * weights['total_spend_weight'] / 1000  # Normalize spend
                )
                
                scores['spending'] += behavioral_score
                scores['geographic'] += behavioral_score * 0.5  # Moderate geographic signal
                scores['lifestyle'] += behavioral_score * 0.3  # Moderate lifestyle signal
                
                neutral_usage[category] = {
                    'apps': matches,
                    'frequency': frequency,
                    'aov': aov,
                    'monthly_spend': monthly_spend,
                    'behavioral_score': behavioral_score
                }
        
        scores['usage_patterns'] = neutral_usage
        return scores
    
    def _get_spending_tier(self, score: float) -> str:
        """Determine spending tier based on score."""
        if score >= 4.0:  # Very high threshold for Premium (needs Tier A apps)
            return "Premium (₹50,000+ monthly)"
        elif score >= 1.5:  # Moderate threshold for Standard
            return "Standard (₹15,000-50,000 monthly)"
        else:
            return "Basic (₹5,000-15,000 monthly)"
    
    def _get_geographic_tier(self, score: float) -> str:
        """Determine geographic tier based on score."""
        if score >= 0.8:
            return "Tier 1 City (Mumbai, Delhi, Bangalore, etc.)"
        elif score >= 0.3:
            return "Tier 2 City (Pune, Hyderabad, Chennai, etc.)"
        else:
            return "Tier 3 City (Smaller cities and towns)"
    
    def _get_lifestyle_category(self, score: float) -> str:
        """Determine lifestyle category based on score."""
        if score >= 0.6:
            return "Professional/Urban"
        elif score >= 0.2:
            return "Entertainment/Social"
        else:
            return "Basic/Conservative"
    
    def _calculate_confidence(self, discriminator_scores: Dict, neutral_scores: Dict) -> float:
        """Calculate confidence based on signal strength."""
        # Higher confidence with more discriminator apps
        discriminator_strength = sum(len(apps) for apps in discriminator_scores['matched_apps'].values())
        
        # Moderate confidence from neutral app usage patterns
        neutral_strength = len(neutral_scores.get('usage_patterns', {}))
        
        # Weight discriminator apps more heavily
        total_strength = discriminator_strength * 2 + neutral_strength
        confidence = min(total_strength / 10, 1.0) * 100  # Normalize to 100%
        
        return confidence
    
    def _get_recommendations(self, spending_tier: str, geographic_tier: str, lifestyle_category: str) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        if "Premium" in spending_tier:
            recommendations.extend([
                "Consider premium financial apps like CRED or Zerodha",
                "High-value investment opportunities available",
                "Premium lifestyle services recommended"
            ])
        elif "Standard" in spending_tier:
            recommendations.extend([
                "Balanced financial planning recommended",
                "Standard investment options suitable",
                "Moderate lifestyle services appropriate"
            ])
        else:
            recommendations.extend([
                "Budget-friendly financial tools recommended",
                "Basic investment options suitable",
                "Essential lifestyle services recommended"
            ])
        
        return recommendations

def create_synthetic_datasets():
    """Create synthetic datasets based on the redesigned framework."""
    
    print("Creating synthetic datasets based on neutral vs discriminator framework...")
    
    # Dataset 1: Neutral App Usage with Behavioral Features
    np.random.seed(42)
    n_users = 1000
    
    neutral_data = []
    for i in range(n_users):
        # Generate user demographics
        age = np.random.randint(18, 65)
        gender = np.random.choice(['Male', 'Female', 'Other'])
        city_tier = np.random.choice(['Tier1', 'Tier2', 'Tier3'], p=[0.3, 0.4, 0.3])
        phone_brand = np.random.choice(['Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Realme'], 
                                     p=[0.15, 0.25, 0.15, 0.3, 0.15])
        
        # Phone price correlates with income
        if phone_brand == 'Apple':
            phone_price_band = np.random.choice(['₹50k+', '₹40-50k'], p=[0.7, 0.3])
            monthly_income_band = np.random.choice(['₹60k+', '₹40-60k'], p=[0.8, 0.2])
        elif phone_brand in ['Samsung', 'OnePlus']:
            phone_price_band = np.random.choice(['₹30-40k', '₹20-30k'], p=[0.6, 0.4])
            monthly_income_band = np.random.choice(['₹25-40k', '₹15-25k'], p=[0.7, 0.3])
        else:  # Xiaomi, Realme
            phone_price_band = np.random.choice(['₹15-20k', '₹10-15k'], p=[0.6, 0.4])
            monthly_income_band = np.random.choice(['₹10-20k', '₹5-10k'], p=[0.7, 0.3])
        
        # Generate behavioral data for neutral apps
        user_data = {
            'user_id': f'U{i+1:04d}',
            'age': age,
            'gender': gender,
            'city_tier': city_tier,
            'phone_brand': phone_brand,
            'phone_price_band': phone_price_band,
            'monthly_income_band': monthly_income_band,
            
            # Food delivery behavior
            'zomato_orders_per_month': np.random.poisson(6) if city_tier in ['Tier1', 'Tier2'] else np.random.poisson(3),
            'zomato_avg_order_value': np.random.normal(300, 100) if monthly_income_band.startswith('₹60k') else 
                                    np.random.normal(200, 80) if monthly_income_band.startswith('₹25k') else 
                                    np.random.normal(150, 60),
            
            'blinkit_orders_per_month': np.random.poisson(8) if city_tier == 'Tier1' else np.random.poisson(4),
            'blinkit_avg_order_value': np.random.normal(250, 80) if monthly_income_band.startswith('₹60k') else 
                                     np.random.normal(180, 60) if monthly_income_band.startswith('₹25k') else 
                                     np.random.normal(120, 40),
            
            # Transportation behavior
            'uber_rides_per_month': np.random.poisson(10) if city_tier == 'Tier1' else np.random.poisson(5),
            'uber_avg_ride_value': np.random.normal(200, 80) if monthly_income_band.startswith('₹60k') else 
                                 np.random.normal(120, 50) if monthly_income_band.startswith('₹25k') else 
                                 np.random.normal(80, 30),
            
            # E-commerce behavior
            'amazon_orders_per_month': np.random.poisson(4),
            'amazon_avg_order_value': np.random.normal(800, 300) if monthly_income_band.startswith('₹60k') else 
                                    np.random.normal(500, 200) if monthly_income_band.startswith('₹25k') else 
                                    np.random.normal(300, 150),
            
            # Payment behavior
            'upi_txn_count': np.random.poisson(20),
            'upi_txn_avg_value': np.random.normal(500, 200) if monthly_income_band.startswith('₹60k') else 
                               np.random.normal(300, 150) if monthly_income_band.startswith('₹25k') else 
                               np.random.normal(200, 100),
        }
        
        # Ensure positive values
        for key, value in user_data.items():
            if isinstance(value, (int, float)) and value < 0:
                user_data[key] = abs(value)
        
        neutral_data.append(user_data)
    
    neutral_df = pd.DataFrame(neutral_data)
    neutral_df.to_csv('data/neutral_app_usage.csv', index=False)
    print(f"Created data/neutral_app_usage.csv with {len(neutral_df)} users")
    
    # Dataset 2: Tier Discriminator Apps
    tier_data = []
    for i, user in enumerate(neutral_data):
        # Determine tier based on income and phone
        if user['monthly_income_band'].startswith('₹60k') or user['phone_brand'] == 'Apple':
            tier = 'Tier1'
            spend_cap = np.random.normal(80000, 20000)
        elif user['monthly_income_band'].startswith('₹25k') or user['phone_brand'] in ['Samsung', 'OnePlus']:
            tier = 'Tier2'
            spend_cap = np.random.normal(30000, 10000)
        else:
            tier = 'Tier3'
            spend_cap = np.random.normal(12000, 5000)
        
        # Generate discriminator app usage based on tier
        discriminator_apps = {
            # Tier A Premium Apps
            'cred': 1 if tier == 'Tier1' and np.random.random() > 0.7 else 0,
            'indmoney': 1 if tier == 'Tier1' and np.random.random() > 0.6 else 0,
            'zerodha': 1 if tier == 'Tier1' and np.random.random() > 0.5 else 0,
            'airbnb': 1 if tier == 'Tier1' and np.random.random() > 0.8 else 0,
            'urban_company': 1 if tier == 'Tier1' and np.random.random() > 0.7 else 0,
            'linkedin_premium': 1 if tier == 'Tier1' and np.random.random() > 0.8 else 0,
            
            # Tier B Mainstream Apps
            'meesho': 1 if tier == 'Tier2' and np.random.random() > 0.4 else 0,
            'ajio': 1 if tier == 'Tier2' and np.random.random() > 0.5 else 0,
            'tata_neu': 1 if tier == 'Tier2' and np.random.random() > 0.6 else 0,
            'byju': 1 if tier == 'Tier2' and np.random.random() > 0.5 else 0,
            'unacademy': 1 if tier in ['Tier1', 'Tier2'] and np.random.random() > 0.4 else 0,
            'sony_liv': 1 if tier == 'Tier2' and np.random.random() > 0.6 else 0,
            
            # Tier C Budget Apps
            'ludo_king': 1 if tier == 'Tier3' and np.random.random() > 0.3 else 0,
            'sharechat': 1 if tier == 'Tier3' and np.random.random() > 0.4 else 0,
            'moj': 1 if tier == 'Tier3' and np.random.random() > 0.5 else 0,
            'kreditbee': 1 if tier == 'Tier3' and np.random.random() > 0.6 else 0,
            'cashbean': 1 if tier == 'Tier3' and np.random.random() > 0.7 else 0,
            'winzo': 1 if tier == 'Tier3' and np.random.random() > 0.8 else 0,
        }
        
        tier_user_data = {
            'user_id': user['user_id'],
            'spend_cap': max(spend_cap, 5000),  # Minimum spend cap
            'tier_label': tier,
            **discriminator_apps
        }
        
        tier_data.append(tier_user_data)
    
    tier_df = pd.DataFrame(tier_data)
    tier_df.to_csv('data/tier_app_indicators.csv', index=False)
    print(f"Created data/tier_app_indicators.csv with {len(tier_df)} users")
    
    # Merge datasets
    merged_df = pd.merge(neutral_df, tier_df, on='user_id')
    merged_df.to_csv('data/master_tier_dataset.csv', index=False)
    print(f"Created data/master_tier_dataset.csv with {len(merged_df)} users")
    
    return neutral_df, tier_df, merged_df

if __name__ == "__main__":
    # Create synthetic datasets
    neutral_df, tier_df, merged_df = create_synthetic_datasets()
    
    # Test the redesigned classifier
    classifier = RedesignedAppTierClassifier()
    
    # Example with your apps + behavioral data
    sample_apps = ['zomato', 'urban_company', 'blinkit', 'instagram', 'reddit', 'unacademy', 'paytm']
    behavioral_data = {
        'zomato_orders_per_month': 8,
        'zomato_avg_order_value': 350,
        'blinkit_orders_per_month': 6,
        'blinkit_avg_order_value': 280,
        'paytm_txn_count': 15,
        'paytm_txn_avg_value': 400
    }
    
    result = classifier.analyze_apps_with_behavior(sample_apps, behavioral_data)
    
    print("\n" + "="*80)
    print("REDESIGNED APP TIER ANALYSIS")
    print("="*80)
    print(f"Spending Tier: {result['spending_tier']}")
    print(f"Geographic Tier: {result['geographic_tier']}")
    print(f"Lifestyle Category: {result['lifestyle_category']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    
    print(f"\nDiscriminator Analysis:")
    for category, apps in result['discriminator_analysis']['matched_apps'].items():
        if apps:
            print(f"  {category}: {apps}")
    
    print(f"\nNeutral App Usage:")
    for category, usage in result['neutral_analysis']['usage_patterns'].items():
        print(f"  {category}: {usage['apps']} - {usage['frequency']} orders/month, ₹{usage['aov']:.0f} AOV")
    
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
