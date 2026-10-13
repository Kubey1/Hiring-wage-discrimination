"""
Detecting Wage Discrimination in Hiring Text
--------------------------------------------
Uses an LLM (OpenAI gpt-4o-mini) and keyword extraction to infer candidate
demographics and score bias in hiring/interview text, then tests whether
wages differ across groups (t-tests, OLS, text-wage correlations).

Data is not included (course dataset) - see data/README.md.
API key is read from a local .env file (OPENAI_API_KEY=...), never hard-coded.

Usage:
    python src/discrimination_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from scipy import stats
import time
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# OPENAI API CONFIGURATION (using .env file - same as previous assignments)
# =============================================================================

# Load API key from .env file
from dotenv import load_dotenv
load_dotenv()  # Reads the .env file (kept out of git)

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY is None:
    print("WARNING: OPENAI_API_KEY not found in .env file!")
    print("   Create a .env file with: OPENAI_API_KEY=<your key>")
    print("   Running without LLM analysis...")
    USE_LLM = False
else:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    USE_LLM = True
    print("SUCCESS: OpenAI API key loaded")

MODEL_NAME = "gpt-4o-mini"  # Cheap and effective

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

df = pd.read_csv(os.path.join(DATA_DIR, "wage_simulation_data.csv"))

print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)

# Identify columns
wage_col = None
text_cols = []

for col in df.columns:
    col_lower = col.lower()
    if 'wage' in col_lower or 'salary' in col_lower or 'pay' in col_lower:
        wage_col = col
    elif df[col].dtype == 'object':
        text_cols.append(col)

# If wage column not found, assume first numeric column
if wage_col is None:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            wage_col = col
            break

print(f"\nIdentified wage column: {wage_col}")
print(f"Identified text columns: {text_cols}")

# CRITICAL FIX: Convert wage to numeric, treating "not hired" as NaN
print(f"\nOriginal wage statistics (raw):")
print(df[wage_col].describe())

# Convert wage to numeric, coercing errors (like "not hired") to NaN
df['wage_numeric'] = pd.to_numeric(df[wage_col], errors='coerce')

# Separate hired vs not hired
df['was_hired'] = ~df['wage_numeric'].isna()
not_hired_count = (~df['was_hired']).sum()
hired_count = df['was_hired'].sum()

print(f"\nHiring status:")
print(f"  Hired: {hired_count} ({hired_count/len(df)*100:.1f}%)")
print(f"  Not hired: {not_hired_count} ({not_hired_count/len(df)*100:.1f}%)")

print(f"\nWage statistics (hired candidates only):")
print(df[df['was_hired']]['wage_numeric'].describe())

# Combine all text columns
df['all_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)

# =============================================================================
# LLM-BASED DEMOGRAPHIC EXTRACTION
# =============================================================================

def extract_demographics_llm(text, max_retries=3):
    """
    Use GPT-4o-mini to extract demographic information from text
    Returns: dict with gender, race, age indicators
    """
    if not USE_LLM or pd.isna(text) or len(str(text).strip()) < 10:
        return None
    
    prompt = f"""Analyze this hiring/interview text and extract any demographic information about the candidate.

Text: {str(text)[:1500]}

Respond ONLY in this exact JSON format (no other text):
{{"gender": "male/female/unknown", "race": "white/black/hispanic/asian/unknown", "age": "young/middle/senior/unknown", "bias_detected": "yes/no", "bias_type": "none/gender/race/age/multiple"}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting demographic information and potential bias in hiring texts. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            # Clean up response if needed
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            return json.loads(result)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    
    return None


def analyze_bias_llm(text, max_retries=3):
    """
    Use GPT-4o-mini to detect discrimination/bias in hiring text
    Returns: dict with bias score and explanation
    """
    if not USE_LLM or pd.isna(text) or len(str(text).strip()) < 10:
        return None
    
    prompt = f"""Analyze this hiring decision text for signs of discrimination or bias.

Text: {str(text)[:1500]}

Respond ONLY in this exact JSON format:
{{"bias_score": 0-10, "bias_type": "none/gender/race/age/multiple", "evidence": "brief explanation"}}

Where bias_score: 0=no bias, 10=extreme bias"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting workplace discrimination. Be objective and evidence-based."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            return json.loads(result)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    
    return None


# =============================================================================
# RUN LLM ANALYSIS ON SAMPLE
# =============================================================================
print("\n" + "="*60)
print("LLM-BASED DEMOGRAPHIC & BIAS ANALYSIS")
print("="*60)

if USE_LLM:
    # Sample size (adjust based on budget - each call ~$0.001)
    SAMPLE_SIZE = min(200, len(df))  # Analyze up to 200 observations
    
    print(f"Analyzing {SAMPLE_SIZE} observations with GPT-4o-mini...")
    print(f"Estimated cost: ~${SAMPLE_SIZE * 0.002:.2f}")
    
    # Sample the data
    sample_indices = np.random.choice(len(df), size=SAMPLE_SIZE, replace=False)
    
    # Extract demographics
    demographics_results = []
    bias_results = []
    
    for i, idx in enumerate(sample_indices):
        text = df.loc[idx, 'all_text']
        
        # Extract demographics
        demo = extract_demographics_llm(text)
        if demo:
            demo['idx'] = idx
            # Only include wage if candidate was hired
            wage_value = df.loc[idx, 'wage_numeric']
            if pd.notna(wage_value):
                demo['wage'] = wage_value
                demographics_results.append(demo)
        
        # Analyze bias
        bias = analyze_bias_llm(text)
        if bias:
            bias['idx'] = idx
            # Only include wage if candidate was hired
            wage_value = df.loc[idx, 'wage_numeric']
            if pd.notna(wage_value):
                bias['wage'] = wage_value
                bias_results.append(bias)
        
        # Progress update
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{SAMPLE_SIZE}...")
        
        # Rate limiting
        time.sleep(0.3)
    
    # Convert to DataFrames
    demo_df = pd.DataFrame(demographics_results)
    bias_df = pd.DataFrame(bias_results)
    
    print(f"\nSuccessfully extracted demographics for {len(demo_df)} observations")
    print(f"Successfully analyzed bias for {len(bias_df)} observations")
    
else:
    print("LLM analysis skipped (no API key)")
    demo_df = pd.DataFrame()
    bias_df = pd.DataFrame()

# =============================================================================
# KEYWORD-BASED ANALYSIS (BACKUP/SUPPLEMENT)
# =============================================================================
print("\n" + "="*60)
print("KEYWORD-BASED DEMOGRAPHIC EXTRACTION")
print("="*60)

# Define keyword dictionaries
gender_keywords = {
    'female': ['she', 'her', 'woman', 'female', 'mother', 'wife', 'girl', 'ms.', 'mrs.', 'miss', 'lady', 'daughter'],
    'male': ['he', 'him', 'man', 'male', 'father', 'husband', 'boy', 'mr.', 'gentleman', 'son']
}

race_keywords = {
    'black': ['black', 'african american', 'african-american'],
    'hispanic': ['hispanic', 'latino', 'latina', 'latinx', 'mexican', 'spanish'],
    'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian', 'vietnamese'],
    'white': ['white', 'caucasian', 'european']
}

age_keywords = {
    'young': ['young', 'recent graduate', 'entry level', 'junior', 'fresh', 'millennial', 'gen z', 'new grad'],
    'senior': ['senior', 'experienced', 'veteran', 'mature', 'older', 'retirement', 'decades of experience']
}

def extract_keyword_demographics(text):
    """Extract demographics using keyword matching"""
    if pd.isna(text):
        return {}
    
    text_lower = str(text).lower()
    result = {}
    
    # Gender
    female_count = sum(1 for kw in gender_keywords['female'] if kw in text_lower)
    male_count = sum(1 for kw in gender_keywords['male'] if kw in text_lower)
    result['is_female'] = int(female_count > male_count and female_count > 0)
    result['is_male'] = int(male_count > female_count and male_count > 0)
    
    # Race
    for race, keywords in race_keywords.items():
        result[f'is_{race}'] = int(any(kw in text_lower for kw in keywords))
    
    # Age
    for age, keywords in age_keywords.items():
        result[f'is_{age}'] = int(any(kw in text_lower for kw in keywords))
    
    return result

# Apply keyword extraction to all data
keyword_results = df['all_text'].apply(extract_keyword_demographics)
keyword_df = pd.DataFrame(keyword_results.tolist())

# Merge with main dataframe
for col in keyword_df.columns:
    df[col] = keyword_df[col]

print("Keyword extraction results:")
for col in keyword_df.columns:
    print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")

# =============================================================================
# FIGURE 1: WAGE DISTRIBUTIONS BY DEMOGRAPHIC GROUP
# =============================================================================
print("\n" + "="*60)
print("CREATING FIGURES")
print("="*60)

# Create df_hired subset for keyword-based analysis (used throughout)
df_hired = df[df['was_hired']].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# If we have LLM results, use those; otherwise use keyword results
if USE_LLM and len(demo_df) > 0:
    # Plot 1: Wage by Gender (LLM)
    ax1 = axes[0, 0]
    gender_groups = demo_df.groupby('gender')['wage'].apply(list).to_dict()
    if len(gender_groups) > 0:
        data = [v for v in gender_groups.values()]
        labels = list(gender_groups.keys())
        bp = ax1.boxplot(data, labels=labels, patch_artist=True)
        colors = ['pink' if 'female' in l else 'lightblue' if 'male' in l else 'gray' for l in labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for i, d in enumerate(data):
            ax1.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5)
    ax1.set_title('Wage by Gender (LLM Detected)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wage')
    
    # Plot 2: Wage by Race (LLM)
    ax2 = axes[0, 1]
    race_groups = demo_df.groupby('race')['wage'].apply(list).to_dict()
    if len(race_groups) > 0:
        data = [v for v in race_groups.values()]
        labels = list(race_groups.keys())
        ax2.boxplot(data, labels=labels)
        for i, d in enumerate(data):
            ax2.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5)
    ax2.set_title('Wage by Race (LLM Detected)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Wage')
    
else:
    # Use keyword-based results
    ax1 = axes[0, 0]
    gender_data = []
    gender_labels = []
    if df_hired['is_female'].sum() > 0:
        gender_data.append(df_hired[df_hired['is_female'] == 1]['wage_numeric'].dropna().values)
        gender_labels.append('Female')
    if df_hired['is_male'].sum() > 0:
        gender_data.append(df_hired[df_hired['is_male'] == 1]['wage_numeric'].dropna().values)
        gender_labels.append('Male')
    if len(gender_data) > 0:
        bp = ax1.boxplot(gender_data, labels=gender_labels, patch_artist=True)
        colors = ['pink', 'lightblue']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        for i, d in enumerate(gender_data):
            ax1.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5, label='Mean' if i==0 else '')
        ax1.legend()
    ax1.set_title('Wage by Gender (Keyword Detected)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wage')
    
    ax2 = axes[0, 1]
    race_data = []
    race_labels = []
    for race in ['white', 'black', 'hispanic', 'asian']:
        col = f'is_{race}'
        if df_hired[col].sum() > 0:
            race_data.append(df_hired[df_hired[col] == 1]['wage_numeric'].dropna().values)
            race_labels.append(race.capitalize())
    if len(race_data) > 0:
        ax2.boxplot(race_data, labels=race_labels)
        for i, d in enumerate(race_data):
            ax2.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5)
    ax2.set_title('Wage by Race (Keyword Detected)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Wage')

# Plot 3: Bias Score Distribution (if LLM available)
ax3 = axes[1, 0]
if USE_LLM and len(bias_df) > 0 and 'bias_score' in bias_df.columns:
    ax3.hist(bias_df['bias_score'], bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='coral')
    ax3.axvline(bias_df['bias_score'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {bias_df["bias_score"].mean():.2f}')
    ax3.set_xlabel('Bias Score (0-10)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of LLM-Detected Bias Scores', fontsize=12, fontweight='bold')
    ax3.legend()
else:
    # Age-based wage distribution instead
    age_data = []
    age_labels = []
    if df_hired['is_young'].sum() > 0:
        age_data.append(df_hired[df_hired['is_young'] == 1]['wage_numeric'].dropna().values)
        age_labels.append('Young')
    if df_hired['is_senior'].sum() > 0:
        age_data.append(df_hired[df_hired['is_senior'] == 1]['wage_numeric'].dropna().values)
        age_labels.append('Senior')
    if len(age_data) > 0:
        ax3.boxplot(age_data, labels=age_labels)
        for i, d in enumerate(age_data):
            ax3.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5)
    ax3.set_title('Wage by Age (Keyword Detected)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Wage')

# Plot 4: Overall wage distribution (hired candidates only)
ax4 = axes[1, 1]
wages_hired = df_hired['wage_numeric'].dropna()
ax4.hist(wages_hired, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax4.axvline(wages_hired.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${wages_hired.mean():.0f}')
ax4.axvline(wages_hired.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${wages_hired.median():.0f}')
ax4.set_xlabel('Wage')
ax4.set_ylabel('Frequency')
ax4.set_title('Wage Distribution (Hired Candidates Only)', fontsize=12, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_wage_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_wage_distributions.png")

# =============================================================================
# FIGURE 2: WAGE GAPS AND BIAS ANALYSIS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate wage gaps
ax1 = axes[0]
wage_gaps = {}

if USE_LLM and len(demo_df) > 0:
    # Use LLM results
    gender_wages = demo_df.groupby('gender')['wage'].mean()
    if 'female' in gender_wages.index and 'male' in gender_wages.index:
        wage_gaps['Gender\n(F vs M)'] = (gender_wages['female'] - gender_wages['male']) / gender_wages['male'] * 100
    
    race_wages = demo_df.groupby('race')['wage'].mean()
    if 'white' in race_wages.index:
        for race in ['black', 'hispanic', 'asian']:
            if race in race_wages.index:
                wage_gaps[f'Race\n({race.capitalize()} vs White)'] = (race_wages[race] - race_wages['white']) / race_wages['white'] * 100
else:
    # Use keyword results (hired candidates only)
    if df_hired['is_female'].sum() > 0 and df_hired['is_male'].sum() > 0:
        female_wage = df_hired[df_hired['is_female'] == 1]['wage_numeric'].mean()
        male_wage = df_hired[df_hired['is_male'] == 1]['wage_numeric'].mean()
        wage_gaps['Gender\n(F vs M)'] = (female_wage - male_wage) / male_wage * 100

    if df_hired['is_white'].sum() > 0:
        white_wage = df_hired[df_hired['is_white'] == 1]['wage_numeric'].mean()
        for race in ['black', 'hispanic', 'asian']:
            if df_hired[f'is_{race}'].sum() > 0:
                race_wage = df_hired[df_hired[f'is_{race}'] == 1]['wage_numeric'].mean()
                wage_gaps[f'Race\n({race.capitalize()} vs W)'] = (race_wage - white_wage) / white_wage * 100

if len(wage_gaps) > 0:
    colors = ['green' if v > 0 else 'red' for v in wage_gaps.values()]
    bars = ax1.bar(wage_gaps.keys(), wage_gaps.values(), color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel('Wage Gap (%)', fontsize=11)
    ax1.set_title('Wage Gaps by Demographic Group', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, wage_gaps.values()):
        ypos = bar.get_height() + (1 if val >= 0 else -3)
        ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'Insufficient demographic data\nfor wage gap calculation', ha='center', va='center', 
             transform=ax1.transAxes, fontsize=12)

# Plot 2: Bias type distribution (if LLM available)
ax2 = axes[1]
if USE_LLM and len(bias_df) > 0 and 'bias_type' in bias_df.columns:
    bias_counts = bias_df['bias_type'].value_counts()
    colors_map = {'none': 'green', 'gender': 'pink', 'race': 'orange', 'age': 'purple', 'multiple': 'red'}
    colors = [colors_map.get(x, 'gray') for x in bias_counts.index]
    ax2.bar(bias_counts.index, bias_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Bias Type')
    ax2.set_ylabel('Count')
    ax2.set_title('Types of Bias Detected (LLM Analysis)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
else:
    # Mean wages comparison (hired candidates only)
    mean_wages = {}
    for col, label in [('is_female', 'Female'), ('is_male', 'Male'), ('is_white', 'White'),
                       ('is_black', 'Black'), ('is_hispanic', 'Hispanic'), ('is_asian', 'Asian')]:
        if df_hired[col].sum() > 0:
            mean_wages[label] = df_hired[df_hired[col] == 1]['wage_numeric'].mean()

    if len(mean_wages) > 0:
        bars = ax2.bar(mean_wages.keys(), mean_wages.values(), color='steelblue', alpha=0.7, edgecolor='black')
        overall_mean = df_hired['wage_numeric'].mean()
        ax2.axhline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: ${overall_mean:.0f}')
        ax2.set_ylabel('Mean Wage')
        ax2.set_title('Mean Wages by Demographic Group (Hired)', fontsize=12, fontweight='bold')
        ax2.legend()
        plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_wage_gaps_bias_types.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_wage_gaps_bias_types.png")

# =============================================================================
# FIGURE 3: REGRESSION & TEXT ANALYSIS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression coefficients
ax1 = axes[0]
demo_features = ['is_female', 'is_male', 'is_black', 'is_hispanic', 'is_asian', 'is_white', 'is_young', 'is_senior']
valid_features = [f for f in demo_features if f in df.columns and df[f].sum() > 0 and df[f].sum() < len(df)]

if len(valid_features) >= 2:
    X = df_hired[valid_features].values
    y = df_hired['wage_numeric'].values

    reg = LinearRegression()
    reg.fit(X, y)
    
    coef_df = pd.DataFrame({'Feature': valid_features, 'Coefficient': reg.coef_})
    coef_df = coef_df.sort_values('Coefficient')
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    ax1.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_xlabel('Coefficient (Effect on Wage)')
    ax1.set_title('OLS Regression: Demographics → Wage', fontsize=12, fontweight='bold')
    
    print("\n" + "="*60)
    print("REGRESSION RESULTS")
    print("="*60)
    print(f"R-squared: {reg.score(X, y):.4f}")
    for feat, coef in zip(valid_features, reg.coef_):
        print(f"  {feat}: {coef:.4f}")
else:
    ax1.text(0.5, 0.5, 'Insufficient features for regression', ha='center', va='center', transform=ax1.transAxes)

# TF-IDF correlation with wage
ax2 = axes[1]
try:
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    text_features = tfidf.fit_transform(df_hired['all_text'].fillna(''))

    correlations = []
    feature_names = tfidf.get_feature_names_out()
    for i in range(text_features.shape[1]):
        corr = np.corrcoef(text_features[:, i].toarray().flatten(), df_hired['wage_numeric'])[0, 1]
        correlations.append(corr)
    
    corr_df = pd.DataFrame({'Feature': feature_names, 'Correlation': correlations}).dropna()
    top_pos = corr_df.nlargest(8, 'Correlation')
    top_neg = corr_df.nsmallest(8, 'Correlation')
    top_features = pd.concat([top_neg, top_pos])
    
    colors = ['green' if c > 0 else 'red' for c in top_features['Correlation']]
    ax2.barh(top_features['Feature'], top_features['Correlation'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('Correlation with Wage')
    ax2.set_title('Text Features Most Correlated with Wage', fontsize=12, fontweight='bold')
except Exception as e:
    ax2.text(0.5, 0.5, f'TF-IDF failed: {e}', ha='center', va='center', transform=ax2.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_regression_text_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3_regression_text_features.png")

# =============================================================================
# FIGURE 4: BIAS SCORE VS WAGE (if LLM available)
# =============================================================================
if USE_LLM and len(bias_df) > 0 and 'bias_score' in bias_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Bias score vs Wage
    ax1 = axes[0]
    ax1.scatter(bias_df['bias_score'], bias_df['wage'], alpha=0.5, edgecolors='black')
    
    # Add regression line
    z = np.polyfit(bias_df['bias_score'], bias_df['wage'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(bias_df['bias_score'].min(), bias_df['bias_score'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend line')
    
    corr = bias_df['bias_score'].corr(bias_df['wage'])
    ax1.set_xlabel('Bias Score (LLM Detected)', fontsize=11)
    ax1.set_ylabel('Wage', fontsize=11)
    ax1.set_title(f'Bias Score vs Wage (r = {corr:.3f})', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Box plot: Wage by bias level
    ax2 = axes[1]
    bias_df['bias_level'] = pd.cut(bias_df['bias_score'], bins=[-0.1, 2, 5, 10], labels=['Low (0-2)', 'Medium (3-5)', 'High (6-10)'])
    bias_groups = bias_df.groupby('bias_level')['wage'].apply(list).to_dict()
    if len(bias_groups) > 0:
        data = [v for v in bias_groups.values() if len(v) > 0]
        labels = [k for k, v in bias_groups.items() if len(v) > 0]
        ax2.boxplot(data, labels=labels)
        for i, d in enumerate(data):
            ax2.scatter(i+1, np.mean(d), color='red', marker='D', s=80, zorder=5)
    ax2.set_xlabel('Bias Level')
    ax2.set_ylabel('Wage')
    ax2.set_title('Wage Distribution by Bias Level', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_bias_score_vs_wage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_bias_score_vs_wage.png")

# =============================================================================
# STATISTICAL TESTS
# =============================================================================
print("\n" + "="*60)
print("STATISTICAL TESTS FOR DISCRIMINATION")
print("="*60)

# T-tests (only for hired candidates)
if df_hired['is_female'].sum() > 10 and df_hired['is_male'].sum() > 10:
    female_wages = df_hired[df_hired['is_female'] == 1]['wage_numeric'].dropna()
    male_wages = df_hired[df_hired['is_male'] == 1]['wage_numeric'].dropna()
    t_stat, p_val = stats.ttest_ind(female_wages, male_wages)
    print(f"\nGender Wage Gap Test:")
    print(f"  Female mean: ${female_wages.mean():.2f}, Male mean: ${male_wages.mean():.2f}")
    print(f"  Gap: ${female_wages.mean() - male_wages.mean():.2f} ({(female_wages.mean() - male_wages.mean())/male_wages.mean()*100:.1f}%)")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    print(f"  Significant at 5%: {'YES - Evidence of discrimination' if p_val < 0.05 else 'NO'}")

if df_hired['is_white'].sum() > 10 and df_hired['is_black'].sum() > 10:
    white_wages = df_hired[df_hired['is_white'] == 1]['wage_numeric'].dropna()
    black_wages = df_hired[df_hired['is_black'] == 1]['wage_numeric'].dropna()
    t_stat, p_val = stats.ttest_ind(white_wages, black_wages)
    print(f"\nRace Wage Gap Test (Black vs White):")
    print(f"  White mean: ${white_wages.mean():.2f}, Black mean: ${black_wages.mean():.2f}")
    print(f"  Gap: ${black_wages.mean() - white_wages.mean():.2f} ({(black_wages.mean() - white_wages.mean())/white_wages.mean()*100:.1f}%)")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    print(f"  Significant at 5%: {'YES - Evidence of discrimination' if p_val < 0.05 else 'NO'}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary_data = [['Overall (Hired)', hired_count, df_hired['wage_numeric'].mean(), df_hired['wage_numeric'].std()]]

for col, label in [('is_female', 'Female'), ('is_male', 'Male'),
                   ('is_white', 'White'), ('is_black', 'Black'),
                   ('is_hispanic', 'Hispanic'), ('is_asian', 'Asian')]:
    if col in df_hired.columns and df_hired[col].sum() > 0:
        subset = df_hired[df_hired[col] == 1]
        summary_data.append([label, len(subset), subset['wage_numeric'].mean(), subset['wage_numeric'].std()])

summary_table = pd.DataFrame(summary_data, columns=['Group', 'N', 'Mean Wage', 'Std Dev'])
print(summary_table.to_string(index=False))
summary_table.to_csv(os.path.join(RES_DIR, 'summary_table.csv'), index=False)
print("\nSaved: results/summary_table.csv")

# =============================================================================
# LLM BIAS SUMMARY (if available)
# =============================================================================
if USE_LLM and len(bias_df) > 0:
    print("\n" + "="*60)
    print("LLM BIAS DETECTION SUMMARY")
    print("="*60)
    print(f"Observations analyzed: {len(bias_df)}")
    print(f"Mean bias score: {bias_df['bias_score'].mean():.2f}/10")
    print(f"Observations with bias detected: {(bias_df['bias_score'] > 2).sum()} ({(bias_df['bias_score'] > 2).mean()*100:.1f}%)")
    print(f"\nBias type breakdown:")
    print(bias_df['bias_type'].value_counts().to_string())

# =============================================================================
# FINAL OUTPUT
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("""
Generated files:
- fig1_wage_distributions.png
- fig2_wage_gaps_bias_types.png
- fig3_regression_text_features.png
- fig4_bias_score_vs_wage.png (if LLM enabled)
- results/summary_table.csv

""")
