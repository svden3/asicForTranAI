# Algorithm Specifications
## Gazillioner FQ Platform - Core Algorithms

**Version:** 1.0
**Date:** December 10, 2025
**Purpose:** Detailed mathematical specifications for all core algorithms

---

## Table of Contents

1. [FQ Scoring Algorithm](#1-fq-scoring-algorithm)
2. [Tax-Loss Harvesting Optimizer](#2-tax-loss-harvesting-optimizer)
3. [Portfolio Rebalancing Engine](#3-portfolio-rebalancing-engine)
4. [Behavioral Pattern Detection](#4-behavioral-pattern-detection)
5. [Thesis Strength Analyzer](#5-thesis-strength-analyzer)
6. [Exercise Difficulty Adapter](#6-exercise-difficulty-adapter)

---

## 1. FQ Scoring Algorithm

### 1.1 Overall Score Calculation

```
FQ_total = Knowledge + Discipline + Wisdom
         = (0-300)   + (0-400)     + (0-300)
         = 0-1000 points
```

**Percentile Calculation:**
```python
def calculate_percentile(score):
    """
    Assumes normal distribution:
    - Mean: 500
    - Standard Deviation: 150
    """
    mean = 500.0
    std_dev = 150.0

    z_score = (score - mean) / std_dev
    percentile = 50 + 34.13 * z_score  # Empirical rule approximation

    return max(1, min(99, int(percentile)))
```

### 1.2 Knowledge Score (0-300)

**Formula:**
```
Knowledge = Σ(CategoryScore_i) for i in [1..6]
CategoryScore_i = min(100, CompletionRate * 50 + AccuracyRate * 30 + DifficultyBonus * 20)
```

**Categories** (each 0-100):
1. Asset Allocation
2. Tax Optimization
3. Options Strategies
4. Behavioral Finance
5. Crypto Fundamentals
6. Market Cycles

**Detailed Calculation:**

```python
def calculate_category_score(exercises, category, total_available=30):
    """
    Calculate knowledge score for a single category.

    Args:
        exercises: List of completed exercises in this category
        category: Category enum
        total_available: Total exercises available (default 30)

    Returns:
        Score 0-100
    """
    if not exercises:
        return 0

    # 1. Completion Rate (0-50 points)
    completed = len(exercises)
    completion_rate = completed / total_available
    completion_points = completion_rate * 50

    # 2. Accuracy Rate (0-30 points)
    # Only count first-try correct answers
    first_try_correct = sum(1 for ex in exercises if ex.attempts == 1 and ex.correct)
    accuracy_rate = first_try_correct / completed
    accuracy_points = accuracy_rate * 30

    # 3. Difficulty Bonus (0-20 points)
    difficulty_multipliers = {
        'beginner': 0.5,
        'intermediate': 0.75,
        'advanced': 1.0,
        'expert': 1.25
    }

    avg_difficulty = sum(difficulty_multipliers[ex.difficulty] for ex in exercises) / completed
    difficulty_points = avg_difficulty * 20

    total = completion_points + accuracy_points + difficulty_points
    return min(100, int(total))
```

### 1.3 Discipline Score (0-400)

**Formula:**
```
Discipline = Contribution + Rebalancing + TaxEfficiency + EmotionalControl + PlanAdherence
           = (0-100)      + (0-100)     + (0-100)        + (0-100)          + (0-100)
           = 0-500 (capped at 400)
```

#### 1.3.1 Contribution Score (0-100)

```python
def calculate_contribution_score(streak_days, target_monthly, actual_monthly):
    """
    Rewards consistent investing.

    Args:
        streak_days: Current contribution streak (days)
        target_monthly: User's target monthly contribution
        actual_monthly: Actual contribution last month

    Returns:
        Score 0-100
    """
    points = 0

    # Streak bonuses
    if streak_days >= 365:    # 1 year
        points += 40
    elif streak_days >= 180:  # 6 months
        points += 30
    elif streak_days >= 90:   # 3 months
        points += 20
    elif streak_days >= 30:   # 1 month
        points += 10

    # Amount vs target
    ratio = actual_monthly / target_monthly
    if ratio >= 1.0:          # Met/exceeded
        points += 30
    elif ratio >= 0.8:        # 80%+
        points += 20
    elif ratio >= 0.5:        # 50%+
        points += 10

    # Auto-invest bonus
    if has_auto_invest_enabled():
        points += 10

    return min(100, points)
```

#### 1.3.2 Emotional Control Score (0-100)

```python
def calculate_emotional_control(panic_sells, fomo_buys, disciplined_holds):
    """
    Penalize emotional decisions, reward discipline.

    Args:
        panic_sells: Count in last 90 days
        fomo_buys: Count in last 90 days
        disciplined_holds: Count (held through >20% drawdown)

    Returns:
        Score 0-100
    """
    points = 50  # Start neutral

    # Penalties
    points -= panic_sells * 20    # Severe penalty
    points -= fomo_buys * 15      # Moderate penalty

    # Rewards
    points += disciplined_holds * 10  # Hold through volatility

    # Bonus for zero emotional decisions in last 90 days
    if panic_sells == 0 and fomo_buys == 0:
        points += 20

    return max(0, min(100, points))
```

### 1.4 Wisdom Score (0-300)

**Formula:**
```
Wisdom = Σ(DecisionPoints_i) for all decisions in last 12 months
       = Capped at 300
```

**Decision Point Awards:**

| Decision Type | Points | Criteria |
|---------------|--------|----------|
| Held through drawdown (thesis intact) | +50 | >30% drawdown, thesis validated |
| Sold winner with thesis broken | +30 | Recognized thesis invalidation |
| Bought dip with conviction | +40 | Added to position at -20%+ |
| Harvested tax loss correctly | +15 | Avoided wash sale |
| Rebalanced during volatility | +25 | Sold high, bought low |
| Panic sold (thesis intact) | -25 | Emotion-driven |
| FOMO bought at peak | -20 | Chasing momentum |
| Triggered wash sale | -15 | Tax inefficiency |

**Learning Rate Calculation:**

```python
def calculate_learning_rate(decisions):
    """
    Measures if user is getting wiser over time.

    Returns:
        Points gained per month (can be negative)
    """
    now = datetime.now()
    six_months_ago = now - timedelta(days=180)
    twelve_months_ago = now - timedelta(days=365)

    # Recent decisions (last 6 months)
    recent_points = sum(d.points for d in decisions if d.date >= six_months_ago)

    # Older decisions (6-12 months ago)
    older_points = sum(d.points for d in decisions
                      if twelve_months_ago <= d.date < six_months_ago)

    # Learning rate = change per month
    learning_rate = (recent_points - older_points) / 6

    return int(learning_rate)
```

---

## 2. Tax-Loss Harvesting Optimizer

### 2.1 Opportunity Detection

```python
def find_tax_loss_opportunities(portfolio, tax_rate=0.21):
    """
    Identify positions where harvesting losses would save taxes.

    Args:
        portfolio: Current portfolio positions
        tax_rate: User's marginal tax rate (default 21% capital gains)

    Returns:
        List of harvest recommendations
    """
    opportunities = []

    for position in portfolio.positions:
        # Only harvest losses > $1,000 (worth the effort)
        if position.unrealized_gain < -1000:

            # Check wash sale risk
            has_recent_sale = any(
                (datetime.now() - sale.date).days < 30
                for sale in position.recent_sales
            )

            if not has_recent_sale:
                tax_savings = abs(position.unrealized_gain) * tax_rate

                opportunities.append({
                    'symbol': position.symbol,
                    'loss': position.unrealized_gain,
                    'tax_savings': tax_savings,
                    'shares': position.shares,
                    'cost_basis': position.cost_basis,
                    'current_price': position.current_price,
                    'recommendation': 'Sell now, rebuy in 31 days',
                    'fq_points': 15
                })

    # Sort by tax savings (highest first)
    return sorted(opportunities, key=lambda x: x['tax_savings'], reverse=True)
```

### 2.2 Wash Sale Detection

**IRS Rule:** Cannot rebuy "substantially identical" security within 30 days before or after sale at a loss.

```python
def check_wash_sale_risk(symbol, portfolio, proposed_sale_date):
    """
    Check if selling this position would trigger wash sale rule.

    Returns:
        (is_wash_sale, details)
    """
    position = portfolio.get_position(symbol)

    # Check purchases in last 30 days
    recent_buys = [tx for tx in position.transactions
                   if tx.type == 'BUY'
                   and (proposed_sale_date - tx.date).days < 30]

    if recent_buys:
        return (True, {
            'reason': 'Purchased within last 30 days',
            'purchase_date': recent_buys[0].date,
            'days_until_safe': 30 - (proposed_sale_date - recent_buys[0].date).days
        })

    return (False, None)
```

### 2.3 Optimal Harvesting Strategy

```python
def optimize_tax_harvesting(portfolio, available_cash, tax_rate):
    """
    Maximize tax savings while respecting constraints.

    Constraints:
    - No wash sales
    - Keep portfolio allocation within ±5% of target
    - Don't harvest if expecting bounce (check thesis)

    Returns:
        Optimal set of positions to harvest
    """
    opportunities = find_tax_loss_opportunities(portfolio, tax_rate)
    selected = []
    total_savings = 0

    for opp in opportunities:
        # Check if harvesting would break allocation
        new_allocation = simulate_allocation_after_harvest(
            portfolio, opp['symbol'], opp['shares']
        )

        drift = calculate_allocation_drift(new_allocation, portfolio.target_allocation)

        if drift < 5:  # Acceptable drift
            selected.append(opp)
            total_savings += opp['tax_savings']

    return {
        'recommended_harvests': selected,
        'total_tax_savings': total_savings,
        'fq_points': len(selected) * 15
    }
```

---

## 3. Portfolio Rebalancing Engine

### 3.1 Drift Detection

```python
def calculate_allocation_drift(current, target):
    """
    Measure how far portfolio has drifted from target.

    Args:
        current: Current allocation (dict: {asset_class: percentage})
        target: Target allocation (dict: {asset_class: percentage})

    Returns:
        Total drift percentage
    """
    total_drift = 0

    for asset_class in ['stocks', 'crypto', 'bonds', 'cash']:
        drift = abs(current[asset_class] - target[asset_class])
        total_drift += drift

    return total_drift
```

### 3.2 Rebalancing Recommendation

```python
def generate_rebalancing_trades(portfolio, target_allocation, threshold=5):
    """
    Generate trades to restore target allocation.

    Args:
        portfolio: Current portfolio
        target_allocation: Target percentages
        threshold: Only rebalance if drift > threshold (default 5%)

    Returns:
        List of trades to execute
    """
    current = portfolio.current_allocation
    drift = calculate_allocation_drift(current, target_allocation)

    if drift < threshold:
        return []  # No rebalancing needed

    trades = []
    total_value = portfolio.total_value

    for asset_class in ['stocks', 'crypto', 'bonds', 'cash']:
        current_pct = current[asset_class]
        target_pct = target_allocation[asset_class]
        difference = target_pct - current_pct

        if abs(difference) > 1:  # Meaningful difference
            dollar_amount = (difference / 100) * total_value

            if dollar_amount > 0:
                action = 'BUY'
            else:
                action = 'SELL'
                dollar_amount = abs(dollar_amount)

            trades.append({
                'asset_class': asset_class,
                'action': action,
                'amount': dollar_amount,
                'reason': f'Rebalance: {current_pct:.1f}% → {target_pct:.1f}%'
            })

    return trades
```

### 3.3 Tax-Aware Rebalancing

```python
def tax_aware_rebalancing(portfolio, target_allocation):
    """
    Rebalance while minimizing tax impact.

    Strategy:
    1. Use new contributions to rebalance (no taxes)
    2. Sell positions with losses first (harvest losses)
    3. Sell long-term gains (lower tax rate)
    4. Avoid short-term gains if possible

    Returns:
        Optimized trade list
    """
    trades = generate_rebalancing_trades(portfolio, target_allocation)
    optimized_trades = []

    for trade in trades:
        if trade['action'] == 'SELL':
            # Find best positions to sell (minimize taxes)
            positions = get_positions_in_class(portfolio, trade['asset_class'])

            # Sort by tax efficiency
            positions_sorted = sorted(positions, key=lambda p: tax_score(p))

            # Allocate sells across tax-efficient positions
            remaining = trade['amount']
            for position in positions_sorted:
                if remaining <= 0:
                    break

                sell_amount = min(remaining, position.current_value)
                optimized_trades.append({
                    'symbol': position.symbol,
                    'action': 'SELL',
                    'amount': sell_amount,
                    'tax_impact': calculate_tax(position, sell_amount)
                })
                remaining -= sell_amount

        else:  # BUY
            optimized_trades.append(trade)

    return optimized_trades

def tax_score(position):
    """
    Lower score = sell first (more tax efficient).

    Priority:
    1. Losses (negative tax impact)
    2. Long-term gains (15-20% rate)
    3. Short-term gains (up to 37% rate) - avoid!
    """
    if position.unrealized_gain < 0:
        return -1  # Best: losses
    elif position.is_long_term():
        return 1   # OK: long-term gains
    else:
        return 2   # Worst: short-term gains
```

---

## 4. Behavioral Pattern Detection

### 4.1 Panic Selling Detection

```python
def detect_panic_selling(trade, portfolio, theses):
    """
    Identify if proposed sale is emotionally driven.

    Panic indicators:
    - Selling at >15% loss
    - No clear reason provided
    - Thesis still intact
    - Market is also down (following crowd)
    - Stock was falling recently (catching falling knife fear)

    Returns:
        (is_panic, confidence_score, context)
    """
    if trade.action != 'SELL':
        return (False, 0, None)

    position = portfolio.get_position(trade.symbol)
    drawdown = position.unrealized_gain_pct

    panic_score = 0

    # 1. Significant loss
    if drawdown < -15:
        panic_score += 30

    # 2. No clear reason
    if not trade.reason or len(trade.reason) < 20:
        panic_score += 25

    # 3. Thesis intact
    thesis = find_thesis_for_symbol(theses, trade.symbol)
    if thesis and thesis.strength in ['strengthening', 'stable']:
        panic_score += 30

    # 4. Market correlation
    market_drawdown = get_market_drawdown()
    if market_drawdown < -10 and drawdown < market_drawdown:
        panic_score += 15  # Following market down

    # 5. Falling knife fear
    recent_price_trend = get_price_trend(trade.symbol, days=7)
    if recent_price_trend < -10:
        panic_score += 20  # Scared of continued fall

    is_panic = panic_score >= 50
    confidence = min(100, panic_score)

    if is_panic:
        context = {
            'original_thesis': thesis.title if thesis else "No thesis",
            'drawdown': drawdown,
            'market_drawdown': market_drawdown,
            'thesis_intact': thesis.strength != 'broken' if thesis else False,
            'panic_score': panic_score
        }
        return (True, confidence, context)

    return (False, confidence, None)
```

### 4.2 FOMO Detection

```python
def detect_fomo_chasing(trade, portfolio):
    """
    Identify if proposed purchase is FOMO-driven.

    FOMO indicators:
    - Stock up >20% recently
    - High social media mentions
    - No thesis or weak thesis
    - Oversized position (>10% of portfolio)
    - Recent news spike

    Returns:
        (is_fomo, confidence_score, context)
    """
    if trade.action != 'BUY':
        return (False, 0, None)

    fomo_score = 0

    # 1. Recent price surge
    price_7d = get_price_change(trade.symbol, days=7)
    price_30d = get_price_change(trade.symbol, days=30)

    if price_7d > 20:
        fomo_score += 35
    if price_30d > 50:
        fomo_score += 30

    # 2. Social media hype
    social_mentions = get_social_mentions(trade.symbol, days=7)
    avg_mentions = get_avg_social_mentions(trade.symbol, days=90)

    if social_mentions > avg_mentions * 3:
        fomo_score += 25  # 3x normal buzz

    # 3. Weak or no thesis
    if not trade.thesis or len(trade.thesis) < 50:
        fomo_score += 20

    # 4. Position size too large
    position_pct = (trade.amount / portfolio.total_value) * 100
    if position_pct > 10:
        fomo_score += 15

    # 5. News spike
    news_volume = get_news_volume(trade.symbol, days=7)
    if news_volume > 20:  # >20 articles in a week
        fomo_score += 15

    is_fomo = fomo_score >= 50
    confidence = min(100, fomo_score)

    if is_fomo:
        context = {
            'price_change_7d': price_7d,
            'price_change_30d': price_30d,
            'social_mentions': social_mentions,
            'has_thesis': len(trade.thesis) > 50 if trade.thesis else False,
            'position_size_pct': position_pct,
            'fomo_score': fomo_score
        }
        return (True, confidence, context)

    return (False, confidence, None)
```

---

## 5. Thesis Strength Analyzer

### 5.1 Thesis Scoring

```python
def score_thesis_quality(thesis):
    """
    Evaluate quality of investment thesis (0-100).

    Criteria:
    - Specific TAM projection
    - Multiple key drivers identified
    - Falsification criteria defined
    - Diversified across value chain
    - Appropriate position sizing

    Returns:
        Score 0-100
    """
    score = 0

    # 1. TAM projection specificity (0-20)
    if thesis.tam_10_year:
        if has_numeric_projection(thesis.tam_10_year):
            score += 20
        else:
            score += 10

    # 2. Key drivers (0-25)
    driver_count = len(thesis.key_drivers)
    if driver_count >= 3:
        score += 25
    elif driver_count >= 2:
        score += 15
    elif driver_count >= 1:
        score += 5

    # 3. Falsification criteria (0-25)
    if thesis.falsification_criteria:
        if len(thesis.falsification_criteria) >= 2:
            score += 25
        else:
            score += 15

    # 4. Winner diversification (0-15)
    if thesis.potential_winners:
        winner_count = len(thesis.potential_winners)
        if winner_count >= 3:
            score += 15
        elif winner_count >= 2:
            score += 10
        else:
            score += 5

    # 5. Position sizing (0-15)
    allocation_pct = thesis.allocation_pct
    if 3 <= allocation_pct <= 15:  # Sweet spot
        score += 15
    elif 1 <= allocation_pct < 3:  # Too small (not enough conviction)
        score += 5
    elif 15 < allocation_pct <= 25:  # Large (high conviction)
        score += 10
    else:  # >25% (too concentrated)
        score += 0

    return score
```

### 5.2 Thesis Strength Tracking

```python
def update_thesis_strength(thesis, news_events, price_changes):
    """
    Track if thesis is strengthening or weakening over time.

    Factors:
    - Milestone achievements
    - Supporting news
    - Price performance vs market
    - Key driver validation

    Returns:
        Updated ThesisStrength enum
    """
    strength_score = 0  # -100 to +100

    # 1. Milestones (each +10)
    recent_milestones = [m for m in thesis.milestones
                        if (datetime.now() - m.date).days < 90]

    for milestone in recent_milestones:
        if milestone.impact == 'supports':
            strength_score += 10
        elif milestone.impact == 'challenges':
            strength_score -= 10

    # 2. News sentiment
    for event in news_events:
        if event.sentiment == 'positive':
            strength_score += 5
        elif event.sentiment == 'negative':
            strength_score -= 5

    # 3. Price performance (outperformance suggests market agrees)
    for symbol in thesis.positions:
        stock_return = price_changes.get(symbol, 0)
        market_return = price_changes.get('SPY', 0)
        outperformance = stock_return - market_return

        strength_score += outperformance * 2  # Scale factor

    # 4. Classify strength
    if strength_score > 30:
        return 'strengthening'
    elif strength_score > -30:
        return 'stable'
    elif strength_score > -60:
        return 'weakening'
    else:
        return 'broken'
```

---

## 6. Exercise Difficulty Adapter

### 6.1 Adaptive Learning

```python
def recommend_next_exercise(user_fq_score, completed_exercises):
    """
    Recommend exercise at appropriate difficulty level.

    Strategy:
    - If user struggling (low accuracy): easier exercises
    - If user succeeding (high accuracy): harder exercises
    - Focus on weakest category

    Returns:
        Next recommended exercise
    """
    # Identify weakest category
    category_scores = calculate_category_breakdown(completed_exercises)
    weakest_category = min(category_scores, key=category_scores.get)

    # Determine appropriate difficulty
    recent_accuracy = calculate_recent_accuracy(completed_exercises, n=10)

    if recent_accuracy > 0.8:
        target_difficulty = 'advanced'
    elif recent_accuracy > 0.6:
        target_difficulty = 'intermediate'
    else:
        target_difficulty = 'beginner'

    # Find uncompleted exercise matching criteria
    available_exercises = get_exercises(
        category=weakest_category,
        difficulty=target_difficulty,
        exclude=completed_exercises
    )

    if not available_exercises:
        # Fall back to next difficulty level
        target_difficulty = next_difficulty_level(target_difficulty)
        available_exercises = get_exercises(
            category=weakest_category,
            difficulty=target_difficulty,
            exclude=completed_exercises
        )

    return available_exercises[0] if available_exercises else None
```

### 6.2 FQ Point Awards

```python
def calculate_exercise_fq_points(exercise, user_performance):
    """
    Award FQ points based on difficulty and performance.

    Base points by difficulty:
    - Beginner: 5 points
    - Intermediate: 8 points
    - Advanced: 12 points
    - Expert: 20 points

    Multipliers:
    - First try correct: 1.5x
    - Second try correct: 1.0x
    - Third+ try: 0.5x

    Returns:
        FQ points awarded
    """
    base_points = {
        'beginner': 5,
        'intermediate': 8,
        'advanced': 12,
        'expert': 20
    }

    points = base_points[exercise.difficulty]

    # Apply performance multiplier
    if user_performance.attempts == 1 and user_performance.correct:
        points *= 1.5
    elif user_performance.attempts == 2 and user_performance.correct:
        points *= 1.0
    elif user_performance.correct:
        points *= 0.5
    else:
        points = 0  # Incorrect answer

    return int(points)
```

---

## Appendix: Performance Benchmarks

### Expected Computation Times (iPad Pro M4)

| Algorithm | Input Size | Expected Time | Max Time |
|-----------|------------|---------------|----------|
| FQ Score Calculation | 1 portfolio + 100 exercises | <100ms | <500ms |
| Tax Harvest Detection | 50 positions | <50ms | <200ms |
| Rebalancing Calculation | 100 positions | <100ms | <300ms |
| Behavioral Analysis | 1 trade + portfolio | <200ms | <1s |
| Thesis Strength Update | 1 thesis + 30 days news | <500ms | <2s |
| Exercise Recommendation | 200 completed | <50ms | <100ms |

### Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| FQ Engine | <10 MB |
| Behavioral Analyzer | <20 MB |
| Portfolio Data (100 positions) | <5 MB |
| Exercise Database (500 exercises) | <50 MB |
| AI Model (LLaMA-13B) | 5.7 GB |
| **Total** | **~6 GB** |

---

**Document Version:** 1.0
**Last Updated:** December 10, 2025
**Maintained By:** Engineering Team
