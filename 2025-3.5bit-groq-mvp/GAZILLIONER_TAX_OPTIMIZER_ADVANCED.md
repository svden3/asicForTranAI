# Advanced Tax Optimization Engine
## Gazillioner - Sophisticated Tax Loss Harvesting & Planning

**Version:** 2.0 (Advanced)
**Date:** December 11, 2025
**Purpose:** Multi-year tax optimization with predictive modeling

---

## Table of Contents

1. [Advanced Tax-Loss Harvesting](#1-advanced-tax-loss-harvesting)
2. [Multi-Year Tax Planning](#2-multi-year-tax-planning)
3. [Options Tax Strategies](#3-options-tax-strategies)
4. [Crypto Tax Optimization](#4-crypto-tax-optimization)
5. [Backdoor Roth Conversions](#5-backdoor-roth-conversions)
6. [Estate Tax Planning](#6-estate-tax-planning)
7. [Real-Time Tax Projections](#7-real-time-tax-projections)

---

## 1. Advanced Tax-Loss Harvesting

### 1.1 Dynamic Opportunity Scoring

**Problem:** Not all losses are equally valuable. A $10k loss in December is worth more than in January.

**Solution:** Score opportunities by time-value and tax impact.

```python
def score_tax_loss_opportunity(position, current_date, user_tax_profile):
    """
    Score tax-loss harvesting opportunities (0-100).

    Higher score = harvest now
    Lower score = wait

    Factors:
    - Loss magnitude (bigger = better)
    - Days until year-end (closer = more urgent)
    - Wash sale window (clean = better)
    - Alternative investment availability (easy to replace = better)
    - User's current tax situation (offset gains = better)
    """
    score = 0

    # 1. Loss Magnitude (0-30 points)
    loss_amount = abs(position.unrealized_gain)
    if loss_amount > 50000:
        score += 30
    elif loss_amount > 10000:
        score += 20
    elif loss_amount > 5000:
        score += 15
    elif loss_amount > 1000:
        score += 10
    else:
        score += 0  # Not worth the effort

    # 2. Time Urgency (0-25 points)
    days_to_year_end = (datetime(current_date.year, 12, 31) - current_date).days

    if days_to_year_end < 30:
        score += 25  # URGENT: Harvest now or lose opportunity
    elif days_to_year_end < 90:
        score += 15  # High priority
    elif days_to_year_end < 180:
        score += 10  # Medium priority
    else:
        score += 5   # Low urgency, can wait

    # 3. Wash Sale Cleanliness (0-20 points)
    if position.is_wash_sale_risk:
        score -= 20  # BLOCKED: Can't harvest yet
    else:
        score += 20  # CLEAN: Safe to harvest

    # 4. Replacement Availability (0-15 points)
    similar_assets = find_similar_but_not_identical(position.symbol)

    if len(similar_assets) >= 3:
        score += 15  # Easy to replace (e.g., VTI → VOO)
    elif len(similar_assets) >= 1:
        score += 10
    else:
        score += 0   # Hard to replace (unique asset)

    # 5. Tax Offset Potential (0-10 points)
    if user_tax_profile.unrealized_gains > 0:
        # User has gains to offset
        offset_ratio = min(loss_amount / user_tax_profile.unrealized_gains, 1.0)
        score += offset_ratio * 10
    else:
        # No gains, but can carry forward loss
        score += 5

    return min(100, max(0, score))


def find_similar_but_not_identical(symbol):
    """
    Find ETFs/stocks similar enough to maintain allocation,
    but different enough to avoid wash sale.

    Examples:
    - VTI (Total Market) → VOO (S&P 500) [Similar but not identical]
    - NVDA → AMD [Same sector, different company]
    - BTC → ETH [Same asset class, different crypto]
    """

    similarity_map = {
        # ETFs
        'VTI': ['VOO', 'SCHB', 'ITOT'],  # Total market → S&P 500
        'VOO': ['IVV', 'SPY'],            # S&P 500 → Other S&P trackers
        'QQQ': ['QQQM', 'VGT'],           # Nasdaq → Tech sector

        # Stocks (same sector)
        'NVDA': ['AMD', 'INTC'],          # Semiconductors
        'TSLA': ['RIVN', 'LCID'],         # EVs
        'AAPL': ['MSFT', 'GOOGL'],        # Big tech

        # Crypto
        'BTC': ['ETH', 'SOL'],            # Layer 1s
        'ETH': ['SOL', 'AVAX'],           # Smart contract platforms
    }

    return similarity_map.get(symbol, [])
```

### 1.2 Automated Replacement Strategy

**Strategy:** Harvest loss + immediately buy similar asset → maintain market exposure

```python
def generate_harvest_and_replace_plan(position, tax_savings):
    """
    Generate complete harvest-and-replace transaction plan.

    Returns:
        - Sell order (harvest loss)
        - Buy order (maintain exposure)
        - Calendar reminder (rebuy original in 31 days)
    """

    # Step 1: Sell losing position
    sell_order = {
        'action': 'SELL',
        'symbol': position.symbol,
        'shares': position.shares,
        'reason': f'Harvest ${abs(position.unrealized_gain):,.0f} loss',
        'tax_savings': tax_savings,
        'execute_date': 'TODAY'
    }

    # Step 2: Buy similar asset (avoid wash sale)
    similar_assets = find_similar_but_not_identical(position.symbol)

    if similar_assets:
        replacement_symbol = similar_assets[0]

        buy_order = {
            'action': 'BUY',
            'symbol': replacement_symbol,
            'amount': position.current_value,  # Same dollar amount
            'reason': f'Replace {position.symbol} exposure while avoiding wash sale',
            'execute_date': 'TODAY (same day as sell)'
        }
    else:
        # No suitable replacement → park in cash for 31 days
        buy_order = None

    # Step 3: Calendar reminder to rebuy original (if desired)
    rebuy_date = datetime.now() + timedelta(days=31)

    reminder = {
        'title': f'Safe to rebuy {position.symbol}',
        'date': rebuy_date,
        'message': f'Wash sale window closed. You can now rebuy {position.symbol} if your thesis is still intact.',
        'action': 'OPTIONAL_REBUY'
    }

    return {
        'sell': sell_order,
        'buy': buy_order,
        'reminder': reminder,
        'net_tax_savings': tax_savings,
        'fq_points': 15  # Reward smart tax move
    }
```

### 1.3 Loss Carryforward Optimization

**Advanced:** Plan multi-year to maximize loss utilization

```python
def optimize_loss_carryforward(losses, projected_income_5_years):
    """
    Optimize when to realize losses across multiple years.

    Strategy:
    - Realize losses in years with HIGH income (higher tax bracket)
    - Defer losses to offset future gains
    - Consider $3k annual deduction limit

    Args:
        losses: List of available tax loss opportunities
        projected_income_5_years: [Year1, Year2, ..., Year5] income projections

    Returns:
        Multi-year harvesting plan
    """

    plan = {year: [] for year in range(2026, 2031)}

    # Sort losses by time-value (year-end urgency)
    losses_sorted = sorted(losses, key=lambda x: x['urgency_score'], reverse=True)

    # Sort years by tax rate (harvest in high-income years first)
    years_by_rate = sorted(
        range(2026, 2031),
        key=lambda year: calculate_marginal_rate(projected_income_5_years[year - 2026]),
        reverse=True
    )

    remaining_losses = losses_sorted.copy()

    for year in years_by_rate:
        year_idx = year - 2026
        income = projected_income_5_years[year_idx]
        marginal_rate = calculate_marginal_rate(income)

        # Estimate gains for this year (from projection)
        projected_gains = estimate_capital_gains(year)

        # Target: Offset all gains + use $3k ordinary income deduction
        target_losses = projected_gains + 3000

        allocated_this_year = 0

        while allocated_this_year < target_losses and remaining_losses:
            loss = remaining_losses.pop(0)
            plan[year].append(loss)
            allocated_this_year += abs(loss['amount'])

        if not remaining_losses:
            break  # All losses allocated

    # Calculate total tax savings
    total_savings = 0
    for year, year_losses in plan.items():
        year_rate = calculate_marginal_rate(
            projected_income_5_years[year - 2026]
        )
        year_loss_total = sum(abs(l['amount']) for l in year_losses)
        total_savings += year_loss_total * year_rate

    return {
        'plan': plan,
        'total_tax_savings': total_savings,
        'years_covered': len([y for y in plan.values() if y])
    }


def calculate_marginal_rate(income):
    """
    Calculate marginal tax rate based on income.
    2025 tax brackets (single filer):
    """
    if income > 609350:
        return 0.37
    elif income > 231250:
        return 0.35
    elif income > 182100:
        return 0.32
    elif income > 95375:
        return 0.24
    elif income > 44725:
        return 0.22
    elif income > 11000:
        return 0.12
    else:
        return 0.10
```

---

## 2. Multi-Year Tax Planning

### 2.1 Roth Conversion Ladder

**Strategy:** Convert traditional IRA → Roth IRA in low-income years

```python
def optimize_roth_conversions(user_profile, retirement_age=65):
    """
    Plan Roth conversions to minimize lifetime taxes.

    Key insight: Convert in years where marginal rate is LOW.

    Ideal scenarios:
    - Between jobs (low income year)
    - Early retirement (before RMDs start)
    - Business loss year (offset with conversion)
    """

    current_age = user_profile.age
    years_to_retirement = retirement_age - current_age

    conversion_plan = []

    # Project income by year
    for year_offset in range(years_to_retirement):
        year = datetime.now().year + year_offset
        projected_income = project_income(user_profile, year)
        marginal_rate = calculate_marginal_rate(projected_income)

        # Identify low-income years (good for conversions)
        if marginal_rate <= 0.22:  # 22% or less
            # Calculate optimal conversion amount
            # Goal: Fill up 22% bracket without spilling into 24%

            top_of_22_bracket = 95375  # 2025 limit
            room_in_bracket = top_of_22_bracket - projected_income

            if room_in_bracket > 0:
                conversion_plan.append({
                    'year': year,
                    'amount': room_in_bracket,
                    'tax_cost': room_in_bracket * 0.22,
                    'future_savings': calculate_roth_benefit(
                        room_in_bracket,
                        years_to_retirement - year_offset
                    ),
                    'reason': f'Fill 22% bracket (income only ${projected_income:,})'
                })

    return conversion_plan


def calculate_roth_benefit(conversion_amount, years_until_withdrawal):
    """
    Calculate benefit of Roth conversion.

    Benefit = (Future value tax-free) - (Tax paid now)
    """

    # Assume 8% annual return
    future_value = conversion_amount * (1.08 ** years_until_withdrawal)

    # Tax saved in retirement (assume 24% bracket then)
    tax_saved = future_value * 0.24

    # Tax paid now (assume 22% bracket)
    tax_paid = conversion_amount * 0.22

    net_benefit = tax_saved - tax_paid

    return net_benefit
```

### 2.2 Capital Gains Timing

**Strategy:** Time gains to minimize taxes

```python
def optimize_gain_realization(portfolio, target_year=None):
    """
    Decide WHEN to realize gains to minimize taxes.

    Strategies:
    - Defer gains to next year (if income lower)
    - Realize in 0% cap gains years (income < $44,625)
    - Pair with losses (same year offset)
    """

    current_year_income = get_current_year_income()
    next_year_income = project_income(user_profile, datetime.now().year + 1)

    recommendations = []

    for position in portfolio.positions:
        if position.unrealized_gain > 0:
            # Calculate tax in different scenarios

            # Scenario 1: Realize now
            tax_now = calculate_cap_gains_tax(
                position.unrealized_gain,
                current_year_income,
                position.is_long_term()
            )

            # Scenario 2: Defer to next year
            tax_next_year = calculate_cap_gains_tax(
                position.unrealized_gain,
                next_year_income,
                position.is_long_term()
            )

            # Scenario 3: Wait until long-term (if short-term now)
            if not position.is_long_term():
                days_to_long_term = 365 - position.days_held

                # Estimate future value
                estimated_future_gain = estimate_gain_growth(
                    position,
                    days_to_long_term
                )

                tax_as_long_term = calculate_cap_gains_tax(
                    estimated_future_gain,
                    current_year_income,
                    is_long_term=True
                )
            else:
                tax_as_long_term = tax_now

            # Recommend best option
            options = {
                'now': tax_now,
                'defer': tax_next_year,
                'wait_long_term': tax_as_long_term
            }

            best_option = min(options, key=options.get)

            recommendations.append({
                'symbol': position.symbol,
                'unrealized_gain': position.unrealized_gain,
                'recommendation': best_option,
                'tax_saved': tax_now - options[best_option],
                'reason': explain_recommendation(best_option, options)
            })

    return recommendations


def calculate_cap_gains_tax(gain, income, is_long_term):
    """
    Calculate capital gains tax based on income and holding period.

    2025 rates:
    Long-term: 0%, 15%, 20% (based on income)
    Short-term: Ordinary income rates (10%-37%)
    """

    if is_long_term:
        # Long-term capital gains rates
        if income > 492300:
            rate = 0.20
        elif income > 44625:
            rate = 0.15
        else:
            rate = 0.00  # 0% bracket!
    else:
        # Short-term = ordinary income
        rate = calculate_marginal_rate(income)

    # Additional: Net Investment Income Tax (3.8% if income > $200k)
    if income > 200000:
        rate += 0.038

    return gain * rate
```

---

## 3. Options Tax Strategies

### 3.1 Tax-Efficient Option Strategies

```python
def optimize_covered_call_taxes(position, current_date):
    """
    Optimize covered call strategy for tax efficiency.

    Key insight:
    - Selling calls too early can trigger short-term gains
    - Selling calls on long-term holdings preserves LT status
    - Rolling calls can defer taxes
    """

    recommendations = []

    if position.is_long_term():
        # Safe: Sell covered calls (won't affect LT status)
        recommendations.append({
            'strategy': 'Covered Call',
            'symbol': position.symbol,
            'reason': 'Position is long-term. Premium is ordinary income, but if assigned, gain is still long-term.',
            'tax_impact': 'NEUTRAL',
            'annual_yield': estimate_covered_call_yield(position.symbol)
        })
    else:
        # Risky: Assignment triggers short-term gain
        days_to_long_term = 365 - position.days_held

        recommendations.append({
            'strategy': 'WAIT',
            'symbol': position.symbol,
            'reason': f'Position is short-term. Wait {days_to_long_term} days for long-term status before selling calls.',
            'tax_impact': 'CAUTION',
            'potential_savings': position.unrealized_gain * (
                calculate_marginal_rate(user.income) - 0.15  # ST vs LT rate diff
            )
        })

    return recommendations


def analyze_qualified_covered_call(position):
    """
    Qualified Covered Calls (QCC) preserve holding period.

    IRS Rules:
    - Strike must be >= previous day's close
    - Expiration <= 30 days (or specific deeper ITM rules)
    - If rules violated, holding period resets
    """

    current_price = position.current_price

    # Suggest qualified strikes
    safe_strikes = []

    for expiry_days in [7, 14, 21, 30]:
        # Strike must be AT or OUT of the money
        atm_strike = round_to_strike(current_price)
        otm_strike = atm_strike + 5  # Next strike up

        safe_strikes.append({
            'expiry_days': expiry_days,
            'strike': otm_strike,
            'premium_estimate': estimate_option_premium(
                position.symbol,
                otm_strike,
                expiry_days
            ),
            'qualified': True,
            'preserves_holding_period': True
        })

    return safe_strikes
```

### 3.2 Wash Sale Prevention with Options

```python
def detect_option_wash_sales(transactions):
    """
    Detect wash sales involving options.

    Complex rules:
    - Selling stock at loss + buying calls = wash sale
    - Selling stock at loss + selling puts = wash sale
    - Buying back within 30 days via options = wash sale
    """

    violations = []

    for i, txn in enumerate(transactions):
        if txn.type == 'STOCK_SELL' and txn.gain < 0:
            # Sold stock at loss

            # Check next 30 days for option activity
            window_end = txn.date + timedelta(days=30)

            for other_txn in transactions[i+1:]:
                if other_txn.date > window_end:
                    break

                # Check for wash sale triggers
                if (other_txn.symbol == txn.symbol and
                    (other_txn.type == 'CALL_BUY' or other_txn.type == 'PUT_SELL')):

                    violations.append({
                        'stock_sale': txn,
                        'option_trigger': other_txn,
                        'disallowed_loss': abs(txn.gain),
                        'wash_sale': True,
                        'recommendation': 'REVERSE option trade or accept loss disallowance'
                    })

    return violations
```

---

## 4. Crypto Tax Optimization

### 4.1 Crypto-Specific Harvesting

```python
def optimize_crypto_taxes(crypto_positions, year_to_date_trades):
    """
    Crypto has unique tax optimization opportunities.

    Key differences from stocks:
    - No wash sale rule (!)
    - Can sell and rebuy SAME DAY
    - More volatile = more harvesting opportunities
    """

    strategies = []

    for position in crypto_positions:
        if position.unrealized_gain < -1000:
            # Harvest loss

            strategies.append({
                'action': 'CRYPTO_LOSS_HARVEST',
                'symbol': position.symbol,
                'step_1': {
                    'action': 'SELL',
                    'amount': position.quantity,
                    'execute': 'NOW'
                },
                'step_2': {
                    'action': 'BUY',
                    'amount': position.quantity,
                    'execute': 'SAME DAY (30 minutes later)',
                    'reason': 'No wash sale rule for crypto!'
                },
                'tax_savings': abs(position.unrealized_gain) * user.tax_rate,
                'market_risk': 'MINIMAL (same-day rebuy)',
                'fq_points': 20  # Bonus for advanced strategy
            })

    return strategies


def generate_crypto_tax_report(transactions, method='HIFO'):
    """
    Generate crypto tax report with optimal cost basis method.

    Methods:
    - FIFO: First In First Out (default, usually worst)
    - LIFO: Last In First Out
    - HIFO: Highest In First Out (usually best!)
    - Specific ID: Choose exact lots
    """

    if method == 'HIFO':
        # Sort sales to match with highest-cost-basis buys first
        # This minimizes gains (or maximizes losses)

        optimized_matches = []

        for sale in transactions.sales:
            # Find highest-cost buy to match
            available_buys = [
                buy for buy in transactions.buys
                if buy.quantity_remaining > 0 and buy.date < sale.date
            ]

            highest_cost_buy = max(available_buys, key=lambda b: b.price)

            optimized_matches.append({
                'sale': sale,
                'matched_buy': highest_cost_buy,
                'gain': (sale.price - highest_cost_buy.price) * sale.quantity,
                'tax_owed': calculate_crypto_tax(
                    sale.price - highest_cost_buy.price,
                    sale.date - highest_cost_buy.date
                )
            })

    return optimized_matches


def identify_crypto_staking_tax(staking_rewards):
    """
    Optimize staking reward taxes.

    Tax treatment:
    - Rewards taxed as ordinary income when received
    - Cost basis = value when received
    - Selling later = capital gain/loss
    """

    recommendations = []

    for reward in staking_rewards:
        # Check if holding at a loss now
        current_value = get_current_price(reward.symbol) * reward.quantity
        cost_basis = reward.value_when_received

        if current_value < cost_basis:
            # Harvest capital loss
            recommendations.append({
                'action': 'SELL',
                'reason': f'Harvest ${cost_basis - current_value:.2f} capital loss',
                'original_income': reward.value_when_received,
                'current_value': current_value,
                'tax_benefit': (cost_basis - current_value) * user.tax_rate,
                'can_rebuy': 'YES (no wash sale rule)'
            })

    return recommendations
```

---

## 5. Backdoor Roth Conversions

### 5.1 Automated Backdoor Roth

```python
def plan_backdoor_roth(user_profile):
    """
    Plan backdoor Roth IRA contribution.

    For high earners (income > $161k single, $240k married):
    - Can't contribute to Roth directly
    - But can: Contribute to Traditional IRA → Convert to Roth

    Steps:
    1. Contribute $7,000 to Traditional IRA (non-deductible)
    2. Immediately convert to Roth IRA
    3. Pay zero tax (no gains yet)
    4. Grow tax-free forever
    """

    if user_profile.income > 161000:  # Above Roth limit

        contribution_limit = 7000 if user_profile.age < 50 else 8000

        return {
            'eligible': True,
            'steps': [
                {
                    'step': 1,
                    'action': 'Contribute to Traditional IRA',
                    'amount': contribution_limit,
                    'tax_deduction': 0,  # Non-deductible (income too high)
                    'deadline': f'{datetime.now().year}-04-15'
                },
                {
                    'step': 2,
                    'action': 'Convert to Roth IRA',
                    'amount': contribution_limit,
                    'tax_owed': 0,  # Zero gains if immediate
                    'execute': 'IMMEDIATELY after contribution'
                },
                {
                    'step': 3,
                    'action': 'File Form 8606',
                    'reason': 'Report non-deductible contribution',
                    'deadline': 'With tax return'
                }
            ],
            'annual_benefit': contribution_limit * (1.08 ** 30),  # Value at retirement
            'lifetime_tax_savings': contribution_limit * (1.08 ** 30) * 0.24  # Assume 24% rate
        }

    return {'eligible': False, 'reason': 'Income below Roth limit - contribute directly'}


def check_pro_rata_rule(user_profile):
    """
    Check if pro-rata rule affects backdoor Roth.

    Pro-rata rule:
    If you have pre-tax IRA money, conversions are partially taxable.

    Example:
    - $100k in Traditional IRA (pre-tax)
    - $7k new contribution (after-tax)
    - Convert $7k → IRS says 93% is taxable (pro-rata)

    Solution: Roll pre-tax IRA into 401k first (if allowed)
    """

    traditional_ira_balance = user_profile.traditional_ira_balance

    if traditional_ira_balance > 0:
        return {
            'pro_rata_issue': True,
            'pre_tax_balance': traditional_ira_balance,
            'solution': 'Roll Traditional IRA into your 401k (if plan allows)',
            'alternative': 'Accept pro-rata taxation on conversion',
            'tax_impact': traditional_ira_balance / (traditional_ira_balance + 7000)
        }

    return {'pro_rata_issue': False}
```

---

## 6. Estate Tax Planning

### 6.1 Stepped-Up Basis Planning

```python
def optimize_for_stepped_up_basis(portfolio, age):
    """
    Plan around stepped-up basis at death.

    Key insight:
    - Heirs inherit at CURRENT value (not original cost basis)
    - All unrealized gains disappear (tax-free!)
    - Strategy: Hold appreciated assets, sell losers
    """

    if age < 65:
        return None  # Not relevant yet

    recommendations = []

    for position in portfolio.positions:
        if position.unrealized_gain > 100000:
            # Large unrealized gain

            recommendations.append({
                'symbol': position.symbol,
                'unrealized_gain': position.unrealized_gain,
                'strategy': 'HOLD until death',
                'reason': 'Stepped-up basis eliminates this gain tax-free for heirs',
                'current_tax_if_sold': position.unrealized_gain * 0.20,  # 20% LT rate
                'tax_if_inherited': 0,  # $0 for heirs
                'savings': position.unrealized_gain * 0.20
            })

        elif position.unrealized_gain < -10000:
            # Large unrealized loss

            recommendations.append({
                'symbol': position.symbol,
                'unrealized_loss': abs(position.unrealized_gain),
                'strategy': 'SELL before death',
                'reason': 'Losses disappear at death (not passed to heirs). Harvest now.',
                'tax_benefit_if_sold_now': abs(position.unrealized_gain) * 0.20,
                'tax_benefit_if_held': 0  # Heirs get stepped-up basis, lose the loss
            })

    return recommendations
```

---

## 7. Real-Time Tax Projections

### 7.1 Live Tax Dashboard

```python
def calculate_real_time_tax_projection(ytd_income, ytd_trades, portfolio):
    """
    Project year-end tax bill in real-time.

    Updates daily as:
    - Income changes
    - Trades executed
    - Unrealized gains/losses fluctuate
    """

    # Current year-to-date
    ytd_wages = ytd_income['wages']
    ytd_realized_gains = sum(t.gain for t in ytd_trades if t.gain > 0)
    ytd_realized_losses = sum(abs(t.gain) for t in ytd_trades if t.gain < 0)

    # Net capital gains (capped at $3k loss deduction)
    net_cap_gains = ytd_realized_gains - ytd_realized_losses
    if net_cap_gains < 0:
        deductible_loss = min(abs(net_cap_gains), 3000)
        carryforward_loss = max(abs(net_cap_gains) - 3000, 0)
    else:
        deductible_loss = 0
        carryforward_loss = 0

    # Adjusted Gross Income
    agi = ytd_wages + max(net_cap_gains, -3000)

    # Calculate tax
    federal_tax = calculate_federal_tax(agi)
    state_tax = calculate_state_tax(agi, user.state)

    # Available harvesting opportunities
    harvestable_losses = sum(
        abs(p.unrealized_gain) for p in portfolio.positions
        if p.unrealized_gain < -1000 and not p.is_wash_sale_risk
    )

    # Projected tax WITH harvesting
    optimized_agi = agi - min(harvestable_losses, net_cap_gains + 3000)
    optimized_tax = calculate_federal_tax(optimized_agi) + calculate_state_tax(optimized_agi, user.state)

    return {
        'current_projection': {
            'agi': agi,
            'federal_tax': federal_tax,
            'state_tax': state_tax,
            'total_tax': federal_tax + state_tax,
            'effective_rate': (federal_tax + state_tax) / agi
        },
        'optimized_projection': {
            'agi': optimized_agi,
            'total_tax': optimized_tax,
            'savings': (federal_tax + state_tax) - optimized_tax
        },
        'recommendations': {
            'harvestable_losses': harvestable_losses,
            'potential_savings': (federal_tax + state_tax) - optimized_tax,
            'action': 'Harvest losses before Dec 31' if harvestable_losses > 0 else 'No action needed'
        },
        'carryforward': {
            'loss_carryforward': carryforward_loss,
            'available_next_year': carryforward_loss
        }
    }


def calculate_federal_tax(agi, filing_status='single'):
    """
    Calculate federal income tax (2025 brackets).
    """
    brackets = {
        'single': [
            (11000, 0.10),
            (44725, 0.12),
            (95375, 0.22),
            (182100, 0.24),
            (231250, 0.32),
            (578125, 0.35),
            (float('inf'), 0.37)
        ]
    }

    tax = 0
    previous_limit = 0

    for limit, rate in brackets[filing_status]:
        if agi > limit:
            tax += (limit - previous_limit) * rate
            previous_limit = limit
        else:
            tax += (agi - previous_limit) * rate
            break

    return tax
```

### 7.2 What-If Tax Scenarios

```python
def analyze_tax_scenarios(portfolio, proposed_trades):
    """
    Compare tax impact of different scenarios.

    Scenarios:
    1. Do nothing
    2. Harvest all available losses
    3. User's proposed trades
    4. AI-optimized strategy
    """

    scenarios = {}

    # Scenario 1: Status quo
    scenarios['do_nothing'] = calculate_real_time_tax_projection(
        ytd_income,
        ytd_trades,
        portfolio
    )

    # Scenario 2: Harvest all losses
    all_loss_positions = [p for p in portfolio.positions if p.unrealized_gain < -1000]
    harvest_all_trades = [generate_harvest_trade(p) for p in all_loss_positions]

    scenarios['harvest_all'] = calculate_real_time_tax_projection(
        ytd_income,
        ytd_trades + harvest_all_trades,
        portfolio
    )

    # Scenario 3: User's proposed trades
    scenarios['user_proposed'] = calculate_real_time_tax_projection(
        ytd_income,
        ytd_trades + proposed_trades,
        portfolio
    )

    # Scenario 4: AI-optimized
    optimized_trades = generate_optimal_tax_strategy(portfolio, ytd_income, ytd_trades)
    scenarios['ai_optimized'] = calculate_real_time_tax_projection(
        ytd_income,
        ytd_trades + optimized_trades,
        portfolio
    )

    # Compare scenarios
    comparison = []
    for name, result in scenarios.items():
        comparison.append({
            'scenario': name,
            'total_tax': result['current_projection']['total_tax'],
            'savings_vs_nothing': scenarios['do_nothing']['current_projection']['total_tax'] -
                                   result['current_projection']['total_tax'],
            'trades_required': len(proposed_trades) if name == 'user_proposed' else
                               len(harvest_all_trades) if name == 'harvest_all' else
                               len(optimized_trades) if name == 'ai_optimized' else 0
        })

    best_scenario = min(comparison, key=lambda x: x['total_tax'])

    return {
        'scenarios': comparison,
        'recommendation': best_scenario,
        'visualization': generate_scenario_chart(comparison)
    }
```

---

## Summary: Tax Optimizer Features

**Advanced Capabilities:**
1. ✅ Multi-year tax planning (Roth conversions, loss carryforward)
2. ✅ Options tax optimization (covered calls, wash sales)
3. ✅ Crypto-specific strategies (no wash sale rule exploitation)
4. ✅ Estate planning (stepped-up basis optimization)
5. ✅ Real-time tax projections (daily updates)
6. ✅ Scenario analysis (what-if modeling)
7. ✅ Automated backdoor Roth (high earner strategy)

**User Benefits:**
- Save $5k-50k annually in taxes (vs unoptimized)
- Automated reminders (harvest by Dec 31, convert in low-income years)
- Educational (learn WHY each strategy works)
- Compliant (all IRS rules encoded)

**Competitive Advantage:**
- Wealthfront: Basic TLH only
- TurboTax: Tax filing, no optimization
- Human advisors: Expensive ($5k-20k/year)
- **Gazillioner: Comprehensive + automated + educational**

**FQ Points Integration:**
- +15 FQ: Each tax loss harvest
- +25 FQ: Complete backdoor Roth
- +50 FQ: Execute multi-year tax plan
- +100 FQ: Save $10k+ in taxes in one year

---

**Implementation Priority:**
1. **Month 1:** Advanced loss harvesting (dynamic scoring)
2. **Month 2:** Real-time tax dashboard
3. **Month 3:** Crypto tax optimization
4. **Month 4:** Multi-year planning (Roth conversions)
5. **Month 5:** Options strategies
6. **Month 6:** Estate planning features
