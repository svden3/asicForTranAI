# MVP Specification - Gazillioner FQ Platform
## Minimum Viable Product: AI Wealth Advisor

**Version:** 1.0
**Date:** December 10, 2025
**Timeline:** 4 Months to Launch
**Website:** https://gazillioner.com/

---

## Executive Summary

**MVP Goal:** Launch a functional AI wealth advisor that proves the core value proposition:
> "Gazillioner helps you avoid costly mistakes (missing opportunities, panic-selling) by improving your Financial Quotient (FQ) through daily coaching and education."

**Success Criteria (Month 6):**
- ✅ 2,000 paying users ($30/month)
- ✅ $500k ARR run-rate
- ✅ 70%+ DAU/MAU (daily engagement)
- ✅ NPS >40
- ✅ <5% monthly churn
- ✅ Users save avg. $1,000+ in first 3 months (tax + avoided mistakes)

**What's In MVP:**
1. FQ Assessment & Scoring System
2. Daily AI Briefing (personalized portfolio insights + education)
3. Portfolio Tracking (stocks + crypto via broker integrations)
4. Tax-Loss Harvesting Suggestions
5. Automated Rebalancing Recommendations
6. Learning Platform (50 exercises + 10 case studies)
7. Thesis Tracker (track long-term investment convictions)
8. Behavioral Alerts (detect FOMO/panic before mistakes happen)

**What's NOT in MVP (Phase 2):**
- ❌ Options strategies (covered calls, spreads)
- ❌ On-device AI inference (Tier 2 hardware)
- ❌ Mobile apps (iOS/Android)
- ❌ Community forum & leaderboards
- ❌ Advanced features (multi-account optimization, scenario planning)

---

## MVP Feature Breakdown

### 1. User Onboarding & Authentication

#### 1.1 Account Creation
**Features:**
- Email/password signup OR OAuth (Google, Apple)
- Email verification (required before full access)
- 2FA optional (but encouraged)
- GDPR/CCPA-compliant consent flow

**Technical Implementation:**
- Next.js frontend with server-side rendering
- NextAuth.js for authentication (supports multiple providers)
- PostgreSQL for user accounts
- SendGrid for email verification
- Password hashing: bcrypt (cost factor 12)

**User Flow:**
1. Land on gazillioner.com
2. Click "Get Started"
3. Enter email + password (or OAuth)
4. Verify email (click link)
5. Redirected to FQ assessment

**Acceptance Criteria:**
- Sign up takes <2 minutes
- Email verification link works reliably
- Secure: passwords hashed, HTTPS only, CSRF protection

---

#### 1.2 Initial FQ Assessment

**Features:**
- 30-question quiz (15 minutes)
- Questions cover Knowledge, Discipline, Wisdom dimensions
- Real-time progress bar
- Immediate FQ score upon completion (0-1000)
- Breakdown by category (Knowledge /300, Discipline /400, Wisdom /300)
- Personalized insights ("You're strong in Knowledge but need work on Discipline")

**Question Categories (30 total):**
- **Knowledge (10 questions):**
  - Asset allocation concepts (stocks, bonds, crypto)
  - Valuation basics (P/E ratio, DCF intuition)
  - Tax fundamentals (capital gains, wash sales)
  - Options basics (calls, puts, covered calls)

- **Discipline (10 questions):**
  - "How often do you rebalance your portfolio?"
  - "Do you have an emergency fund (6 months expenses)?"
  - "Have you ever panic-sold during a market crash?"
  - "Do you invest consistently (dollar-cost averaging)?"

- **Wisdom (10 questions):**
  - Scenario-based: "NVDA is down 30%. Your thesis (AI revolution) is intact. What do you do?"
  - Pattern recognition: "Which of these is most like Apple in 2010?" (show 4 current stocks)
  - Historical: "Why did most investors miss Amazon's 100x run?" (multiple choice)

**Scoring Algorithm:**
```python
def calculate_fq(answers):
    knowledge_score = sum([q.points for q in knowledge_questions]) / 10 * 300
    discipline_score = sum([q.points for q in discipline_questions]) / 10 * 400
    wisdom_score = sum([q.points for q in wisdom_questions]) / 10 * 300

    total_fq = knowledge_score + discipline_score + wisdom_score
    return {
        'total': round(total_fq),
        'knowledge': round(knowledge_score),
        'discipline': round(discipline_score),
        'wisdom': round(wisdom_score),
        'percentile': calculate_percentile(total_fq),  # vs other users
        'tier': get_tier(total_fq)  # "Beginner" / "Building Wealth" / etc.
    }
```

**UI/UX:**
- One question per page (focus, no overwhelm)
- Progress bar at top (14/30 complete)
- "Save and Continue Later" option
- Visual results page (pie chart, bar graphs)
- Celebration animation when score reveals

**Acceptance Criteria:**
- Quiz completable in <15 minutes
- Score calculation accurate and deterministic
- Results page loads in <2 seconds
- Mobile-responsive (works on phone)

---

#### 1.3 Goal Setting

**Features:**
- Add up to 5 financial goals
- For each goal, specify:
  - Type (retirement, house, emergency fund, etc.)
  - Target amount (e.g., $2.5M)
  - Target date (e.g., age 60, year 2027)
  - Priority (critical, important, nice-to-have)
- System calculates required monthly savings
- Visual timeline showing all goals

**UI/UX:**
- Simple form with dropdowns and date pickers
- Real-time calculation: "To reach $2.5M by 2055, save $3,200/month"
- Adjustable: move sliders to see impact (earlier retirement = higher savings needed)
- Visual: Timeline with milestones

**Acceptance Criteria:**
- Add goal in <1 minute
- Calculations accurate (compound interest, inflation-adjusted)
- Responsive on mobile

---

#### 1.4 Broker/Exchange Connections

**Features:**
- Connect stock brokers: Robinhood, Schwab, Fidelity, Interactive Brokers (via Plaid)
- Connect crypto exchanges: Coinbase, Binance, Kraken (direct API keys)
- Manual entry fallback (for unsupported accounts)
- OAuth flow for Plaid (secure, no password sharing)

**Plaid Integration:**
```javascript
// Frontend: Plaid Link component
import { usePlaidLink } from 'react-plaid-link';

const { open, ready } = usePlaidLink({
  token: linkToken,  // from backend
  onSuccess: (public_token, metadata) => {
    // Exchange public token for access token (backend call)
    exchangePublicToken(public_token);
  },
});
```

**Backend: Exchange Token & Fetch Holdings**
```javascript
// Exchange public token for access token
const response = await plaidClient.itemPublicTokenExchange({
  public_token: publicToken,
});
const accessToken = response.data.access_token;

// Fetch holdings
const holdings = await plaidClient.investmentsHoldingsGet({
  access_token: accessToken,
});

// Store in database
await savePortfolio(userId, holdings);
```

**Crypto Exchange Integration:**
- Users provide API keys (read-only permissions)
- Keys stored encrypted in AWS Secrets Manager
- Fetch balances via exchange APIs (Coinbase API, Binance API)

**Acceptance Criteria:**
- Plaid connection works for top 4 brokers
- Crypto API keys stored securely (encrypted, not in database plaintext)
- Portfolio syncs within 10 seconds
- Error handling: graceful failure if API down

---

### 2. Daily Briefing System

#### 2.1 AI-Generated Daily Briefing

**Goal:** Every morning at 6 AM (user's timezone), generate a personalized 2-minute briefing.

**Content Blocks (Template):**

```markdown
# Good morning, [Name]. Here's your financial snapshot:

## YOUR PORTFOLIO
- Current value: $[X] ([+/-]$[Y] today, [+/-]Z% YTD)
- Allocation: [X]% stocks, [Y]% crypto, [Z]% bonds vs. target ([A]% / [B]% / [C]%)

## TODAY'S ACTIONS (max 3)
1. **Rebalance:** Sell $[X] [TICKER], buy $[Y] [TICKER] (back to target allocation)
   - Why: Your stocks rallied to 75% (target: 70%)
   - Tax impact: Selling long-term gains (15% tax on $[X] profit)

2. **Tax-Loss Harvest:** Sell [TICKER] at $[X] loss → save $[Y] in taxes
   - Replace with [SIMILAR_TICKER] to maintain exposure
   - Wash sale clear: Last sold 45 days ago ✓

3. **Contribute:** Auto-invest $[X] to Roth IRA today (your monthly schedule)

## LEARNING MOMENT (2-min read)
**Why people sold Apple too early**
In 2011, Steve Jobs died. Stock fell 30%. Many sold, thinking "Apple is done."
Reality: Tim Cook was an excellent operator. iPhone kept growing. Services revenue took off.
Lesson: Founder risk is real, but ecosystem moats often survive leadership changes.

Pattern to remember: "Founder departure ≠ broken thesis if moat is intact"

## THESIS CHECK
**Your NVDA holding:** Stock down 8% this week after earnings.
- Your thesis: "AI revolution will drive data center demand for 10+ years"
- Evidence this week:
  ✓ Data center revenue up 217% YoY (accelerating, not slowing)
  ✓ Guidance: Next quarter revenue $[X]B (beat expectations)
  ✗ Stock fell due to "high expectations" (sentiment, not fundamentals)
- Verdict: **Thesis STRENGTHENING** (revenue growth accelerating)
- Recommendation: **HOLD** (or add on dips if you have cash)

## BEHAVIORAL ALERT
⚠️ You searched for "[MEMECOIN]" yesterday after it pumped 100%.
Pattern detected: **FOMO** (buying momentum without thesis)
Historical lesson: Remember when you bought SHIB at peak in 2021? Result: -$3,200
Framework: Ask yourself: "Do I understand the 10-year thesis, or am I just chasing?"
Recommendation: Wait 48 hours. If you still want to buy after research, allocate <1% of portfolio.

---
See you tomorrow! Reply with questions or "Done" when you've completed today's actions.
```

**AI Generation Process:**

1. **Fetch Data (6 AM daily cron job):**
   ```python
   user = get_user(user_id)
   portfolio = get_portfolio(user_id)  # from broker APIs
   market_data = get_market_data()  # VIX, yields, BTC price, etc.
   user_goals = get_goals(user_id)
   user_theses = get_theses(user_id)
   recent_actions = get_recent_actions(user_id, days=7)  # searches, trades
   ```

2. **Generate Briefing (AI):**
   ```python
   # Use template + AI to fill in blanks
   prompt = f"""
   Generate a daily financial briefing for this user:

   Portfolio: {portfolio}
   Goals: {user_goals}
   Theses: {user_theses}
   Recent actions: {recent_actions}
   Market context: {market_data}

   Include:
   1. Portfolio snapshot
   2. 1-3 action items (rebalance, tax harvest, contribute)
   3. Learning moment (1 concept, 200 words, tie to user's holdings if possible)
   4. Thesis check (pick 1-2 holdings, is thesis intact?)
   5. Behavioral alert (if user showed FOMO/panic patterns in recent_actions)

   Style: Concise, actionable, educational. 2-minute read (<500 words).
   """

   briefing = ai_model.generate(prompt)  # GPT-4 or fine-tuned Llama
   ```

3. **Deliver:**
   - Save to database (`daily_briefings` table)
   - Send email (if user opted in)
   - Push notification (if mobile app installed)
   - Display in dashboard when user logs in

**Acceptance Criteria:**
- Briefing generates in <5 seconds (p95)
- Delivered by 6 AM user timezone (100% reliability)
- Content is accurate (portfolio values within 1%, tax calculations correct)
- Readable (<500 words, Grade 10 reading level)
- Actionable (at least 1 specific thing to do)

---

#### 2.2 Thesis Tracker

**Goal:** Let users define long-term investment theses and track if they're still valid.

**Features:**
- Create a thesis: Name, related holdings, core beliefs, falsification criteria
- Monthly updates: AI analyzes news/data and reports if thesis is strengthening/intact/weakening/broken
- Alerts: If major event contradicts thesis (e.g., "NVDA loses CUDA dominance to AMD")

**Example Thesis:**
```yaml
Name: "AI Revolution"
Holdings: [NVDA, MSFT, GOOGL]
Core Beliefs:
  - "AI will transform computing over next 10 years"
  - "NVDA has 10-year lead with CUDA ecosystem"
  - "Hyperscalers (MSFT, GOOGL) will spend $100B+/year on AI infrastructure"
Falsification Criteria:
  - "Competitor breaks CUDA moat (PyTorch runs faster on AMD/Intel)"
  - "AI hype collapses (spending drops 50%+)"
  - "Regulation bans AI development"
Allocation: 15% of portfolio (appropriate for high-conviction, high-risk bet)
```

**Monthly Update (AI-Generated):**
```markdown
## Thesis Update: "AI Revolution" (December 2025)

**Status:** STRENGTHENING ↗️

**Evidence (Last 30 Days):**
✓ NVDA data center revenue up 217% YoY (accelerating growth)
✓ MSFT announced $50B AI infrastructure spend in 2026 (up from $30B in 2025)
✓ ChatGPT hit 200M weekly active users (mainstream adoption)
✓ Google DeepMind AlphaFold 3 breakthrough (AI proving value in biology)

✗ AMD launched MI300X chip claiming 2× performance vs H100 (potential threat to NVDA moat)
  → Analysis: AMD chip faster for inference, but CUDA ecosystem still dominant for training.
            Threat is real but 3-5 years away from breaking moat.

**Recommendation:**
Your thesis remains intact. The AI TAM is expanding faster than expected (good for all players).
NVDA moat is challenged but not broken. If AMD takes 20% market share, NVDA still grows 100%+
due to TAM expansion.

**Action:** HOLD current 15% allocation. Consider trimming to 12% if NVDA rallies another 30%
(prevent over-concentration risk).
```

**Technical Implementation:**
- Thesis stored in PostgreSQL (JSON field for flexibility)
- News monitoring: NewsAPI, Google News RSS, Twitter API (for $NVDA mentions)
- AI analysis: GPT-4 reads top 10 news articles, checks if they support/contradict thesis
- Monthly cron job generates updates
- Push notification if status changes (Intact → Weakening)

**Acceptance Criteria:**
- Users can create thesis in <3 minutes
- Monthly updates are accurate (no hallucinations)
- Alerts trigger within 24 hours of major events

---

#### 2.3 Behavioral Alerts

**Goal:** Detect when user is about to make an emotional decision and intervene.

**Patterns to Detect:**

1. **FOMO (Fear of Missing Out):**
   - Trigger: User searches for or adds to watchlist a stock/crypto that's up 30%+ in 30 days
   - Alert: "You're chasing a pump. Remember when you FOMO'd into SHIB? -$3k. Wait 48 hours and research."

2. **Panic Selling:**
   - Trigger: User's portfolio down 20%+, they're viewing "Sell" page
   - Alert: "You're considering selling in a drawdown. Check your thesis first. Is it broken or just volatility?"

3. **Premature Profit-Taking:**
   - Trigger: User selling a position that's up 50%+ with thesis still intact
   - Alert: "You want to lock in gains. But your thesis is strengthening. Consider trimming 20%, not 100%."

4. **Loss Aversion:**
   - Trigger: User holding a position down 30%+ with broken thesis (refusing to realize loss)
   - Alert: "You're avoiding a painful loss. But thesis is broken. Sell, reinvest in better opportunity."

**Technical Implementation:**
```python
def detect_behavioral_patterns(user_id):
    recent_searches = get_search_history(user_id, days=7)
    portfolio = get_portfolio(user_id)
    watchlist = get_watchlist(user_id)
    pending_trades = get_pending_trades(user_id)

    alerts = []

    # FOMO Detection
    for search in recent_searches:
        ticker = extract_ticker(search)
        price_change_30d = get_price_change(ticker, days=30)
        if price_change_30d > 0.30:  # up 30%+
            historical_fomo = get_user_fomo_history(user_id)  # past mistakes
            alerts.append({
                'type': 'FOMO',
                'ticker': ticker,
                'message': f"You're chasing {ticker} after a 30% pump...",
                'historical_lesson': historical_fomo,
                'recommendation': "Wait 48 hours, research the thesis"
            })

    # Panic Selling Detection
    portfolio_drawdown = calculate_drawdown(portfolio)
    if portfolio_drawdown < -0.20 and pending_trades:  # down 20%+, about to sell
        for trade in pending_trades:
            if trade.action == 'SELL':
                holding = get_holding(trade.ticker)
                thesis = get_thesis_for_holding(user_id, holding)
                if thesis and thesis.status in ['STRENGTHENING', 'INTACT']:
                    alerts.append({
                        'type': 'PANIC',
                        'ticker': trade.ticker,
                        'message': f"Your {holding} thesis is still intact. Don't panic sell.",
                        'thesis_status': thesis.status,
                        'recommendation': "Review your thesis. If intact, hold through volatility."
                    })

    # ... similar logic for other patterns

    return alerts
```

**UI/UX:**
- Modal popup (blocks navigation until user acknowledges)
- Shows pattern detected, historical lesson, recommendation
- Options: "Proceed Anyway" / "Cancel" / "Remind Me Tomorrow"
- User response logged for Wisdom scoring

**Acceptance Criteria:**
- Alerts trigger within 1 minute of user action
- No false positives (>90% accuracy)
- User can dismiss or snooze (don't be annoying)

---

### 3. Portfolio Management Features

#### 3.1 Portfolio Dashboard

**Features:**
- Total portfolio value (real-time or daily sync)
- Performance: % change (today, week, month, YTD, all-time)
- Allocation chart (pie chart: stocks, crypto, bonds, cash)
- Target allocation vs. actual (visual drift indicator)
- Holdings table (ticker, shares, cost basis, current value, gain/loss, %)
- Recent transactions log

**Technical Implementation:**
```javascript
// Frontend: React component
function PortfolioDashboard({ userId }) {
  const { data: portfolio } = useQuery(['portfolio', userId], fetchPortfolio);

  return (
    <div>
      <PortfolioSummary value={portfolio.total_value} change={portfolio.change_pct} />
      <AllocationChart current={portfolio.allocation} target={portfolio.target_allocation} />
      <HoldingsTable holdings={portfolio.holdings} />
      <TransactionsLog transactions={portfolio.recent_transactions} />
    </div>
  );
}
```

**Data Update:**
- Real-time: WebSocket for live price updates (optional, Tier 1 users get daily sync)
- Daily sync: Cron job fetches holdings from broker APIs every morning
- On-demand: "Refresh" button triggers immediate sync

**Acceptance Criteria:**
- Dashboard loads in <2 seconds
- Data accurate (matches broker statements within 1%)
- Responsive on mobile

---

#### 3.2 Tax-Loss Harvesting

**Goal:** Automatically find opportunities to sell losing positions and replace with similar assets.

**Algorithm:**
```python
def find_tax_loss_harvesting_opportunities(portfolio, tax_bracket):
    opportunities = []

    for holding in portfolio.holdings:
        unrealized_loss = holding.cost_basis - holding.current_value

        if unrealized_loss > 500:  # worth the effort
            # Check wash sale window
            last_sold_date = get_last_sale_date(holding.ticker)
            days_since_sale = (today - last_sold_date).days if last_sold_date else 999

            if days_since_sale >= 31:  # wash sale clear
                # Find similar security
                similar_ticker = find_similar_security(holding.ticker)

                if similar_ticker:
                    tax_savings = unrealized_loss * tax_bracket

                    opportunities.append({
                        'sell': holding.ticker,
                        'sell_shares': holding.shares,
                        'loss': unrealized_loss,
                        'buy': similar_ticker,
                        'tax_savings': tax_savings,
                        'reasoning': f"Harvest ${unrealized_loss} loss, save ${tax_savings} in taxes"
                    })

    return opportunities

def find_similar_security(ticker):
    # Database of substantially similar securities
    similar_map = {
        'SPY': 'VOO',  # S&P 500 ETFs
        'VOO': 'SPY',
        'QQQ': 'QQQM',  # NASDAQ ETFs
        'VTI': 'ITOT',  # Total market ETFs
        # ... etc
    }
    return similar_map.get(ticker)
```

**User Experience:**
1. Daily briefing includes: "Tax-loss harvest: Sell SPY (-$840 loss) → buy VOO. Save $202 in taxes."
2. User clicks "Review"
3. Modal shows:
   - Current position: 10 shares SPY at $480 (cost basis $564)
   - Unrealized loss: -$840
   - Tax savings: $202 (assuming 24% tax bracket)
   - Replacement: Buy 10 shares VOO at $481 (same S&P 500 exposure)
   - Wash sale status: ✓ Clear (last sold SPY 45 days ago)
4. User clicks "Execute" → trades submitted to broker API

**Acceptance Criteria:**
- Finds all opportunities >$500 loss
- Accurately tracks wash sale windows (no violations)
- Tax savings calculations correct (based on user's tax bracket)

---

#### 3.3 Automated Rebalancing

**Goal:** Suggest trades to bring portfolio back to target allocation when drift >5%.

**Algorithm:**
```python
def calculate_rebalancing_trades(portfolio, target_allocation):
    current_allocation = {
        'stocks': sum([h.value for h in portfolio.holdings if h.asset_class == 'stock']) / portfolio.total_value,
        'crypto': sum([h.value for h in portfolio.holdings if h.asset_class == 'crypto']) / portfolio.total_value,
        'bonds': sum([h.value for h in portfolio.holdings if h.asset_class == 'bond']) / portfolio.total_value,
    }

    drift = {
        asset: current_allocation[asset] - target_allocation[asset]
        for asset in target_allocation
    }

    # Only rebalance if drift >5%
    if max(abs(d) for d in drift.values()) < 0.05:
        return None  # no rebalancing needed

    # Calculate trades
    trades = []
    for asset, drift_pct in drift.items():
        if drift_pct > 0:  # overweight, sell
            sell_amount = drift_pct * portfolio.total_value
            # Pick specific holding to sell (prefer long-term gains for tax efficiency)
            holding_to_sell = pick_tax_efficient_holding(portfolio, asset, sell_amount)
            trades.append({'action': 'SELL', 'ticker': holding_to_sell.ticker, 'amount': sell_amount})
        elif drift_pct < 0:  # underweight, buy
            buy_amount = abs(drift_pct) * portfolio.total_value
            ticker_to_buy = get_preferred_ticker(asset)  # e.g., VTI for stocks, BTC for crypto
            trades.append({'action': 'BUY', 'ticker': ticker_to_buy, 'amount': buy_amount})

    return trades

def pick_tax_efficient_holding(portfolio, asset_class, amount_needed):
    # Prefer selling long-term gains (lower tax) or losses (tax benefit)
    holdings = [h for h in portfolio.holdings if h.asset_class == asset_class]
    holdings.sort(key=lambda h: (h.holding_period < 365, -h.unrealized_gain))  # prioritize long-term, then losses
    return holdings[0]
```

**User Experience:**
1. Daily briefing: "Your stocks are 75% (target 70%). Rebalance: Sell $2,500 VTI, buy $1,500 BTC + $1,000 BND."
2. User reviews suggested trades
3. System shows:
   - Current vs. target allocation (visual)
   - Tax implications (selling VTI = $45 long-term capital gains tax)
   - Impact on risk (lowering stock exposure = lower volatility)
4. User approves → trades execute

**Acceptance Criteria:**
- Rebalancing suggestions are correct (brings allocation within 1% of target)
- Tax-aware (prefers long-term gains over short-term)
- User can customize (e.g., "Don't sell AAPL" exception rule)

---

### 4. Learning Platform

#### 4.1 Exercise Library (50 Exercises in MVP)

**Goal:** Provide bite-sized learning modules to improve FQ score.

**Exercise Categories (50 total for MVP):**
- Asset Allocation (10): Risk vs return, diversification, correlation
- Valuation (8): P/E ratio, PEG ratio, DCF intuition, growth vs value
- Tax Optimization (10): Capital gains, wash sales, tax-loss harvesting, account types (Roth vs Traditional)
- Behavioral Finance (8): FOMO, panic, loss aversion, confirmation bias
- Crypto Fundamentals (8): DeFi, staking, gas fees, wallet security
- Market Cycles (6): Bull/bear patterns, recessions, sector rotation

**Exercise Format:**
```yaml
Exercise ID: tax-001
Title: "Understanding Wash Sale Rules"
Category: Tax Optimization
Difficulty: Intermediate
Time: 8 minutes
FQ Points: +10

Content:
  - Explanation (200 words): What is a wash sale, why does it exist, when does it apply?
  - Example: "You sell SPY at a loss on Jan 1. You buy VOO on Jan 15. Is this a wash sale?"
  - Question: Multiple choice or scenario-based
  - Feedback: Immediate explanation of correct answer

Question:
  "You sold 10 shares of Tesla at a $2,000 loss on March 1. When is the earliest you can
   repurchase Tesla without triggering a wash sale?"

  A) March 15 (14 days later)
  B) April 1 (31 days later)  ← CORRECT
  C) May 1 (61 days later)
  D) You can never buy Tesla again (incorrect understanding)

Explanation:
  "Correct! The wash sale rule prohibits repurchasing the same or 'substantially identical'
   security within 30 days before or after the sale. Since you sold on March 1, you must
   wait until April 1 (31 days) to repurchase. Note: You COULD buy a similar stock (like
   Rivian or Lucid) immediately without violating the rule."

Spaced Repetition:
  - Review in 7 days (reinforcement)
  - Review in 30 days (long-term retention)
```

**Technical Implementation:**
```javascript
// Exercise Component
function Exercise({ exercise, onComplete }) {
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);

  const handleSubmit = () => {
    setShowFeedback(true);
    const isCorrect = selectedAnswer === exercise.correct_answer;
    onComplete(exercise.id, isCorrect);  // Update FQ score
  };

  return (
    <div>
      <h2>{exercise.title}</h2>
      <div>{exercise.content}</div>
      <div>{exercise.question}</div>
      <AnswerOptions options={exercise.options} onSelect={setSelectedAnswer} />
      <button onClick={handleSubmit}>Submit</button>
      {showFeedback && <Feedback correct={selectedAnswer === exercise.correct_answer} explanation={exercise.explanation} />}
    </div>
  );
}
```

**Acceptance Criteria:**
- 50 exercises ready at MVP launch
- Each exercise takes 5-15 minutes
- Immediate feedback (no waiting)
- FQ points awarded correctly (+10 per exercise)
- Spaced repetition: exercises resurface after 7, 30, 90 days

---

#### 4.2 Historical Case Studies (10 in MVP)

**Goal:** Learn from past investment scenarios.

**MVP Case Studies:**
1. **Apple 2010-2024:** Why investors sold too early
2. **NVDA 2015-2024:** Recognizing expanding TAMs before the crowd
3. **Bitcoin 2020:** Buying during maximum fear (COVID crash to $16k)
4. **Tesla 2019:** Holding through 60% drawdown when thesis intact
5. **Dotcom Bubble 2000:** Distinguishing winners (Amazon) from losers (Pets.com)
6. **Netflix 2011:** Qwikster disaster (stock fell 80%, recovered to 50x)
7. **Amazon 2014:** "Overvalued at $300" (now $3,000+ split-adjusted)
8. **Microsoft 2014:** Satya Nadella turnaround (cloud transformation)
9. **Ethereum 2018:** Holding through 95% crash (ICO bubble pop)
10. **Google IPO 2004:** "$85 is too expensive" (now $2,800+ split-adjusted)

**Case Study Format:**
```markdown
# Case Study: Apple 2010 - Why Investors Sold Too Early

## Timeline
**August 2010:** Steve Jobs announced iPhone 4 (stock at $10/share split-adjusted)
**October 2011:** Steve Jobs passed away → stock fell 30% to $13
**2013:** Stock flat for 2 years, "smartphone market saturated" fears
**2016:** First YoY iPhone sales decline → panic
**2024:** Stock at $180/share (18× from 2010, 13× from Jobs' death)

## Decision Points

### Point 1: October 2011 (Jobs Dies)
Stock fell from $18 to $13 (-28% in 3 months).

**What would you do?**
A) Sell immediately (founder risk, Apple is done)
B) Hold, but stop adding new money (cautious)
C) Add to position (ecosystem moat intact)
D) Go all-in (max conviction)

**What most investors did:** A or B (sold or stopped buying)

**What happened:** Stock went from $13 → $180 (13×)

**Why selling was wrong:**
- Mistook founder risk for product risk
- iPhone ecosystem already had 200M users (switching costs high)
- Tim Cook was strong operator (supply chain genius)
- Services revenue (App Store) just getting started

**Lesson:** Founder departure ≠ broken thesis if moat is intact

### Point 2: 2013 (Market Saturation Fears)
Stock flat for 2 years. Analysts: "Smartphone penetration at 50%, growth over."

**What would you do?**
A) Sell (growth story over, become a value stock)
B) Hold (price appreciation done, collect dividends)
C) Add (services revenue underappreciated)

**What most investors did:** A or B (sold or just held)

**What happened:** Stock tripled in next 3 years as Services grew 20%/year

**Lesson:** Mature products → look for new revenue streams (Services, Wearables)

## Pattern Recognition
This pattern repeats constantly:
- **Netflix 2011:** Qwikster disaster → stock fell 80% → recovered 50×
- **Microsoft 2014:** "Dead company" → cloud transformation → 10×
- **Amazon 2014:** "Overvalued, no profits" → AWS dominance → 10×

**How to recognize:**
✓ Strong moat (ecosystem, network effects)
✓ New revenue streams emerging (Services for Apple, AWS for Amazon)
✓ Temporary setback (Qwikster, founder death) not structural

**Red flags (when to actually sell):**
✗ Moat is breaking (users leaving ecosystem)
✗ No new products/revenue streams (just milking old products)
✗ Management failing (missing every target, poor capital allocation)

## Your Turn
Apply this pattern to today's market.

Which current stock is most like Apple in 2010?
A) TSLA (after 60% crash, but EV moat unclear)
B) NVDA (after 60% crash, but AI moat intact)
C) META (after metaverse flop, but ad moat intact)
D) COIN (after crypto crash, but exchange moat weak)

[User selects B]

**Analysis:**
Correct! NVDA most resembles Apple 2010:
✓ Temporary setback (Fed rate hikes, crypto crash) not structural
✓ Expanding TAM (AI is like iPhone in 2010: early innings)
✓ Moat widening (CUDA ecosystem has network effects)
✓ New revenue streams (data center now 80% of revenue, was 20% in 2015)

**FQ Points:** +25 (excellent pattern recognition!)
```

**Acceptance Criteria:**
- 10 case studies ready at MVP launch
- Each case study 10-15 minutes to complete
- Interactive (user makes decisions, sees outcomes)
- Awards +25 to +50 FQ points

---

### 5. FQ Scoring & Gamification

#### 5.1 Real-Time FQ Updates

**Events that change FQ:**

| Action | FQ Points | Category |
|--------|-----------|----------|
| Complete exercise | +5 to +15 | Knowledge |
| Complete case study | +25 to +50 | Wisdom |
| Rebalance when suggested | +15 | Discipline |
| Tax-loss harvest | +10 | Discipline |
| Auto-invest on schedule | +5 | Discipline |
| Hold through 20% drawdown (thesis intact) | +25 | Wisdom |
| Add to position during fear (contrarian) | +30 | Wisdom |
| **Panic sell** (thesis intact) | **-20** | Wisdom |
| **FOMO buy** (chase pump) | **-15** | Wisdom |
| **Miss rebalancing** (ignore suggestion) | **-5** | Discipline |
| **Violate wash sale** | **-10** | Knowledge |

**UI/UX:**
- Toast notification: "+10 FQ: Completed tax-loss harvesting exercise"
- Dashboard sparkline: 7-day FQ trend
- Celebration animation when crossing milestones (500 → 600)

---

#### 5.2 Streaks & Achievements

**Streaks:**
- Daily engagement streak (check in, read briefing, or complete exercise)
- Contribution streak (invest on schedule for X consecutive months)
- Learning streak (complete 1 exercise per week for X weeks)

**Achievements (Badges):**
- 🏆 **"Tax Ninja"** - Harvested $5,000+ in losses
- 💎 **"Diamond Hands"** - Held through 50%+ drawdown (thesis intact)
- 📈 **"DCA Champion"** - 12 months consecutive contributions
- 🎓 **"Knowledge Seeker"** - Completed 100 exercises
- 🧠 **"Wisdom Master"** - FQ Wisdom score >250
- 💰 **"Millionaire"** - Net worth >$1M
- 🔥 **"30-Day Streak"** - Daily engagement for 30 days

**Technical Implementation:**
```python
def check_achievements(user_id):
    user = get_user(user_id)
    portfolio = get_portfolio(user_id)
    fq_history = get_fq_history(user_id)

    new_achievements = []

    # Check Tax Ninja
    total_losses_harvested = sum([a.loss_amount for a in user.actions if a.type == 'tax_harvest'])
    if total_losses_harvested >= 5000 and 'tax_ninja' not in user.achievements:
        new_achievements.append('tax_ninja')

    # Check Diamond Hands
    max_drawdown = calculate_max_drawdown(portfolio.history)
    if max_drawdown < -0.50 and user.held_through_drawdown and 'diamond_hands' not in user.achievements:
        new_achievements.append('diamond_hands')

    # ... similar logic for other achievements

    if new_achievements:
        award_achievements(user_id, new_achievements)
        send_notification(user_id, f"🏆 Achievement Unlocked: {new_achievements[0]}")
```

---

### 6. Technical Architecture (MVP)

#### 6.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  Next.js 14 (React) + TypeScript + Tailwind CSS             │
│  - Dashboard, Portfolio, Learning, Settings pages            │
│  - Real-time updates via WebSocket                           │
│  - Responsive (mobile-first design)                          │
└───────────────┬─────────────────────────────────────────────┘
                │ HTTPS / WebSocket
┌───────────────▼─────────────────────────────────────────────┐
│                     API GATEWAY                              │
│  - Authentication (NextAuth.js)                              │
│  - Rate limiting (100 req/min per user)                      │
│  - Request logging                                           │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬──────────────┐
    │           │           │              │
┌───▼────┐ ┌───▼────┐ ┌───▼────┐   ┌─────▼──────┐
│Portfolio│ │   FQ   │ │   AI   │   │  Learning  │
│ Service │ │Service │ │Service │   │  Service   │
│         │ │        │ │        │   │            │
│- Sync   │ │-Scoring│ │-Briefing│  │- Exercises │
│- Rebal. │ │-Exer.  │ │-Thesis  │  │- Case      │
│- Tax    │ │-Achiev.│ │-Alerts  │  │  Studies   │
└────┬────┘ └───┬────┘ └───┬────┘   └─────┬──────┘
     │          │          │              │
     └──────────┴──────────┴──────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  PostgreSQL (users, portfolios, fq_scores, exercises)        │
│  Redis (caching, session management, real-time data)         │
│  S3 (exercise content, case study images)                    │
└─────────────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                   EXTERNAL APIs                              │
│  - Plaid (broker connections)                                │
│  - Alpaca / IB (trading APIs)                                │
│  - CoinGecko / CoinMarketCap (crypto prices)                 │
│  - IEX Cloud (stock market data)                             │
│  - OpenAI GPT-4 / Claude API (AI generation)                 │
│  - NewsAPI (thesis tracking)                                 │
└─────────────────────────────────────────────────────────────┘
```

#### 6.2 Database Schema (Key Tables)

**users**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- bcrypt
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    fq_score INTEGER DEFAULT 0,
    fq_knowledge INTEGER DEFAULT 0,
    fq_discipline INTEGER DEFAULT 0,
    fq_wisdom INTEGER DEFAULT 0,
    target_allocation JSONB,  -- {"stocks": 0.70, "crypto": 0.20, "bonds": 0.10}
    tax_bracket DECIMAL(4,3),  -- 0.24 for 24% bracket
    timezone VARCHAR(50) DEFAULT 'America/New_York'
);
```

**portfolios**
```sql
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    broker_name VARCHAR(100),  -- "Robinhood", "Coinbase", etc.
    account_id VARCHAR(255),  -- from Plaid or exchange
    holdings JSONB,  -- [{"ticker": "AAPL", "shares": 10, "cost_basis": 1500, ...}]
    total_value DECIMAL(15,2),
    last_synced_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**fq_history**
```sql
CREATE TABLE fq_history (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    fq_score INTEGER,
    fq_knowledge INTEGER,
    fq_discipline INTEGER,
    fq_wisdom INTEGER,
    UNIQUE(user_id, date)
);
```

**exercises**
```sql
CREATE TABLE exercises (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    category VARCHAR(100),  -- "Tax Optimization", "Behavioral Finance", etc.
    difficulty VARCHAR(50),  -- "Beginner", "Intermediate", "Advanced"
    content TEXT,  -- Markdown
    question TEXT,
    options JSONB,  -- [{"id": "A", "text": "..."}, ...]
    correct_answer VARCHAR(10),
    explanation TEXT,
    fq_points INTEGER DEFAULT 10,
    estimated_time_minutes INTEGER
);
```

**user_exercise_completions**
```sql
CREATE TABLE user_exercise_completions (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exercise_id UUID REFERENCES exercises(id),
    completed_at TIMESTAMP DEFAULT NOW(),
    selected_answer VARCHAR(10),
    is_correct BOOLEAN,
    time_spent_seconds INTEGER,
    next_review_at TIMESTAMP  -- for spaced repetition
);
```

**daily_briefings**
```sql
CREATE TABLE daily_briefings (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    content TEXT,  -- Markdown
    action_items JSONB,  -- [{"type": "rebalance", "trades": [...]}]
    generated_at TIMESTAMP DEFAULT NOW(),
    read_at TIMESTAMP,
    UNIQUE(user_id, date)
);
```

**theses**
```sql
CREATE TABLE theses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    holdings JSONB,  -- ["NVDA", "MSFT", "GOOGL"]
    core_beliefs TEXT[],
    falsification_criteria TEXT[],
    target_allocation DECIMAL(4,3),
    status VARCHAR(50) DEFAULT 'INTACT',  -- STRENGTHENING / INTACT / WEAKENING / BROKEN
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated_at TIMESTAMP
);
```

#### 6.3 Tech Stack Details

**Frontend:**
- Next.js 14 (App Router, RSC)
- TypeScript (strict mode)
- Tailwind CSS (already used on gazillioner.com)
- React Query (data fetching, caching)
- Zustand (client state management)
- Chart.js / Recharts (visualizations)
- Framer Motion (animations)

**Backend:**
- Node.js 20 LTS + Express (API server)
- Python 3.11 + FastAPI (AI inference service)
- PostgreSQL 15 (primary database)
- Redis 7 (caching, sessions, pub/sub for real-time)
- Prisma (TypeScript ORM)

**AI/ML:**
- OpenAI GPT-4 Turbo (daily briefing generation)
- PyTorch 2.0 (fine-tuning custom models later)
- ONNX Runtime (quantized inference)
- Existing 3.5-bit quantization code (from repo)

**Infrastructure:**
- AWS (ECS for containers, RDS for Postgres, ElastiCache for Redis)
- Docker + Docker Compose (local dev)
- GitHub Actions (CI/CD)
- Cloudflare (CDN, DDoS protection)
- Sentry (error tracking)
- PostHog (product analytics)

**External Services:**
- Plaid (broker integrations) - $0.60 per connected item/month
- IEX Cloud (stock prices) - $9/month (hobbyist tier)
- CoinGecko API (crypto prices) - Free tier OK for MVP
- SendGrid (transactional emails) - Free tier (100/day)
- Stripe (payments) - 2.9% + $0.30 per transaction

---

### 7. MVP Development Plan (4 Months)

#### Month 1: Foundation

**Week 1-2: Setup & Auth**
- [ ] Initialize Next.js project
- [ ] Set up PostgreSQL + Prisma schema
- [ ] Implement NextAuth.js (email/password + OAuth)
- [ ] Design system (Tailwind components)
- [ ] CI/CD pipeline (GitHub Actions → AWS ECS)

**Week 3-4: Broker Integrations**
- [ ] Plaid integration (connect stock brokers)
- [ ] Coinbase API integration (crypto exchange)
- [ ] Manual entry fallback
- [ ] Portfolio sync service (fetch holdings daily)
- [ ] Portfolio dashboard UI

**Deliverables:**
- Users can sign up, log in, connect accounts
- Portfolio syncs and displays accurately

---

#### Month 2: FQ System & Daily Briefing

**Week 1-2: FQ Assessment**
- [ ] Write 30 assessment questions (10 Knowledge, 10 Discipline, 10 Wisdom)
- [ ] Build quiz UI (one question per page, progress bar)
- [ ] Implement scoring algorithm
- [ ] Results page (FQ breakdown, percentile, tier)
- [ ] Goal-setting flow

**Week 3-4: Daily Briefing**
- [ ] AI briefing generation service (GPT-4 integration)
- [ ] Template system (portfolio snapshot, action items, learning, thesis check)
- [ ] Cron job (6 AM daily generation)
- [ ] Email delivery (SendGrid)
- [ ] Briefing dashboard UI

**Deliverables:**
- Users complete FQ assessment, get personalized score
- Daily briefing generates and delivers every morning

---

#### Month 3: Portfolio Features

**Week 1-2: Tax-Loss Harvesting**
- [ ] Algorithm: Find harvesting opportunities
- [ ] Similar securities database (SPY→VOO, etc.)
- [ ] Wash sale tracking
- [ ] Tax savings calculation
- [ ] UI: Review and execute trades

**Week 3-4: Rebalancing & Thesis Tracker**
- [ ] Rebalancing algorithm (drift detection, trade generation)
- [ ] Tax-aware trade selection
- [ ] Thesis CRUD (create, read, update, delete)
- [ ] Monthly thesis update generation (AI + NewsAPI)
- [ ] Behavioral alert detection (FOMO, panic, etc.)

**Deliverables:**
- Tax-loss harvesting suggestions work end-to-end
- Rebalancing suggestions accurate
- Users can create and track theses

---

#### Month 4: Learning & Polish

**Week 1-2: Learning Platform**
- [ ] Write 50 exercises (content + questions + explanations)
- [ ] Build exercise UI (question → answer → feedback)
- [ ] Spaced repetition scheduling
- [ ] Write 10 case studies (historical scenarios)
- [ ] Case study UI (timeline, decision points, outcomes)

**Week 3: Beta Testing & Bug Fixes**
- [ ] Recruit 50 alpha testers (friends, family)
- [ ] Fix critical bugs
- [ ] Performance optimization (page load <2s)
- [ ] Security audit (basic: SQL injection, XSS, CSRF)

**Week 4: Launch Prep**
- [ ] Legal: Privacy policy, Terms of Service (lawyer review)
- [ ] Stripe integration (subscription payments)
- [ ] Onboarding flow polish (7-day activation sequence)
- [ ] Launch landing page update
- [ ] Press kit (screenshots, demo video, pitch)

**Deliverables:**
- 50 exercises + 10 case studies live
- Private alpha with 50 users (feedback collected)
- Ready for closed beta launch

---

### 8. MVP Success Metrics (Month 6)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Paying Users** | 2,000 | Stripe subscriptions |
| **MRR** | $50k+ | Monthly recurring revenue |
| **ARR (Run-Rate)** | $600k+ | MRR × 12 |
| **Churn Rate** | <5%/month | Cancellations / active users |
| **DAU/MAU** | >70% | Daily logins / monthly actives |
| **NPS** | >40 | Net Promoter Score survey |
| **Onboarding Completion** | >85% | % who complete FQ assessment + connect account |
| **Daily Briefing Open Rate** | >70% | Email opens + dashboard views |
| **Exercise Completion** | >60% | % of users who complete ≥1 exercise/week |
| **Avg. FQ Improvement** | +50 pts | Avg. FQ gain in first 90 days |
| **User-Reported Savings** | $1,000+ | Tax savings + avoided mistakes (surveyed) |

---

### 9. Budget (4-Month MVP)

| Category | Cost | Notes |
|----------|------|-------|
| **Engineering** | $120k | 2 full-stack engineers × $15k/month × 4 months |
| **AI/ML Engineer** | $25k | Contractor for AI integration (part-time) |
| **Product/Design** | $20k | Contractor for UX/UI design |
| **DevOps/Infrastructure** | $8k | AWS (ECS, RDS, ElastiCache): $2k/month |
| **External APIs** | $2k | Plaid, IEX Cloud, OpenAI, SendGrid |
| **Legal/Compliance** | $15k | Privacy policy, terms, regulatory review |
| **Marketing (Beta)** | $5k | Influencer partnerships, ads for beta testers |
| **Miscellaneous** | $5k | Tools (GitHub, Sentry, PostHog, etc.) |
| **Total** | **$200k** | 4-month runway to closed beta (500 users) |

---

### 10. Post-MVP Roadmap (Phase 2 - Months 5-8)

**Features to Add:**
- Options strategies (covered calls, protective puts)
- Mobile apps (iOS, Android via React Native)
- Community forum (anonymous, education-focused)
- Leaderboards (opt-in, privacy-preserving)
- 200 more exercises (total 250)
- 40 more case studies (total 50)
- On-device AI inference (Tier 2 hardware device)
- Multi-account optimization (taxable, IRA, 401k)

**Growth Initiatives:**
- Scale to 10,000 users (from 2,000)
- Raise Series A ($5-10M) OR continue bootstrapped
- Hire: 2 more engineers, 1 growth marketer, 1 customer success
- Partnerships: Ledger/Trezor (hardware wallets), TurboTax (tax integration)

---

### 11. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **Plaid API unreliable** | Implement retry logic, fallback to manual entry, multi-provider support |
| **GPT-4 costs too high** | Fine-tune cheaper model (Llama 3), cache common briefings, use templates |
| **Low engagement** | A/B test notifications, improve onboarding, gamification tweaks |
| **Users don't pay** | Free tier with limits, 30-day trial, show tangible savings early |
| **Regulatory issues** | Legal review, stay educational (not advice), compliance monitoring |
| **Data breach** | Penetration testing, SOC 2 certification, bug bounty, insurance |

---

### 12. Go / No-Go Checklist (Before Public Launch)

**Must Have (Blockers):**
- [ ] FQ assessment works (scoring accurate)
- [ ] Daily briefing generates reliably (100% uptime)
- [ ] Broker integrations work (Plaid + 1 crypto exchange)
- [ ] Tax-loss harvesting accurate (no wash sale violations)
- [ ] Payment system works (Stripe subscriptions)
- [ ] Legal docs reviewed (privacy policy, terms)
- [ ] Security audit passed (no critical vulnerabilities)

**Nice to Have (Can Launch Without):**
- [ ] 50 exercises (can launch with 30)
- [ ] 10 case studies (can launch with 5)
- [ ] Behavioral alerts (can add post-launch)
- [ ] Thesis tracker (can add post-launch)

**Success Criteria (Private Alpha → Closed Beta):**
- [ ] 50 alpha users tested for 2 weeks
- [ ] NPS >30 (from alpha testers)
- [ ] <10 critical bugs reported
- [ ] At least 5 testimonials collected

---

## Next Steps

**Week 1 (Immediate):**
1. Get stakeholder approval for MVP scope and budget
2. Recruit engineering team (2 full-stack + 1 AI/ML contractor)
3. Finalize tech stack decisions
4. Set up project repo, CI/CD, infrastructure

**Week 2-4:**
- Sprint 1: Authentication, database, broker integrations
- Daily standups, weekly demos to stakeholders

**Month 2-4:**
- Continue 2-week sprints
- Monthly stakeholder reviews
- Recruit alpha testers (Week 12)

**Month 5-6:**
- Closed beta (500 users)
- Iterate based on feedback
- Prepare for public launch (PR, partnerships)

---

**Document Owner:** Product Team
**Last Updated:** December 10, 2025
**Status:** Ready for Approval

---

*This MVP specification is the source of truth for the first 4 months of development. All feature requests must be evaluated against this scope to prevent scope creep.*
