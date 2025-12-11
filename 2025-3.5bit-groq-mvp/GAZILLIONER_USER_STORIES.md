# User Stories - Gazillioner FQ Platform
## AI-Powered Wealth Advisor

**Version:** 1.0
**Date:** December 10, 2025
**Website:** https://gazillioner.com/

---

## Story Format

Each user story follows this template:
```
As a [persona],
I want to [action],
So that [benefit].

Acceptance Criteria:
- Given [context]
- When [action]
- Then [expected outcome]

Priority: [Critical / High / Medium / Low]
Story Points: [1, 2, 3, 5, 8, 13]
Sprint: [MVP / Phase 2 / Future]
```

---

## Epic 1: User Onboarding & FQ Assessment

### US-001: Initial FQ Assessment

**As a** new user,
**I want to** complete an initial FQ assessment quiz,
**So that** I can understand my current financial decision-making capability and get a baseline score.

**Acceptance Criteria:**
- Given I've just signed up for Gazillioner
- When I navigate to the onboarding flow
- Then I see a 30-question quiz covering Knowledge, Discipline, and Wisdom
- And the quiz takes 15 minutes or less to complete
- And I receive my FQ score (0-1000) immediately after completion
- And I see a breakdown by category (Knowledge /300, Discipline /400, Wisdom /300)
- And I receive personalized suggestions for improvement

**Priority:** Critical
**Story Points:** 8
**Sprint:** MVP

**Technical Notes:**
- Quiz questions stored in PostgreSQL
- Scoring algorithm calculates weighted average across categories
- UI shows progress bar during quiz
- Results page shows visual breakdown (pie chart, bar graphs)

---

### US-002: Personalized Onboarding Journey

**As a** new user,
**I want to** go through a personalized onboarding flow based on my FQ assessment,
**So that** I can quickly get value from the platform tailored to my specific needs.

**Acceptance Criteria:**
- Given I've completed my FQ assessment
- When I proceed through onboarding
- Then I'm asked to set my financial goals (retirement age, major purchases, income needs)
- And I'm asked to input my risk tolerance (quiz-based, 1-10 scale)
- And I'm guided to connect at least one broker or exchange account
- And I complete my first exercise within 5 minutes
- And I receive my first daily briefing the next morning
- And the entire onboarding takes <15 minutes

**Priority:** Critical
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Multi-step wizard with progress tracking
- Risk tolerance quiz (10 questions)
- Plaid integration for broker connections
- Automatic exercise assignment based on lowest FQ category
- Onboarding completion tracked in user profile

---

### US-003: Goal Setting

**As a** user,
**I want to** define my long-term financial goals with specific timelines and amounts,
**So that** the AI can provide personalized advice aligned with my objectives.

**Acceptance Criteria:**
- Given I'm in the goal-setting step of onboarding (or accessing settings later)
- When I add a financial goal
- Then I can specify:
  - Goal type (retirement, house purchase, emergency fund, college fund, etc.)
  - Target amount (e.g., $2.5M for retirement)
  - Target date (e.g., age 60, year 2027)
  - Priority (critical, important, nice-to-have)
- And the system calculates required monthly savings to reach the goal
- And I see a visual timeline showing all my goals
- And I can add up to 5 goals initially

**Priority:** High
**Story Points:** 5
**Sprint:** MVP

---

## Epic 2: Daily Briefing & Insights

### US-004: Morning Daily Briefing

**As a** user,
**I want to** receive a personalized daily briefing every morning,
**So that** I know exactly what actions to take today and understand my long-term progress.

**Acceptance Criteria:**
- Given I'm a registered user with at least one connected account
- When it's 6 AM in my local timezone
- Then I receive a daily briefing via:
  - Email (if opted in)
  - App push notification
  - In-app dashboard (always available)
- And the briefing includes:
  - Portfolio snapshot (current value, % change, allocation)
  - Action items (max 3: rebalance, tax harvest, contribute)
  - Learning moment (1 concept, 2-minute read)
  - Thesis check (1-2 of my holdings, is the thesis intact?)
  - Behavioral alert (if applicable: FOMO/panic detected)
- And the briefing is <500 words (2-minute read)
- And I can mark action items as complete

**Priority:** Critical
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Background job runs at 6 AM user timezone
- AI generates briefing using portfolio data + market context + user goals
- Template system with dynamic content blocks
- Email rendered using MJML (responsive email framework)
- Push notifications via Firebase Cloud Messaging

---

### US-005: Thesis Tracking

**As a** user,
**I want to** define and track my long-term investment theses,
**So that** I can maintain conviction through volatility and know when my thesis has changed.

**Acceptance Criteria:**
- Given I hold an investment (e.g., NVDA, Bitcoin)
- When I navigate to "My Theses" page
- Then I can create a new thesis by specifying:
  - Name (e.g., "AI Revolution")
  - Related holdings (NVDA, MSFT, GOOGL)
  - Core beliefs (10-year TAM expansion, winner-take-most, etc.)
  - Falsification criteria (what would prove me wrong?)
  - Appropriate allocation (% of portfolio)
- And the system monitors news, earnings, and relevant data
- And I receive monthly thesis updates (strengthening / intact / weakening / broken)
- And I'm alerted if major events challenge my thesis
- And I can review my thesis history (when I created it, how it's evolved)

**Priority:** High
**Story Points:** 8
**Sprint:** MVP

**Technical Notes:**
- Thesis stored in PostgreSQL with JSON metadata
- News monitoring via NewsAPI or similar
- AI analyzes news sentiment and relevance to thesis
- Monthly cron job generates thesis updates
- Alert system triggers on major contradicting events

---

### US-006: Behavioral Alerts

**As a** user,
**I want to** be warned when I'm about to make an emotional decision,
**So that** I can pause and reconsider before acting on FOMO or panic.

**Acceptance Criteria:**
- Given I'm logged into the platform
- When the system detects a potential emotional decision:
  - FOMO: I'm searching for or adding a position that's up 30%+ in 30 days
  - Panic: I'm considering selling a position down 20%+ with intact thesis
  - Premature profit-taking: Selling a winner that's up 50%+ with strengthening thesis
  - Loss aversion: Holding a loser down 30%+ with broken thesis
- Then I receive an immediate alert (modal popup, push notification)
- And the alert shows:
  - Pattern detected ("You're chasing a pump")
  - Historical lesson ("Remember when you FOMO'd into X? Outcome: -$5k")
  - Framework to apply ("Is the thesis sound? Or are you just chasing?")
  - Recommendation ("Wait 48 hours, do research, then decide")
- And I can dismiss or "Remind me tomorrow"
- And my response is tracked for Wisdom scoring

**Priority:** High
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Real-time detection based on user actions (search queries, watchlist additions, trade intentions)
- Historical pattern matching from user's decision journal
- Alert system with configurable cooldown (don't spam)
- A/B test different intervention styles (aggressive warning vs gentle nudge)

---

## Epic 3: Portfolio Management

### US-007: Multi-Account Portfolio Sync

**As a** user,
**I want to** connect multiple broker and exchange accounts,
**So that** I have a unified view of my entire portfolio across stocks, crypto, and options.

**Acceptance Criteria:**
- Given I'm on the "Connect Accounts" page
- When I click "Add Account"
- Then I can choose from supported providers:
  - Stock brokers: Robinhood, Schwab, Fidelity, Interactive Brokers
  - Crypto exchanges: Coinbase, Binance, Kraken
  - Manual entry (for unsupported accounts)
- And I authenticate via OAuth (Plaid for stocks, exchange API keys for crypto)
- And my portfolio syncs within 10 seconds
- And I see all holdings aggregated in one dashboard
- And I can sync on-demand or set auto-sync (daily, hourly, real-time)
- And my API keys are encrypted and stored securely

**Priority:** Critical
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Plaid integration for stock brokers
- Direct API integration for major crypto exchanges
- Encryption: API keys stored in AWS Secrets Manager or HashiCorp Vault
- Background sync job runs based on user preference
- Handle rate limits and API failures gracefully

---

### US-008: Automated Rebalancing Suggestions

**As a** user,
**I want to** receive automated suggestions when my portfolio drifts from my target allocation,
**So that** I can maintain my desired risk profile without constant monitoring.

**Acceptance Criteria:**
- Given I've set a target allocation (e.g., 70% stocks, 20% crypto, 10% bonds)
- When my portfolio drifts by more than 5% from target (e.g., stocks become 75% due to rally)
- Then I receive a rebalancing suggestion in my daily briefing
- And the suggestion specifies exact trades:
  - "Sell $2,500 of VTI (stocks)"
  - "Buy $1,500 of BTC (crypto)"
  - "Buy $1,000 of BND (bonds)"
- And the suggestion is tax-aware (prefers selling losses or long-term gains)
- And I can review the suggestion with projected outcomes
- And I can execute the trades with one click (if broker API supports trading)
- And I can dismiss or customize the suggestion

**Priority:** High
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Rebalancing algorithm calculates optimal trades
- Tax-aware logic: consider holding periods, unrealized gains/losses
- Integration with broker trading APIs (Alpaca, Interactive Brokers)
- Confirmation modal before executing trades
- Log all rebalancing actions for Discipline scoring

---

### US-009: Tax-Loss Harvesting Automation

**As a** user,
**I want to** automatically identify tax-loss harvesting opportunities,
**So that** I can reduce my tax bill without spending hours analyzing my portfolio.

**Acceptance Criteria:**
- Given I have holdings with unrealized losses
- When the system scans my portfolio daily
- Then it identifies opportunities where:
  - Position has unrealized loss >$500 (worth the effort)
  - No wash sale violation (31-day window clear)
  - Substantially similar security available (e.g., SPY → VOO)
- And I receive a notification with:
  - "Sell X shares of ABC at $Y loss"
  - "Buy X shares of DEF (similar exposure)"
  - "Tax savings: $Z (based on your tax bracket)"
- And I can execute the trades with one click
- And the system tracks wash sale windows to prevent violations
- And I see year-to-date tax savings on my dashboard

**Priority:** High
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Tax-loss harvesting algorithm with wash sale detection
- Database of substantially similar securities (SPY/VOO, QQQ/QQQM, etc.)
- User's tax bracket stored in profile (for savings calculation)
- 31-day tracking: block repurchase of sold security
- Annual tax report generation (CSV export for TurboTax)

---

### US-010: Options Strategy Recommendations

**As a** user with options trading experience,
**I want to** receive personalized options strategy suggestions for my holdings,
**So that** I can generate income or protect my positions without complex analysis.

**Acceptance Criteria:**
- Given I own 100+ shares of a stock (enough for covered calls)
- When the system analyzes my holdings
- Then it suggests options strategies:
  - Covered calls (generate income, cap upside)
  - Cash-secured puts (acquire stock at discount, earn premium)
  - Protective puts (downside protection, cost premium)
  - Collars (protect downside, limit upside, low/no cost)
- And each suggestion includes:
  - Strategy name and description
  - Recommended strike price and expiration
  - Projected outcomes (best case, worst case, expected case)
  - Probability of assignment / exercise
  - Expected return (annualized %)
- And I can filter by goal (income, protection, acquisition)
- And I can execute the trade (if supported by broker)

**Priority:** Medium
**Story Points:** 13
**Sprint:** Phase 2

**Technical Notes:**
- Options chain data from broker API or IEX Cloud
- Black-Scholes model for pricing and Greeks
- Strategy optimizer based on user goals and risk tolerance
- Backtesting: show historical performance of similar strategies
- Requires user to acknowledge options risk (options agreement)

---

## Epic 4: Learning & Education

### US-011: Daily FQ Exercises

**As a** user,
**I want to** complete short daily exercises to improve my financial knowledge,
**So that** I can systematically increase my FQ score and make better decisions.

**Acceptance Criteria:**
- Given I'm logged into Gazillioner
- When I navigate to the "Learning" tab
- Then I see 2-3 recommended exercises for today (personalized to my weaknesses)
- And each exercise:
  - Takes 5-15 minutes to complete
  - Tests a specific concept (e.g., P/E ratio interpretation, wash sale rules)
  - Provides immediate feedback on my answer
  - Explains the correct answer and why
  - Awards +5 to +50 FQ points upon completion
- And I can browse the full exercise library (300+ exercises)
- And completed exercises are marked with checkmarks
- And exercises use spaced repetition (revisit key concepts after 2, 7, 30, 90 days)

**Priority:** High
**Story Points:** 13
**Sprint:** MVP

**Technical Notes:**
- Exercise database with 300+ questions (start with 50 for MVP)
- Categories: Asset allocation, Valuation, Tax, Options, Crypto, Behavioral Finance, etc.
- Spaced repetition algorithm (similar to Anki)
- Progress tracking: % complete per category
- Gamification: streaks, badges for completing categories

---

### US-012: Historical Case Studies

**As a** user,
**I want to** study historical investment scenarios and test my decision-making,
**So that** I can learn from past patterns and avoid repeating common mistakes.

**Acceptance Criteria:**
- Given I'm in the "Learning" section
- When I select "Case Studies"
- Then I see 50+ historical scenarios:
  - "Apple 2010: Why investors sold too early"
  - "NVDA 2015: Recognizing expanding TAMs before the crowd"
  - "Bitcoin 2020: Buying during maximum fear (COVID crash)"
  - "Dotcom 2000: Distinguishing winners from hype"
- And each case study includes:
  - Timeline of events (with dates, prices, key milestones)
  - Decision points ("What would you do here?")
  - Multiple choice or free text response
  - Reveal of actual outcome
  - Lessons learned (pattern to recognize)
  - +25 to +50 FQ points for correct reasoning
- And I can filter by category (tech stocks, crypto, market crashes, etc.)
- And I track my performance (how often I'd make the right call)

**Priority:** Medium
**Story Points:** 13
**Sprint:** Phase 2

**Technical Notes:**
- Case study content stored as markdown with embedded questions
- Historical price data from APIs (for charts)
- Scoring based on reasoning quality, not just binary correct/wrong
- User answers logged for Wisdom tracking

---

### US-013: Pattern Recognition Library

**As a** user,
**I want to** access a library of recurring investment patterns,
**So that** I can recognize opportunities and risks earlier than others.

**Acceptance Criteria:**
- Given I'm in the "Learning" section
- When I access "Pattern Library"
- Then I see patterns organized by type:
  - **Opportunity Patterns:** Expanding TAMs, network effects, misunderstood innovation
  - **Risk Patterns:** Hype cycles, frauds, broken theses
  - **Market Patterns:** Bull/bear cycles, sector rotation, sentiment extremes
  - **Behavioral Patterns:** FOMO, panic, confirmation bias, loss aversion
- And each pattern includes:
  - Name and description
  - Historical examples (3-5 instances)
  - How to recognize it today (indicators, signals)
  - What to do (action framework)
  - Common mistakes (what NOT to do)
- And I can search for patterns relevant to current market conditions
- And the AI highlights patterns detected in my portfolio or watchlist

**Priority:** Medium
**Story Points:** 8
**Sprint:** Phase 2

---

## Epic 5: Community & Social Features

### US-014: Anonymous Community Forum

**As a** user,
**I want to** participate in a community of other investors learning together,
**So that** I can share insights, get feedback, and stay accountable without revealing my identity.

**Acceptance Criteria:**
- Given I'm a registered user
- When I access the "Community" tab
- Then I can:
  - Browse discussions (sorted by recent, popular, trending)
  - Post questions or share learnings (anonymous by default)
  - Reply to others' posts
  - Upvote/downvote content (quality signal)
  - Report inappropriate content (spam, stock tips, abuse)
- And I can optionally display my FQ score (badge flair)
- And I CANNOT share my portfolio details or specific holdings (enforced by moderation)
- And the focus is education, not stock tips ("Why did I miss X?" not "Buy Y now!")
- And I can mute or block users

**Priority:** Medium
**Story Points:** 13
**Sprint:** Phase 2

**Technical Notes:**
- Forum backend: Node.js + PostgreSQL (posts, comments, votes)
- Moderation: keyword filters + manual review for reported content
- Anonymous usernames: auto-generated (e.g., "Investor_4721")
- FQ badge flair (optional): shows user's FQ tier (0-200, 201-400, etc.)
- Rate limiting to prevent spam

---

### US-015: FQ Leaderboards (Opt-In)

**As a** competitive user,
**I want to** see how my FQ score ranks against others,
**So that** I can stay motivated to improve and benchmark my progress.

**Acceptance Criteria:**
- Given I've opted into leaderboards (privacy setting)
- When I access the "Leaderboard" tab
- Then I see rankings:
  - **Global:** Top 100 FQ scores (anonymous usernames)
  - **Age Group:** My ranking vs. others in my age bracket (25-30, 31-35, etc.)
  - **Friends:** My ranking vs. connected friends (if they've opted in)
  - **Biggest Gainers:** Top 20 users by FQ increase this month
- And I see my current rank and percentile (e.g., "You're #247, top 35%")
- And leaderboards refresh daily
- And I can opt out anytime (removes me from all leaderboards)
- And usernames are NEVER linked to real identities publicly

**Priority:** Low
**Story Points:** 5
**Sprint:** Phase 2

**Technical Notes:**
- Leaderboard data cached in Redis (fast reads)
- Privacy: only users who opt in are shown
- Anonymized: use generated usernames, not real names
- Age group bucketing (5-year ranges)
- Friend connections: optional social graph (users can connect accounts)

---

## Epic 6: FQ Scoring & Gamification

### US-016: Real-Time FQ Score Updates

**As a** user,
**I want to** see my FQ score update in real-time as I take actions,
**So that** I get immediate feedback and understand what behaviors improve my score.

**Acceptance Criteria:**
- Given I'm logged into Gazillioner
- When I complete an action that affects my FQ:
  - Complete an exercise (+10 FQ)
  - Rebalance portfolio when suggested (+15 FQ)
  - Harvest a tax loss (+10 FQ)
  - Hold through 20% drawdown without selling (+25 FQ)
  - Panic sell (−20 FQ)
  - FOMO buy after 30% pump (−15 FQ)
- Then my FQ score updates within 2 seconds
- And I see a notification: "+10 FQ: Completed exercise on tax-loss harvesting"
- And I see my updated rank/percentile
- And the dashboard shows a sparkline of my FQ trend (7 days, 30 days, 1 year)

**Priority:** High
**Story Points:** 8
**Sprint:** MVP

**Technical Notes:**
- FQ calculation service (Node.js microservice)
- Event-driven architecture: actions trigger FQ updates
- WebSocket for real-time dashboard updates
- Notification system (toast messages in app)

---

### US-017: Streaks & Achievements

**As a** user,
**I want to** earn badges and maintain streaks for consistent behavior,
**So that** I stay motivated and build long-term habits.

**Acceptance Criteria:**
- Given I'm using Gazillioner consistently
- When I achieve milestones:
  - **Streaks:** 7, 30, 90, 365 days of daily engagement
  - **Achievements:** "Tax Ninja" (harvest $5k+ losses), "Diamond Hands" (hold through 50% drawdown), "Options Apprentice" (first covered call), etc.
- Then I receive a congratulatory notification
- And the badge appears on my profile
- And I see progress toward locked achievements ("Millionaire: 62% there")
- And streak bonuses award extra FQ points (30 days: +10 FQ, 90 days: +50 FQ)
- And I'm warned if my streak is about to break ("Don't forget to check in today!")

**Priority:** Medium
**Story Points:** 8
**Sprint:** MVP

**Technical Notes:**
- Achievement definitions stored in database (criteria, rewards)
- Daily cron job checks for streak continuity
- Push notifications for streak warnings (evening reminder)
- Badge icons designed (or use icon library like Font Awesome)

---

### US-018: Personalized FQ Improvement Plan

**As a** user,
**I want to** receive a customized 12-month roadmap to improve my FQ,
**So that** I have clear next steps and can track my progress toward mastery.

**Acceptance Criteria:**
- Given I've completed my initial FQ assessment
- When I view my "FQ Improvement Plan"
- Then I see a personalized roadmap:
  - Current FQ: 560
  - 3-month goal: 680 (+120 points)
  - 12-month goal: 800 (+240 points)
  - Milestones broken down by phase:
    - Phase 1 (Months 1-3): Knowledge foundation (+120 FQ)
    - Phase 2 (Months 4-6): Discipline mastery (+80 FQ)
    - Phase 3 (Months 7-12): Wisdom development (+40 FQ)
- And each phase lists specific actions:
  - "Complete 10 tax optimization exercises"
  - "Maintain 90-day contribution streak"
  - "Hold through next market correction without panic selling"
- And I see progress toward each milestone
- And the plan adapts based on my actual performance (if I'm ahead/behind schedule)

**Priority:** High
**Story Points:** 8
**Sprint:** MVP

**Technical Notes:**
- Improvement plan generator (ML model or rule-based)
- Plan stored in database, updated monthly
- Progress tracking dashboard
- Adaptive plan: recalculates if user diverges from expected path

---

## Epic 7: Privacy & Security

### US-019: On-Device AI Inference (Tier 2)

**As a** privacy-conscious user,
**I want to** run the AI model locally on my own hardware,
**So that** my financial data never leaves my device.

**Acceptance Criteria:**
- Given I've purchased the Tier 2 hardware device (Jetson Nano or custom ASIC)
- When I set up the device
- Then the personalized AI model downloads to my device (encrypted)
- And all daily briefings are generated locally (no cloud API calls)
- And portfolio data is encrypted at rest on the device
- And only anonymized FQ scores sync to cloud (if I opt in)
- And I can verify no data is sent to cloud (network traffic logs available)
- And the device runs 24/7 with <10W power consumption

**Priority:** High
**Story Points:** 21
**Sprint:** Phase 2 (after cloud MVP proven)

**Technical Notes:**
- Custom 3.5-bit quantized models (from existing repo work)
- NVIDIA Jetson Nano support (for MVP hardware)
- Local inference server (Python FastAPI + ONNX Runtime)
- Encrypted model distribution (download from cloud, decrypt locally)
- Hardware wallet integration (Ledger, Trezor for crypto signing)

---

### US-020: Data Export & Deletion

**As a** user,
**I want to** export all my data or delete my account permanently,
**So that** I have full control over my information (GDPR/CCPA compliance).

**Acceptance Criteria:**
- Given I'm in my account settings
- When I click "Export Data"
- Then I receive a ZIP file within 24 hours containing:
  - Portfolio history (CSV)
  - FQ score history (JSON)
  - Exercise completion records (JSON)
  - All my theses and decision journal entries (JSON)
  - Account settings and preferences (JSON)
- When I click "Delete Account"
- Then I'm warned of consequences (data loss, subscription cancellation)
- And I must confirm via email link (prevent accidental deletion)
- And my account and ALL data is hard-deleted within 24 hours
- And I receive confirmation email

**Priority:** High
**Story Points:** 5
**Sprint:** MVP (required for GDPR/CCPA compliance)

**Technical Notes:**
- Data export: background job generates ZIP
- Data deletion: cascade delete across all tables
- 30-day soft delete (recoverable) before hard delete
- Audit log of deletion requests (for compliance)

---

## Epic 8: Mobile App (Future)

### US-021: Mobile Daily Briefing

**As a** mobile user,
**I want to** receive and read my daily briefing on my phone,
**So that** I can stay on top of my finances on the go.

**Acceptance Criteria:**
- Given I have the Gazillioner mobile app installed (iOS or Android)
- When it's 6 AM my local time
- Then I receive a push notification with the briefing headline
- And I can open the app and read the full briefing (optimized for mobile)
- And I can mark action items as complete with one tap
- And I can execute suggested trades (if broker supports mobile API)
- And the app works offline (shows cached data from last sync)

**Priority:** Medium
**Story Points:** 13
**Sprint:** Phase 3

**Technical Notes:**
- React Native (cross-platform iOS + Android)
- Push notifications via Firebase Cloud Messaging
- Offline mode: cache last 7 days of briefings
- Broker API integration (mobile SDKs)

---

### US-022: Mobile FQ Exercises

**As a** mobile user,
**I want to** complete FQ exercises on my phone during downtime,
**So that** I can improve my score during my commute or breaks.

**Acceptance Criteria:**
- Given I have the mobile app
- When I open the "Learning" tab
- Then I see today's recommended exercises (optimized for mobile)
- And I can complete exercises with swipe gestures and taps
- And exercises sync across devices (progress saved)
- And I earn FQ points immediately (with celebration animation)

**Priority:** Medium
**Story Points:** 8
**Sprint:** Phase 3

---

## Epic 9: Advanced Features (Phase 3+)

### US-023: Multi-Account Tax Optimization

**As an** advanced user with multiple account types,
**I want to** optimize my portfolio across taxable, Roth IRA, 401k, and other accounts,
**So that** I minimize taxes while maintaining my target allocation.

**Acceptance Criteria:**
- Given I have multiple accounts (e.g., taxable brokerage, Roth IRA, 401k)
- When the system analyzes my overall portfolio
- Then it suggests tax-efficient placement:
  - High-growth stocks in Roth IRA (tax-free growth)
  - Bonds in 401k (tax-deferred interest)
  - Tax-efficient ETFs in taxable (low dividends, qualified dividends)
- And it suggests rebalancing across accounts (not just within)
- And it estimates tax savings from optimal placement

**Priority:** Low
**Story Points:** 21
**Sprint:** Phase 3

---

### US-024: AI Scenario Planning

**As a** user with a major life event approaching,
**I want to** model different financial scenarios (house purchase, job change, retirement),
**So that** I can make informed decisions about my future.

**Acceptance Criteria:**
- Given I'm planning a major financial decision
- When I access "Scenario Planner"
- Then I can model scenarios:
  - "What if I buy a $600k house in 2027?"
  - "What if I retire at 55 instead of 65?"
  - "What if my income drops 30% (career change)?"
- And the system projects outcomes:
  - Impact on retirement savings
  - Required monthly savings changes
  - Probability of meeting goals
- And I can compare scenarios side-by-side
- And I can save and revisit scenarios

**Priority:** Low
**Story Points:** 13
**Sprint:** Phase 3

---

## Story Prioritization Summary

### MVP (Sprint 1-4, Months 1-4)
**Critical (Must-Have):**
- US-001: Initial FQ Assessment
- US-002: Personalized Onboarding
- US-004: Morning Daily Briefing
- US-007: Multi-Account Portfolio Sync

**High (Should-Have):**
- US-003: Goal Setting
- US-005: Thesis Tracking
- US-006: Behavioral Alerts
- US-008: Automated Rebalancing Suggestions
- US-009: Tax-Loss Harvesting Automation
- US-011: Daily FQ Exercises
- US-016: Real-Time FQ Score Updates
- US-018: Personalized FQ Improvement Plan
- US-020: Data Export & Deletion

**Medium (Nice-to-Have):**
- US-017: Streaks & Achievements

### Phase 2 (Months 5-8)
- US-010: Options Strategy Recommendations
- US-012: Historical Case Studies
- US-013: Pattern Recognition Library
- US-014: Anonymous Community Forum
- US-015: FQ Leaderboards
- US-019: On-Device AI Inference

### Phase 3 (Months 9-12+)
- US-021: Mobile Daily Briefing
- US-022: Mobile FQ Exercises
- US-023: Multi-Account Tax Optimization
- US-024: AI Scenario Planning

---

## Definition of Done

A user story is considered "Done" when:
1. ✅ Code is written and peer-reviewed
2. ✅ Unit tests pass (>80% coverage)
3. ✅ Integration tests pass
4. ✅ UI/UX reviewed and approved by design team
5. ✅ Acceptance criteria validated by product owner
6. ✅ Deployed to staging environment
7. ✅ QA testing completed (no critical bugs)
8. ✅ Documentation updated (user-facing and technical)
9. ✅ Product analytics tracking implemented
10. ✅ Deployed to production

---

## Story Point Reference

- **1 point:** Trivial (e.g., text change, simple UI tweak) - 1-2 hours
- **2 points:** Simple (e.g., new API endpoint, basic form) - 4 hours
- **3 points:** Small (e.g., CRUD feature, simple integration) - 1 day
- **5 points:** Medium (e.g., complex form, basic algorithm) - 2-3 days
- **8 points:** Large (e.g., new subsystem, external API integration) - 1 week
- **13 points:** Very Large (e.g., major feature, complex algorithm) - 2 weeks
- **21 points:** Epic (too large, should be broken down) - 3+ weeks

---

**Total Story Points (MVP):** ~140 points
**Estimated Team Velocity:** 30-40 points/sprint (2 weeks)
**Estimated MVP Timeline:** 4-5 sprints (8-10 weeks)

---

*This document is a living artifact and will be updated as product requirements evolve.*
