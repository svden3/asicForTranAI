# v0.app Prompts for Gazillioner FQ Platform
## Step-by-Step UI Generation Guide

**Purpose:** This document contains ready-to-use prompts for v0.app (Vercel's AI UI generator) to build the Gazillioner FQ platform frontend.

**How to Use:**
1. Visit https://v0.dev/
2. Copy each prompt below (one at a time)
3. Paste into v0.app chat
4. Review generated code
5. Click "Add to Codebase" to download/integrate
6. Iterate with follow-up prompts if needed

**Tech Stack (v0.app Default):**
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- shadcn/ui components

---

## Phase 1: Core Pages & Layout

### Prompt 1: Main Layout with Navigation

```
Create a modern, professional layout for a financial AI platform called "Gazillioner" with these requirements:

DESIGN STYLE:
- Dark mode primary (dark navy/black background with green and white accents)
- Professional financial aesthetic (think Bloomberg meets modern SaaS)
- Green (#10b981) as primary accent color
- White text on dark backgrounds
- Clean, spacious layout

COMPONENTS NEEDED:
1. Top Navigation Bar:
   - Logo: "Gazillioner" text with small icon ($ or brain icon)
   - Navigation links: Dashboard, Portfolio, Learning, Community, Settings
   - Right side: Notification bell icon, FQ score badge (e.g., "FQ: 560"), user avatar dropdown
   - Sticky on scroll

2. Sidebar (Desktop only, collapsible):
   - Quick stats: Portfolio value, Today's change, FQ score
   - Quick actions: "Daily Briefing", "Complete Exercise", "Rebalance"
   - Streak counter: "🔥 47 day streak"

3. Main Content Area:
   - Full width on mobile
   - Max-width container on desktop (centered)
   - Padding for readability

4. Footer:
   - Links: Privacy, Terms, Help, Contact
   - Copyright: "© 2025 Gazillioner"

RESPONSIVE:
- Mobile: Hide sidebar, show hamburger menu
- Tablet: Show navigation, hide sidebar
- Desktop: Show full layout

ACCESSIBILITY:
- High contrast (WCAG AA compliant)
- Keyboard navigation support
- Screen reader friendly

Generate the main layout component with Next.js App Router structure.
```

---

### Prompt 2: Dashboard Page (Landing After Login)

```
Create a financial dashboard page for Gazillioner with the following sections:

PAGE STRUCTURE:
Top-to-bottom layout with cards/sections:

1. HEADER GREETING:
   - "Good morning, [Name]"
   - Current date and time
   - Weather icon (optional)

2. DAILY BRIEFING CARD (Prominent, top of page):
   - Title: "Today's Briefing" with date
   - 2-3 sentence summary preview
   - "Read Full Briefing" button (green, prominent)
   - "Mark as Read" checkbox
   - Small "✨ AI Generated" badge

3. PORTFOLIO SNAPSHOT CARD:
   - Total value: Large, bold number (e.g., "$127,400")
   - Today's change: "+$1,200 (+0.95%)" in green or red based on positive/negative
   - Small sparkline chart (7-day trend)
   - "View Full Portfolio" link

4. ACTION ITEMS CARD:
   - Title: "Today's Actions (3)"
   - Checklist format:
     ☐ Rebalance: Sell $800 VTI, Buy $800 BTC
     ☐ Tax-loss harvest: Sell ARKK (-$840 loss)
     ☐ Contribute $500 to Roth IRA
   - Each item has "Review" button
   - Progress bar: "1/3 completed"

5. FQ SCORE CARD:
   - Large circular progress indicator showing FQ score (e.g., 560/1000)
   - Breakdown: Knowledge 145/300, Discipline 235/400, Wisdom 180/300
   - Small trend: "+14 points this week ↗️"
   - "View Improvement Plan" link

6. LEARNING CARD:
   - Title: "Recommended for You"
   - 2-3 exercise cards (thumbnail, title, time estimate, FQ points)
   - "Browse All Exercises" button

DESIGN:
- Card-based layout with subtle shadows
- Dark mode (dark cards on darker background)
- Green accents for positive values, red for negative
- Icons for each section (portfolio 💼, actions ✅, learning 📚)
- Skeleton loading states for async data

INTERACTIVITY:
- Hover effects on cards (slight lift, border glow)
- Click to expand cards for more details
- Smooth transitions

Generate the dashboard page component with sample data.
```

---

### Prompt 3: FQ Assessment Quiz UI

```
Create an engaging, multi-step quiz interface for the Financial Quotient (FQ) assessment with these requirements:

QUIZ FLOW:
1. Welcome screen:
   - Title: "Discover Your Financial Quotient (FQ)"
   - Subtitle: "30 questions • 15 minutes • Get your personalized score"
   - "Start Assessment" button (large, green)
   - Visual: Brain icon or graph illustration

2. Question screen (repeated for each question):
   - Progress bar at top: "Question 14/30" with 47% filled bar
   - Category badge: "Tax Optimization" or "Behavioral Finance" (small pill)
   - Question text: Large, readable (e.g., "You sold Tesla at a loss on March 1. When can you repurchase without triggering a wash sale?")
   - 4 answer options (A, B, C, D):
     - Radio button cards (full width on mobile)
     - Hover effect (border highlight)
     - Selected state (green border, checkmark)
   - Navigation: "Previous" and "Next" buttons
   - "Save & Continue Later" link (small, subtle)

3. Results screen:
   - Celebration animation (confetti or sparkles)
   - Large FQ score: "Your FQ Score: 560" (animated count-up)
   - Percentile: "You're in the top 35% of users"
   - Tier badge: "Building Wealth" (with icon)
   - Breakdown chart:
     - Knowledge: 145/300 (⭐⭐⭐⚪⚪)
     - Discipline: 235/400 (⭐⭐⭐⭐⚪)
     - Wisdom: 180/300 (⭐⭐⭐⭐⚪)
   - Insights: "You're strong in Discipline but need work on Knowledge"
   - "View Improvement Plan" button (primary CTA)
   - "Share Score" button (secondary, social sharing)

DESIGN:
- One question per screen (no scrolling)
- Clean, minimal design (reduce cognitive load)
- Dark mode with green accents
- Large touch targets for mobile
- Smooth page transitions (fade/slide)

ACCESSIBILITY:
- Keyboard navigation (arrow keys to select, Enter to submit)
- Screen reader labels
- High contrast

Generate the quiz component with state management for current question, selected answers, and score calculation.
```

---

## Phase 2: Portfolio Management

### Prompt 4: Portfolio Dashboard

```
Create a detailed portfolio management dashboard with the following components:

LAYOUT:
Top-to-bottom sections on the page:

1. PORTFOLIO SUMMARY (Top):
   - Total Value: Large, bold "$127,400"
   - Performance metrics in a row:
     - Today: "+$1,200 (+0.95%)" (green)
     - This Week: "+$3,400 (+2.7%)"
     - This Month: "+$5,800 (+4.8%)"
     - YTD: "+$18,200 (+16.4%)"
   - Small area chart showing 30-day value trend

2. ALLOCATION OVERVIEW:
   - Doughnut chart (center shows total value)
   - Segments:
     - Stocks: 72% ($91,728) - Blue
     - Crypto: 8% ($10,192) - Orange
     - Bonds: 13% ($16,562) - Purple
     - Cash: 7% ($8,918) - Gray
   - Target vs Actual comparison:
     - "Stocks: 72% (Target: 70%)" with small ⚠️ icon if drift >5%
   - "Rebalance Now" button if drift detected

3. HOLDINGS TABLE:
   Columns: Asset | Shares/Amount | Cost Basis | Current Value | Gain/Loss | % | Actions
   Rows (sample data):
   - AAPL | 50 shares | $7,500 | $9,250 | +$1,750 (+23.3%) | 7.3% | [View] [Sell]
   - BTC | 0.25 BTC | $8,000 | $10,500 | +$2,500 (+31.3%) | 8.2% | [View] [Sell]
   - NVDA | 25 shares | $12,000 | $20,000 | +$8,000 (+66.7%) | 15.7% | [View] [Sell]
   - VOO | 30 shares | $13,500 | $14,400 | +$900 (+6.7%) | 11.3% | [View] [Sell]

   Features:
   - Sortable columns (click header to sort)
   - Search/filter bar (filter by asset class, ticker)
   - Pagination (10 rows per page)
   - Expandable rows (click to see more details: purchase history, performance chart)

4. RECENT TRANSACTIONS:
   - Last 5 transactions in a list:
     - Date | Type | Asset | Amount | Price
     - "Dec 9 | Buy | BTC | 0.05 | $4,750"
     - "Dec 5 | Sell | TSLA | 10 shares | $2,400"
   - "View All Transactions" link

5. CONNECTED ACCOUNTS:
   - List of connected brokers/exchanges:
     - Robinhood (✓ Connected) [Sync Now] [Disconnect]
     - Coinbase (✓ Connected) [Sync Now] [Disconnect]
   - "Add Account" button
   - Last synced: "2 hours ago"

DESIGN:
- Dark mode, card-based layout
- Green for gains, red for losses
- Responsive table (stack columns on mobile)
- Loading skeletons for async data
- Empty state if no holdings

INTERACTIVITY:
- Hover effects on table rows
- Click row to expand details
- Live updates (WebSocket if available)

Generate the portfolio dashboard component with sample data and TypeScript types.
```

---

### Prompt 5: Tax-Loss Harvesting Modal

```
Create a modal dialog for reviewing and executing a tax-loss harvesting opportunity:

MODAL STRUCTURE:

1. HEADER:
   - Title: "Tax-Loss Harvesting Opportunity"
   - Subtitle: "Save on taxes while maintaining your portfolio exposure"
   - Close button (X in top-right)

2. OPPORTUNITY SUMMARY CARD:
   - Current Position:
     - "You own 10 shares of SPY"
     - "Purchase price: $564/share ($5,640 total)"
     - "Current price: $480/share ($4,800 total)"
     - "Unrealized loss: -$840"
   - Tax Impact:
     - "Tax savings: $202" (large, green, bold)
     - "Based on your 24% tax bracket"
   - Wash Sale Status:
     - "✓ No wash sale violation"
     - "Last sold SPY: 45 days ago"

3. REPLACEMENT RECOMMENDATION:
   - "Replace with VOO to maintain S&P 500 exposure"
   - Comparison table:
     | | SPY | VOO |
     |---|---|---|
     | Expense Ratio | 0.09% | 0.03% |
     | Assets Under Mgmt | $480B | $360B |
     | Holdings | 503 | 503 |
     | Performance (YTD) | +26.4% | +26.5% |
   - "VOO tracks the same index (S&P 500) but is legally distinct"

4. PROPOSED TRADES:
   - Trade 1: "Sell 10 shares SPY @ $480" → "Proceeds: $4,800"
   - Trade 2: "Buy 10 shares VOO @ $481" → "Cost: $4,810"
   - Net cost: "$10" (fees/spread)

5. CONFIRMATION CHECKLIST:
   - ☑ "I understand I'm selling SPY at a loss to harvest tax savings"
   - ☑ "I understand VOO provides similar (but not identical) exposure"
   - ☑ "I will not repurchase SPY for 31 days to avoid wash sale"

6. ACTIONS:
   - Primary button: "Execute Trades" (green, large)
   - Secondary button: "Save for Later" (gray outline)
   - Link: "Learn More About Tax-Loss Harvesting"

DESIGN:
- Modal overlay (dark semi-transparent background)
- Card-style modal (max-width 600px, centered)
- Dark mode (dark card on darker overlay)
- Icons: ✓ for checkmarks, ⚠️ for warnings, 💰 for tax savings
- Smooth entrance animation (fade + scale)

INTERACTIVITY:
- Checkbox validation (must check all boxes to enable "Execute" button)
- Loading state on button click
- Success toast after execution: "✓ Trades executed. Tax loss of $840 harvested."

Generate the tax-loss harvesting modal component with form validation.
```

---

## Phase 3: Learning Platform

### Prompt 6: Exercise Browser & Card

```
Create a learning platform interface for browsing and completing FQ exercises:

PAGE LAYOUT:

1. HEADER:
   - Title: "FQ Exercises"
   - Subtitle: "Build your financial knowledge • Earn FQ points"
   - Filter bar:
     - Category dropdown: "All Categories", "Tax Optimization", "Behavioral Finance", etc.
     - Difficulty: "All Levels", "Beginner", "Intermediate", "Advanced"
     - Status: "All", "Not Started", "In Progress", "Completed"
   - Search box: "Search exercises..."

2. EXERCISE GRID:
   Display exercises as cards in a 3-column grid (responsive: 1 col mobile, 2 col tablet, 3 col desktop)

   Each card shows:
   - Category badge (top-left, small pill): "Tax Optimization"
   - Difficulty badge (top-right): "Intermediate" with star rating ⭐⭐⭐
   - Title: "Understanding Wash Sale Rules"
   - Description: "Learn how the IRS wash sale rule works and how to avoid violations" (2 lines max, ellipsis)
   - Metadata row:
     - Time estimate: "⏱ 8 minutes"
     - FQ points: "+10 FQ"
     - Status: "Not Started" or "✓ Completed" or "In Progress"
   - Button: "Start Exercise" (green) or "Review" (gray outline) if completed
   - Hover effect: Lift card slightly, show subtle shadow

3. EXERCISE DETAIL MODAL (when clicked):

   Step 1 - Content:
   - Title: "Understanding Wash Sale Rules"
   - Progress: "1/3 - Learning" (progress bar)
   - Content section:
     - Markdown text (200-300 words explaining the concept)
     - Code example or diagram if relevant
     - "Next: Question" button

   Step 2 - Question:
   - Progress: "2/3 - Question"
   - Question text: "You sold 10 shares of Tesla at a $2,000 loss on March 1. When is the earliest you can repurchase Tesla without triggering a wash sale?"
   - Answer options (4 radio buttons):
     - A) March 15 (14 days later)
     - B) April 1 (31 days later)
     - C) May 1 (61 days later)
     - D) You can never buy Tesla again
   - "Submit Answer" button

   Step 3 - Feedback:
   - Progress: "3/3 - Complete"
   - Result: "✓ Correct!" (green) or "✗ Incorrect" (red)
   - Explanation: Detailed explanation of correct answer (200 words)
   - FQ points earned: "+10 FQ" with celebration animation
   - Next steps:
     - "Next Exercise" button (suggests related exercise)
     - "Back to Library" button
   - Spaced repetition reminder: "We'll quiz you again in 7 days for retention"

DESIGN:
- Dark mode, card-based
- Green for correct answers, red for incorrect
- Category color coding (Tax = blue, Behavioral = purple, Crypto = orange)
- Smooth transitions between modal steps

INTERACTIVITY:
- Filter updates grid in real-time (no page reload)
- Modal keyboard shortcuts (ESC to close, Enter to submit)
- Progress saved automatically (localStorage or API)

Generate the exercise browser page and modal component with sample exercises data.
```

---

### Prompt 7: Historical Case Study Viewer

```
Create an interactive historical case study viewer for learning from past investment scenarios:

LAYOUT:

1. CASE STUDY HEADER:
   - Title: "Apple 2010-2024: Why Investors Sold Too Early"
   - Category badge: "Pattern Recognition"
   - Difficulty: "Intermediate"
   - FQ Points: "+50"
   - Estimated time: "15 minutes"

2. TIMELINE VISUALIZATION:
   - Horizontal timeline from 2010 to 2024
   - Key events marked with icons/dots:
     - 2010: iPhone 4 launch (📱)
     - 2011: Steve Jobs passes (💀)
     - 2013: Market saturation fears (📉)
     - 2016: First sales decline (⚠️)
     - 2024: Stock at $180 (🚀)
   - Stock price line chart overlaid on timeline
   - Zoomable/scrollable on mobile

3. DECISION POINTS (Interactive):

   Decision Point 1: October 2011 (Jobs Dies)
   - Context card:
     - "Steve Jobs passed away. Stock fell from $18 to $13 (-28%)."
     - "Media headline: 'Can Apple Survive Without Jobs?'"
     - "Stock price chart showing decline"
   - Question: "What would you do?"
   - Answer options (4 buttons):
     - "Sell immediately (founder risk too high)"
     - "Hold but stop adding"
     - "Add to position (ecosystem moat intact)"
     - "Go all-in (maximum opportunity)"
   - User selects → Show outcome:
     - "Most investors chose: Sell or Hold"
     - "What actually happened: Stock went 13× from here"
     - "Why selling was wrong: [explanation]"
     - "Pattern learned: Founder departure ≠ broken thesis if moat intact"
     - FQ impact: "+15 FQ" if correct reasoning

   Decision Point 2: 2013 (Saturation Fears)
   - [Similar structure]

   Decision Point 3: 2016 (Sales Decline)
   - [Similar structure]

4. LESSONS SUMMARY:
   - Title: "What You Learned"
   - Key patterns identified:
     - "✓ Moat strength > founder dependency"
     - "✓ New revenue streams (Services) can offset mature products"
     - "✓ Temporary setbacks ≠ broken thesis"
   - Application: "Which current stock is most like Apple 2010?"
     - Multiple choice with analysis
   - Final FQ award: "+50 FQ" (if completed all decision points correctly)

5. RELATED CASE STUDIES:
   - "Similar patterns:"
     - "Netflix 2011: Qwikster Disaster" (card preview)
     - "Microsoft 2014: Cloud Transformation" (card preview)

DESIGN:
- Story-driven layout (scroll to progress through timeline)
- Timeline visualization (interactive, animated)
- Decision point cards (prominent, full-width)
- Dark mode with gold/yellow accent for historical theme
- Historical stock price charts embedded

INTERACTIVITY:
- Click timeline events to jump to that section
- Decision reveals outcome with smooth animation
- Progress saved (can resume later)
- Share button: "Share your result (got 2/3 decisions right!)"

Generate the case study viewer component with Apple 2010 example data and timeline visualization.
```

---

## Phase 4: Daily Briefing

### Prompt 8: Daily Briefing Page

```
Create a daily briefing page that displays the AI-generated personalized financial insights:

PAGE STRUCTURE:

1. HEADER:
   - Date: "Monday, December 10, 2025"
   - Greeting: "Good morning, Alex"
   - Status badge: "✨ AI Generated at 6:02 AM"
   - Actions:
     - "Mark as Read" checkbox
     - "Email This" button
     - "⋮" menu (Archive, Share)

2. PORTFOLIO SNAPSHOT CARD:
   - Title: "Your Portfolio"
   - Current value: "$127,400" (large, bold)
   - Change metrics:
     - Today: "+$1,200 (+0.95%)" (green, with ↗️ icon)
     - YTD: "+$18,200 (+16.4%)"
   - Allocation status:
     - "Your allocation: 72% stocks, 8% crypto, 13% bonds, 7% cash"
     - "Target: 70% / 10% / 15% / 5%"
     - Visual bars showing current vs target
     - "⚠️ Stocks 2% overweight" (yellow warning)

3. TODAY'S ACTIONS SECTION:
   - Title: "Today's Actions (3)"
   - Numbered list with expandable cards:

   Action 1: Rebalance
   - Summary: "Sell $800 VTI, buy $800 BTC (back to target allocation)"
   - Why: "Your stocks rallied to 72% (target: 70%). Crypto is 8% (target: 10%)."
   - Tax impact: "Selling long-term gains (15% tax on $85 profit = $13)"
   - Buttons: [Review Details] [Execute]

   Action 2: Tax-Loss Harvest
   - Summary: "Sell ARKK (-$840 loss) → save $202 in taxes"
   - Why: "ARKK is down 21% from your purchase. No wash sale violation (bought 45 days ago)."
   - Replacement: "Buy QQQM to maintain tech exposure"
   - Buttons: [Review Details] [Execute]

   Action 3: Contribute
   - Summary: "Auto-invest $500 to Roth IRA today"
   - Why: "Your monthly contribution schedule (2nd Monday of each month)"
   - Progress: "You've contributed $3,500 of $7,000 yearly limit (50%)"
   - Buttons: [Skip This Month] [Confirm]

4. LEARNING MOMENT SECTION:
   - Title: "📚 Today's Learning: Why People Sold Apple Too Early"
   - Content: Markdown-formatted text (200-300 words)
   - Example:
     ```
     In 2011, Steve Jobs died. Apple stock fell 30%. Many investors sold,
     thinking "Apple is done without Jobs."

     Reality: Tim Cook was an excellent operator. iPhone kept growing.
     Services revenue exploded. Stock went up 13× from that low.

     Lesson: **Founder risk is real, but ecosystem moats often survive
     leadership changes.**

     Pattern to remember: "Founder departure ≠ broken thesis if moat intact"
     ```
   - "Read Full Case Study" link

5. THESIS CHECK SECTION:
   - Title: "📊 Thesis Check: Your NVDA Holding"
   - Your thesis (user-defined): "AI revolution will drive data center demand"
   - This week's evidence:
     - ✓ Data center revenue up 217% YoY (accelerating)
     - ✓ Guidance beat expectations
     - ✗ Stock fell 8% (sentiment, not fundamentals)
   - Verdict: "Thesis STRENGTHENING" (green badge with ↗️)
   - Recommendation: "HOLD (or add on dips)"
   - "Review Full Thesis" link

6. BEHAVIORAL ALERT (conditional):
   - Title: "⚠️ Behavioral Alert"
   - Pattern detected: "You searched for [MEMECOIN] after it pumped 100%"
   - Historical lesson: "Remember when you FOMO'd into SHIB in 2021? Result: -$3,200"
   - Framework: "Ask yourself: Do I understand the 10-year thesis, or am I chasing?"
   - Recommendation: "Wait 48 hours. Research first. If still interested, allocate <1% of portfolio."
   - Dismiss button

7. FOOTER:
   - "See you tomorrow! 👋"
   - Feedback buttons: "👍 Helpful" "👎 Not Useful"
   - "View Previous Briefings" link

DESIGN:
- Long-form reading layout (like a newsletter)
- Card-based sections (clearly separated)
- Dark mode with green accents
- Readable typography (18px body text, generous line height)
- Markdown rendering for content
- Icons for each section (💼 portfolio, ✅ actions, 📚 learning, 📊 thesis)

INTERACTIVITY:
- Expand/collapse action cards
- Inline execution of actions (modal confirmation)
- Feedback recorded (for AI improvement)
- Bookmark/save briefing for later

Generate the daily briefing page component with sample data and markdown rendering.
```

---

## Phase 5: Onboarding Flow

### Prompt 9: Multi-Step Onboarding Wizard

```
Create a multi-step onboarding wizard for new users:

WIZARD STRUCTURE:

Progress bar at top: Shows steps 1/5, 2/5, etc.
Steps: Welcome → FQ Assessment → Goals → Connect Accounts → Complete

---

STEP 1: Welcome
- Hero section:
  - Title: "Welcome to Gazillioner! 👋"
  - Subtitle: "Your AI-powered wealth advisor that helps you avoid costly mistakes"
- Benefits list (with icons):
  - 💰 "Tax optimization: Save thousands annually"
  - 📈 "Behavioral coaching: Avoid FOMO and panic selling"
  - 🧠 "FQ scoring: Track your financial decision-making ability"
  - 📚 "Daily learning: 5 minutes a day to build wealth"
- "Get Started" button (large, green)
- "Already have an account? Log in" link

---

STEP 2: FQ Assessment
- Title: "Let's discover your Financial Quotient (FQ)"
- Subtitle: "30 questions • 15 minutes • No judgment, just insight"
- "Start Assessment" button
- [Launches FQ quiz from Prompt 3]
- After completion: "Next: Set Your Goals" button

---

STEP 3: Goal Setting
- Title: "What are your financial goals?"
- Subtitle: "We'll personalize your daily briefings based on these goals"
- Form with up to 3 goals:

  Goal 1 (required):
  - Goal type: Dropdown (Retirement, House, Emergency Fund, Pay Off Debt, Other)
  - Target amount: "$2,500,000"
  - Target date: Date picker (Year 2055, Age 60)
  - Priority: Radio buttons (Critical, Important, Nice-to-have)
  - Auto-calculated: "You need to save $3,200/month to reach this goal"

  Goal 2 (optional): [Same fields]
  Goal 3 (optional): [Same fields]

- Visual: Timeline showing all goals
- "Continue" button

---

STEP 4: Connect Accounts
- Title: "Connect your investment accounts"
- Subtitle: "We'll sync your portfolio to provide personalized advice"
- Connection options (card grid):

  Card 1: Stock Brokers (via Plaid)
  - Logo: Plaid logo
  - "Connect Robinhood, Schwab, Fidelity, and 100+ brokers"
  - "Connect via Plaid" button
  - [Opens Plaid Link modal]

  Card 2: Crypto Exchanges
  - Logo: Coinbase, Binance logos
  - "Connect Coinbase, Binance, Kraken"
  - "Connect Exchange" button
  - [Opens API key input modal]

  Card 3: Manual Entry
  - Icon: ✍️
  - "Enter holdings manually"
  - "Add Manually" button

- Connected accounts list (appears after connection):
  - "✓ Robinhood connected (10 holdings)"
  - "✓ Coinbase connected (3 holdings)"
- "Skip for now" link (allow later connection)
- "Continue" button

---

STEP 5: Complete
- Celebration animation (confetti)
- Title: "You're all set! 🎉"
- Summary:
  - "Your FQ Score: 560"
  - "Goals set: 2"
  - "Accounts connected: 2"
  - "Portfolio value: $127,400"
- Next steps preview:
  - "✓ Your first daily briefing arrives tomorrow at 6 AM"
  - "✓ Complete 3 exercises this week to improve your FQ"
  - "✓ Review your personalized 12-month improvement plan"
- "Go to Dashboard" button (large, primary)
- "Start a quick tour" link (optional tutorial)

---

DESIGN:
- Full-screen wizard (no navigation bar)
- Progress bar at top (sticky)
- "Back" button on all steps except first
- Dark mode, clean and minimal
- Each step centered, max-width 600px
- Smooth transitions between steps (slide animation)

VALIDATION:
- Step 2 requires FQ assessment completion
- Step 3 requires at least 1 goal
- Step 4 can be skipped (but encouraged)

Generate the onboarding wizard component with step navigation and validation.
```

---

## Phase 6: Settings & Profile

### Prompt 10: Settings Page

```
Create a comprehensive settings page with tabbed navigation:

LAYOUT:
Sidebar tabs (left) + Content area (right):

---

TAB 1: Profile
- Section: Basic Info
  - Avatar upload (circular, with default if not set)
  - Name: [Input field]
  - Email: [Input field] (verified badge if verified)
  - Change Password: [Button opens modal]

- Section: Preferences
  - Timezone: [Dropdown]
  - Daily briefing time: [Time picker] (default 6:00 AM)
  - Email notifications: [Checkboxes]
    - ☑ Daily briefing
    - ☑ Action item reminders
    - ☐ Weekly summary
    - ☐ FQ milestones
  - Push notifications: [Checkboxes]

- Section: Financial Info
  - Tax bracket: [Dropdown] (22%, 24%, 32%, 35%, 37%)
  - Risk tolerance: [Slider 1-10] (1=Conservative, 10=Aggressive)
  - Investment experience: [Dropdown] (Beginner, Intermediate, Advanced)

---

TAB 2: Portfolio
- Section: Target Allocation
  - Sliders for each asset class (must sum to 100%):
    - Stocks: [Slider] 70%
    - Crypto: [Slider] 10%
    - Bonds: [Slider] 15%
    - Cash: [Slider] 5%
  - Visual: Pie chart updates in real-time
  - Rebalancing threshold: [Slider] (alert when drift > 5%)

- Section: Connected Accounts
  - List of connections:
    - Robinhood (✓ Connected) | Last synced: 2 hours ago | [Sync Now] [Disconnect]
    - Coinbase (✓ Connected) | Last synced: 1 day ago | [Sync Now] [Disconnect]
  - "Add Account" button

- Section: Tax Settings
  - Tax-loss harvesting: [Toggle ON/OFF]
  - Wash sale tracking: [Toggle ON/OFF] (always recommended ON)
  - Tax year: 2025 (read-only, auto-detected)

---

TAB 3: Goals
- List of current goals (editable):

  Goal 1: Retire at 60
  - Target: $2.5M by 2055
  - Priority: Critical
  - Progress: $127k / $2.5M (5%)
  - [Edit] [Delete]

  Goal 2: Buy house
  - Target: $120k down payment by 2027
  - Priority: Important
  - Progress: $12k / $120k (10%)
  - [Edit] [Delete]

- "Add New Goal" button

---

TAB 4: Privacy & Data
- Section: Data Export
  - "Download all your data (portfolio history, FQ scores, exercises)"
  - [Download Data] button → generates ZIP file

- Section: FQ Score Visibility
  - Leaderboard opt-in: [Toggle ON/OFF]
  - "If enabled, your FQ score (not portfolio details) will be shown on anonymous leaderboards"

- Section: AI Settings
  - Daily briefing personalization: [Toggle ON/OFF]
  - Behavioral alerts: [Toggle ON/OFF]
  - Data used for AI training: [Toggle ON/OFF]
  - "Your financial data is encrypted and never sold to third parties"

- Section: Account Management
  - Delete account: [Button opens modal with confirmation]
  - "Deleting your account will permanently erase all data within 24 hours"

---

TAB 5: Subscription
- Current plan: "AI Advisor - $30/month"
- Status: "Active" (green badge)
- Billing cycle: "Renews on Jan 10, 2026"
- Payment method: "Visa •••• 4242" [Update]
- Plan options:

  Tier 1: AI Advisor (Current)
  - $30/month or $300/year
  - Features list
  - [Current Plan]

  Tier 2: Privacy Edition
  - $299 device + $10/month
  - Features list
  - [Upgrade] button

- "Cancel Subscription" link (opens modal)
- Billing history: [View Invoices]

---

DESIGN:
- Dark mode, card-based sections
- Sidebar tabs (vertical on desktop, horizontal dropdown on mobile)
- Form fields with clear labels
- Toggle switches (green when ON)
- Danger zone (delete account) in red outline card at bottom

INTERACTIVITY:
- Unsaved changes warning ("You have unsaved changes" toast)
- Auto-save or manual "Save Changes" button
- Validation on all forms
- Success toast: "✓ Settings saved"

Generate the settings page with tabbed navigation and form handling.
```

---

## Phase 7: Mobile Responsive Components

### Prompt 11: Mobile Navigation & Bottom Tab Bar

```
Create mobile-optimized navigation components:

COMPONENT 1: Mobile Header (Top)
- Fixed to top of screen
- Contains:
  - Left: Hamburger menu icon (opens drawer)
  - Center: "Gazillioner" logo
  - Right:
    - Notification bell icon (with red dot if unread)
    - FQ score badge "560" (small, green)
- Height: 56px
- Background: Dark with subtle border-bottom
- z-index: 50 (always on top)

COMPONENT 2: Mobile Drawer Menu (Slides from left)
- Overlay: Semi-transparent dark background
- Drawer: Full-height sidebar (300px width)
- Contents:
  - User profile section:
    - Avatar (small)
    - Name
    - Email
    - FQ score badge
  - Navigation links:
    - Dashboard (with icon)
    - Portfolio
    - Daily Briefing
    - Learning
    - Community
    - Settings
  - Bottom section:
    - "Upgrade to Privacy Edition" card (if Tier 1)
    - Help & Support link
    - Log Out button
- Swipe gesture to close
- Tap overlay to close

COMPONENT 3: Bottom Tab Bar (Fixed to bottom)
- Fixed to bottom of screen (iOS safe area aware)
- 5 tabs in a row:

  Tab 1: Dashboard
  - Icon: 🏠 (or home icon)
  - Label: "Dashboard"
  - Active state: Green icon + text, background highlight

  Tab 2: Portfolio
  - Icon: 💼
  - Label: "Portfolio"

  Tab 3: Briefing
  - Icon: 📰 (with red dot if unread)
  - Label: "Briefing"

  Tab 4: Learning
  - Icon: 📚
  - Label: "Learn"

  Tab 5: More
  - Icon: ⋮
  - Label: "More"
  - Opens: Sheet with additional options (Community, Settings, Help)

- Height: 60px (+ safe area)
- Background: Dark with subtle border-top
- Active tab highlighted (green icon + text)

DESIGN:
- Dark mode
- Icons: Use lucide-react or heroicons
- Active state: Green (#10b981)
- Inactive state: Gray (#6b7280)
- Smooth transitions on tab change
- Haptic feedback on tap (mobile)

INTERACTIONS:
- Tap tab to navigate (Next.js router)
- Tap active tab to scroll to top
- Bottom sheet animation for "More" tab
- Badge notifications on tabs

Generate the mobile navigation components with Next.js App Router integration.
```

---

## Additional Prompts (As Needed)

### Prompt 12: Empty States

```
Create beautiful empty state components for various scenarios:

1. No Portfolio Holdings:
   - Illustration: Empty wallet or piggy bank icon
   - Title: "No investments yet"
   - Description: "Connect your brokerage account or add holdings manually to get started"
   - Primary CTA: "Connect Account" button
   - Secondary CTA: "Add Manually" link

2. No Exercises Completed:
   - Illustration: Open book or lightbulb icon
   - Title: "Start your learning journey"
   - Description: "Complete exercises to improve your FQ score and build wealth"
   - CTA: "Browse Exercises" button

3. No Daily Briefing (First Day):
   - Illustration: Calendar or sunrise icon
   - Title: "Your first briefing arrives tomorrow"
   - Description: "We'll analyze your portfolio overnight and deliver personalized insights at 6 AM"
   - CTA: "Customize Briefing Time" link

4. No Goals Set:
   - Illustration: Target or mountain icon
   - Title: "Set your financial goals"
   - Description: "Tell us what you're working towards and we'll help you get there"
   - CTA: "Add Goal" button

5. Search No Results:
   - Illustration: Magnifying glass icon
   - Title: "No results for '[search query]'"
   - Description: "Try different keywords or browse all exercises"
   - CTA: "Clear Search" button

DESIGN:
- Centered layout (vertically and horizontally)
- Icon/illustration (64px, gray or muted color)
- Title (20px, semibold)
- Description (14px, gray)
- CTA button (primary green)
- Subtle background pattern or gradient

Generate reusable empty state components with props for customization.
```

---

### Prompt 13: Loading Skeletons

```
Create loading skeleton screens for async content:

1. Portfolio Dashboard Skeleton:
   - Portfolio summary: Shimmer rectangles for value and chart
   - Holdings table: 5 rows of shimmer rectangles (ticker, value, gain/loss)
   - Allocation chart: Circular shimmer

2. Daily Briefing Skeleton:
   - Title: Shimmer line (200px width)
   - Portfolio snapshot: Shimmer card
   - 3 action item cards: Shimmer rectangles
   - Learning section: Shimmer paragraph lines

3. Exercise Card Skeleton:
   - Category badge: Shimmer pill
   - Title: Shimmer line
   - Description: 2 shimmer lines
   - Button: Shimmer rectangle

DESIGN:
- Dark mode: Shimmer from #1f2937 to #374151 (gray-800 to gray-700)
- Smooth animation (1.5s linear infinite)
- Rounded corners matching actual components
- Correct aspect ratios (match real content)

IMPLEMENTATION:
- Use CSS keyframes for shimmer animation
- Component accepts props: width, height, variant (circle, rectangle, text)
- Compose skeletons to match real layouts

Generate reusable skeleton loading components.
```

---

## Master Prompt (For Overall Architecture)

### Prompt 14: Complete Application Shell

```
Create a complete Next.js 14 application shell for Gazillioner FQ Platform with the following architecture:

PROJECT STRUCTURE:
```
gazillioner-app/
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   ├── signup/
│   │   └── onboarding/
│   ├── (dashboard)/
│   │   ├── layout.tsx          # Main layout with nav
│   │   ├── page.tsx             # Dashboard
│   │   ├── portfolio/
│   │   ├── briefing/
│   │   ├── learning/
│   │   ├── community/
│   │   └── settings/
│   ├── api/
│   │   ├── auth/
│   │   ├── portfolio/
│   │   ├── fq/
│   │   └── briefing/
│   └── layout.tsx               # Root layout
├── components/
│   ├── ui/                      # shadcn/ui components
│   ├── portfolio/
│   ├── learning/
│   ├── fq/
│   └── layout/
├── lib/
│   ├── db.ts                    # Prisma client
│   ├── auth.ts                  # NextAuth config
│   └── utils.ts
└── prisma/
    └── schema.prisma
```

TECH STACK:
- Next.js 14 (App Router, TypeScript)
- shadcn/ui (component library)
- Tailwind CSS (styling)
- Prisma (ORM for PostgreSQL)
- NextAuth.js (authentication)
- React Query (data fetching)
- Zustand (client state)

KEY FEATURES:
1. Authentication:
   - Email/password + OAuth (Google, Apple)
   - Protected routes (dashboard requires auth)
   - Session management

2. Layout:
   - Root layout: Dark mode by default, global styles
   - Dashboard layout: Navigation + sidebar + main content
   - Mobile responsive (bottom tab bar on mobile)

3. Theme:
   - Dark mode primary (customizable in settings)
   - Green (#10b981) accent color
   - Professional financial aesthetic

4. API Routes:
   - RESTful endpoints for portfolio, FQ, briefing
   - Authentication middleware
   - Rate limiting

5. Database Schema (Prisma):
   - Users, Portfolios, FQ History, Exercises, Briefings, Theses

Generate the complete application shell with:
- Folder structure
- Root layout with dark mode
- Dashboard layout with navigation
- Basic API routes (auth, portfolio)
- Prisma schema for key tables
- Environment variable setup (.env.example)

Include TypeScript types and proper error handling.
```

---

## Usage Instructions

### Step 1: Start with Master Prompt
1. Go to https://v0.dev/
2. Use **Prompt 14** (Complete Application Shell) first
3. This generates the foundational structure
4. Download the generated code to your project

### Step 2: Build Pages Iteratively
Use prompts in order:
1. **Prompt 1** → Main Layout
2. **Prompt 2** → Dashboard
3. **Prompt 3** → FQ Assessment
4. **Prompt 4** → Portfolio Dashboard
5. Continue through prompts 5-13 as needed

### Step 3: Integrate Components
- Copy generated code from v0.app
- Paste into appropriate files in your Next.js project
- Adjust imports and types as needed
- Test in browser

### Step 4: Refine with Follow-up Prompts

If v0.app doesn't get it right on first try, use follow-ups like:
- "Make the colors darker and increase contrast"
- "Add a loading skeleton for the portfolio table"
- "Make this mobile responsive with a different layout on small screens"
- "Add hover effects and smooth transitions"

### Step 5: Connect to Backend

After UI is built:
1. Create API routes (use Prompt 14 as starting point)
2. Set up Prisma with PostgreSQL
3. Connect React Query to fetch data
4. Replace mock data with real API calls

---

## Tips for Best Results

**Be Specific:**
- Include exact colors (#10b981 for green)
- Specify component library (shadcn/ui)
- Mention responsive breakpoints

**Iterate:**
- Start simple, add complexity in follow-ups
- Ask for one component at a time (not entire pages)
- Refine with "make this darker" or "add animation"

**Copy Existing Gazillioner Design:**
- Mention "similar to current gazillioner.com dark theme"
- Reference the green accent color already in use
- Keep professional financial aesthetic

**Mobile-First:**
- Always ask for responsive design
- Specify mobile layouts explicitly
- Test on mobile viewport in v0.app preview

---

## Next Steps After v0.app

Once you have UI components from v0.app:

1. **Backend Setup:**
   - Set up PostgreSQL database
   - Configure Prisma ORM
   - Create API routes in Next.js

2. **Integrations:**
   - Plaid (broker connections)
   - OpenAI API (daily briefings)
   - Stripe (payments)

3. **Testing:**
   - Jest + React Testing Library
   - Playwright for E2E tests
   - Manual QA with real users

4. **Deployment:**
   - Deploy to Vercel (recommended for Next.js)
   - Set up environment variables
   - Configure custom domain (gazillioner.com)

---

**You now have a complete prompt library to build the entire Gazillioner FQ platform UI using v0.app!**

Start with Prompt 14, then work through 1-13 to build all features systematically.
