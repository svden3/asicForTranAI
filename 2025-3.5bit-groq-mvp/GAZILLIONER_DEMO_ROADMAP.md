# Gazillioner Demo Prototype Roadmap
## Build a Clickable Demo in 5 Days for Investor Meetings

**Goal:** Create a working prototype that demonstrates the core FQ platform value proposition to investors.

**Timeline:** 5 days (working 4-6 hours/day)

**Output:** Clickable Next.js app deployed to Vercel with:
- Working FQ assessment
- Mock daily briefing
- Portfolio dashboard (with sample data)
- Professional design matching your brand

---

## Why This Approach Works for Fundraising

**Investors want to see:**
1. ✅ **Visual proof** - Not just slides, but actual product
2. ✅ **Product thinking** - How you've designed the UX/UI
3. ✅ **Technical competence** - Can you actually build this?
4. ✅ **Vision clarity** - What's the end experience?

**A working demo answers all 4 questions in 30 seconds.**

Most pre-seed founders show slides. You'll show a working app. **Massive advantage.**

---

## Day 1: Setup & Core Shell (4 hours)

### Morning (2 hours): Project Setup

**Step 1: Create Next.js Project**
```bash
npx create-next-app@latest gazillioner-demo
# Choose these options when prompted:
# ✓ TypeScript: Yes
# ✓ ESLint: Yes
# ✓ Tailwind CSS: Yes
# ✓ src/ directory: No
# ✓ App Router: Yes
# ✓ Import alias: No (or default @/*)

cd gazillioner-demo
npm install
npm run dev
```

**Step 2: Install shadcn/ui**
```bash
npx shadcn-ui@latest init
# Choose:
# ✓ Style: Default
# ✓ Color: Slate
# ✓ CSS variables: Yes
```

**Step 3: Install Additional Dependencies**
```bash
npm install lucide-react recharts framer-motion
```

### Afternoon (2 hours): Generate App Shell

**v0.app Prompt to Use: Prompt 14 (Master Prompt)**

Copy the entire "Prompt 14: Complete Application Shell" from `GAZILLIONER_V0_PROMPTS.md` and paste into v0.dev.

**What you'll get:**
- Root layout with dark mode
- Navigation bar (top)
- Sidebar (desktop only)
- Mobile responsive layout
- Basic routing structure

**Integration Steps:**
1. Download code from v0.app
2. Copy `app/layout.tsx` to your project
3. Copy components to `components/layout/`
4. Test: `npm run dev` → Check http://localhost:3000
5. Adjust colors to match your brand (green #10b981)

**Expected Output:**
- Professional dark mode layout
- Navigation works (even if pages are empty)
- Mobile responsive header with hamburger menu

**Git Checkpoint:**
```bash
git add .
git commit -m "feat: Add app shell with navigation"
```

---

## Day 2: FQ Assessment (Demo's "Wow" Moment) (5 hours)

### Morning (3 hours): Generate Quiz Interface

**v0.app Prompt to Use: Prompt 3 (FQ Assessment Quiz)**

This is your **key differentiator** - the FQ scoring system. Make this impressive.

**Steps:**
1. Copy Prompt 3 from `GAZILLIONER_V0_PROMPTS.md`
2. Paste into v0.dev
3. Generate and preview
4. If not perfect, iterate:
   - "Make the progress bar more prominent"
   - "Add a celebration animation on results page with confetti"
   - "Make the score display larger and more dramatic"

**Integration:**
1. Create `app/onboarding/fq-assessment/page.tsx`
2. Copy generated quiz component
3. Create `data/quiz-questions.ts` with 10 sample questions (for demo):

```typescript
// data/quiz-questions.ts
export const quizQuestions = [
  {
    id: 1,
    category: "Knowledge",
    question: "What is a P/E ratio?",
    options: [
      { id: "A", text: "Price to Earnings ratio" },
      { id: "B", text: "Profit to Equity ratio" },
      { id: "C", text: "Portfolio Efficiency ratio" },
      { id: "D", text: "Price to Expense ratio" }
    ],
    correctAnswer: "A",
    points: 10
  },
  // ... add 9 more questions
  // Use questions from MVP_SPEC.md or make them up for demo
];
```

### Afternoon (2 hours): Add Results Page with Animation

**Make the results page memorable:**
- Large FQ score reveal (animated count-up)
- Confetti animation (use `canvas-confetti` package)
- Breakdown by category (Knowledge, Discipline, Wisdom)
- Personalized insight ("You're strong in X, need work on Y")

**Quick Win:**
```bash
npm install canvas-confetti
```

Then add confetti to results page:
```typescript
import confetti from 'canvas-confetti';

useEffect(() => {
  confetti({
    particleCount: 100,
    spread: 70,
    origin: { y: 0.6 }
  });
}, []);
```

**Expected Output:**
- Working 10-question quiz
- Smooth transitions between questions
- Dramatic score reveal with confetti
- Professional, polished feel

**Git Checkpoint:**
```bash
git commit -m "feat: Add FQ assessment with score reveal"
```

**Demo Script:**
> "Let me show you our FQ assessment. In 5 minutes, users get a personalized Financial Quotient score that measures their decision-making ability - not their net worth. Watch this..."
>
> [Click through quiz, show results with confetti]
>
> "This gamification is key to retention. Users come back daily to improve their FQ score."

---

## Day 3: Daily Briefing (The Core Value Prop) (5 hours)

### Morning (3 hours): Generate Briefing Page

**v0.app Prompt to Use: Prompt 8 (Daily Briefing Page)**

This shows **why users will pay $30/month** - personalized AI insights daily.

**Steps:**
1. Copy Prompt 8 from `GAZILLIONER_V0_PROMPTS.md`
2. Generate in v0.app
3. Create `app/dashboard/briefing/page.tsx`

**Add Sample Briefing Data:**

Create `data/sample-briefing.ts`:
```typescript
export const sampleBriefing = {
  date: "December 10, 2025",
  greeting: "Good morning, Alex",
  portfolio: {
    value: 127400,
    change: 1200,
    changePercent: 0.95,
    allocation: {
      stocks: 72,
      crypto: 8,
      bonds: 13,
      cash: 7
    },
    target: {
      stocks: 70,
      crypto: 10,
      bonds: 15,
      cash: 5
    }
  },
  actions: [
    {
      type: "rebalance",
      summary: "Sell $800 VTI, buy $800 BTC",
      why: "Your stocks rallied to 72% (target: 70%). Crypto is 8% (target: 10%).",
      taxImpact: "Selling long-term gains (15% tax on $85 profit = $13)"
    },
    {
      type: "tax-harvest",
      summary: "Sell ARKK (-$840 loss) → save $202 in taxes",
      why: "ARKK is down 21%. No wash sale violation.",
      replacement: "Buy QQQM to maintain tech exposure"
    },
    {
      type: "contribute",
      summary: "Auto-invest $500 to Roth IRA today",
      why: "Your monthly contribution schedule"
    }
  ],
  learningMoment: {
    title: "Why People Sold Apple Too Early",
    content: `In 2011, Steve Jobs died. Apple stock fell 30%. Many investors sold,
    thinking "Apple is done without Jobs."

    Reality: Tim Cook was an excellent operator. iPhone kept growing.
    Services revenue exploded. Stock went up 13× from that low.

    **Lesson:** Founder risk is real, but ecosystem moats often survive
    leadership changes.

    **Pattern to remember:** "Founder departure ≠ broken thesis if moat intact"`
  },
  thesisCheck: {
    holding: "NVDA",
    userThesis: "AI revolution will drive data center demand",
    evidence: [
      { type: "positive", text: "Data center revenue up 217% YoY" },
      { type: "positive", text: "Guidance beat expectations" },
      { type: "negative", text: "Stock fell 8% (sentiment, not fundamentals)" }
    ],
    verdict: "STRENGTHENING",
    recommendation: "HOLD (or add on dips)"
  }
};
```

### Afternoon (2 hours): Add Interactivity

**Make it feel real:**
- Expandable action items (click to see full details)
- "Execute" buttons (show modal: "Coming soon - connect your broker")
- Markdown rendering for learning moment
- Green/red color coding (gains vs losses)

**Expected Output:**
- Beautiful daily briefing page
- Looks like AI-generated content (even though it's hardcoded for demo)
- Clear value proposition: "This is what you get every morning"

**Git Checkpoint:**
```bash
git commit -m "feat: Add daily briefing page with sample data"
```

**Demo Script:**
> "Every morning at 6 AM, users get a personalized briefing like this. It's not generic - it's based on THEIR portfolio, THEIR goals, THEIR theses. Watch what happens when I click an action item..."
>
> [Expand rebalancing suggestion]
>
> "The AI explains WHY, calculates tax impact, and can even execute trades. This is the behavioral coaching that helps users avoid costly mistakes."

---

## Day 4: Portfolio Dashboard (Show Multi-Asset Intelligence) (5 hours)

### Morning (3 hours): Generate Portfolio Page

**v0.app Prompt to Use: Prompt 4 (Portfolio Dashboard)**

This demonstrates **multi-asset support** (stocks + crypto) - a key differentiator.

**Steps:**
1. Copy Prompt 4 from `GAZILLIONER_V0_PROMPTS.md`
2. Generate in v0.app
3. Create `app/dashboard/portfolio/page.tsx`

**Add Sample Portfolio Data:**

Create `data/sample-portfolio.ts`:
```typescript
export const sampleHoldings = [
  {
    ticker: "AAPL",
    name: "Apple Inc.",
    assetClass: "stock",
    shares: 50,
    costBasis: 7500,
    currentValue: 9250,
    currentPrice: 185,
    gainLoss: 1750,
    gainLossPercent: 23.3,
    portfolioPercent: 7.3
  },
  {
    ticker: "NVDA",
    name: "NVIDIA Corp.",
    assetClass: "stock",
    shares: 25,
    costBasis: 12000,
    currentValue: 20000,
    currentPrice: 800,
    gainLoss: 8000,
    gainLossPercent: 66.7,
    portfolioPercent: 15.7
  },
  {
    ticker: "BTC",
    name: "Bitcoin",
    assetClass: "crypto",
    shares: 0.25,
    costBasis: 8000,
    currentValue: 10500,
    currentPrice: 42000,
    gainLoss: 2500,
    gainLossPercent: 31.3,
    portfolioPercent: 8.2
  },
  {
    ticker: "VOO",
    name: "Vanguard S&P 500 ETF",
    assetClass: "stock",
    shares: 30,
    costBasis: 13500,
    currentValue: 14400,
    currentPrice: 480,
    gainLoss: 900,
    gainLossPercent: 6.7,
    portfolioPercent: 11.3
  },
  // Add 4-5 more holdings for realism
];
```

### Afternoon (2 hours): Add Charts

**Install charting library:**
```bash
npm install recharts
```

**Add two charts:**
1. **Allocation Doughnut Chart** (Current vs Target)
2. **Performance Line Chart** (30-day trend)

Use Recharts documentation or ask v0.app:
> "Create a doughnut chart showing portfolio allocation with Recharts. Segments: Stocks 72%, Crypto 8%, Bonds 13%, Cash 7%. Dark mode, green accent."

**Expected Output:**
- Holdings table with sortable columns
- Doughnut chart showing allocation
- Performance trend chart
- Professional financial dashboard aesthetic

**Git Checkpoint:**
```bash
git commit -m "feat: Add portfolio dashboard with charts"
```

**Demo Script:**
> "Here's where users see their entire portfolio - stocks AND crypto in one place. Bloomberg doesn't do crypto well. Robo-advisors don't do crypto at all. We're the only platform that optimizes across both."
>
> [Point to holdings table]
>
> "Notice the tax-loss harvesting opportunity on ARKK? The AI flagged that automatically. Users save thousands in taxes without thinking about it."

---

## Day 5: Dashboard Home & Polish (6 hours)

### Morning (3 hours): Generate Dashboard Home

**v0.app Prompt to Use: Prompt 2 (Dashboard Page)**

This is the **landing page** after login - ties everything together.

**Steps:**
1. Copy Prompt 2 from `GAZILLIONER_V0_PROMPTS.md`
2. Generate in v0.app
3. Create `app/dashboard/page.tsx`

**Add Card Components:**
- Daily Briefing Preview (link to full briefing)
- Portfolio Snapshot (total value, change)
- FQ Score Card (with trend)
- Action Items Checklist (3 items from briefing)
- Recommended Exercises (2-3 cards)

### Afternoon (3 hours): Polish & Deploy

**Polish Checklist:**
- [ ] Fix any layout issues on mobile
- [ ] Ensure all links work (navigation between pages)
- [ ] Add loading states (simple spinners)
- [ ] Add empty states (if needed)
- [ ] Fix any TypeScript errors
- [ ] Test in Chrome, Safari, Firefox
- [ ] Take screenshots for pitch deck (see below)

**Deploy to Vercel:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow prompts, choose "gazillioner-demo" as project name
# Vercel will give you a URL: https://gazillioner-demo.vercel.app
```

**Set Custom Domain (Optional):**
If you own `gazillioner.com`, you can set up a subdomain:
- Create `demo.gazillioner.com` in DNS
- Point to Vercel in your domain settings
- Vercel auto-configures HTTPS

**Expected Output:**
- Fully working demo deployed to the internet
- Shareable link you can send to investors
- Mobile responsive (works on phone)
- Professional, polished feel

**Git Checkpoint:**
```bash
git commit -m "feat: Add dashboard home and final polish"
git push
```

---

## Demo Day Checklist (Before Showing Investors)

### Functionality Test
- [ ] Navigation works (all links go somewhere)
- [ ] FQ assessment completes (shows score at end)
- [ ] Daily briefing displays correctly
- [ ] Portfolio dashboard shows holdings
- [ ] Charts render properly
- [ ] Mobile layout works (test on phone)
- [ ] No console errors (press F12 in browser)
- [ ] Fast loading (<2 seconds per page)

### Visual Polish
- [ ] Logo is clear (top-left)
- [ ] Colors match brand (green accent #10b981)
- [ ] Typography is readable
- [ ] Dark mode looks professional
- [ ] Icons are consistent
- [ ] Spacing feels balanced
- [ ] Animations are smooth (not janky)

### Demo Script Preparation
- [ ] Write 2-minute walkthrough script
- [ ] Practice demo 5 times (get smooth)
- [ ] Have backup plan (screenshots if internet fails)
- [ ] Prepare answers to "Can it do X?" questions

---

## Taking Screenshots for Pitch Deck

**Use these pages for slides:**

1. **Dashboard Home** → Slide 5 (Product Overview)
   - Full-screen screenshot
   - Crop to remove browser chrome
   - Highlight: FQ score, daily briefing preview, action items

2. **FQ Assessment Results** → Slide 7 (FQ Scoring System)
   - Screenshot of results page with confetti
   - Highlight: 560 score, breakdown by category

3. **Daily Briefing** → Slide 8 (Daily Briefing Feature)
   - Screenshot of full briefing
   - Highlight: Personalized insights, action items, learning moment

4. **Portfolio Dashboard** → Slide 9 (Multi-Asset Intelligence)
   - Screenshot with holdings table and charts
   - Highlight: Stocks + crypto in one view

**Screenshot Tools:**
- Mac: Cmd+Shift+4 (select area)
- Windows: Win+Shift+S (Snipping Tool)
- Or use browser extension: "Full Page Screen Capture"

**Image Optimization:**
- Save as PNG (better quality than JPG for UI)
- Use TinyPNG.com to compress (reduces file size)
- Minimum resolution: 1920x1080 (HD)

---

## Sample Demo Script (2 Minutes)

**Opening (15 seconds):**
> "Let me show you how Gazillioner works. I'll walk through the user journey from onboarding to daily use."

**FQ Assessment (30 seconds):**
> "First, users take a 15-minute Financial Quotient assessment. This isn't about net worth - it measures decision-making ability."
>
> [Click through 2-3 quiz questions]
>
> "They get an instant score with personalized insights. This gamification drives engagement - users come back daily to improve their FQ."

**Daily Briefing (45 seconds):**
> "Every morning, they receive an AI-generated briefing tailored to THEIR portfolio and goals."
>
> [Navigate to briefing page]
>
> "It tells them exactly what to do: rebalance, harvest tax losses, contribute to retirement. The AI explains WHY and calculates tax impact."
>
> [Scroll through learning moment]
>
> "There's also a daily learning moment - pattern recognition from historical cases like Apple 2011. This builds the wisdom to avoid costly mistakes."

**Portfolio Dashboard (30 seconds):**
> "Here's their portfolio - stocks AND crypto in one view. No other platform does this well. Bloomberg doesn't understand crypto. Robo-advisors don't support it at all."
>
> [Point to holdings]
>
> "The AI monitors for tax-loss harvesting opportunities automatically. Users save thousands without thinking about it."

**Closing (15 seconds):**
> "That's Gazillioner. We help users avoid the mistakes that cost them $50k-500k over their lifetime - missing opportunities like Apple and Bitcoin, or panic-selling at bottoms. Questions?"

**Total: 2 minutes 15 seconds** (perfect for pitch)

---

## Advanced: Add "AI Generated" Shimmer Effect

**Make it clear the briefing is AI-generated:**

Add a subtle shimmer badge to the daily briefing:

```tsx
// components/AIBadge.tsx
export function AIBadge() {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
      <Sparkles className="w-4 h-4 text-purple-400" />
      <span className="text-sm text-purple-300">AI Generated</span>
    </div>
  );
}
```

This makes it **visually obvious** that you're using AI (investors love this in 2025).

---

## Backup Plan (If v0.app Doesn't Work Perfectly)

**If you get stuck or v0 generates buggy code:**

1. **Use shadcn/ui directly:**
   - Browse: https://ui.shadcn.com/
   - Copy/paste pre-built components
   - Customize colors and layout

2. **Use templates:**
   - Vercel templates: https://vercel.com/templates
   - Find a dark mode dashboard template
   - Customize with your content

3. **Hire Upwork developer for 1 week:**
   - Post: "Need Next.js developer to build demo from Figma/specs"
   - Budget: $500-1000
   - Provide v0-generated code as starting point

**But honestly, v0.app is good enough that you won't need backup.**

---

## What Investors Will Ask

**Be ready for these questions:**

**Q: "How does the AI work?"**
A: "We use GPT-4 fine-tuned on financial patterns. For the demo, I've hardcoded sample briefings, but in production it's fully AI-generated based on portfolio data, market conditions, and user goals."

**Q: "What if the AI gives bad advice?"**
A: "We're not providing investment advice - we're educational. Users make their own decisions. Every suggestion includes reasoning so they learn WHY. Plus, we're building in human review for the first 1000 users."

**Q: "How do you compete with Betterment/Wealthfront?"**
A: "They're robo-advisors focused on passive ETF portfolios. We're behavioral coaches focused on decision-making. We support crypto (they don't), options (they don't), and education (they don't). Different markets."

**Q: "What's your moat?"**
A: "Three layers: 1) FQ scoring system (proprietary), 2) Community network effects (users learn together), 3) On-device AI via custom 3.5-bit quantization (privacy moat)."

**Q: "Why will users pay $30/month?"**
A: "If we save them $1000/year in taxes ALONE, that's 3× ROI. Plus behavioral coaching to avoid $50k+ mistakes. We're targeting users with $100k+ portfolios where $30/month is noise."

**Q: "What's your GTM strategy?"**
A: "Content-driven SEO ('Why did I sell Apple too early?'), finance YouTube partnerships, Reddit communities (r/personalfinance, r/fatFIRE), and product-led growth (free FQ assessment as lead magnet)."

---

## Success Metrics for Demo

**Good demo achieves:**
- ✅ Investor says "This is cool" or "I'd use this"
- ✅ Investor asks about business model (means they're interested)
- ✅ Investor introduces you to someone else
- ✅ Investor asks for follow-up meeting

**Great demo achieves:**
- ✅ Investor pulls out checkbook (term sheet discussion)
- ✅ Investor asks if they can invest personally (use the product)
- ✅ Investor says "Who else have you talked to?" (FOMO)

**With a working demo, you're 10× more likely to get a great outcome than slides alone.**

---

## After Building Demo: What's Next?

**Week 2:**
- Create pitch deck (next document I'll make for you)
- Practice demo until smooth
- Set up investor meetings (10-20 meetings)

**Week 3-4:**
- Pitch, iterate based on feedback
- Track investor interest (who's warm/hot)
- Negotiate terms if you get multiple interested

**Month 2:**
- Close pre-seed round (target: $1M)
- Hire team (2 engineers, 1 designer)
- Build real MVP (not just demo)

**You're on track to raise and launch within 3-4 months total.**

---

## You're Ready to Build!

**Day 1:** App shell + navigation
**Day 2:** FQ assessment (with confetti!)
**Day 3:** Daily briefing (core value prop)
**Day 4:** Portfolio dashboard (multi-asset)
**Day 5:** Dashboard home + deploy

**By end of Week 1:** Working demo live on the internet, ready to show investors.

**Start today!** Open `GAZILLIONER_V0_PROMPTS.md`, copy Prompt 14, go to v0.dev, and generate your first component.

You'll be shocked how fast this goes. 🚀
