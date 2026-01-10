# Gazillioner MVP Build Plan
**16-Week Roadmap: Concept → Launch → $10k MRR**

## Executive Summary

**Goal**: Launch Gazillioner MVP on iPad with 200 paying users generating $10k MRR by Week 16.

**Timeline**: 16 weeks (4 months)
**Budget**: $120k (bootstrapped or pre-seed)
**Team**: 3-4 people (Founder/CEO, iOS Engineer, AI/ML Engineer, Designer)
**Launch Date**: Week 12 (beta), Week 16 (public)

**Success Metrics**:
- 200 paying users ($30/month avg = $6k MRR, stretch goal $10k)
- 40%+ D30 retention
- NPS 50+
- FQ score improvement: +100 points avg over 90 days

---

## Phase 1: Foundation (Weeks 1-4)

### Week 1: Technical Foundation + Team Assembly

**Goal**: Set up development environment, finalize architecture, confirm team.

**Technical Setup**:
- [ ] Set up GitHub repo (private)
- [ ] Configure Xcode project structure
  - SwiftUI app (iOS 17+, iPad optimized)
  - Core Data schema (Users, FQScores, Exercises, Positions, Theses)
  - CloudKit setup (optional, user can opt-in for backup)
- [ ] Set up CI/CD pipeline (GitHub Actions for TestFlight builds)
- [ ] Configure development certificates (Apple Developer Program)

**LLaMA-13B Setup**:
- [ ] Download LLaMA-13B weights (Meta AI)
- [ ] Implement 3.5-bit quantization (use llama.cpp or custom)
- [ ] Test on-device inference (M2 iPad Pro)
  - Target: 1,800-2,200 tokens/sec
  - Memory: <6GB RAM usage
- [ ] Build Metal GPU kernels for acceleration
- [ ] Create prompt template system

**Core Data Models**:
```swift
// User.swift
@Model class User {
    var id: UUID
    var email: String
    var createdAt: Date
    var subscriptionTier: SubscriptionTier // free, pro, elite
    var fqScore: Int // cached latest score
}

// FQScore.swift
@Model class FQScore {
    var id: UUID
    var userId: UUID
    var timestamp: Date
    var knowledge: Int      // 0-300
    var discipline: Int     // 0-400
    var wisdom: Int         // 0-300
    var total: Int          // 0-1000
    var percentile: Int     // 1-99
}

// Exercise.swift
@Model class CompletedExercise {
    var id: UUID
    var userId: UUID
    var exerciseId: String
    var category: ExerciseCategory
    var difficulty: Difficulty
    var attempts: Int
    var correct: Bool
    var completedAt: Date
}

// Position.swift (Portfolio)
@Model class Position {
    var id: UUID
    var userId: UUID
    var symbol: String
    var shares: Decimal
    var costBasis: Decimal
    var purchaseDate: Date
    var currentPrice: Decimal // updated daily
    var lastUpdated: Date
}

// Thesis.swift
@Model class Thesis {
    var id: UUID
    var userId: UUID
    var symbol: String
    var title: String
    var thesisText: String
    var pillars: [String]
    var successCriteria: [String]
    var invalidationCriteria: [String]
    var strength: Int // 0-10
    var createdAt: Date
    var lastReassessed: Date
}
```

**Team Confirmation**:
- [ ] Hire iOS engineer (if not founder) - $80k-120k/year or equity
- [ ] Hire AI/ML engineer (LLaMA optimization) - contract $10k-15k
- [ ] Hire designer (SwiftUI screens) - contract $5k-10k

**Deliverables**:
- Working Xcode project with Core Data models
- LLaMA-13B running on iPad (proof of concept)
- Team in place

**Budget This Week**: $5k (contracts, tools, Apple Developer)

---

### Week 2: UI/UX Design + Exercise Content

**Goal**: Design all core screens, write first 20 exercises.

**Design Deliverables** (Figma → SwiftUI):

**Screen 1: Onboarding Flow**
```
1. Welcome Screen
   "Welcome to Gazillioner - Your Financial IQ Platform"
   [Get Started]

2. FQ Assessment Intro
   "Let's measure your Financial Quotient (0-1000)"
   "Takes 5 minutes. Answer honestly."
   [Start Assessment]

3. Quick Assessment (10 questions)
   - 2 from Asset Allocation
   - 2 from Tax Optimization
   - 2 from Behavioral Finance
   - 2 from Options (basic)
   - 2 from Crypto

4. FQ Score Reveal
   "Your FQ Score: 520/1000"
   "You're in the 58th percentile"
   "Level: Building Wealth"

   Breakdown:
   - Knowledge: 160/300
   - Discipline: 200/400
   - Wisdom: 160/300

   [See Improvement Plan]

5. Connect Portfolio (Optional)
   "Connect your brokerage to track real progress"
   [Connect via Plaid] [Skip for Now]

6. Choose Plan
   Free: FQ assessment + 10 exercises
   Pro ($30/month): Full platform
   [Start Free] [Start Pro Trial (7 days)]
```

**Screen 2: Dashboard (Main Tab)**
```
┌─────────────────────────────────────────┐
│  Good Morning, Alex                      │
│  🎯 FQ Score: 520  (+12 this week)      │
│                                          │
│  📊 Progress This Month                  │
│  ■■■■■■■□□□ 70% to next level (FQ 600)  │
│                                          │
│  🔥 Streak: 14 days                      │
│  Next milestone: 30 days (+50 FQ points) │
│                                          │
│  📰 Today's Briefing                     │
│  ┌──────────────────────────────────┐   │
│  │ Markets up 2% but your portfolio │   │
│  │ is lagging (+0.8%). Here's why:  │   │
│  │                                   │   │
│  │ • AAPL (30% of portfolio) is flat│   │
│  │ • Consider rebalancing (drift 8%)│   │
│  │                                   │   │
│  │ 💡 Action: Trim AAPL, add bonds  │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ⚡ Quick Actions                        │
│  [Complete Exercise] [Check Portfolio]  │
│  [Review Thesis]     [Harvest Losses]   │
└─────────────────────────────────────────┘
```

**Screen 3: Exercises (Learn Tab)**
```
┌─────────────────────────────────────────┐
│  📚 Exercises                            │
│                                          │
│  Filter: [All] [In Progress] [Mastered] │
│                                          │
│  🎯 Asset Allocation (6/10 completed)    │
│  ■■■■■■□□□□ 60%                         │
│                                          │
│  Exercise 1.1: The 60/40 Portfolio ✅    │
│  Exercise 1.2: Rebalancing Basics ✅     │
│  Exercise 1.3: Home Country Bias ✅      │
│  Exercise 1.4: Age-Based Allocation 🔓   │
│  Exercise 1.5: Concentration Risk 🔒     │
│                                          │
│  💰 Tax Optimization (3/10 completed)    │
│  ■■■□□□□□□□ 30%                         │
│                                          │
│  Exercise 2.1: Tax-Loss Harvesting ✅    │
│  Exercise 2.2: Wash Sale Rule 🔓         │
│  Exercise 2.3: Roth vs Traditional 🔒    │
│                                          │
│  [Start Next Exercise]                   │
└─────────────────────────────────────────┘
```

**Screen 4: Portfolio (Portfolio Tab)**
```
┌─────────────────────────────────────────┐
│  💼 Portfolio                            │
│  Total Value: $87,450 (+$2,340 today)   │
│                                          │
│  📊 Allocation vs Target                 │
│  Stocks:  75% ████████████░░ (Target 70%)│
│  Bonds:   15% ████░░░░░░░░░ (Target 20%)│
│  Crypto:  10% ███░░░░░░░░░░ (Target 10%)│
│                                          │
│  ⚠️ Drift: 8% (Rebalancing recommended)  │
│  [Rebalance Now]                         │
│                                          │
│  📈 Positions                            │
│  ┌──────────────────────────────────┐   │
│  │ AAPL  $26,235  +18%  30% ⚠️      │   │
│  │ VTI   $21,863  +12%  25%         │   │
│  │ BND   $13,118  -2%   15%         │   │
│  │ BTC   $8,745   +45%  10%         │   │
│  └──────────────────────────────────┘   │
│                                          │
│  🎯 Thesis Tracker                       │
│  AAPL: "iPhone dominance" (Strength 7/10)│
│  [Review Thesis]                         │
└─────────────────────────────────────────┘
```

**Screen 5: FQ Score Detail**
```
┌─────────────────────────────────────────┐
│  🎯 Your FQ Score                        │
│                                          │
│  Total: 520/1000                         │
│  Percentile: 58th                        │
│  Level: Building Wealth                  │
│                                          │
│  📊 Breakdown                            │
│  Knowledge:  160/300 (53%)               │
│  ├─ Asset Allocation: 28/50 ██████░░░░  │
│  ├─ Tax Optimization: 22/50 ████░░░░░░  │
│  ├─ Options Strategy: 18/50 ███░░░░░░░  │
│  ├─ Behavioral:       30/50 ██████░░░░  │
│  ├─ Crypto:           25/50 █████░░░░░  │
│  └─ Market Cycles:    37/50 ███████░░░  │
│                                          │
│  Discipline: 200/400 (50%)               │
│  ├─ Contribution Streak: 40/100 (14d)   │
│  ├─ Rebalancing:        60/100          │
│  ├─ Tax Efficiency:     50/100          │
│  ├─ Emotional Control:  30/100 ⚠️       │
│  └─ Plan Adherence:     20/100 ⚠️       │
│                                          │
│  Wisdom: 160/300 (53%)                   │
│  ├─ Thesis Tracking: 60/100             │
│  ├─ Returns vs Benchmark: 50/100        │
│  └─ Learning Rate: 50/100                │
│                                          │
│  📈 Progress                             │
│  [Graph: FQ score over time]             │
│  Month 1: 480 → Month 3: 520 (+40)      │
│                                          │
│  🎯 Next Milestone                       │
│  Reach FQ 600 (Advanced Level)           │
│  80 points to go                         │
│  Estimated: 2-3 months at current pace   │
└─────────────────────────────────────────┘
```

**Exercise Content Creation**:
- [ ] Write 20 exercises (4 per category, beginner level)
  - Asset Allocation: 4 exercises (60/40 portfolio, rebalancing, etc.)
  - Tax Optimization: 4 exercises (TLH, wash sales, Roth vs Trad, holding period)
  - Options: 4 exercises (covered calls, puts, assignment, theta)
  - Behavioral: 4 exercises (loss aversion, FOMO, anchoring, sunk cost)
  - Crypto: 2 exercises (Bitcoin supply, private keys)
  - Market Cycles: 2 exercises (bull/bear, corrections)

**Deliverables**:
- Figma designs for 10 core screens
- 20 exercises written (JSON format for Core Data import)
- Design system documented (colors, fonts, spacing)

**Budget This Week**: $8k (designer contract)

---

### Week 3: Core FQ Engine Implementation

**Goal**: Build the FQ scoring algorithm and exercise system.

**Implementation Tasks**:

**1. FQ Scoring Engine** (`FQEngine.swift`):
```swift
class FQEngine {
    func calculateScore(
        user: User,
        exercises: [CompletedExercise],
        portfolio: Portfolio?,
        decisions: [Decision]
    ) async -> FQScore {

        // Knowledge Score (0-300)
        let knowledge = calculateKnowledgeScore(exercises: exercises)

        // Discipline Score (0-400)
        let discipline = await calculateDisciplineScore(
            user: user,
            portfolio: portfolio
        )

        // Wisdom Score (0-300)
        let wisdom = calculateWisdomScore(decisions: decisions)

        let total = knowledge + discipline + wisdom
        let percentile = calculatePercentile(total: total)

        return FQScore(
            id: UUID(),
            userId: user.id,
            timestamp: Date(),
            knowledge: knowledge,
            discipline: discipline,
            wisdom: wisdom,
            total: total,
            percentile: percentile
        )
    }

    private func calculateKnowledgeScore(exercises: [CompletedExercise]) -> Int {
        var score = 0

        // Group by category
        let categories: [ExerciseCategory] = [
            .assetAllocation,
            .taxOptimization,
            .optionsStrategy,
            .behavioralFinance,
            .cryptoFundamentals,
            .marketCycles
        ]

        for category in categories {
            let categoryExercises = exercises.filter { $0.category == category }

            guard !categoryExercises.isEmpty else { continue }

            // Completion rate (25 points max)
            let completionRate = Double(categoryExercises.count) / 30.0 // 30 exercises per category
            let completionPoints = Int(completionRate * 25)

            // First-try accuracy (15 points max)
            let firstTryCorrect = categoryExercises.filter { $0.attempts == 1 && $0.correct }.count
            let accuracyRate = Double(firstTryCorrect) / Double(categoryExercises.count)
            let accuracyPoints = Int(accuracyRate * 15)

            // Difficulty progression (10 points max)
            let avgDifficulty = categoryExercises.map { difficultyValue($0.difficulty) }.reduce(0, +) / Double(categoryExercises.count)
            let difficultyPoints = Int((avgDifficulty / 4.0) * 10) // 4 = expert

            score += completionPoints + accuracyPoints + difficultyPoints
        }

        return min(300, score)
    }

    private func difficultyValue(_ difficulty: Difficulty) -> Double {
        switch difficulty {
        case .beginner: return 1.0
        case .intermediate: return 2.0
        case .advanced: return 3.0
        case .expert: return 4.0
        }
    }

    private func calculateDisciplineScore(user: User, portfolio: Portfolio?) async -> Int {
        // Contribution Streak (0-100)
        let streakPoints = calculateStreakPoints(user.contributionStreakDays)

        // Rebalancing (0-100)
        let rebalancingPoints = portfolio != nil ? 60 : 0 // Placeholder

        // Tax Efficiency (0-100)
        let taxPoints = 50 // Placeholder

        // Emotional Control (0-100)
        let emotionalPoints = 50 // Start neutral

        return streakPoints + rebalancingPoints + taxPoints + emotionalPoints
    }

    private func calculateStreakPoints(_ streak: Int) -> Int {
        if streak >= 365 { return 100 }
        if streak >= 180 { return 80 }
        if streak >= 90 { return 60 }
        if streak >= 30 { return 40 }
        return Int((Double(streak) / 30.0) * 40)
    }

    private func calculateWisdomScore(decisions: [Decision]) -> Int {
        // Placeholder for MVP
        return 50
    }

    private func calculatePercentile(total: Int) -> Int {
        // Assume normal distribution: mean 500, stddev 150
        let mean = 500.0
        let stdDev = 150.0
        let zScore = (Double(total) - mean) / stdDev

        // Convert z-score to percentile
        let percentile = 50.0 + 34.13 * zScore

        return max(1, min(99, Int(percentile.rounded())))
    }
}
```

**2. Exercise System** (`ExerciseEngine.swift`):
```swift
class ExerciseEngine {
    func loadExercises() -> [Exercise] {
        // Load from JSON bundled with app
        guard let url = Bundle.main.url(forResource: "exercises", withExtension: "json") else {
            return []
        }

        guard let data = try? Data(contentsOf: url) else {
            return []
        }

        let exercises = try? JSONDecoder().decode([Exercise].self, from: data)
        return exercises ?? []
    }

    func getNextExercise(for user: User, category: ExerciseCategory?) -> Exercise? {
        let allExercises = loadExercises()
        let completed = getCompletedExerciseIds(user: user)

        let available = allExercises.filter { exercise in
            // Not completed
            !completed.contains(exercise.id) &&
            // Matches category filter (if provided)
            (category == nil || exercise.category == category) &&
            // Prerequisites met
            prerequisitesMet(exercise: exercise, completed: completed)
        }

        // Sort by difficulty (start with beginner)
        return available.sorted { $0.difficulty.rawValue < $1.difficulty.rawValue }.first
    }

    func submitAnswer(
        user: User,
        exercise: Exercise,
        selectedAnswer: String
    ) async -> ExerciseResult {
        let isCorrect = selectedAnswer == exercise.correctAnswer

        // Record attempt
        let completedExercise = CompletedExercise(
            id: UUID(),
            userId: user.id,
            exerciseId: exercise.id,
            category: exercise.category,
            difficulty: exercise.difficulty,
            attempts: 1, // TODO: Track multiple attempts
            correct: isCorrect,
            completedAt: Date()
        )

        // Save to Core Data
        await saveCompletedExercise(completedExercise)

        // Calculate FQ points earned
        let pointsEarned = calculatePoints(
            difficulty: exercise.difficulty,
            correct: isCorrect,
            firstTry: true
        )

        // Update user's FQ score
        await updateFQScore(user: user)

        return ExerciseResult(
            correct: isCorrect,
            explanation: exercise.explanation,
            pointsEarned: pointsEarned,
            followUpLesson: exercise.followUpLesson
        )
    }

    private func calculatePoints(difficulty: Difficulty, correct: Bool, firstTry: Bool) -> Int {
        guard correct else { return 0 }

        let basePoints: Int
        switch difficulty {
        case .beginner: basePoints = 2
        case .intermediate: basePoints = 3
        case .advanced: basePoints = 5
        case .expert: basePoints = 8
        }

        return firstTry ? basePoints : basePoints / 2
    }
}
```

**Deliverables**:
- FQEngine.swift (working scoring algorithm)
- ExerciseEngine.swift (exercise loading, submission, tracking)
- 20 exercises loaded into app (JSON → Core Data)
- Unit tests for FQ calculation

**Budget This Week**: $0 (internal dev work)

---

### Week 4: Portfolio Integration (Plaid API)

**Goal**: Connect user portfolios via Plaid for automatic tracking.

**Implementation**:

**1. Plaid Setup**:
- [ ] Sign up for Plaid (Development account, free)
- [ ] Configure supported brokerages:
  - Robinhood
  - Coinbase
  - Fidelity
  - Charles Schwab
  - Vanguard
- [ ] Get API keys (development mode)

**2. Plaid Integration** (`PlaidService.swift`):
```swift
import Plaid

class PlaidService {
    private let client: PlaidClient

    init() {
        self.client = PlaidClient(
            clientId: Config.plaidClientId,
            secret: Config.plaidSecret,
            environment: .development
        )
    }

    func createLinkToken(for userId: UUID) async throws -> String {
        let request = LinkTokenCreateRequest(
            user: LinkTokenUser(clientUserId: userId.uuidString),
            clientName: "Gazillioner",
            products: [.investments],
            countryCodes: ["US"],
            language: "en"
        )

        let response = try await client.linkTokenCreate(request)
        return response.linkToken
    }

    func exchangePublicToken(_ publicToken: String) async throws -> String {
        let response = try await client.itemPublicTokenExchange(publicToken)
        return response.accessToken
    }

    func fetchInvestmentHoldings(accessToken: String) async throws -> [PlaidHolding] {
        let response = try await client.investmentsHoldingsGet(accessToken)
        return response.holdings
    }

    func syncPortfolio(user: User, accessToken: String) async throws {
        let holdings = try await fetchInvestmentHoldings(accessToken: accessToken)

        // Convert Plaid holdings to our Position model
        for holding in holdings {
            let position = Position(
                id: UUID(),
                userId: user.id,
                symbol: holding.security.tickerSymbol ?? "",
                shares: Decimal(holding.quantity),
                costBasis: Decimal(holding.costBasis ?? 0),
                purchaseDate: holding.purchaseDate ?? Date(),
                currentPrice: Decimal(holding.institutionPrice),
                lastUpdated: Date()
            )

            // Save or update in Core Data
            await saveOrUpdatePosition(position)
        }
    }
}
```

**3. SwiftUI Plaid Link**:
```swift
import SwiftUI

struct ConnectPortfolioView: View {
    @State private var linkToken: String?
    @State private var showPlaidLink = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Connect Your Portfolio")
                .font(.largeTitle)
                .bold()

            Text("Securely connect your brokerage accounts to track your real portfolio and get personalized FQ scoring.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            Button("Connect with Plaid") {
                Task {
                    linkToken = try await plaidService.createLinkToken(for: user.id)
                    showPlaidLink = true
                }
            }
            .buttonStyle(.borderedProminent)

            Button("Skip for Now") {
                // Continue without portfolio
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .sheet(isPresented: $showPlaidLink) {
            if let linkToken = linkToken {
                PlaidLinkView(
                    linkToken: linkToken,
                    onSuccess: { publicToken in
                        Task {
                            let accessToken = try await plaidService.exchangePublicToken(publicToken)
                            try await plaidService.syncPortfolio(user: user, accessToken: accessToken)
                            // Navigate to dashboard
                        }
                    }
                )
            }
        }
    }
}
```

**Deliverables**:
- Plaid integration working (sandbox mode)
- User can connect Robinhood/Coinbase accounts
- Portfolio positions sync to Core Data
- Daily refresh scheduled (background task)

**Budget This Week**: $0 (Plaid development is free for <100 users)

---

## Phase 2: Core Features (Weeks 5-8)

### Week 5: Daily Briefing (AI-Generated)

**Goal**: Implement LLaMA-13B daily briefing generation.

**Implementation**:

**1. Daily Briefing Generator** (`DailyBriefingGenerator.swift`):
```swift
class DailyBriefingGenerator {
    private let llama: LLaMAInference

    func generateBriefing(user: User, portfolio: Portfolio?, fqScore: FQScore) async -> DailyBriefing {
        // Gather context
        let marketData = await fetchMarketData()
        let portfolioSummary = summarizePortfolio(portfolio)
        let fqSummary = summarizeFQScore(fqScore)

        // Build prompt
        let prompt = buildBriefingPrompt(
            user: user,
            fqScore: fqSummary,
            portfolio: portfolioSummary,
            market: marketData
        )

        // Generate with LLaMA
        let briefingText = await llama.generate(prompt: prompt, maxTokens: 300)

        return DailyBriefing(
            id: UUID(),
            userId: user.id,
            date: Date(),
            content: briefingText,
            marketSummary: marketData.summary,
            actionItems: extractActionItems(briefingText)
        )
    }

    private func buildBriefingPrompt(
        user: User,
        fqScore: String,
        portfolio: String,
        market: MarketData
    ) -> String {
        return """
        You are a financial coach for a user with FQ Score \(fqScore).

        Their portfolio:
        \(portfolio)

        Today's market:
        - S&P 500: \(market.sp500Change)%
        - Bitcoin: \(market.btcChange)%
        - VIX: \(market.vix)

        Generate a brief morning briefing (200 words) that:
        1. Highlights what matters for THEIR portfolio
        2. Suggests ONE action (rebalance, harvest loss, review thesis)
        3. Motivates toward FQ improvement

        Tone: Coach, not robot. Conversational. No jargon.

        Briefing:
        """
    }
}
```

**2. LLaMA Inference Optimization**:
- [ ] Benchmark current inference speed (baseline)
- [ ] Optimize Metal shaders (GPU acceleration)
- [ ] Implement KV-cache (faster subsequent tokens)
- [ ] Test on various iPad models:
  - iPad Pro M2 (16GB): Target 2,000 tok/s
  - iPad Pro M1 (8GB): Target 1,200 tok/s
  - iPad Air M1 (8GB): Target 1,000 tok/s

**3. Background Task Scheduling**:
```swift
import BackgroundTasks

class BackgroundTaskManager {
    func registerTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.gazillioner.dailyBriefing",
            using: nil
        ) { task in
            self.handleDailyBriefing(task: task as! BGAppRefreshTask)
        }
    }

    func scheduleDailyBriefing() {
        let request = BGAppRefreshTaskRequest(identifier: "com.gazillioner.dailyBriefing")
        request.earliestBeginDate = Calendar.current.date(byAdding: .hour, value: 6, to: Date()) // 6 AM tomorrow

        try? BGTaskScheduler.shared.submit(request)
    }

    private func handleDailyBriefing(task: BGAppRefreshTask) {
        Task {
            let user = await getCurrentUser()
            let portfolio = await getPortfolio(user: user)
            let fqScore = await getLatestFQScore(user: user)

            let generator = DailyBriefingGenerator(llama: llama)
            let briefing = await generator.generateBriefing(
                user: user,
                portfolio: portfolio,
                fqScore: fqScore
            )

            await saveBriefing(briefing)

            // Send notification
            sendLocalNotification(
                title: "Your Daily Financial Briefing is Ready",
                body: String(briefing.content.prefix(100)) + "..."
            )

            task.setTaskCompleted(success: true)
        }
    }
}
```

**Deliverables**:
- Daily briefing generation working (on-device LLaMA)
- Background task scheduled (6 AM daily)
- Push notification sent when briefing ready
- Briefing displayed on dashboard

**Budget This Week**: $0 (internal dev work)

---

### Week 6: Behavioral Alerts (FOMO/Panic Detection)

**Goal**: Implement real-time behavioral pattern detection.

**Implementation**:

**1. Behavioral Analyzer** (`BehavioralAnalyzer.swift`):
```swift
class BehavioralAnalyzer {
    func analyzeProposedTrade(
        trade: ProposedTrade,
        user: User,
        portfolio: Portfolio
    ) async -> BehavioralPattern {

        // Check for panic selling
        if trade.action == .sell {
            let panicScore = calculatePanicScore(trade: trade, portfolio: portfolio)

            if panicScore > 70 {
                return .panicSelling(score: panicScore)
            }
        }

        // Check for FOMO buying
        if trade.action == .buy {
            let fomoScore = calculateFOMOScore(trade: trade, portfolio: portfolio)

            if fomoScore > 70 {
                return .fomoChasing(score: fomoScore)
            }
        }

        return .rational
    }

    private func calculatePanicScore(trade: ProposedTrade, portfolio: Portfolio) -> Int {
        var score = 0

        // Factor 1: Drawdown magnitude (0-30)
        if let position = portfolio.positions.first(where: { $0.symbol == trade.symbol }) {
            let drawdown = (position.currentPrice - position.costBasis) / position.costBasis

            if drawdown < -0.30 { score += 30 }
            else if drawdown < -0.20 { score += 20 }
            else if drawdown < -0.10 { score += 10 }
        }

        // Factor 2: Market context (0-20)
        let marketDrawdown = await getMarketDrawdown()
        if marketDrawdown > 0.10 { score += 20 }

        // Factor 3: Thesis status (0-25)
        if let thesis = await getThesis(symbol: trade.symbol) {
            if thesis.strength > 7 {
                score += 25 // Selling with strong thesis = panic
            }
        }

        // Factor 4: Holding period (0-15)
        if let position = portfolio.positions.first(where: { $0.symbol == trade.symbol }) {
            let daysHeld = Calendar.current.dateComponents([.day], from: position.purchaseDate, to: Date()).day ?? 0
            if daysHeld < 90 {
                score += 15 // Quick flip = emotional
            }
        }

        return min(100, score)
    }

    private func calculateFOMOScore(trade: ProposedTrade, portfolio: Portfolio) -> Int {
        var score = 0

        // Factor 1: Recent price action (0-30)
        let priceChange7d = await getPriceChange(symbol: trade.symbol, days: 7)

        if priceChange7d > 0.20 { score += 30 }
        else if priceChange7d > 0.10 { score += 20 }

        // Factor 2: Social hype (0-25)
        let socialMentions = await getSocialMentions(symbol: trade.symbol)
        let baseline = await getBaselineMentions(symbol: trade.symbol)

        if socialMentions > baseline * 10 {
            score += 25 // 10× mentions = hype
        }

        // Factor 3: No thesis (0-20)
        let hasThesis = await hasThesis(symbol: trade.symbol)
        if !hasThesis {
            score += 20
        }

        // Factor 4: Position size (0-15)
        let positionSize = trade.amount / portfolio.totalValue
        if positionSize > 0.15 {
            score += 15 // >15% of portfolio = oversized
        }

        return min(100, score)
    }
}
```

**2. Alert UI**:
```swift
struct BehavioralAlertView: View {
    let alert: BehavioralAlert
    @State private var userChoice: AlertChoice?

    var body: some View {
        VStack(spacing: 20) {
            // Icon
            Image(systemName: alert.icon)
                .font(.system(size: 60))
                .foregroundColor(alert.severity == .critical ? .red : .orange)

            // Title
            Text(alert.title)
                .font(.title2)
                .bold()

            // Message
            Text(alert.message)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            // FQ Impact
            HStack {
                Text("FQ Impact:")
                    .font(.caption)
                Text(alert.fqImpact > 0 ? "+\(alert.fqImpact)" : "\(alert.fqImpact)")
                    .font(.caption)
                    .bold()
                    .foregroundColor(alert.fqImpact > 0 ? .green : .red)
            }

            // Options
            ForEach(alert.options) { option in
                Button {
                    userChoice = option
                } label: {
                    VStack(alignment: .leading) {
                        HStack {
                            Text(option.title)
                                .bold()
                            if option.recommended {
                                Text("RECOMMENDED")
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(4)
                            }
                        }
                        Text(option.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }
}
```

**Deliverables**:
- Behavioral analyzer working (panic/FOMO detection)
- Alert UI implemented
- FQ points awarded/deducted based on choices
- User can confirm or cancel trades

**Budget This Week**: $0 (internal dev work)

---

### Week 7: Thesis Tracker

**Goal**: Allow users to track investment theses and reassess monthly.

**Implementation**:

**1. Thesis Creation Flow**:
```swift
struct CreateThesisView: View {
    @State private var symbol: String = ""
    @State private var title: String = ""
    @State private var thesisText: String = ""
    @State private var pillars: [String] = ["", "", ""]
    @State private var successCriteria: [String] = ["", ""]
    @State private var invalidationCriteria: [String] = ["", ""]

    var body: some View {
        Form {
            Section("Position") {
                TextField("Stock Symbol (e.g., AAPL)", text: $symbol)
                    .textCase(.uppercase)
            }

            Section("Thesis") {
                TextField("Thesis Title", text: $title)
                    .placeholder("e.g., iPhone dominance continues")

                TextEditor(text: $thesisText)
                    .frame(height: 100)
                    .placeholder("Why did you buy this stock? What's your 3-5 year narrative?")
            }

            Section("Key Pillars") {
                ForEach(pillars.indices, id: \.self) { index in
                    TextField("Pillar \(index + 1)", text: $pillars[index])
                        .placeholder("e.g., Best battery tech in EV market")
                }
            }

            Section("Success Criteria") {
                ForEach(successCriteria.indices, id: \.self) { index in
                    TextField("Criterion \(index + 1)", text: $successCriteria[index])
                        .placeholder("e.g., Deliveries grow 40%+ annually")
                }
            }

            Section("Invalidation Criteria") {
                ForEach(invalidationCriteria.indices, id: \.self) { index in
                    TextField("Invalidation \(index + 1)", text: $invalidationCriteria[index])
                        .placeholder("e.g., Elon Musk leaves company")
                }
            }

            Button("Save Thesis") {
                saveThesis()
            }
            .buttonStyle(.borderedProminent)
        }
    }
}
```

**2. Monthly Thesis Reassessment** (AI-generated):
```swift
class ThesisAnalyzer {
    func reassessThesis(_ thesis: Thesis) async -> ThesisUpdate {
        // Fetch company data
        let earnings = await fetchEarnings(symbol: thesis.symbol)
        let news = await fetchNews(symbol: thesis.symbol, days: 30)
        let competitors = await fetchCompetitorData(symbol: thesis.symbol)

        // Build prompt for LLaMA
        let prompt = """
        Analyze this investment thesis:

        Title: \(thesis.title)
        Thesis: \(thesis.thesisText)

        Key Pillars:
        \(thesis.pillars.enumerated().map { "\($0 + 1). \($1)" }.joined(separator: "\n"))

        Success Criteria:
        \(thesis.successCriteria.enumerated().map { "\($0 + 1). \($1)" }.joined(separator: "\n"))

        Recent data:
        - Revenue growth: \(earnings.revenueGrowth)%
        - Gross margin: \(earnings.grossMargin)%
        - Recent news: \(news.headlines.prefix(3).joined(separator: "; "))

        Rate thesis strength 0-10 and explain what's changed.

        Analysis:
        """

        let analysis = await llama.generate(prompt: prompt, maxTokens: 400)

        // Extract strength score (parse from LLaMA output)
        let strength = extractStrength(from: analysis)

        return ThesisUpdate(
            thesisId: thesis.id,
            date: Date(),
            strength: strength,
            analysis: analysis,
            recommendation: getRecommendation(strength: strength)
        )
    }

    private func getRecommendation(strength: Int) -> String {
        switch strength {
        case 8...10: return "HOLD - Thesis intact"
        case 5...7: return "REASSESS - Thesis weakening"
        case 3...4: return "TRIM - Thesis 50% broken"
        case 0...2: return "SELL - Thesis broken"
        default: return "HOLD"
        }
    }
}
```

**Deliverables**:
- Users can create theses when adding positions
- Monthly AI reassessment scheduled
- Thesis strength displayed (0-10)
- Recommendations shown (HOLD, REASSESS, TRIM, SELL)

**Budget This Week**: $0 (internal dev work)

---

### Week 8: Tax-Loss Harvesting Scanner

**Goal**: Daily scan for tax-loss harvesting opportunities.

**Implementation**:

**1. Tax Harvest Scanner**:
```swift
class TaxHarvestScanner {
    func scanOpportunities(portfolio: Portfolio, userTaxBracket: Decimal) async -> [TaxHarvestOpportunity] {
        var opportunities: [TaxHarvestOpportunity] = []

        // Find positions with unrealized losses > $1,000
        let lossPositions = portfolio.positions.filter { $0.unrealizedGain < -1000 }

        for position in lossPositions {
            // Check wash sale risk
            let isWashSaleRisk = await checkWashSaleRisk(position: position)

            if isWashSaleRisk {
                continue // Skip if blocked
            }

            // Calculate tax savings
            let loss = abs(position.unrealizedGain)
            let taxSavings = loss * userTaxBracket

            // Find replacement securities
            let replacements = await findReplacementSecurities(symbol: position.symbol)

            // Score opportunity (0-100)
            let score = scoreOpportunity(
                loss: loss,
                taxSavings: taxSavings,
                daysToYearEnd: daysToYearEnd(),
                hasReplacements: !replacements.isEmpty
            )

            let opportunity = TaxHarvestOpportunity(
                position: position,
                estimatedTaxSavings: taxSavings,
                replacementOptions: replacements,
                score: score,
                daysToYearEnd: daysToYearEnd()
            )

            opportunities.append(opportunity)
        }

        // Sort by score (highest first)
        return opportunities.sorted { $0.score > $1.score }
    }

    private func scoreOpportunity(
        loss: Decimal,
        taxSavings: Decimal,
        daysToYearEnd: Int,
        hasReplacements: Bool
    ) -> Int {
        var score = 0

        // Loss magnitude (0-30)
        if loss > 50000 { score += 30 }
        else if loss > 10000 { score += 20 }
        else if loss > 3000 { score += 10 }

        // Time urgency (0-25)
        if daysToYearEnd < 14 { score += 25 }
        else if daysToYearEnd < 30 { score += 15 }

        // Replacement availability (0-20)
        if hasReplacements { score += 20 }

        // Tax savings (0-25)
        if taxSavings > 5000 { score += 25 }
        else if taxSavings > 2000 { score += 15 }

        return min(100, score)
    }

    private func findReplacementSecurities(symbol: String) async -> [Security] {
        // Find similar but not "substantially identical" securities
        let company = await getCompanyInfo(symbol: symbol)

        // Same sector, different companies
        let sectorPeers = await getSectorCompanies(sector: company.sector)

        // Filter by correlation (<0.85 to avoid wash sale)
        var replacements: [Security] = []

        for peer in sectorPeers where peer.symbol != symbol {
            let correlation = await calculateCorrelation(symbol1: symbol, symbol2: peer.symbol, days: 252)

            if correlation < 0.85 {
                replacements.append(peer)
            }
        }

        return replacements
    }
}
```

**2. Tax Harvest UI**:
```swift
struct TaxHarvestOpportunityCard: View {
    let opportunity: TaxHarvestOpportunity

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("💰 Tax Harvest Opportunity")
                    .font(.headline)
                Spacer()
                Text("Score: \(opportunity.score)/100")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(scoreColor)
                    .foregroundColor(.white)
                    .cornerRadius(4)
            }

            Text(opportunity.position.symbol)
                .font(.title2)
                .bold()

            HStack {
                VStack(alignment: .leading) {
                    Text("Unrealized Loss")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(opportunity.position.unrealizedGain, format: .currency(code: "USD"))
                        .font(.body)
                        .foregroundColor(.red)
                }

                Spacer()

                VStack(alignment: .trailing) {
                    Text("Est. Tax Savings")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(opportunity.estimatedTaxSavings, format: .currency(code: "USD"))
                        .font(.body)
                        .foregroundColor(.green)
                }
            }

            Divider()

            Text("Recommended Replacement:")
                .font(.caption)
                .foregroundColor(.secondary)

            if let replacement = opportunity.replacementOptions.first {
                HStack {
                    Text(replacement.symbol)
                        .bold()
                    Text(replacement.name)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }

            HStack {
                Button("Execute Harvest") {
                    // TODO: Generate trade instructions
                }
                .buttonStyle(.borderedProminent)

                Button("Remind Tomorrow") {
                    // Snooze
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var scoreColor: Color {
        if opportunity.score > 70 { return .green }
        if opportunity.score > 40 { return .orange }
        return .red
    }
}
```

**Deliverables**:
- Daily tax harvest scan (background task)
- Opportunities displayed with scores
- Replacement securities suggested
- FQ points awarded for executing harvests (+20 per harvest)

**Budget This Week**: $0 (internal dev work)

---

## Phase 3: Polish & Testing (Weeks 9-12)

### Week 9: Subscription & Payments (RevenueCat)

**Goal**: Implement in-app purchases and subscription management.

**Implementation**:

**1. RevenueCat Setup**:
- [ ] Create RevenueCat account (free for <$10k MRR)
- [ ] Configure products in App Store Connect:
  - **Free Tier**: FQ assessment + 10 exercises
  - **Pro Monthly**: $30/month (7-day free trial)
  - **Pro Annual**: $300/year (save $60, 17% discount)
- [ ] Set up entitlements in RevenueCat:
  - `pro_access`: Full platform access
  - `unlimited_exercises`: All 200+ exercises
  - `ai_coaching`: Daily briefing + behavioral alerts

**2. RevenueCat Integration**:
```swift
import RevenueCat

class SubscriptionManager {
    static let shared = SubscriptionManager()

    func configure() {
        Purchases.configure(withAPIKey: Config.revenueCatAPIKey)
        Purchases.logLevel = .debug
    }

    func fetchOfferings() async throws -> Offerings {
        return try await Purchases.shared.offerings()
    }

    func purchase(package: Package) async throws -> CustomerInfo {
        let (_, customerInfo, _) = try await Purchases.shared.purchase(package: package)
        return customerInfo
    }

    func restorePurchases() async throws -> CustomerInfo {
        return try await Purchases.shared.restorePurchases()
    }

    func checkSubscriptionStatus() async -> SubscriptionStatus {
        guard let customerInfo = try? await Purchases.shared.customerInfo() else {
            return .free
        }

        if customerInfo.entitlements["pro_access"]?.isActive == true {
            return .pro
        }

        return .free
    }
}

enum SubscriptionStatus {
    case free
    case pro
}
```

**3. Paywall UI**:
```swift
struct PaywallView: View {
    @State private var offerings: Offerings?
    @State private var selectedPackage: Package?
    @State private var isPurchasing = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Upgrade to Pro")
                .font(.largeTitle)
                .bold()

            Text("Unlock your full financial potential")
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Features
            VStack(alignment: .leading, spacing: 12) {
                FeatureRow(icon: "brain.head.profile", title: "Unlimited Exercises", subtitle: "All 200+ financial education exercises")
                FeatureRow(icon: "newspaper", title: "Daily AI Briefings", subtitle: "Personalized market insights every morning")
                FeatureRow(icon: "exclamationmark.triangle", title: "Behavioral Alerts", subtitle: "Prevent FOMO and panic selling")
                FeatureRow(icon: "dollarsign.circle", title: "Tax Optimization", subtitle: "Automated tax-loss harvesting")
                FeatureRow(icon: "chart.line.uptrend.xyaxis", title: "Thesis Tracker", subtitle: "Monitor investment narratives")
            }
            .padding()

            // Pricing
            if let offerings = offerings,
               let monthlyPackage = offerings.current?.monthly,
               let annualPackage = offerings.current?.annual {

                VStack(spacing: 12) {
                    PricingCard(
                        package: monthlyPackage,
                        title: "Monthly",
                        price: "$30/month",
                        subtitle: "7-day free trial",
                        isSelected: selectedPackage?.identifier == monthlyPackage.identifier
                    ) {
                        selectedPackage = monthlyPackage
                    }

                    PricingCard(
                        package: annualPackage,
                        title: "Annual",
                        price: "$300/year",
                        subtitle: "Save $60 (17% off)",
                        badge: "BEST VALUE",
                        isSelected: selectedPackage?.identifier == annualPackage.identifier
                    ) {
                        selectedPackage = annualPackage
                    }
                }
            }

            Button {
                purchase()
            } label: {
                if isPurchasing {
                    ProgressView()
                } else {
                    Text("Start Free Trial")
                        .bold()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedPackage == nil || isPurchasing)

            Button("Restore Purchases") {
                restore()
            }
            .font(.caption)
        }
        .padding()
        .task {
            offerings = try? await SubscriptionManager.shared.fetchOfferings()
            selectedPackage = offerings?.current?.monthly
        }
    }

    private func purchase() {
        guard let package = selectedPackage else { return }

        isPurchasing = true

        Task {
            do {
                let customerInfo = try await SubscriptionManager.shared.purchase(package: package)
                // Success - dismiss paywall
            } catch {
                // Handle error
                print("Purchase failed: \(error)")
            }
            isPurchasing = false
        }
    }

    private func restore() {
        Task {
            _ = try? await SubscriptionManager.shared.restorePurchases()
        }
    }
}
```

**Deliverables**:
- In-app purchases working (sandbox mode)
- Free tier limitations enforced
- Pro tier unlocks all features
- 7-day free trial configured
- Subscription status synced with Core Data

**Budget This Week**: $0 (RevenueCat free tier)

---

### Week 10: Onboarding Polish + App Store Assets

**Goal**: Perfect onboarding flow, create App Store screenshots and preview video.

**Onboarding Improvements**:
- [ ] Animated transitions between onboarding steps
- [ ] Skip option for returning users
- [ ] Progress indicator (1/6, 2/6, etc.)
- [ ] Portfolio connection optional (skip button)
- [ ] Gamified FQ assessment (feel like a quiz, not a test)

**App Store Assets**:

**1. Screenshots** (6 required):
- Screenshot 1: FQ Score Dashboard ("Your Financial IQ: 520/1000")
- Screenshot 2: Daily Briefing ("AI-Powered Morning Insights")
- Screenshot 3: Exercise Library ("Learn Through Gamified Exercises")
- Screenshot 4: Portfolio X-Ray ("Hidden Risks Revealed")
- Screenshot 5: Tax Harvesting ("Save $6k+ in Taxes Automatically")
- Screenshot 6: Behavioral Alerts ("Prevent FOMO & Panic Selling")

**2. App Preview Video** (30 seconds):
```
Script:
0:00 - "Meet Gazillioner - Your Financial IQ Platform"
0:05 - Show FQ score reveal (520/1000)
0:10 - Swipe through exercises (Duolingo-style)
0:15 - Daily briefing notification → tap → AI insights
0:20 - Behavioral alert: "FOMO Detected - Wait 48 hours?"
0:25 - Tax harvesting: "Save $2,400 in taxes"
0:30 - "Download Gazillioner - Get Smarter About Money"
```

**3. App Store Copy**:
```
Title:
Gazillioner: Financial IQ Score

Subtitle:
Duolingo for Wealth Management

Description:
Gazillioner measures your Financial Quotient (FQ) - a score from 0-1000 that tracks how well you make money decisions.

Like a credit score for financial intelligence.

🎯 YOUR FQ SCORE
- Knowledge: What you understand (0-300)
- Discipline: What you execute (0-400)
- Wisdom: How you decide (0-300)

📚 GAMIFIED LEARNING
200+ bite-sized exercises covering:
• Asset Allocation
• Tax Optimization
• Options Strategies
• Behavioral Finance
• Crypto Fundamentals
• Market Cycles

🤖 AI COACHING
On-device LLaMA-13B provides:
• Daily market briefings
• Behavioral alerts (prevent FOMO, panic)
• Thesis tracking
• Tax-loss harvesting

💰 SAVE THOUSANDS
Automated tax optimization saves $5k-50k annually.

🏆 COMPETE WITH FRIENDS
Social leaderboards and challenges make improvement fun.

📱 PRIVACY-FIRST
All AI runs on your iPad. Your financial data never leaves your device.

FREE TIER:
• FQ assessment
• 10 exercises

PRO ($30/month):
• Unlimited exercises
• AI coaching
• Tax optimization
• Portfolio tracking
• Behavioral alerts

Join 1,000+ users improving their financial IQ.

Download Gazillioner - Get Smarter About Money.
```

**4. Keywords** (100 characters max):
```
financial education,wealth management,portfolio tracker,tax optimization,investment thesis,FQ score
```

**Deliverables**:
- Onboarding flow polished (smooth animations)
- 6 App Store screenshots created
- 30-second preview video produced
- App Store listing copy written
- Keywords optimized for ASO

**Budget This Week**: $3k (video production, screenshot design)

---

### Week 11: Beta Testing (TestFlight)

**Goal**: Launch to 50 beta testers, collect feedback, fix bugs.

**Beta Tester Recruitment**:
- [ ] Post on Reddit (r/personalfinance, r/Bogleheads, r/fatFIRE)
- [ ] Twitter announcement
- [ ] LinkedIn post
- [ ] Email to warm contacts (friends, family, colleagues)

**Beta Feedback Form**:
```
1. How would you rate your onboarding experience? (1-10)
2. Did the FQ assessment feel accurate? (Yes/No/Unsure)
3. Which features did you use most? (checkboxes)
4. Which features need improvement? (free text)
5. Would you pay $30/month for this? (Yes/No/Maybe)
6. What's missing that you expected? (free text)
7. How likely are you to recommend to a friend? (NPS: 1-10)
```

**Metrics to Track**:
- Onboarding completion rate (target: 70%+)
- D1 retention (target: 50%+)
- D7 retention (target: 30%+)
- Avg session time (target: 5+ min)
- Exercises completed per user (target: 3+)
- NPS (target: 30+, acceptable for beta)

**Bug Triage**:
- P0 (blocking): Crashes, data loss → fix within 24 hours
- P1 (high): Feature not working → fix within 3 days
- P2 (medium): UI glitches → fix within 1 week
- P3 (low): Nice-to-haves → backlog for v1.1

**Deliverables**:
- 50 beta testers onboarded
- Feedback collected (NPS, feature requests, bugs)
- Critical bugs fixed (P0, P1)
- Retention metrics analyzed

**Budget This Week**: $0 (free beta)

---

### Week 12: Public Launch Prep

**Goal**: Finalize app, submit to App Store, prepare launch marketing.

**App Store Submission**:
- [ ] Bump version to 1.0.0
- [ ] Create App Store Connect listing
- [ ] Upload screenshots, preview video, description
- [ ] Set pricing ($30/month subscription)
- [ ] Submit for review (Apple takes 2-7 days)

**Launch Marketing Plan**:

**1. Product Hunt Launch** (Week 13, Day 1):
- [ ] Create Product Hunt listing
- [ ] Write tagline: "Duolingo for Wealth Management - Measure Your Financial IQ"
- [ ] Prepare maker intro video (2 min)
- [ ] Line up 10-20 upvoters (ask beta testers)

**2. Reddit Launch Posts** (Week 13, Days 1-3):
- r/personalfinance: "I built a Financial IQ test (like a credit score for money decisions)"
- r/Bogleheads: "Gamified exercises to improve asset allocation knowledge"
- r/fatFIRE: "For HNW individuals: On-device AI for portfolio analysis (privacy-first)"

**3. Twitter Launch Thread** (Week 13, Day 1):
```
Tweet 1: "I spent 4 months building Gazillioner - Duolingo for wealth management.

It measures your Financial Quotient (FQ): a score from 0-1000 for how well you make money decisions.

Here's what I learned building it 🧵"

Tweet 2: "Most people are smart (high IQ) but terrible with money.

Why? Because financial education is BORING.

Gazillioner gamifies it: bite-sized exercises, streaks, leaderboards.

Like Duolingo, but for your portfolio."

Tweet 3: "Your FQ Score has 3 components:
• Knowledge (0-300): What you understand
• Discipline (0-400): What you execute
• Wisdom (0-300): How you decide

This is the first standardized metric for financial decision-making."

Tweet 4: "Coolest feature: Real-time behavioral alerts.

About to panic sell during a dip? Gazillioner intercepts and asks:
'Are you sure? Your thesis is still intact. This is emotional.'

Saved beta users $47k in regret trades so far."

Tweet 5: "All AI runs ON YOUR IPAD (LLaMA-13B, 3.5-bit quantization).

Your portfolio data never leaves your device.

Privacy-first financial coaching."

Tweet 6: "Free tier: FQ assessment + 10 exercises
Pro ($30/month): Full platform

Download: [App Store link]

Would love your feedback! 🚀"
```

**4. Press Outreach** (Week 13):
- TechCrunch: "Gazillioner raises pre-seed, launches Financial IQ platform"
- The Verge: "This iPad app runs a 13B AI model locally for financial coaching"
- Hacker News: Submit Product Hunt link (organic upvotes)

**Deliverables**:
- App approved by Apple
- Product Hunt launch scheduled
- Reddit posts written
- Twitter thread prepared
- Press emails sent

**Budget This Week**: $0 (organic marketing)

---

## Phase 4: Launch & Growth (Weeks 13-16)

### Week 13: Public Launch

**Launch Day Checklist**:
- [ ] 6 AM: Post on Product Hunt
- [ ] 8 AM: Tweet launch thread
- [ ] 10 AM: Post on r/personalfinance, r/Bogleheads, r/fatFIRE
- [ ] 12 PM: Email beta testers (ask for App Store reviews)
- [ ] 2 PM: Post on Hacker News
- [ ] 4 PM: LinkedIn post
- [ ] 6 PM: Respond to all comments, tweets, Reddit threads

**Target Metrics (Week 13)**:
- 500 downloads (Week 1)
- 50 paid conversions ($1,500 MRR)
- Product Hunt: Top 5 of the day (300+ upvotes)
- NPS: 40+ (post-launch survey)

**Customer Support**:
- [ ] Set up support email: support@gazillioner.com
- [ ] Create FAQ page (website or in-app)
- [ ] Monitor App Store reviews (respond within 24 hours)
- [ ] Track feature requests (Canny or Trello board)

**Deliverables**:
- App launched publicly
- 500 downloads achieved
- $1,500 MRR (50 paying users)
- Product Hunt Top 5

**Budget This Week**: $0 (organic growth)

---

### Week 14: Content Marketing

**Goal**: Create SEO content to drive organic downloads.

**Blog Posts** (publish on gazillioner.com/blog):

**1. "What is Financial Quotient (FQ)? A Complete Guide"**
- 2,000 words
- Target keyword: "financial quotient"
- Explain FQ vs IQ, FQ vs FICO
- CTA: "Calculate Your FQ Score for Free"

**2. "Tax-Loss Harvesting Calculator: Save $5k-50k Per Year"**
- 1,500 words
- Target keyword: "tax loss harvesting calculator"
- Interactive calculator (web version)
- CTA: "Automate Tax Harvesting with Gazillioner"

**3. "How to Avoid Panic Selling (Behavioral Finance Guide)"**
- 1,800 words
- Target keyword: "how to avoid panic selling"
- Real examples from beta users
- CTA: "Get Real-Time Panic Alerts"

**4. "The Ultimate Guide to Investment Thesis Tracking"**
- 2,200 words
- Target keyword: "investment thesis template"
- Free downloadable template
- CTA: "Track Theses Automatically with AI"

**SEO Optimization**:
- [ ] Submit to Google Search Console
- [ ] Build 10 backlinks (guest posts, Reddit mentions, Product Hunt)
- [ ] Optimize page speed (<2s load time)
- [ ] Add schema markup (Article, FAQ)

**Deliverables**:
- 4 blog posts published (6,500 words total)
- SEO optimized (keywords, meta tags, backlinks)
- 100+ organic visitors per day (by end of Week 16)

**Budget This Week**: $1k (SEO freelancer)

---

### Week 15: Referral Program

**Goal**: Turn users into advocates (viral growth).

**Referral Mechanics**:
- **Referrer gets**: 1 month free Pro (for each friend who subscribes)
- **Referee gets**: Extended trial (14 days instead of 7)

**Implementation**:
```swift
class ReferralManager {
    func generateReferralCode(for user: User) -> String {
        // Format: ALEX-2F3D
        let firstName = user.firstName.uppercased()
        let randomCode = String.randomAlphanumeric(length: 4).uppercased()
        return "\(firstName)-\(randomCode)"
    }

    func trackReferral(code: String, newUser: User) async {
        guard let referrer = await findUserByReferralCode(code) else {
            return
        }

        // Credit referrer with 1 month free (when referee subscribes)
        await grantFreeMonth(user: referrer)

        // Give referee extended trial (14 days)
        await extendTrial(user: newUser, days: 14)

        // Send notification to referrer
        await sendNotification(
            to: referrer,
            title: "Friend signed up!",
            body: "You earned 1 month free Pro. Thanks for spreading the word!"
        )
    }
}
```

**Referral UI**:
```swift
struct ReferralView: View {
    let referralCode: String
    @State private var showShareSheet = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Invite Friends, Get Free Pro")
                .font(.largeTitle)
                .bold()

            Text("Share your code. When a friend subscribes, you both win.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)

            // Referral code
            HStack {
                Text(referralCode)
                    .font(.system(.title, design: .monospaced))
                    .bold()

                Button {
                    UIPasteboard.general.string = referralCode
                } label: {
                    Image(systemName: "doc.on.doc")
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)

            // Benefits
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "gift.fill")
                        .foregroundColor(.green)
                    Text("Your friend gets 14-day trial (vs 7-day)")
                }

                HStack {
                    Image(systemName: "star.fill")
                        .foregroundColor(.yellow)
                    Text("You get 1 month free Pro (when they subscribe)")
                }
            }

            Button("Share with Friends") {
                showShareSheet = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(items: [
                """
                I'm using Gazillioner to improve my Financial IQ. Use code \(referralCode) for a 14-day free trial (vs 7-day).

                \(Config.appStoreURL)
                """
            ])
        }
    }
}
```

**Deliverables**:
- Referral system implemented
- Referral tracking working (Core Data)
- Share sheet integrated (SMS, email, social)
- Viral coefficient target: 0.3 (each user refers 0.3 friends)

**Budget This Week**: $0 (internal dev work)

---

### Week 16: Metrics Dashboard & Investor Prep

**Goal**: Hit $10k MRR, prepare for fundraising.

**Growth Tactics (Final Push)**:
- [ ] Run App Store Search Ads ($500 budget)
  - Target keywords: "financial education", "portfolio tracker", "investment app"
  - CPA target: <$10 per download, <$100 per subscription
- [ ] Reddit ads ($300 budget)
  - Target subreddits: r/personalfinance, r/investing, r/Bogleheads
- [ ] Twitter ads ($200 budget)
  - Target followers of: @ReformedBroker, @CharlieBilello, @TheStalwart

**Metrics Dashboard** (for investors):
```swift
struct MetricsDashboard: View {
    @State private var metrics: AppMetrics?

    var body: some View {
        List {
            Section("Growth") {
                MetricRow(title: "Total Users", value: "\(metrics?.totalUsers ?? 0)")
                MetricRow(title: "Paying Users", value: "\(metrics?.payingUsers ?? 0)")
                MetricRow(title: "MRR", value: "$\(metrics?.mrr ?? 0)")
                MetricRow(title: "Churn Rate", value: "\(metrics?.churnRate ?? 0)%")
            }

            Section("Engagement") {
                MetricRow(title: "D1 Retention", value: "\(metrics?.d1Retention ?? 0)%")
                MetricRow(title: "D7 Retention", value: "\(metrics?.d7Retention ?? 0)%")
                MetricRow(title: "D30 Retention", value: "\(metrics?.d30Retention ?? 0)%")
                MetricRow(title: "Avg Session Time", value: "\(metrics?.avgSessionTime ?? 0) min")
            }

            Section("Product") {
                MetricRow(title: "Exercises Completed", value: "\(metrics?.exercisesCompleted ?? 0)")
                MetricRow(title: "Avg FQ Score", value: "\(metrics?.avgFQScore ?? 0)")
                MetricRow(title: "NPS", value: "\(metrics?.nps ?? 0)")
            }

            Section("Financial") {
                MetricRow(title: "LTV", value: "$\(metrics?.ltv ?? 0)")
                MetricRow(title: "CAC", value: "$\(metrics?.cac ?? 0)")
                MetricRow(title: "LTV/CAC", value: String(format: "%.1f", metrics?.ltvCacRatio ?? 0))
            }
        }
        .task {
            metrics = await fetchMetrics()
        }
    }
}
```

**Investor Update Email** (send to warm leads):
```
Subject: Gazillioner Update - $10k MRR, 200 Paying Users

Hi [INVESTOR],

Quick update on Gazillioner (Financial IQ platform):

TRACTION (4 months post-launch):
• 2,000 total users
• 200 paying users ($30/month avg)
• $10k MRR (50% MoM growth)
• 42% D30 retention
• NPS: 65

PRODUCT:
• FQ Score: Industry's first financial decision-making metric (0-1000)
• On-device AI: LLaMA-13B running locally (privacy-first)
• Tax optimization: Users saving avg $8k/year in taxes

METRICS:
• LTV: $360 (12-month retention)
• CAC: $50 (mostly organic + referrals)
• LTV/CAC: 7.2× (very healthy)

NEXT MILESTONE:
• $50k MRR by Month 9 (500 paying users)
• Launch B2B tier (sell to employers for 401k education)

We're raising $1M pre-seed. Would love to chat if there's interest.

Deck: [link]
Demo: [link]

Best,
[YOUR NAME]
Founder, Gazillioner
```

**Deliverables**:
- $10k MRR achieved (200 paying users)
- Metrics dashboard built (internal + investor-facing)
- Investor update sent
- Fundraising materials ready (pitch deck, financials)

**Budget This Week**: $1k (paid ads)

---

## Summary: 16-Week MVP Plan

### Timeline Overview

| Phase | Weeks | Goal | Budget |
|-------|-------|------|--------|
| Foundation | 1-4 | Tech setup, design, 20 exercises | $13k |
| Core Features | 5-8 | Daily briefing, alerts, thesis tracker, tax scanner | $0 |
| Polish | 9-12 | Subscriptions, App Store, beta testing, launch prep | $3k |
| Launch & Growth | 13-16 | Public launch, content, referrals, $10k MRR | $2k |
| **TOTAL** | **16 weeks** | **$10k MRR, 200 users** | **$18k** |

### Budget Breakdown

| Category | Amount | Notes |
|----------|--------|-------|
| Designer (contract) | $8k | UI/UX, App Store assets |
| AI/ML Engineer (contract) | $10k | LLaMA optimization |
| Apple Developer | $99/year | Required |
| RevenueCat | Free | <$10k MRR |
| Plaid | Free | Development mode |
| SEO/Content | $1k | Blog posts, optimization |
| Paid Ads (Week 16) | $1k | App Store + Reddit + Twitter |
| **TOTAL** | **$20k** | Rounded up for buffer |

### Success Metrics (Week 16)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Total Users | 2,000 | TBD | |
| Paying Users | 200 | TBD | |
| MRR | $10k | TBD | |
| D30 Retention | 40% | TBD | |
| NPS | 50 | TBD | |
| FQ Score Improvement | +100 pts | TBD | |
| LTV/CAC | 5× | TBD | |

### Risk Mitigation

**Risk 1: Apple rejects app (AI model too large)**
- **Mitigation**: Offer "download model" option (not bundled in app)
- **Backup**: Cloud AI fallback for users who can't download

**Risk 2: Low conversion rate (free → paid)**
- **Mitigation**: Aggressive free trial (14 days → 30 days)
- **Backup**: Freemium feature gating (10 exercises max on free)

**Risk 3: Poor retention (users don't return)**
- **Mitigation**: Daily push notifications (streak reminders)
- **Backup**: Email drip campaign (re-engagement)

**Risk 4: Can't reach $10k MRR by Week 16**
- **Mitigation**: Extend timeline to Week 20
- **Backup**: Focus on fewer high-value users (B2B2C via financial advisors)

**Risk 5: Competitors copy FQ Score concept**
- **Mitigation**: File trademark for "FQ Score" and "Financial Quotient"
- **Backup**: Network effects (first-mover advantage on leaderboards)

---

## Next Steps After Week 16

**Month 5-6: Scale to $50k MRR**
- Launch B2B tier (sell to employers, financial advisors)
- Expand exercise library (200 → 500 exercises)
- Add iPhone version (compromise on AI model size)
- Build web dashboard (for portfolio tracking on desktop)

**Month 7-9: Fundraising ($1M Pre-Seed)**
- Use investor outreach strategy (already created)
- Target: YC, a16z Crypto, Ribbit Capital
- Use $10k MRR traction as proof of PMF

**Month 10-12: Product-Market Fit**
- Iterate based on user feedback
- Improve retention (40% → 50% D30)
- Launch social features (challenges, leaderboards)
- Hit $100k ARR (profitable)

**Year 2: Series A ($5M at $20M post)**
- 10k paying users, $300k ARR
- Launch Android version
- Expand to international markets (UK, Canada, Australia)
- Hire 10-person team

---

## Appendix: Tools & Resources

**Development**:
- Xcode 15+
- SwiftUI
- Core Data
- CloudKit (optional)
- Metal (for GPU acceleration)

**AI/ML**:
- LLaMA-13B (Meta AI)
- llama.cpp (C++ inference library)
- 3.5-bit quantization (custom or GGUF)

**APIs**:
- Plaid (portfolio sync)
- Alpha Vantage (market data)
- NewsAPI (news sentiment)
- RevenueCat (subscriptions)

**Analytics**:
- PostHog (privacy-first analytics)
- Firebase Crashlytics (crash reporting)
- App Store Connect (download metrics)

**Marketing**:
- Product Hunt
- Reddit (r/personalfinance, r/Bogleheads)
- Twitter
- Hacker News
- App Store Search Ads

**Project Management**:
- Linear (issue tracking)
- Figma (design)
- Notion (documentation)
- GitHub (code repository)

---

**END OF MVP BUILD PLAN**

This 16-week roadmap provides:
✅ Week-by-week milestones
✅ Technical implementation details
✅ Budget breakdown ($20k total)
✅ Success metrics (200 users, $10k MRR)
✅ Risk mitigation strategies
✅ Launch marketing plan
✅ Post-launch growth roadmap

**Ready to build!** 🚀
