// FQEngine.swift
// Financial Quotient (FQ) Scoring Algorithm
//
// Calculates 0-1000 score across three dimensions:
// - Knowledge: 0-300 (understanding concepts)
// - Discipline: 0-400 (executing consistently)
// - Wisdom: 0-300 (making good decisions)

import Foundation

class FQEngine {

    // MARK: - Main Calculation

    func calculateScore(
        portfolio: Portfolio?,
        historicalDecisions: [Decision]
    ) async -> FQScore {

        let knowledge = await calculateKnowledge()
        let discipline = await calculateDiscipline(portfolio: portfolio)
        let wisdom = await calculateWisdom(decisions: historicalDecisions)

        let total = knowledge.total + discipline.total + wisdom.totalPoints
        let percentile = calculatePercentile(total)

        // Calculate trends
        let history = await fetchFQHistory()
        let change30Days = total - (history.thirtyDaysAgo?.total ?? total)
        let change90Days = total - (history.ninetyDaysAgo?.total ?? total)

        return FQScore(
            id: UUID(),
            timestamp: Date(),
            knowledge: knowledge.total,
            discipline: discipline.total,
            wisdom: wisdom.totalPoints,
            percentile: percentile,
            knowledgeBreakdown: knowledge,
            disciplineBreakdown: discipline,
            wisdomBreakdown: wisdom,
            change30Days: change30Days,
            change90Days: change90Days
        )
    }

    // MARK: - Knowledge Calculation (0-300)

    func calculateKnowledge() async -> KnowledgeScores {
        // Load exercise completion from Core Data
        let exercises = await fetchCompletedExercises()

        // Calculate category scores
        let assetAllocation = calculateCategoryScore(
            exercises: exercises,
            category: .assetAllocation,
            maxScore: 100
        )

        let taxOptimization = calculateCategoryScore(
            exercises: exercises,
            category: .taxOptimization,
            maxScore: 100
        )

        let optionsStrategies = calculateCategoryScore(
            exercises: exercises,
            category: .optionsStrategy,
            maxScore: 100
        )

        let behavioralFinance = calculateCategoryScore(
            exercises: exercises,
            category: .behavioralTraining,
            maxScore: 100
        )

        let cryptoFundamentals = calculateCategoryScore(
            exercises: exercises,
            category: .cryptoFundamentals,
            maxScore: 100
        )

        let marketCycles = calculateCategoryScore(
            exercises: exercises,
            category: .marketCycles,
            maxScore: 100
        )

        return KnowledgeScores(
            assetAllocation: assetAllocation,
            taxOptimization: taxOptimization,
            optionsStrategies: optionsStrategies,
            behavioralFinance: behavioralFinance,
            cryptoFundamentals: cryptoFundamentals,
            marketCycles: marketCycles
        )
    }

    private func calculateCategoryScore(
        exercises: [CompletedExercise],
        category: ExerciseCategory,
        maxScore: Int
    ) -> Int {
        // Filter exercises for this category
        let categoryExercises = exercises.filter { $0.category == category }

        guard !categoryExercises.isEmpty else { return 0 }

        // Calculate score based on:
        // - Completion rate (50%)
        // - First-try accuracy (30%)
        // - Difficulty level (20%)

        let totalExercises = 30 // Total available per category
        let completed = categoryExercises.count
        let completionRate = Double(completed) / Double(totalExercises)

        let firstTryCorrect = categoryExercises.filter { $0.attempts == 1 && $0.correct }.count
        let accuracyRate = Double(firstTryCorrect) / Double(completed)

        let avgDifficulty = categoryExercises.map { difficultyMultiplier($0.difficulty) }.reduce(0, +) / Double(completed)

        let score = (completionRate * 0.5 + accuracyRate * 0.3 + avgDifficulty * 0.2) * Double(maxScore)

        return Int(score.rounded())
    }

    private func difficultyMultiplier(_ difficulty: Difficulty) -> Double {
        switch difficulty {
        case .beginner: return 0.5
        case .intermediate: return 0.75
        case .advanced: return 1.0
        case .expert: return 1.25
        }
    }

    // MARK: - Discipline Calculation (0-400)

    func calculateDiscipline(portfolio: Portfolio?) async -> DisciplineScores {
        guard let portfolio = portfolio else {
            return DisciplineScores.zero()
        }

        // 1. Contribution Streak (0-100)
        let contributionData = await fetchContributionHistory()
        let streak = contributionData.currentStreak
        let contributionPoints = calculateContributionPoints(
            streak: streak,
            targetAmount: contributionData.targetMonthly,
            actualAmount: contributionData.lastMonthTotal
        )

        // 2. Rebalancing Adherence (0-100)
        let rebalancingData = await fetchRebalancingHistory()
        let rebalancingPoints = calculateRebalancingPoints(
            timesRebalanced: rebalancingData.timesRebalanced,
            timesIgnored: rebalancingData.timesIgnored,
            currentDrift: portfolio.driftFromTarget
        )

        // 3. Tax Efficiency (0-100)
        let taxData = await fetchTaxHistory()
        let taxPoints = calculateTaxEfficiencyPoints(
            lossesHarvested: taxData.lossesHarvested,
            washSalesAvoided: taxData.washSalesAvoided,
            washSalesTriggered: taxData.washSalesTriggered
        )

        // 4. Emotional Control (0-100)
        let emotionalData = await fetchEmotionalDecisions()
        let emotionalPoints = calculateEmotionalControlPoints(
            panicSells: emotionalData.panicSells,
            fomoBuys: emotionalData.fomoBuys,
            disciplinedHolds: emotionalData.disciplinedHolds
        )

        // 5. Plan Adherence (0-100)
        let planPoints = calculatePlanAdherencePoints(
            portfolio: portfolio,
            target: portfolio.targetAllocation
        )

        return DisciplineScores(
            contributionStreak: streak,
            contributionPoints: contributionPoints,
            rebalancingPoints: rebalancingPoints,
            taxEfficiencyPoints: taxPoints,
            emotionalControlPoints: emotionalPoints,
            planAdherencePoints: planPoints
        )
    }

    private func calculateContributionPoints(streak: Int, targetAmount: Decimal, actualAmount: Decimal) -> Int {
        // Base points for consistency
        var points = 0

        // Streak bonuses
        if streak >= 12 { points += 40 }      // 1 year
        else if streak >= 6 { points += 30 }  // 6 months
        else if streak >= 3 { points += 20 }  // 3 months
        else if streak >= 1 { points += 10 }  // 1 month

        // Amount vs target
        let ratio = actualAmount / targetAmount
        if ratio >= 1.0 { points += 30 }      // Met or exceeded
        else if ratio >= 0.8 { points += 20 } // 80%+
        else if ratio >= 0.5 { points += 10 } // 50%+

        return min(points, 100)
    }

    private func calculateRebalancingPoints(timesRebalanced: Int, timesIgnored: Int, currentDrift: Decimal) -> Int {
        var points = 0

        // Rebalancing when needed (+10 per time)
        points += timesRebalanced * 10

        // Penalty for ignoring (-5 per time)
        points -= timesIgnored * 5

        // Current drift penalty
        let driftPct = abs(Double(truncating: currentDrift as NSNumber))
        if driftPct < 3 { points += 20 }      // Well balanced
        else if driftPct < 5 { points += 10 } // Acceptable
        else if driftPct > 10 { points -= 20 }// Way off

        return max(0, min(points, 100))
    }

    private func calculateTaxEfficiencyPoints(
        lossesHarvested: Int,
        washSalesAvoided: Int,
        washSalesTriggered: Int
    ) -> Int {
        var points = 0

        // Harvesting losses (+5 per harvest)
        points += lossesHarvested * 5

        // Avoiding wash sales (+10 per avoided)
        points += washSalesAvoided * 10

        // Triggering wash sales (-15 per triggered)
        points -= washSalesTriggered * 15

        return max(0, min(points, 100))
    }

    private func calculateEmotionalControlPoints(
        panicSells: Int,
        fomoBuys: Int,
        disciplinedHolds: Int
    ) -> Int {
        var points = 50 // Start neutral

        // Penalties for emotional decisions
        points -= panicSells * 20
        points -= fomoBuys * 15

        // Rewards for discipline
        points += disciplinedHolds * 10

        return max(0, min(points, 100))
    }

    private func calculatePlanAdherencePoints(portfolio: Portfolio, target: Allocation) -> Int {
        // Compare current vs target allocation
        let stockDiff = abs(Double(truncating: (portfolio.currentAllocation.stocks - target.stocks) as NSNumber))
        let cryptoDiff = abs(Double(truncating: (portfolio.currentAllocation.crypto - target.crypto) as NSNumber))
        let bondDiff = abs(Double(truncating: (portfolio.currentAllocation.bonds - target.bonds) as NSNumber))

        let totalDrift = stockDiff + cryptoDiff + bondDiff

        // Score inversely proportional to drift
        if totalDrift < 5 { return 100 }
        else if totalDrift < 10 { return 80 }
        else if totalDrift < 15 { return 60 }
        else if totalDrift < 20 { return 40 }
        else { return 20 }
    }

    // MARK: - Wisdom Calculation (0-300)

    func calculateWisdom(decisions: [Decision]) async -> WisdomScores {
        guard !decisions.isEmpty else {
            return WisdomScores(recentDecisions: [], totalPoints: 0, learningRate: 0)
        }

        // Recent decisions (last 12 months)
        let calendar = Calendar.current
        let oneYearAgo = calendar.date(byAdding: .year, value: -1, to: Date())!
        let recentDecisions = decisions.filter { $0.date >= oneYearAgo }

        // Calculate total points from decisions
        let totalPoints = recentDecisions.map { $0.points }.reduce(0, +)

        // Calculate learning rate (points gained per month)
        let sixMonthsAgo = calendar.date(byAdding: .month, value: -6, to: Date())!
        let recentPoints = decisions.filter { $0.date >= sixMonthsAgo }.map { $0.points }.reduce(0, +)
        let olderPoints = decisions.filter { $0.date < sixMonthsAgo && $0.date >= oneYearAgo }.map { $0.points }.reduce(0, +)

        let learningRate = (recentPoints - olderPoints) / 6 // Points per month

        // Cap total at 300
        let cappedTotal = min(totalPoints, 300)

        return WisdomScores(
            recentDecisions: recentDecisions.sorted { $0.date > $1.date }.prefix(10).map { decision in
                WisdomScores.Decision(
                    id: UUID(),
                    date: decision.date,
                    action: decision.action,
                    thesis: decision.thesis,
                    outcome: decision.outcome,
                    points: decision.points,
                    lesson: decision.lesson
                )
            },
            totalPoints: cappedTotal,
            learningRate: learningRate
        )
    }

    // MARK: - Percentile Calculation

    private func calculatePercentile(_ score: Int) -> Int {
        // Percentile based on normal distribution
        // Mean: 500, StdDev: 150

        let mean = 500.0
        let stdDev = 150.0
        let z = (Double(score) - mean) / stdDev

        // Convert z-score to percentile (approximation)
        let percentile = 50.0 + 34.13 * z // 1 std dev ≈ 84th percentile

        return max(1, min(99, Int(percentile.rounded())))
    }

    // MARK: - Data Fetching (Core Data)

    private func fetchCompletedExercises() async -> [CompletedExercise] {
        // TODO: Fetch from Core Data
        return []
    }

    private func fetchContributionHistory() async -> ContributionData {
        // TODO: Fetch from Core Data
        return ContributionData(currentStreak: 0, targetMonthly: 1000, lastMonthTotal: 0)
    }

    private func fetchRebalancingHistory() async -> RebalancingData {
        // TODO: Fetch from Core Data
        return RebalancingData(timesRebalanced: 0, timesIgnored: 0)
    }

    private func fetchTaxHistory() async -> TaxData {
        // TODO: Fetch from Core Data
        return TaxData(lossesHarvested: 0, washSalesAvoided: 0, washSalesTriggered: 0)
    }

    private func fetchEmotionalDecisions() async -> EmotionalData {
        // TODO: Fetch from Core Data
        return EmotionalData(panicSells: 0, fomoBuys: 0, disciplinedHolds: 0)
    }

    private func fetchFQHistory() async -> FQHistory {
        // TODO: Fetch from Core Data
        return FQHistory(thirtyDaysAgo: nil, ninetyDaysAgo: nil)
    }
}

// MARK: - Supporting Types

struct CompletedExercise {
    let category: ExerciseCategory
    let difficulty: Difficulty
    let attempts: Int
    let correct: Bool
}

struct ContributionData {
    let currentStreak: Int
    let targetMonthly: Decimal
    let lastMonthTotal: Decimal
}

struct RebalancingData {
    let timesRebalanced: Int
    let timesIgnored: Int
}

struct TaxData {
    let lossesHarvested: Int
    let washSalesAvoided: Int
    let washSalesTriggered: Int
}

struct EmotionalData {
    let panicSells: Int
    let fomoBuys: Int
    let disciplinedHolds: Int
}

struct FQHistory {
    let thirtyDaysAgo: FQScore?
    let ninetyDaysAgo: FQScore?
}

// Decision model (matches models file)
struct Decision {
    let date: Date
    let action: String
    let thesis: String
    let outcome: String
    let points: Int
    let lesson: String
}

extension DisciplineScores {
    static func zero() -> DisciplineScores {
        return DisciplineScores(
            contributionStreak: 0,
            contributionPoints: 0,
            rebalancingPoints: 0,
            taxEfficiencyPoints: 0,
            emotionalControlPoints: 0,
            planAdherencePoints: 0
        )
    }
}

// MARK: - FQ Level Classification

extension FQScore {
    var level: FQLevel {
        switch total {
        case 0..<201:
            return .beginner
        case 201..<401:
            return .foundation
        case 401..<601:
            return .buildingWealth
        case 601..<801:
            return .advanced
        case 801...1000:
            return .master
        default:
            return .beginner
        }
    }

    var levelDescription: String {
        level.description
    }
}

enum FQLevel: String {
    case beginner = "Beginner"
    case foundation = "Foundation"
    case buildingWealth = "Building Wealth"
    case advanced = "Advanced"
    case master = "Master"

    var description: String {
        switch self {
        case .beginner: return "Learning the basics"
        case .foundation: return "Building good habits"
        case .buildingWealth: return "Consistent execution"
        case .advanced: return "Wise long-term decisions"
        case .master: return "Warren Buffett tier"
        }
    }

    var color: String {
        switch self {
        case .beginner: return "gray"
        case .foundation: return "blue"
        case .buildingWealth: return "green"
        case .advanced: return "purple"
        case .master: return "gold"
        }
    }
}
