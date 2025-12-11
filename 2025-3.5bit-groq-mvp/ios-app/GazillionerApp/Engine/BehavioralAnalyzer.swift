// BehavioralAnalyzer.swift
// Detects emotional patterns (FOMO, panic, concentration risk)
// Prevents costly mistakes BEFORE they happen

import Foundation

class BehavioralAnalyzer {

    // MARK: - Pattern Detection

    func analyzePortfolio(_ portfolio: Portfolio) async -> [BehavioralAlert] {
        var alerts: [BehavioralAlert] = []

        // Check for various behavioral patterns
        alerts.append(contentsOf: await detectConcentrationRisk(portfolio))
        alerts.append(contentsOf: await detectDriftFromAllocation(portfolio))
        alerts.append(contentsOf: await detectTaxInefficiency(portfolio))
        alerts.append(contentsOf: await detectUnusualActivity(portfolio))

        return alerts.sorted { $0.severity.rawValue > $1.severity.rawValue }
    }

    func evaluateProposedTrade(_ trade: ProposedTrade, portfolio: Portfolio, theses: [Thesis]) async -> BehavioralPattern {
        // Detect if this trade shows emotional decision-making

        // 1. Check for panic selling
        if let panicContext = await detectPanicSelling(trade, portfolio: portfolio, theses: theses) {
            return .panicSelling(panicContext)
        }

        // 2. Check for FOMO chasing
        if let fomoContext = await detectFOMO(trade, portfolio: portfolio) {
            return .fomoChasing(fomoContext)
        }

        // 3. Check for wash sale risk
        if await detectWashSaleRisk(trade, portfolio: portfolio) {
            return .washSaleRisk
        }

        // Otherwise, seems rational
        return .rational
    }

    // MARK: - Panic Selling Detection

    private func detectPanicSelling(_ trade: ProposedTrade, portfolio: Portfolio, theses: [Thesis]) async -> PanicContext? {
        guard trade.action == .sell else { return nil }

        // Find the position being sold
        guard let position = portfolio.positions.first(where: { $0.symbol == trade.symbol }) else {
            return nil
        }

        // Check if position is down significantly
        let drawdown = position.unrealizedGainPct
        guard drawdown < -15 else { return nil } // Not a big enough loss

        // Check recent price action (is it STILL falling?)
        let recentPriceChange = await getRecentPriceChange(symbol: trade.symbol, days: 7)

        // Check if there's a linked thesis
        let linkedThesis = theses.first(where: { $0.positions.contains(trade.symbol) })

        // Check broader market context
        let marketDrawdown = await getMarketDrawdown()

        // PANIC indicators:
        // - Selling at a loss
        // - Stock down 15%+ from buy price
        // - User didn't specify a clear reason
        // - No thesis invalidation

        if drawdown < -15,
           trade.reason == nil || trade.reason!.isEmpty,
           linkedThesis?.strength != .broken {

            return PanicContext(
                originalThesis: linkedThesis?.title ?? "No thesis tracked",
                drawdown: drawdown,
                marketDrawdown: marketDrawdown,
                daysSinceBottom: await getDaysSinceBottom(symbol: trade.symbol),
                newsEvents: await getRecentNews(symbol: trade.symbol),
                thesisStillIntact: linkedThesis?.strength != .broken
            )
        }

        return nil
    }

    // MARK: - FOMO Detection

    private func detectFOMO(_ trade: ProposedTrade, portfolio: Portfolio) async -> FOMOContext? {
        guard trade.action == .buy else { return nil }

        // Check recent price movement
        let priceChange7d = await getRecentPriceChange(symbol: trade.symbol, days: 7)
        let priceChange30d = await getRecentPriceChange(symbol: trade.symbol, days: 30)

        // FOMO indicators:
        // - Stock up >20% in last 7 days
        // - Stock up >50% in last 30 days
        // - User has no thesis
        // - Position size too large

        let hasThesis = trade.thesis != nil && !trade.thesis!.isEmpty
        let positionSizePct = (trade.amount / portfolio.totalValue) * 100

        if priceChange7d > 20 || priceChange30d > 50 {
            return FOMOContext(
                priceChange7d: priceChange7d,
                priceChange30d: priceChange30d,
                hasThesis: hasThesis,
                positionSizePct: positionSizePct,
                socialMentions: await getSocialMentions(symbol: trade.symbol),
                newsVolume: await getNewsVolume(symbol: trade.symbol, days: 7)
            )
        }

        return nil
    }

    // MARK: - Concentration Risk

    private func detectConcentrationRisk(_ portfolio: Portfolio) async -> [BehavioralAlert] {
        var alerts: [BehavioralAlert] = []

        // Check largest position
        if let largestPosition = portfolio.positions.max(by: { $0.currentValue < $1.currentValue }) {
            let concentration = (largestPosition.currentValue / portfolio.totalValue) * 100

            if concentration > 30 {
                let alert = BehavioralAlert(
                    id: UUID(),
                    type: .concentrationRisk,
                    severity: concentration > 50 ? .critical : .warning,
                    title: "High Concentration Risk",
                    proposedAction: UserAction(
                        symbol: largestPosition.symbol,
                        action: .hold,
                        amount: 0,
                        reason: "Monitoring concentration"
                    ),
                    detectedAt: Date(),
                    options: [
                        AlertOption(
                            id: UUID(),
                            title: "Rebalance - Trim position",
                            description: "Reduce \(largestPosition.symbol) to 20% of portfolio",
                            fqImpact: 15,
                            recommended: true
                        ),
                        AlertOption(
                            id: UUID(),
                            title: "Hold - I'm confident",
                            description: "Keep concentration (high conviction thesis)",
                            fqImpact: 0,
                            recommended: false
                        )
                    ],
                    historicalContext: "Remember: Even great companies can crash. Enron, Lehman, FTX were once 'safe bets.'"
                )
                alerts.append(alert)
            }
        }

        return alerts
    }

    // MARK: - Drift Detection

    private func detectDriftFromAllocation(_ portfolio: Portfolio) async -> [BehavioralAlert] {
        var alerts: [BehavioralAlert] = []

        let drift = abs(Double(truncating: portfolio.driftFromTarget as NSNumber))

        if drift > 5 {
            let alert = BehavioralAlert(
                id: UUID(),
                type: .allocationDrift,
                severity: drift > 10 ? .warning : .info,
                title: "Portfolio Drifted from Target",
                proposedAction: UserAction(
                    symbol: "",
                    action: .rebalance,
                    amount: 0,
                    reason: "Allocation drift: \(String(format: "%.1f", drift))%"
                ),
                detectedAt: Date(),
                options: [
                    AlertOption(
                        id: UUID(),
                        title: "Rebalance Now",
                        description: "Sell winners, buy losers to restore target allocation",
                        fqImpact: 20,
                        recommended: true
                    ),
                    AlertOption(
                        id: UUID(),
                        title: "Wait - Let it ride",
                        description: "Accept drift (may miss rebalancing bonus)",
                        fqImpact: -5,
                        recommended: false
                    )
                ],
                historicalContext: "Rebalancing mechanically sells high, buys low. COVID crash rebalancers outperformed by 15%."
            )
            alerts.append(alert)
        }

        return alerts
    }

    // MARK: - Tax Inefficiency Detection

    private func detectTaxInefficiency(_ portfolio: Portfolio) async -> [BehavioralAlert] {
        var alerts: [BehavioralAlert] = []

        // Find positions with harvestable losses
        let lossPositions = portfolio.positions.filter { $0.unrealizedGain < -1000 }

        for position in lossPositions {
            // Check if wash sale risk
            let hasRecentSale = position.recentSales.contains { sale in
                let daysSince = Calendar.current.dateComponents([.day], from: sale.date, to: Date()).day ?? 0
                return daysSince < 30
            }

            if !hasRecentSale {
                let alert = BehavioralAlert(
                    id: UUID(),
                    type: .taxInefficiency,
                    severity: .info,
                    title: "Tax-Loss Harvesting Opportunity",
                    proposedAction: UserAction(
                        symbol: position.symbol,
                        action: .sell,
                        amount: position.currentValue,
                        reason: "Harvest \(String(format: "$%.0f", Double(truncating: abs(position.unrealizedGain) as NSNumber))) tax loss"
                    ),
                    detectedAt: Date(),
                    options: [
                        AlertOption(
                            id: UUID(),
                            title: "Harvest Loss",
                            description: "Sell now, save ~\(String(format: "$%.0f", Double(truncating: abs(position.unrealizedGain) as NSNumber) * 0.21)) in taxes",
                            fqImpact: 15,
                            recommended: true
                        ),
                        AlertOption(
                            id: UUID(),
                            title: "Hold - Expecting recovery",
                            description: "Keep position (forfeit tax benefit)",
                            fqImpact: 0,
                            recommended: false
                        )
                    ],
                    historicalContext: "You can rebuy in 31 days to avoid wash sale rule."
                )
                alerts.append(alert)
            }
        }

        return alerts
    }

    // MARK: - Wash Sale Detection

    private func detectWashSaleRisk(_ trade: ProposedTrade, portfolio: Portfolio) async -> Bool {
        guard trade.action == .buy else { return false }

        // Check if user sold this symbol in last 30 days
        guard let position = portfolio.positions.first(where: { $0.symbol == trade.symbol }) else {
            return false
        }

        return position.isWashSaleRisk
    }

    // MARK: - Unusual Activity

    private func detectUnusualActivity(_ portfolio: Portfolio) async -> [BehavioralAlert] {
        // Detect patterns like:
        // - Trading too frequently (day trading)
        // - Buying/selling same stock repeatedly
        // - Chasing trends

        // TODO: Implement based on transaction history

        return []
    }

    // MARK: - Market Data Helpers

    private func getRecentPriceChange(symbol: String, days: Int) async -> Decimal {
        // Fetch price change from MarketDataService
        // TODO: Implement
        return 0
    }

    private func getMarketDrawdown() async -> Decimal {
        // Get S&P 500 drawdown from recent high
        // TODO: Implement
        return 0
    }

    private func getDaysSinceBottom(symbol: String) async -> Int {
        // Calculate days since local minimum
        // TODO: Implement
        return 0
    }

    private func getRecentNews(symbol: String) async -> [String] {
        // Fetch recent news headlines
        // TODO: Implement via NewsAPI or similar
        return []
    }

    private func getSocialMentions(symbol: String) async -> Int {
        // Check social media mentions (Twitter, Reddit)
        // High mentions = potential FOMO
        // TODO: Implement
        return 0
    }

    private func getNewsVolume(symbol: String, days: Int) async -> Int {
        // Count news articles in last N days
        // Spike in volume = potential hype
        // TODO: Implement
        return 0
    }
}

// MARK: - Supporting Types

enum BehavioralPattern {
    case panicSelling(PanicContext)
    case fomoChasing(FOMOContext)
    case washSaleRisk
    case rational
}

struct PanicContext {
    let originalThesis: String
    let drawdown: Decimal           // -30%
    let marketDrawdown: Decimal     // -15%
    let daysSinceBottom: Int
    let newsEvents: [String]
    let thesisStillIntact: Bool
}

struct FOMOContext {
    let priceChange7d: Decimal
    let priceChange30d: Decimal
    let hasThesis: Bool
    let positionSizePct: Decimal
    let socialMentions: Int
    let newsVolume: Int
}

struct ProposedTrade {
    let symbol: String
    let action: TradeAction
    let amount: Decimal
    let reason: String?
    let thesis: String?
}

enum TradeAction {
    case buy
    case sell
    case hold
    case rebalance
}

struct BehavioralAlert: Identifiable {
    let id: UUID
    let type: AlertType
    let severity: Severity
    let title: String
    let proposedAction: UserAction
    let detectedAt: Date
    let options: [AlertOption]
    let historicalContext: String?

    var icon: String {
        switch type {
        case .panicSelling: return "exclamationmark.triangle.fill"
        case .fomoChasing: return "flame.fill"
        case .concentrationRisk: return "chart.pie.fill"
        case .taxInefficiency: return "dollarsign.circle.fill"
        case .allocationDrift: return "scale.3d"
        case .washSaleRisk: return "exclamationmark.circle.fill"
        }
    }

    enum AlertType {
        case panicSelling
        case fomoChasing
        case concentrationRisk
        case taxInefficiency
        case allocationDrift
        case washSaleRisk
    }

    enum Severity: Int {
        case critical = 3
        case warning = 2
        case info = 1

        var color: String {
            switch self {
            case .critical: return "red"
            case .warning: return "orange"
            case .info: return "blue"
            }
        }
    }
}

struct UserAction {
    let symbol: String
    let action: TradeAction
    let amount: Decimal
    let reason: String?

    var description: String {
        switch action {
        case .buy:
            return "Buy \(String(format: "$%.0f", Double(truncating: amount as NSNumber))) of \(symbol)"
        case .sell:
            return "Sell \(String(format: "$%.0f", Double(truncating: amount as NSNumber))) of \(symbol)"
        case .hold:
            return "Hold \(symbol)"
        case .rebalance:
            return "Rebalance portfolio"
        }
    }
}

struct AlertOption: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let fqImpact: Int              // +25, -20, etc.
    let recommended: Bool
}
