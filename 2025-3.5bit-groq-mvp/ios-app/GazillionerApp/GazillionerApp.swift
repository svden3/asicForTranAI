// GazillionerApp.swift
// Main application entry point for Gazillioner FQ Platform
//
// Created: December 10, 2025
// Platform: iOS 17+, iPadOS 17+ (optimized for iPad Pro M2/M4)

import SwiftUI
import CoreData

@main
struct GazillionerApp: App {
    @StateObject private var appState = AppState()
    @Environment(\.scenePhase) private var scenePhase

    // Core Data persistence controller
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
                .environmentObject(appState)
                .onAppear {
                    setupApp()
                }
        }
        .onChange(of: scenePhase) { newPhase in
            handleScenePhaseChange(newPhase)
        }
    }

    private func setupApp() {
        // Initialize AI engine on background thread
        Task.detached(priority: .high) {
            await appState.initializeAI()
        }

        // Configure analytics (privacy-preserving)
        configureAnalytics()

        // Check for daily briefing
        Task {
            await appState.checkDailyBriefing()
        }
    }

    private func handleScenePhaseChange(_ phase: ScenePhase) {
        switch phase {
        case .active:
            // App became active - refresh portfolio
            Task {
                await appState.refreshPortfolio()
            }
        case .background:
            // Save state
            persistenceController.save()
        case .inactive:
            break
        @unknown default:
            break
        }
    }

    private func configureAnalytics() {
        // Privacy-first analytics: only track feature usage, never PII
        Analytics.configure(
            privacyMode: .maximum,
            trackCrashes: true,
            trackFeatureUsage: true,
            trackFinancialData: false  // NEVER track portfolio data
        )
    }
}

// MARK: - App State
class AppState: ObservableObject {
    // AI Engine
    @Published var llama: LLaMAInference?
    @Published var isAIReady = false

    // User data
    @Published var fqScore: FQScore?
    @Published var portfolio: Portfolio?
    @Published var dailyBriefing: DailyBriefing?

    // UI state
    @Published var showOnboarding = false
    @Published var showDailyBriefing = false
    @Published var behavioralAlerts: [BehavioralAlert] = []

    // Services
    private let fqEngine = FQEngine()
    private let portfolioService = PortfolioService()
    private let behavioralAnalyzer = BehavioralAnalyzer()

    init() {
        checkIfFirstLaunch()
    }

    // MARK: - Initialization

    func initializeAI() async {
        do {
            print("🤖 Initializing LLaMA-13B (3.5-bit)...")
            let config = LLaMAInference.Config(
                modelSize: .medium,  // 13B
                maxSeqLength: 2048,
                temperature: 0.7,
                topP: 0.95
            )

            let engine = try LLaMAInference(config: config)

            await MainActor.run {
                self.llama = engine
                self.isAIReady = true
                print("✅ AI engine ready!")
            }
        } catch {
            print("❌ Failed to initialize AI: \(error)")
            // Fallback: disable AI features, use rule-based system
            await MainActor.run {
                self.isAIReady = false
            }
        }
    }

    func checkDailyBriefing() async {
        guard isAIReady, let llama = llama else { return }

        // Check if briefing already generated today
        let lastBriefingDate = UserDefaults.standard.object(forKey: "lastBriefingDate") as? Date
        let calendar = Calendar.current

        if let lastDate = lastBriefingDate,
           calendar.isDateInToday(lastDate) {
            // Already have today's briefing
            return
        }

        // Generate new briefing
        print("📰 Generating daily briefing...")
        let generator = DailyBriefingGenerator(llama: llama, portfolio: portfolio, fqScore: fqScore)
        let briefing = await generator.generateBriefing()

        await MainActor.run {
            self.dailyBriefing = briefing
            self.showDailyBriefing = true
            UserDefaults.standard.set(Date(), forKey: "lastBriefingDate")
        }
    }

    func refreshPortfolio() async {
        print("💼 Refreshing portfolio...")

        do {
            // Sync with brokers (via Plaid)
            let updatedPositions = try await portfolioService.syncPositions()

            // Update portfolio
            var newPortfolio = portfolio ?? Portfolio.empty()
            newPortfolio.positions = updatedPositions
            newPortfolio.lastUpdated = Date()

            // Recalculate FQ score
            let newFQScore = await fqEngine.calculateScore(
                portfolio: newPortfolio,
                historicalDecisions: getHistoricalDecisions()
            )

            // Check for behavioral alerts
            let alerts = await behavioralAnalyzer.analyzePortfolio(newPortfolio)

            await MainActor.run {
                self.portfolio = newPortfolio
                self.fqScore = newFQScore
                self.behavioralAlerts = alerts
            }

            print("✅ Portfolio refreshed")
        } catch {
            print("❌ Portfolio sync failed: \(error)")
        }
    }

    // MARK: - Helpers

    private func checkIfFirstLaunch() {
        let hasLaunchedBefore = UserDefaults.standard.bool(forKey: "hasLaunchedBefore")
        if !hasLaunchedBefore {
            showOnboarding = true
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
        }
    }

    private func getHistoricalDecisions() -> [Decision] {
        // Fetch from Core Data
        // TODO: Implement Core Data fetch
        return []
    }
}

// MARK: - Content View
struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedTab = 0

    var body: some View {
        if appState.showOnboarding {
            OnboardingView()
        } else {
            TabView(selection: $selectedTab) {
                DashboardView()
                    .tabItem {
                        Label("Dashboard", systemImage: "chart.line.uptrend.xyaxis")
                    }
                    .tag(0)

                PortfolioView()
                    .tabItem {
                        Label("Portfolio", systemImage: "chart.pie.fill")
                    }
                    .tag(1)

                ExercisesView()
                    .tabItem {
                        Label("Learn", systemImage: "brain.head.profile")
                    }
                    .tag(2)

                ThesisTrackerView()
                    .tabItem {
                        Label("Theses", systemImage: "lightbulb.fill")
                    }
                    .tag(3)

                SettingsView()
                    .tabItem {
                        Label("Settings", systemImage: "gear")
                    }
                    .tag(4)
            }
            .sheet(isPresented: $appState.showDailyBriefing) {
                if let briefing = appState.dailyBriefing {
                    DailyBriefingView(briefing: briefing)
                }
            }
            .overlay {
                if !appState.isAIReady {
                    AILoadingOverlay()
                }
            }
        }
    }
}

struct AILoadingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)

                Text("Loading AI Engine...")
                    .font(.headline)
                    .foregroundColor(.white)

                Text("Preparing your personalized financial coach")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            .padding(40)
            .background(Color.black.opacity(0.8))
            .cornerRadius(20)
        }
    }
}

// MARK: - Analytics (Privacy-Preserving)
struct Analytics {
    enum PrivacyMode {
        case maximum    // No tracking beyond crashes
        case balanced   // Feature usage only
        case standard   // Feature + performance
    }

    static func configure(privacyMode: PrivacyMode, trackCrashes: Bool, trackFeatureUsage: Bool, trackFinancialData: Bool) {
        // NEVER track financial data
        assert(trackFinancialData == false, "Privacy violation: cannot track financial data")

        // Configure Crashlytics or similar
        print("📊 Analytics configured: \(privacyMode)")
    }

    static func track(event: String, properties: [String: Any]? = nil) {
        // Only track feature usage, never PII or financial data
        let allowedEvents = ["exercise_completed", "briefing_viewed", "alert_dismissed"]
        guard allowedEvents.contains(event) else { return }

        print("📊 Event: \(event)")
    }
}

// MARK: - Persistence Controller
class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentCloudKitContainer

    init(inMemory: Bool = false) {
        container = NSPersistentCloudKitContainer(name: "Gazillioner")

        if inMemory {
            container.persistentStoreDescriptions.first!.url = URL(fileURLWithPath: "/dev/null")
        }

        // Enable CloudKit sync (optional, user can disable)
        container.persistentStoreDescriptions.first?.cloudKitContainerOptions = NSPersistentCloudKitContainerOptions(
            containerIdentifier: "iCloud.com.gazillioner.app"
        )

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Core Data failed to load: \(error.localizedDescription)")
            }
        }

        // Auto-merge changes from CloudKit
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }

    func save() {
        let context = container.viewContext
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("❌ Failed to save Core Data: \(error)")
            }
        }
    }
}
