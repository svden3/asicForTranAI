import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-950 to-gray-900 flex items-center justify-center p-6">
      <div className="max-w-4xl mx-auto text-center">
        <div className="mb-8">
          <h1 className="text-6xl font-bold text-white mb-4">
            💰 Gazillioner
          </h1>
          <p className="text-2xl text-green-400 mb-2">
            Your AI Wealth Advisor
          </p>
          <p className="text-gray-400 text-lg">
            Improve your Financial Quotient (FQ) with personalized coaching
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="text-4xl mb-4">🧠</div>
            <h3 className="text-xl font-semibold text-white mb-2">FQ Scoring</h3>
            <p className="text-gray-400 text-sm">
              Measure and improve your financial decision-making (0-1000 scale)
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="text-4xl mb-4">💼</div>
            <h3 className="text-xl font-semibold text-white mb-2">Portfolio AI</h3>
            <p className="text-gray-400 text-sm">
              Stocks, crypto, bonds, options - all in one intelligent dashboard
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="text-4xl mb-4">📈</div>
            <h3 className="text-xl font-semibold text-white mb-2">Behavioral Coaching</h3>
            <p className="text-gray-400 text-sm">
              AI detects FOMO, panic selling, and guides better decisions
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <Link
            href="/chat-test"
            className="inline-block bg-green-600 hover:bg-green-700 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-colors"
          >
            🚀 Try AI Chat Demo
          </Link>
          <p className="text-gray-500 text-sm">
            Powered by Claude 3.5 Sonnet
          </p>
        </div>

        <div className="mt-12 pt-8 border-t border-gray-800">
          <p className="text-gray-600 text-sm">
            Status: MVP in Development | December 2025
          </p>
        </div>
      </div>
    </div>
  );
}
