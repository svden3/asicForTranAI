import { ChatInterface } from '@/components/ChatInterface';

export default function ChatTestPage() {
  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">
            Gazillioner AI Chat Test
          </h1>
          <p className="text-gray-400">
            Testing Claude AI integration with personalized financial advice
          </p>
          <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-lg">
            <p className="text-sm text-gray-300 mb-2">
              <strong className="text-green-400">Demo User Context:</strong>
            </p>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• FQ Score: 560/1000</li>
              <li>• Portfolio Value: $127,400</li>
              <li>• Allocation: 72% stocks, 8% crypto, 13% bonds, 7% cash</li>
              <li>• Top Holdings: NVDA, AAPL, BTC</li>
              <li>• Investment Theses: AI Revolution, Clean Energy Transition</li>
            </ul>
          </div>
        </div>

        <div className="h-[calc(100vh-16rem)]">
          <ChatInterface userId="test-user" />
        </div>
      </div>
    </div>
  );
}
