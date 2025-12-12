# Gazillioner + Claude AI Integration
## Secure Implementation Guide

**Date:** December 10, 2025
**API:** Anthropic Claude (claude-3-5-sonnet-20241022)

---

## ⚠️ SECURITY FIRST: Never Expose API Keys!

### What You Did Wrong (Learning Moment)

**❌ NEVER DO THIS:**
```typescript
// DON'T hardcode API keys
const apiKey = "sk-ant-api03-xxxxx";  // EXPOSED!

// DON'T commit to git
git add config.ts  // Contains API key!

// DON'T share in chat/email
"Here's my key: sk-ant-api03-xxxxx"  // ANYONE CAN USE THIS!
```

**✅ ALWAYS DO THIS:**
```typescript
// Use environment variables
const apiKey = process.env.ANTHROPIC_API_KEY;

// Add to .gitignore
echo ".env.local" >> .gitignore

// Share securely
// Use password managers (1Password, LastPass)
// Or encrypted channels (Signal, PGP)
```

---

## 1. Secure Setup (Step-by-Step)

### Step 1: Create New API Key

1. Go to: https://console.anthropic.com/settings/keys
2. **Revoke the old key** you accidentally exposed
3. Click "Create Key"
4. Name: "Gazillioner Production"
5. **Copy the key** (shows only once!)
6. Save in password manager (1Password, LastPass, etc.)

### Step 2: Create `.env.local` File

In your project root:

```bash
# Create .env.local (this file is git-ignored)
touch .env.local

# Open in editor
code .env.local  # or nano, vim, etc.
```

Add this content:

```bash
# .env.local (NEVER commit this file!)

# Anthropic Claude API
ANTHROPIC_API_KEY=your-new-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/gazillioner

# NextAuth
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-random-secret-here

# Other APIs
PLAID_CLIENT_ID=your-plaid-client-id
PLAID_SECRET=your-plaid-secret
STRIPE_SECRET_KEY=your-stripe-secret
```

### Step 3: Ensure `.gitignore` Protects Secrets

```bash
# Check .gitignore includes:
cat .gitignore | grep .env.local

# If not found, add it:
echo ".env.local" >> .gitignore
echo ".env*.local" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
```

### Step 4: Never Accidentally Commit Secrets

```bash
# Before committing, always check:
git status

# Make sure .env.local is NOT listed
# If it is:
git reset HEAD .env.local

# Use git-secrets (optional but recommended):
brew install git-secrets  # macOS
git secrets --install
git secrets --register-aws
```

---

## 2. Install Anthropic SDK

```bash
npm install @anthropic-ai/sdk
```

**TypeScript types included automatically!**

---

## 3. Create Claude Chat API Route

### File: `app/api/chat/route.ts`

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { NextResponse } from 'next/server';

// Initialize Claude client (API key from environment)
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

export async function POST(request: Request) {
  try {
    const { message, userId } = await request.json();

    // Get user context (portfolio, FQ score, etc.)
    const userContext = await getUserContext(userId);

    // Create system prompt with user context
    const systemPrompt = `You are Gazillioner AI, a personal wealth advisor helping users improve their Financial Quotient (FQ) and make better investment decisions.

User Context:
- FQ Score: ${userContext.fqScore}/1000
- Portfolio Value: $${userContext.portfolioValue.toLocaleString()}
- Allocation: ${userContext.allocation.stocks}% stocks, ${userContext.allocation.crypto}% crypto, ${userContext.allocation.bonds}% bonds
- Top Holdings: ${userContext.topHoldings.join(', ')}
- Active Theses: ${userContext.theses.join(', ')}

Your role:
- Provide personalized financial advice based on their portfolio
- Help improve their FQ score through education
- Detect behavioral mistakes (FOMO, panic selling)
- Suggest tax optimization opportunities
- Explain complex financial concepts simply

Be concise, actionable, and supportive. If suggesting trades, explain WHY.`;

    // Call Claude API
    const response = await anthropic.messages.create({
      model: process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: message,
        },
      ],
    });

    // Extract text from response
    const aiMessage = response.content[0].type === 'text'
      ? response.content[0].text
      : '';

    // Log for analytics (optional)
    await logChatMessage(userId, message, aiMessage);

    return NextResponse.json({
      message: aiMessage,
      usage: response.usage,  // Track token usage for costs
    });

  } catch (error) {
    console.error('Claude API error:', error);

    // Handle different error types
    if (error.status === 401) {
      return NextResponse.json(
        { error: 'Invalid API key. Please check environment variables.' },
        { status: 401 }
      );
    }

    if (error.status === 429) {
      return NextResponse.json(
        { error: 'Rate limit exceeded. Please try again in a moment.' },
        { status: 429 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to get response from AI. Please try again.' },
      { status: 500 }
    );
  }
}

// Helper: Get user context from database
async function getUserContext(userId: string) {
  // Replace with your actual database queries
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      portfolio: true,
      fqScore: true,
      theses: true,
    },
  });

  return {
    fqScore: user.fqScore?.total || 0,
    portfolioValue: user.portfolio?.totalValue || 0,
    allocation: {
      stocks: user.portfolio?.allocation?.stocks || 0,
      crypto: user.portfolio?.allocation?.crypto || 0,
      bonds: user.portfolio?.allocation?.bonds || 0,
    },
    topHoldings: user.portfolio?.holdings
      ?.slice(0, 3)
      .map(h => h.ticker) || [],
    theses: user.theses?.map(t => t.name) || [],
  };
}

// Helper: Log chat messages (for analytics)
async function logChatMessage(userId: string, userMessage: string, aiMessage: string) {
  await prisma.chatMessage.create({
    data: {
      userId,
      userMessage,
      aiMessage,
      timestamp: new Date(),
    },
  });
}
```

---

## 4. Create Chat Frontend Component

### File: `components/ChatInterface.tsx`

```typescript
'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function ChatInterface({ userId }: { userId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          userId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const aiMessage: Message = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        // Error handling
        const errorMessage: Message = {
          role: 'assistant',
          content: data.error || 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Connection error. Please check your internet and try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <h2 className="text-lg font-semibold text-white">Gazillioner AI</h2>
        <p className="text-sm text-gray-400">Your personal wealth advisor</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p>👋 Hi! I'm your AI financial advisor.</p>
            <p className="mt-2">Ask me anything about your portfolio, FQ score, or finances.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === 'user'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-800 text-gray-100'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              <p className="text-xs mt-1 opacity-70">
                {msg.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-lg px-4 py-2 flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
              <span className="text-gray-400">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Ask me anything about your finances..."
            className="flex-1 bg-gray-800 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-700 text-white rounded-lg px-4 py-2 transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

## 5. Add Chat to Dashboard

### File: `app/dashboard/page.tsx`

```typescript
import { ChatInterface } from '@/components/ChatInterface';
import { auth } from '@/lib/auth';

export default async function DashboardPage() {
  const session = await auth();

  if (!session) {
    redirect('/login');
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
      {/* Main Content (2/3 width on desktop) */}
      <div className="lg:col-span-2 space-y-6">
        {/* Portfolio Summary */}
        <PortfolioSummaryCard userId={session.user.id} />

        {/* Daily Briefing Preview */}
        <DailyBriefingCard userId={session.user.id} />

        {/* FQ Score */}
        <FQScoreCard userId={session.user.id} />
      </div>

      {/* Chat Sidebar (1/3 width on desktop, full width on mobile) */}
      <div className="lg:col-span-1">
        <div className="sticky top-6 h-[calc(100vh-3rem)]">
          <ChatInterface userId={session.user.id} />
        </div>
      </div>
    </div>
  );
}
```

---

## 6. Enhanced Features

### 6.1 Add Quick Actions

```typescript
// components/ChatInterface.tsx

const quickActions = [
  { icon: '💰', label: 'Daily Briefing', prompt: 'Show me today\'s briefing' },
  { icon: '📊', label: 'Portfolio', prompt: 'How is my portfolio doing?' },
  { icon: '🧠', label: 'FQ Score', prompt: 'What can I do to improve my FQ score?' },
  { icon: '💡', label: 'Tax Tips', prompt: 'Any tax-loss harvesting opportunities?' },
];

// In the component, before messages:
{messages.length === 0 && (
  <div className="grid grid-cols-2 gap-2 mt-4">
    {quickActions.map((action, idx) => (
      <button
        key={idx}
        onClick={() => {
          setInput(action.prompt);
          sendMessage();
        }}
        className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left transition-colors"
      >
        <span className="text-2xl">{action.icon}</span>
        <p className="text-sm text-gray-300 mt-1">{action.label}</p>
      </button>
    ))}
  </div>
)}
```

### 6.2 Add Rich Card Responses

```typescript
// Detect intents and return structured data
if (message.toLowerCase().includes('portfolio')) {
  const portfolio = await getPortfolio(userId);

  return NextResponse.json({
    message: aiMessage,
    card: {
      type: 'portfolio',
      data: portfolio,
    },
  });
}

// In frontend, render rich cards:
{msg.card?.type === 'portfolio' && (
  <PortfolioCard data={msg.card.data} />
)}
```

### 6.3 Add Conversation History

```typescript
// Store messages in database
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW()
);

// Load history on mount
useEffect(() => {
  async function loadHistory() {
    const res = await fetch(`/api/chat/history?userId=${userId}`);
    const history = await res.json();
    setMessages(history);
  }
  loadHistory();
}, [userId]);
```

---

## 7. Cost Management

### Monitor Claude API Usage

```typescript
// Track token usage
const response = await anthropic.messages.create({...});

console.log('Tokens used:', {
  input: response.usage.input_tokens,
  output: response.usage.output_tokens,
  total: response.usage.input_tokens + response.usage.output_tokens,
});

// Log to database for billing analytics
await prisma.apiUsage.create({
  data: {
    userId,
    service: 'claude',
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
    cost: calculateCost(response.usage),
  },
});
```

### Claude API Pricing (as of Dec 2024)

**Claude 3.5 Sonnet:**
- Input: $3 per million tokens
- Output: $15 per million tokens

**Example costs:**
- 1,000 chat messages (avg 500 tokens in/out each):
  - Input: 500k tokens × $3/1M = $1.50
  - Output: 500k tokens × $15/1M = $7.50
  - **Total: $9/1000 messages**

**At scale (10k users, 10 messages/day):**
- 100k messages/day × $0.009 = **$900/day** = **$27k/month**
- Consider caching, shorter responses, or fine-tuned smaller model

---

## 8. Testing

### Test API Connection

```bash
# Create test script: test-claude.js
cat > test-claude.js << 'EOF'
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function test() {
  try {
    const response = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 100,
      messages: [
        {
          role: 'user',
          content: 'Say hello!',
        },
      ],
    });

    console.log('✅ Claude API working!');
    console.log('Response:', response.content[0].text);
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

test();
EOF

# Run test
node test-claude.js
```

**Expected output:**
```
✅ Claude API working!
Response: Hello! How can I help you today?
```

---

## 9. Deployment (Vercel)

### Set Environment Variables in Vercel

```bash
# Via Vercel CLI
vercel env add ANTHROPIC_API_KEY production
# Paste your NEW API key when prompted

# Or via Vercel Dashboard:
# 1. Go to https://vercel.com/your-project/settings/environment-variables
# 2. Add ANTHROPIC_API_KEY
# 3. Value: your-new-api-key
# 4. Environments: Production, Preview, Development
# 5. Save
```

### Deploy

```bash
git add .
git commit -m "feat: Add Claude AI chat integration"
git push

# Vercel auto-deploys (if connected to GitHub)
# Or manual deploy:
vercel --prod
```

---

## 10. Security Checklist

Before going live:

- [ ] API key revoked (the one you accidentally shared)
- [ ] New API key created and stored in `.env.local`
- [ ] `.env.local` is in `.gitignore`
- [ ] Environment variables set in Vercel dashboard
- [ ] No API keys in any committed code
- [ ] Rate limiting implemented (prevent abuse)
- [ ] Input validation (sanitize user messages)
- [ ] Output validation (check for prompt injection)
- [ ] Cost monitoring (set budget alerts in Anthropic console)
- [ ] Error handling (don't leak sensitive info in errors)

---

## 11. Next Steps

**Week 1:**
- [ ] Revoke old API key ⚠️ URGENT
- [ ] Set up secure environment variables
- [ ] Implement basic chat API route
- [ ] Test with simple messages
- [ ] Deploy to staging

**Week 2:**
- [ ] Add user context (FQ score, portfolio)
- [ ] Implement quick actions
- [ ] Add conversation history
- [ ] Polish UI (animations, loading states)
- [ ] Deploy to production

**Week 3:**
- [ ] Add rich card responses (portfolio summaries, charts)
- [ ] Implement proactive suggestions ("You should rebalance")
- [ ] Add voice input (Whisper API)
- [ ] Monitor costs and optimize

---

## You're Ready to Build (Securely)! 🔒

**Remember:**
1. ✅ Always use environment variables
2. ✅ Never commit secrets to git
3. ✅ Rotate keys if exposed
4. ✅ Monitor usage and costs
5. ✅ Implement rate limiting

**Start here:**
1. Revoke the old key (https://console.anthropic.com/settings/keys)
2. Create new key
3. Add to `.env.local`
4. Test with `test-claude.js`
5. Build the chat interface!

🚀 Let's build this the right way!
