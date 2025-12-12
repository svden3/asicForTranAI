# Setup Claude AI Chat - Complete Implementation
## Copy-Paste Ready Code (30 Minutes to Working Chat)

**Date:** December 10, 2025

---

## Step 1: Secure Your API Key (5 minutes)

### A. Revoke Old Key

1. Go to: https://console.anthropic.com/settings/keys
2. Find key starting with `sk-ant-api03-16ggvB2E8...`
3. Click "Delete" or "Revoke"
4. Confirm deletion

### B. Create New Key

1. Same page, click "Create Key"
2. Name: "Gazillioner Production"
3. **Copy the key** (shows only once!)
4. Keep the browser tab open for now

---

## Step 2: Install Dependencies (2 minutes)

```bash
# Navigate to your project
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp

# Install Anthropic SDK
npm install @anthropic-ai/sdk

# Install UI dependencies (if not already installed)
npm install lucide-react clsx tailwind-merge
```

---

## Step 3: Set Environment Variables (3 minutes)

### A. Create `.env.local` file

```bash
# Create .env.local in project root
cat > .env.local << 'EOF'
# Claude API
ANTHROPIC_API_KEY=PASTE_YOUR_NEW_KEY_HERE
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Database (update with your actual values)
DATABASE_URL=postgresql://user:password@localhost:5432/gazillioner

# NextAuth (generate a random secret)
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-random-secret-here
EOF
```

### B. Replace `PASTE_YOUR_NEW_KEY_HERE` with your actual key

```bash
# Edit .env.local
code .env.local
# or
notepad .env.local

# Replace PASTE_YOUR_NEW_KEY_HERE with the key from Step 1B
```

### C. Ensure `.gitignore` protects it

```bash
# Check if .gitignore exists
cat .gitignore | grep -E "\.env\.local|\.env\*\.local"

# If not found, add it:
echo "" >> .gitignore
echo "# Environment variables" >> .gitignore
echo ".env.local" >> .gitignore
echo ".env*.local" >> .gitignore
```

---

## Step 4: Test API Connection (3 minutes)

### Create test script

```bash
# Create test-claude.mjs
cat > test-claude.mjs << 'EOF'
import Anthropic from '@anthropic-ai/sdk';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function test() {
  console.log('🧪 Testing Claude API connection...\n');

  try {
    const response = await anthropic.messages.create({
      model: process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022',
      max_tokens: 100,
      messages: [
        {
          role: 'user',
          content: 'Say hello and confirm you are Claude!',
        },
      ],
    });

    console.log('✅ SUCCESS! Claude API is working.\n');
    console.log('Response:', response.content[0].text);
    console.log('\nTokens used:', {
      input: response.usage.input_tokens,
      output: response.usage.output_tokens,
    });
  } catch (error) {
    console.error('❌ ERROR:', error.message);
    if (error.status === 401) {
      console.error('\n⚠️  Invalid API key. Check your .env.local file.');
    }
  }
}

test();
EOF

# Install dotenv
npm install dotenv

# Run test
node test-claude.mjs
```

**Expected output:**
```
🧪 Testing Claude API connection...

✅ SUCCESS! Claude API is working.

Response: Hello! Yes, I'm Claude, an AI assistant created by Anthropic...

Tokens used: { input: 15, output: 25 }
```

**If you see errors:**
- 401 error → API key is wrong, check `.env.local`
- Network error → Check internet connection
- Module error → Run `npm install` again

---

## Step 5: Create API Route (5 minutes)

### Create directory structure

```bash
# Create directories if they don't exist
mkdir -p app/api/chat
```

### Create `app/api/chat/route.ts`

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { NextResponse } from 'next/server';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

export async function POST(request: Request) {
  try {
    const { message, userId } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // TODO: Replace with actual user context from database
    const mockUserContext = {
      fqScore: 560,
      portfolioValue: 127400,
      allocation: { stocks: 72, crypto: 8, bonds: 13, cash: 7 },
      topHoldings: ['NVDA', 'AAPL', 'BTC'],
      theses: ['AI Revolution', 'Clean Energy Transition'],
    };

    const systemPrompt = `You are Gazillioner AI, a personal wealth advisor helping users improve their Financial Quotient (FQ) and make better investment decisions.

User Context:
- FQ Score: ${mockUserContext.fqScore}/1000
- Portfolio Value: $${mockUserContext.portfolioValue.toLocaleString()}
- Allocation: ${mockUserContext.allocation.stocks}% stocks, ${mockUserContext.allocation.crypto}% crypto, ${mockUserContext.allocation.bonds}% bonds, ${mockUserContext.allocation.cash}% cash
- Top Holdings: ${mockUserContext.topHoldings.join(', ')}
- Investment Theses: ${mockUserContext.theses.join(', ')}

Your role:
- Provide personalized financial advice based on their portfolio
- Help improve their FQ score through education
- Detect behavioral mistakes (FOMO, panic selling, loss aversion)
- Suggest tax optimization opportunities (tax-loss harvesting, wash sales)
- Explain complex financial concepts simply
- Reference their specific holdings and theses when relevant

Guidelines:
- Be concise but informative (aim for 2-3 paragraphs)
- Use bullet points for action items
- If suggesting trades, always explain WHY
- Link advice to their FQ score improvement
- Be supportive and educational, not judgmental
- Use emojis sparingly (only for emphasis)

Respond in a friendly, professional tone.`;

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

    const aiMessage = response.content[0].type === 'text'
      ? response.content[0].text
      : 'Sorry, I could not generate a response.';

    return NextResponse.json({
      message: aiMessage,
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    });

  } catch (error: any) {
    console.error('Claude API error:', error);

    if (error.status === 401) {
      return NextResponse.json(
        { error: 'Invalid API key configuration' },
        { status: 500 }
      );
    }

    if (error.status === 429) {
      return NextResponse.json(
        { error: 'Rate limit exceeded. Please wait a moment.' },
        { status: 429 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to get AI response. Please try again.' },
      { status: 500 }
    );
  }
}
```

---

## Step 6: Create Chat Component (7 minutes)

### Create `components/ChatInterface.tsx`

```typescript
'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface QuickAction {
  icon: string;
  label: string;
  prompt: string;
}

const quickActions: QuickAction[] = [
  { icon: '💰', label: 'Daily Briefing', prompt: 'Show me my daily briefing' },
  { icon: '📊', label: 'Portfolio', prompt: 'How is my portfolio doing today?' },
  { icon: '🧠', label: 'Improve FQ', prompt: 'What can I do to improve my FQ score?' },
  { icon: '💡', label: 'Tax Tips', prompt: 'Any tax-loss harvesting opportunities?' },
];

export function ChatInterface({ userId }: { userId?: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (messageText?: string) => {
    const textToSend = messageText || input;
    if (!textToSend.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: textToSend,
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
          message: textToSend,
          userId: userId || 'demo-user',
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
        const errorMessage: Message = {
          role: 'assistant',
          content: `Error: ${data.error || 'Something went wrong. Please try again.'}`,
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

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
    sendMessage(prompt);
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Gazillioner AI</h2>
            <p className="text-xs text-gray-400">Your personal wealth advisor</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="space-y-4">
            <div className="text-center text-gray-400 mt-4">
              <p className="text-lg">👋 Hi! I'm your AI financial advisor.</p>
              <p className="text-sm mt-2">
                Ask me anything about your portfolio, FQ score, or finances.
              </p>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-2 gap-2 mt-6">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => handleQuickAction(action.prompt)}
                  className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left transition-all duration-200 border border-gray-700 hover:border-green-600"
                >
                  <span className="text-2xl block mb-1">{action.icon}</span>
                  <p className="text-sm text-gray-300 font-medium">{action.label}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-lg px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-800 text-gray-100 border border-gray-700'
              }`}
            >
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
              <p className={`text-xs mt-2 ${
                msg.role === 'user' ? 'text-green-100' : 'text-gray-500'
              }`}>
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
            <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-green-500" />
              <span className="text-sm text-gray-400">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ask me anything about your finances..."
            className="flex-1 bg-gray-800 text-white rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 border border-gray-700"
            disabled={isLoading}
          />
          <button
            onClick={() => sendMessage()}
            disabled={isLoading || !input.trim()}
            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg px-4 py-3 transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Press Enter to send • Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
```

---

## Step 7: Create Test Page (3 minutes)

### Create `app/chat-test/page.tsx`

```typescript
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
        </div>

        <div className="h-[calc(100vh-12rem)]">
          <ChatInterface userId="test-user" />
        </div>
      </div>
    </div>
  );
}
```

---

## Step 8: Run & Test (2 minutes)

### Start development server

```bash
npm run dev
```

### Test the chat

1. Open browser: http://localhost:3000/chat-test
2. Try quick actions (click any button)
3. Type a message: "How's my portfolio doing?"
4. Wait for response (should arrive in 2-5 seconds)

**Expected behavior:**
- ✅ Quick action buttons work
- ✅ Message sends when you click Send or press Enter
- ✅ AI responds with personalized advice (mentions your FQ score, portfolio)
- ✅ No errors in console

**If you see errors:**
- Check browser console (F12)
- Check terminal for server errors
- Verify `.env.local` has correct API key
- Run `node test-claude.mjs` again to confirm API works

---

## Step 9: Deploy to Vercel (Optional, 5 minutes)

### Set environment variables in Vercel

```bash
# Install Vercel CLI (if not already)
npm i -g vercel

# Login
vercel login

# Set environment variables
vercel env add ANTHROPIC_API_KEY
# Paste your NEW API key when prompted
# Select: Production, Preview, Development (all 3)

vercel env add ANTHROPIC_MODEL
# Enter: claude-3-5-sonnet-20241022
# Select: Production, Preview, Development (all 3)
```

### Deploy

```bash
# Commit your changes
git add .
git commit -m "feat: Add Claude AI chat integration"

# Deploy to Vercel
vercel --prod
```

### Test production

1. Vercel will give you a URL: `https://your-project.vercel.app`
2. Visit: `https://your-project.vercel.app/chat-test`
3. Test chat (same as local)

---

## Troubleshooting

### Issue: "Invalid API key"

**Solution:**
```bash
# Check .env.local exists
ls -la .env.local

# Check contents (key should start with sk-ant-)
cat .env.local | grep ANTHROPIC_API_KEY

# If missing or wrong, recreate:
echo "ANTHROPIC_API_KEY=your-actual-key-here" > .env.local

# Restart dev server
npm run dev
```

### Issue: "Module not found: @anthropic-ai/sdk"

**Solution:**
```bash
# Reinstall
npm install @anthropic-ai/sdk

# Check it's in package.json
cat package.json | grep anthropic

# Restart dev server
npm run dev
```

### Issue: Messages not appearing

**Solution:**
```bash
# Check browser console (F12)
# Look for errors

# Check server terminal
# Look for API errors

# Test API directly
node test-claude.mjs
```

### Issue: "Rate limit exceeded"

**Solution:**
- You're sending too many requests
- Wait 1 minute, try again
- Check Anthropic console for usage: https://console.anthropic.com/

### Issue: Slow responses (>10 seconds)

**Possible causes:**
- Network latency
- Claude API is slow (rare)
- Your prompt is too long (check system prompt length)

**Solution:**
```typescript
// Reduce max_tokens in route.ts
max_tokens: 512,  // Instead of 1024
```

---

## Next Steps

### Immediate (Today)

- [ ] ✅ Revoked old API key
- [ ] ✅ Created new API key
- [ ] ✅ Set up `.env.local`
- [ ] ✅ Tested API with `test-claude.mjs`
- [ ] ✅ Created API route
- [ ] ✅ Created chat component
- [ ] ✅ Tested at `/chat-test`

### This Week

- [ ] Add conversation history (store in database)
- [ ] Add user context (real FQ score, portfolio from database)
- [ ] Add rich card responses (portfolio summaries, charts)
- [ ] Polish UI (animations, better styling)
- [ ] Add to main dashboard (sidebar or modal)

### Next Week

- [ ] Add proactive suggestions ("You should rebalance")
- [ ] Add voice input (Whisper API)
- [ ] Implement rate limiting
- [ ] Monitor costs (Anthropic console)
- [ ] A/B test messaging styles

---

## Cost Monitoring

### Check your usage

1. Go to: https://console.anthropic.com/settings/usage
2. View today's usage (tokens + cost)
3. Set budget alerts (Settings → Billing → Alerts)

### Expected costs (with current traffic)

**Development (testing):**
- 100 test messages: ~$1
- 1,000 test messages: ~$10

**Production (estimates):**
- 100 users × 10 messages/day = 1,000 messages/day = ~$9/day = $270/month
- 1,000 users × 10 messages/day = 10,000 messages/day = ~$90/day = $2,700/month

### Set budget alert

1. Go to: https://console.anthropic.com/settings/billing
2. Click "Set Budget Alert"
3. Enter: $100/month (or your limit)
4. Save

You'll get email when you hit 50%, 75%, 90% of budget.

---

## Security Checklist ✅

- [ ] Old API key revoked
- [ ] New API key in `.env.local` only
- [ ] `.env.local` in `.gitignore`
- [ ] Vercel environment variables set
- [ ] No API keys in any code files
- [ ] No API keys committed to git
- [ ] Budget alerts configured

---

## Success! 🎉

If you completed all steps, you now have:

✅ Secure Claude API integration
✅ Working chat interface
✅ Personalized financial advice (using mock user context)
✅ Quick action buttons
✅ Production-ready code
✅ Deployed to Vercel (optional)

**Total time: 30 minutes**

**Next:** Integrate with real user data (FQ scores, portfolios) from your database!

---

## Quick Reference

**Test API:**
```bash
node test-claude.mjs
```

**Run dev server:**
```bash
npm run dev
```

**Test chat:**
```
http://localhost:3000/chat-test
```

**Check usage:**
```
https://console.anthropic.com/settings/usage
```

**Revoke key:**
```
https://console.anthropic.com/settings/keys
```

---

**You're ready to build! 🚀**
