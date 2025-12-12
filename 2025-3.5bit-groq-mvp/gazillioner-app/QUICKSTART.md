# Gazillioner Quick Start Guide

Get the AI chat working in 10 minutes!

## Step 1: Set Up API Key (3 minutes)

### Get your Claude API key:

1. Go to https://console.anthropic.com/settings/keys
2. Sign in or create account
3. Click "Create Key"
4. Name it: "Gazillioner Development"
5. **Copy the key** (starts with `sk-ant-api03-`)

### Add to environment:

```bash
# Copy the example file
cp .env.example .env.local

# Edit the file (use your favorite editor)
code .env.local
# or
notepad .env.local

# Replace PASTE_YOUR_KEY_HERE with your actual API key
```

## Step 2: Test API Connection (2 minutes)

```bash
node test-claude.mjs
```

**Expected output:**
```
🧪 Testing Claude API connection...

✅ SUCCESS! Claude API is working.

Response: Hello! Yes, I'm Claude, an AI assistant created by Anthropic...

Tokens used: { input: 15, output: 25 }
Cost estimate: ~$ 0.0004

🚀 You can now run: npm run dev
```

**If you see an error:**
- ❌ "ANTHROPIC_API_KEY not found" → Create `.env.local` file
- ❌ "Invalid API key" → Check you copied the key correctly
- ❌ "Rate limit exceeded" → Wait 1 minute, try again

## Step 3: Run the App (1 minute)

```bash
npm run dev
```

**Expected output:**
```
  ▲ Next.js 15.0.0
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

## Step 4: Test the Chat (4 minutes)

### Open your browser:

1. Go to: **http://localhost:3000**
2. Click "🚀 Try AI Chat Demo"
3. Or go directly to: **http://localhost:3000/chat-test**

### Try these quick actions:

1. Click "Daily Briefing" button
2. Click "Portfolio" button
3. Type your own question: "What should I know about tax-loss harvesting?"

**Expected behavior:**
- ✅ AI responds in 2-5 seconds
- ✅ Mentions your FQ score (560)
- ✅ References your portfolio ($127,400)
- ✅ Provides personalized advice

### Test different prompts:

- "How can I improve my FQ score?"
- "Is my portfolio too heavy in tech stocks?"
- "Should I rebalance to my target allocation?"
- "Explain the wash sale rule"

## Troubleshooting

### Issue: "Module not found"

```bash
# Reinstall dependencies
npm install
```

### Issue: "Invalid API key"

```bash
# Check your .env.local file
cat .env.local

# Make sure the key starts with: sk-ant-api03-
# If not, copy the correct key from Anthropic console
```

### Issue: Chat not responding

1. Check browser console (F12 → Console tab)
2. Check terminal for errors
3. Try: `node test-claude.mjs` to verify API works
4. Restart dev server: Ctrl+C, then `npm run dev`

### Issue: "Rate limit exceeded"

- You're sending too many requests
- Wait 1 minute
- Check usage: https://console.anthropic.com/settings/usage

## Success Checklist

- [ ] ✅ API key added to `.env.local`
- [ ] ✅ `node test-claude.mjs` shows success
- [ ] ✅ `npm run dev` runs without errors
- [ ] ✅ Chat interface loads at `/chat-test`
- [ ] ✅ AI responds to messages
- [ ] ✅ Quick action buttons work

## Next Steps

### Customize the experience:

1. **Edit user context** (app/api/chat/route.ts:22-28)
   - Change FQ score
   - Update portfolio value
   - Modify holdings

2. **Add more quick actions** (components/ChatInterface.tsx:16-21)
   ```typescript
   { icon: '🎯', label: 'Goals', prompt: 'Help me set financial goals' },
   ```

3. **Adjust AI personality** (app/api/chat/route.ts:30-60)
   - Make it more/less formal
   - Add industry jargon
   - Change response length

### Deploy to production:

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Set environment variables
vercel env add ANTHROPIC_API_KEY
# Paste your key when prompted

# Deploy
vercel --prod
```

## Cost Monitoring

### Check your usage:

- https://console.anthropic.com/settings/usage

### Set budget alert:

1. Go to: https://console.anthropic.com/settings/billing
2. Click "Set Budget Alert"
3. Enter: $50/month (or your limit)
4. Save

### Expected costs:

- **Testing (100 messages):** ~$1
- **Light use (10 messages/day):** ~$3/month
- **Heavy use (100 messages/day):** ~$30/month

## Questions?

- **Documentation:** See parent directory for comprehensive docs
  - `SETUP_CLAUDE_NOW.md` - Detailed setup guide
  - `GAZILLIONER_MVP_SPEC.md` - Full technical spec
  - `GAZILLIONER_BRD.md` - Business requirements

- **Issues:** Check the logs
  ```bash
  # Server logs (terminal where npm run dev is running)
  # Browser logs (F12 → Console)
  ```

## You're Ready! 🎉

Your Gazillioner AI chat is working! Next:
- Connect to real user data (database integration)
- Add conversation history
- Implement portfolio sync with Plaid
- Add FQ assessment quiz

Happy building! 🚀
