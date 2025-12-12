# Gazillioner - AI Wealth Advisor

Your personal AI-powered financial advisor that helps improve your Financial Quotient (FQ).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Environment Variables

1. Copy `.env.example` to `.env.local`:
   ```bash
   cp .env.example .env.local
   ```

2. Get your Claude API key:
   - Go to https://console.anthropic.com/settings/keys
   - Create a new API key
   - Copy the key (it starts with `sk-ant-api03-`)

3. Edit `.env.local` and paste your API key:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-api03-YOUR_ACTUAL_KEY_HERE
   ```

### 3. Test the API Connection

```bash
node test-claude.mjs
```

Expected output: ✅ SUCCESS! Claude API is working.

### 4. Run Development Server

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

### 5. Test the AI Chat

Go to http://localhost:3000/chat-test

## 🔐 Security

⚠️ **NEVER commit your `.env.local` file to git!**

The `.gitignore` file already excludes it, but always double-check before committing.

## 📚 Features

- ✅ Claude 3.5 Sonnet AI integration
- ✅ Personalized financial advice
- ✅ FQ (Financial Quotient) scoring
- ✅ Portfolio management (stocks, crypto, bonds)
- ✅ Tax optimization suggestions
- ✅ Behavioral coaching (FOMO detection, panic selling alerts)

## 🛠️ Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **AI:** Claude 3.5 Sonnet (Anthropic)
- **UI Components:** Lucide React

## 📖 Documentation

See the parent directory for comprehensive docs:
- `GAZILLIONER_BRD.md` - Business Requirements
- `GAZILLIONER_MVP_SPEC.md` - Technical Specification
- `SETUP_CLAUDE_NOW.md` - Complete setup guide
- `GAZILLIONER_PITCH_DECK.md` - Investor presentation

## 💰 Cost Monitoring

- Average cost: ~$0.009 per message
- Monitor usage: https://console.anthropic.com/settings/usage
- Set budget alerts in Anthropic console

## 🚢 Deployment

Deploy to Vercel:

```bash
# Install Vercel CLI
npm i -g vercel

# Set environment variables
vercel env add ANTHROPIC_API_KEY

# Deploy
vercel --prod
```

## 📝 License

Proprietary - Gazillioner 2025
