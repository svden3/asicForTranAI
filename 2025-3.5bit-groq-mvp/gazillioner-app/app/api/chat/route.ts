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
    // This is mock data for MVP demonstration
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
