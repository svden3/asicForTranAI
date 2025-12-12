import Anthropic from '@anthropic-ai/sdk';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function test() {
  console.log('🧪 Testing Claude API connection...\n');

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('❌ ERROR: ANTHROPIC_API_KEY not found in .env.local');
    console.error('\n⚠️  Please create .env.local and add your API key:');
    console.error('   cp .env.example .env.local');
    console.error('   then edit .env.local and add your key\n');
    process.exit(1);
  }

  if (process.env.ANTHROPIC_API_KEY === 'sk-ant-api03-PASTE_YOUR_KEY_HERE') {
    console.error('❌ ERROR: You need to replace the placeholder API key');
    console.error('\n⚠️  Steps to fix:');
    console.error('   1. Go to https://console.anthropic.com/settings/keys');
    console.error('   2. Create a new API key');
    console.error('   3. Copy the key (starts with sk-ant-api03-)');
    console.error('   4. Edit .env.local and replace PASTE_YOUR_KEY_HERE\n');
    process.exit(1);
  }

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
    console.log('\nCost estimate: ~$', (
      (response.usage.input_tokens * 0.003 / 1000) +
      (response.usage.output_tokens * 0.015 / 1000)
    ).toFixed(4));
    console.log('\n🚀 You can now run: npm run dev');
  } catch (error) {
    console.error('❌ ERROR:', error.message);
    if (error.status === 401) {
      console.error('\n⚠️  Invalid API key. Check your .env.local file.');
      console.error('   Make sure you copied the key correctly from:');
      console.error('   https://console.anthropic.com/settings/keys\n');
    } else if (error.status === 429) {
      console.error('\n⚠️  Rate limit exceeded. Please wait a moment and try again.\n');
    } else {
      console.error('\n⚠️  Unexpected error. Check your internet connection.\n');
    }
    process.exit(1);
  }
}

test();
