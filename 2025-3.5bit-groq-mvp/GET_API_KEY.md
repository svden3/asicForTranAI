# How to Get a Valid Groq API Key

## The API Key Error

If you see:
```
{
    "error": {
        "message": "Invalid API Key",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
    }
}
```

Your API key is either:
- Invalid format
- Expired
- Not properly set in environment

## Get a New API Key (5 minutes)

### Step 1: Visit Groq Console
Go to: **https://console.groq.com**

### Step 2: Sign Up (Free)
- Click "Sign Up"
- Use GitHub, Google, or email
- **Free tier includes:**
  - 500 million tokens
  - Access to LLaMA 3.3 70B, Qwen, and more
  - High-speed inference (3100+ tok/s)

### Step 3: Create API Key
1. After login, go to: **https://console.groq.com/keys**
2. Click "Create API Key"
3. Give it a name (e.g., "asicForTranAI-demo")
4. Click "Create"
5. **Copy the key immediately** (starts with `gsk_`)

⚠️ **Important:** You can only see the key once! Save it somewhere safe.

### Step 4: Set the Environment Variable

**macOS/Linux:**
```bash
export GROQ_API_KEY='gsk_your_actual_key_here'

# Verify it's set
echo $GROQ_API_KEY
```

**To make it permanent** (add to `~/.zshrc` or `~/.bashrc`):
```bash
echo "export GROQ_API_KEY='gsk_your_actual_key_here'" >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_your_actual_key_here"
```

### Step 5: Run the Demo
```bash
cd /Users/jimxiao/ai/asicForTranAI/2025-3.5bit-groq-mvp/groq
./compile_and_run.sh
```

## Verify Your Key Format

A valid Groq API key:
- ✅ Starts with `gsk_`
- ✅ Is around 50-60 characters long
- ✅ Contains only alphanumeric characters

Example format:
```
gsk_abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx
```

## Common Issues

### Issue 1: Key Not Set
```bash
# Check if set
echo $GROQ_API_KEY

# If empty, set it
export GROQ_API_KEY='your_key_here'
```

### Issue 2: Wrong Quotes
```bash
# ❌ Wrong (smart quotes)
export GROQ_API_KEY="gsk_..."

# ✅ Correct (straight quotes)
export GROQ_API_KEY='gsk_...'
```

### Issue 3: Spaces in Key
```bash
# ❌ Wrong (spaces)
export GROQ_API_KEY="gsk_abc def"

# ✅ Correct (no spaces)
export GROQ_API_KEY='gsk_abcdef'
```

### Issue 4: Key Expired
- Keys don't expire automatically
- But you can revoke them in console
- Create a new one if needed

## Alternative: Use .env File

Create `.env` in the groq directory:
```bash
GROQ_API_KEY=gsk_your_actual_key_here
```

Then load it:
```bash
source .env
./compile_and_run.sh
```

## Testing Your Key

Quick test:
```bash
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | head -20
```

If valid, you'll see a JSON response with available models.

## Need Help?

1. **Groq Documentation**: https://console.groq.com/docs
2. **API Status**: https://status.groq.com
3. **Community**: https://discord.gg/groq

---

**Once you have a valid key, the demo will show LLaMA 3.3 70B running at incredible speed on Groq's ASIC!**
