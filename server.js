const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');

const app = express();
app.use(cors());
app.use(express.json());

const COHERE_API_KEY = "KMntogWwwmGEghQ3zzaxKVNpZEeXGOgYapreMmnI"; // Replace with your Cohere API key

app.post('/solve', async (req, res) => {
  const { question, options } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    let prompt = question;
    if (options && options.length > 0) {
      prompt += `\nOptions: ${options.join(', ')}`;
    }

    const response = await fetch("https://api.cohere.ai/v1/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${COHERE_API_KEY}`,
      },
      body: JSON.stringify({
        model: "command",
        prompt: prompt,
        max_tokens: 100,
        temperature: 0.7,
      }),
    });

    const data = await response.json();
    const answer = data.generations[0].text.trim();
    res.json({ answer });
  } catch (error) {
    res.status(500).json({ error: `Error fetching answer: ${error.message}` });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
