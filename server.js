const express = require('express');
const cors = require('cors');
const math = require('mathjs'); // Ensure math.js is installed

const app = express();
app.use(cors());
app.use(express.json());

app.post('/solve-equation', (req, res) => {
  const { equation } = req.body;

  if (!equation) {
    return res.status(400).json({ error: "Equation is required" });
  }

  try {
    const solution = math.evaluate(equation.replace(/\s+/g, '')); // Remove whitespace
    res.json({ solution });
  } catch (error) {
    res.status(500).json({ error: `Error solving equation: ${error.message}` });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
