document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const form = e.target;
  const data = {
    RM: parseFloat(form.RM.value),
    LSTAT: parseFloat(form.LSTAT.value),
    PTRATIO: parseFloat(form.PTRATIO.value)
  };

  document.getElementById("result").innerText = "⏳ Calculating...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error: ${response.status} - ${text}`);
    }

    const result = await response.json();
    document.getElementById("result").innerText = `🏠 Predicted Price: $${result.predicted_price.toFixed(2)}k`;

  } catch (error) {
    console.error("Error:", error);
    document.getElementById("result").innerText = "❌ Error fetching prediction. Check console.";
  }
});
