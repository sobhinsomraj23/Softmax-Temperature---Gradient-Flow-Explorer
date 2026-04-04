import React, { useState, useMemo } from "react";
import Plot from "react-plotly.js";

const classNames = ["Class A", "Class B", "Class C"];

function softmax(logits, T) {
  const scaled = logits.map(v => v / T);
  const maxVal = Math.max(...scaled);
  const exps = scaled.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function gradient(probs, target) {
  return probs.map((p, i) => p - (i === target ? 1 : 0));
}

function entropy(probs) {
  return -probs.reduce((sum, p) => sum + p * Math.log(p + 1e-10), 0);
}

export default function SoftmaxExplorer() {
  const [logitsInput, setLogitsInput] = useState("8,3,-1");
  const [target, setTarget] = useState(0);
  const [temperature, setTemperature] = useState(1);

  const logits = useMemo(
    () => logitsInput.split(",").map(Number),
    [logitsInput]
  );

  const probs = useMemo(
    () => softmax(logits, temperature),
    [logits, temperature]
  );

  const grads = useMemo(
    () => gradient(probs, target),
    [probs, target]
  );

  const temps = useMemo(
    () => Array.from({ length: 100 }, (_, i) => 0.1 + i * 0.1),
    []
  );

  const probCurves = useMemo(() => {
    return logits.map((_, idx) =>
      temps.map(T => softmax(logits, T)[idx])
    );
  }, [logits, temps]);

  const gradMagnitudes = useMemo(() => {
    return temps.map(T => {
      const p = softmax(logits, T);
      const g = gradient(p, target);
      return Math.sqrt(g.reduce((s, v) => s + v * v, 0));
    });
  }, [logits, target, temps]);

  const entropyValues = useMemo(() => {
    return temps.map(T => {
      const p = softmax(logits, T);
      return entropy(p);
    });
  }, [logits, temps]);

  const classGradCurves = useMemo(() => {
    return logits.map((_, idx) =>
      temps.map(T => {
        const p = softmax(logits, T);
        const g = gradient(p, target);
        return g[idx];
      })
    );
  }, [logits, target, temps]);

  
  const commonLayout = {
    paper_bgcolor: "#1e293b",
    plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" },
    title: {
      x: 0.5, // center title
      font: { size: 18, color: "#f8fafc" }
    },
    xaxis: { title: "Temperature" },
  };

  const insight = useMemo(() => {
    if (temperature < 0.5)
      return "Low Temperature → Very confident predictions → Gradients vanish (learning slows)";
    if (temperature > 5)
      return "High Temperature → Uniform probabilities → Weak learning signal";
    return "Balanced Temperature → Optimal learning and gradient flow";
  }, [temperature]);

  return (
    <div style={{
      background: "#0f172a",
      color: "#e2e8f0",
      minHeight: "100vh",
      padding: "20px",
      fontFamily: "Arial"
    }}>
      <h1>Softmax Temperature & Gradient Flow Explorer</h1>

      {/* CONTROLS */}
      <div style={{
        padding: "15px",
        background: "#1e293b",
        borderRadius: "10px",
        marginBottom: "20px"
      }}>
        <h3>Controls</h3>

        <input
          value={logitsInput}
          onChange={e => setLogitsInput(e.target.value)}
          style={{ marginRight: "10px" }}
        />

        <select
          value={target}
          onChange={e => setTarget(Number(e.target.value))}
        >
          {classNames.map((name, i) => (
            <option key={i} value={i}>{name}</option>
          ))}
        </select>

        <div style={{ marginTop: "10px" }}>
          <label>Temperature: {temperature.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="10"
            step="0.1"
            value={temperature}
            onChange={e => setTemperature(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* PROBABILITIES CARDS */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        {probs.map((p, i) => (
          <div key={i} style={{
            flex: 1,
            padding: "10px",
            background: "#1e293b",
            borderRadius: "8px"
          }}>
            <strong>{classNames[i]}</strong>
            <div>{(p * 100).toFixed(2)}%</div>
          </div>
        ))}
      </div>

      {/* INSIGHT PANEL */}
      <div style={{
        marginBottom: "20px",
        padding: "10px",
        background: "#1e293b",
        borderRadius: "8px"
      }}>
        {insight}
      </div>

      {/* GRAPH GRID */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "20px"
      }}>

        {/* GRAPH 1 */}
        <Plot
          data={probCurves.map((curve, i) => ({
            x: temps,
            y: curve,
            type: "scatter",
            mode: "lines",
            name: classNames[i]
          }))}
          layout={{
                        ...commonLayout,
            title: {
              ...commonLayout.title,
              text: "Softmax Output Distribution vs Temperature"
            },
            yaxis: { title: "Probability" },
            // title: "Softmax Output Distribution vs Temperature",
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#e2e8f0" }
            
          }}
        />

        {/* GRAPH 2 */}
        <Plot
          data={[{
            x: temps,
            y: gradMagnitudes,
            type: "scatter",
            mode: "lines",
            name: "Gradient Strength"
          }]}
          layout={{
            ...commonLayout,
            title: {
              ...commonLayout.title,
              text: "Gradient Strength vs Temperature (Learning Signal)"
            },
            yaxis: { title: "Gradient Magnitude" },
            // title: "Gradient Strength vs Temperature (Learning Signal)",
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#e2e8f0" }
          }}
        />

        {/* GRAPH 3 */}
        <Plot
          data={[{
            x: temps,
            y: entropyValues,
            type: "scatter",
            mode: "lines",
            name: "Entropy"
          }]}
          layout={{
                        ...commonLayout,
            title: {
              ...commonLayout.title,
              text: "Prediction Uncertainty (Entropy) vs Temperature"
            },
            yaxis: { title: "Entropy" },
            // title: "Prediction Uncertainty (Entropy) vs Temperature",
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#e2e8f0" }
          }}
        />

        {/* GRAPH 4 */}
        <Plot
          data={classGradCurves.map((curve, i) => ({
            x: temps,
            y: curve,
            type: "scatter",
            mode: "lines",
            name: classNames[i]
          }))}
          layout={{
                        ...commonLayout,
            title: {
              ...commonLayout.title,
              text: "Per-Class Gradient Flow vs Temperature"
            },
            yaxis: { title: "Gradient Value" },
            // title: "Per-Class Gradient Flow vs Temperature",
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#e2e8f0" }
          }}
        />

      </div>
    </div>
  );
}