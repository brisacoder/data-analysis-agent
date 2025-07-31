
// node_pyodide_csv.js
import { loadPyodide } from "@pyodide/pyodide";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

// Use ES module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load local CSV file from disk
const csvPath = path.join(__dirname, "sample.csv");
const csvData = await fs.readFile(csvPath, "utf-8");

// Initialize Pyodide
console.log("Loading Pyodide...");
const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/" });
console.log("Pyodide loaded.");

// Write CSV into Pyodide virtual FS
pyodide.FS.writeFile("/home/pyodide/sample.csv", csvData);

// Run Python code using pandas
const pythonCode = `
import pandas as pd
df = pd.read_csv("/home/pyodide/sample.csv")
print("Head:")
print(df.head())
print("Describe:")
print(df.describe())
`;

console.log("Running Python code...");
try {
  const result = pyodide.runPython(pythonCode);
  console.log(result);
} catch (err) {
  console.error("Execution error:", err);
}
