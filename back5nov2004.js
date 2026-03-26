// CC BY-SA 4.0
// Learning representations by back-propagating errors
// David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams
// Nature volume 323, pages 533–536 (1986)
// https://www.nature.com/articles/323533a0

// ...we still face the same question that were asked in 1953. Turing's question
// was what it would take for machines to begin to think. Von Neuwmann's question
// was what it would take for machines to begin to reproduce.
// Turing's Cathedral. The origins of the digital universe.
// George Dyson. Pantheon books - New York.

// WHAT HATH GOD WROUGHT

// ******** Algorithm for back propagation errors in order
// to train artificial multilayer neural networks (3).

// ─────────────────────────────────────────────────────────────
// Network dimensions
//   ia   = training vector size (31)
//   ia-1 = # of neurons of the first (presentation) layer  → 30
//   ib   = # of neurons of the hidden layer                → 20
//   ic   = # of neurons of the output layer                → 2
// ─────────────────────────────────────────────────────────────
const ia = 31;
const ib = 20;
const ic = 2;

// learning rate (negative → gradient descent)

const eta = -0.1;

// ─────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────

function rnd001(iran, ifin) {
  // Fortran: i = i * 54891  (integer*4 wraps at 32-bit)
  // JavaScript numbers are 64-bit floats, so we force 32-bit wrap.
  //iran = Math.imul(iran, 54891) | 0;        // signed 32-bit multiply
  //let xi = Math.random();
  let xi;
  xi = Math.random();
  xi = xi * ifin;
  return xi;
}

/** Sigmoid activation function  σ(x) = 1 / (1 + e^−x) */
function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

// ─────────────────────────────────────────────────────────────
// Weight matrices  (1-indexed; index 0 unused)
//   pesos1[i][j]  input layer  → first layer   (ia-1) × ia
//   pesos2[i][j]  first layer  → hidden layer   ib    × (ia-1)
//   pesos3[i][j]  hidden layer → output layer   ic    × ib
// Sub-arrays are empty; JS creates slots on first assignment.
// ─────────────────────────────────────────────────────────────
const pesos1 = Array.from({ length: ia }, () => []); // [1..ia-1][1..ia]
const pesos2 = Array.from({ length: ib + 1 }, () => []); // [1..ib]  [1..ia-1]
const pesos3 = Array.from({ length: ic + 1 }, () => []); // [1..ic]  [1..ib]

// Adjustments — fully recomputed each iteration, no need to pre-fill
const ajustew1 = Array.from({ length: ia }, () => []);
const ajustew2 = Array.from({ length: ib + 1 }, () => []);
const ajustew3 = Array.from({ length: ic + 1 }, () => []);

// Training patterns: vector[n][l][j]  n∈{1,2}, l∈[1..10], j∈[1..ia]
const vector = [
  null, // n=0 unused
  Array.from({ length: 11 }, () => []), // n=1
  Array.from({ length: 11 }, () => []), // n=2
];

// Labels / target outputs  etiqueta[n][i]  n∈{1,2}, i∈{1,2}
const etiqueta = [
  null,
  [0, 1, 0], // n=1: [_unused_, target1, target2]
  [0, 0, 1], // n=2
];

// Layer activations — overwritten each forward pass, no pre-fill needed
const x1 = []; // net input,  first layer
const x2 = []; // net input,  hidden layer
const x3 = []; // net input,  output layer
const y1 = []; // output,     first layer  (y1[1]=1 bias set in forwardPass)
const y2 = []; // output,     hidden layer (y2[1]=1 bias set in forwardPass)
const y3 = []; // output,     output layer

// Error gradients — fully assigned each backprop pass
const delta1 = [];
const delta2 = [];
const delta3 = [];

// ─────────────────────────────────────────────────────────────
// Weight initialisation  (random values in (−0.3, 0.3))
// ─────────────────────────────────────────────────────────────
// pesos1 : (ia-1) × ia
for (let j = 1; j <= ia; j++) {
  for (let i = 1; i <= ia - 1; i++) {
    const sr = rnd001(1);
    const s = sr < 0.5 ? -1 : 1;
    pesos1[i][j] = rnd001(1) * 0.3 * s;
  }
}

// pesos2 : ib × (ia-1)
for (let j = 1; j <= ia - 1; j++) {
  for (let i = 1; i <= ib; i++) {
    const sr = rnd001(1);
    const s = sr < 0.5 ? -1 : 1;
    pesos2[i][j] = rnd001(1) * 0.3 * s;
  }
}

// pesos3 : ic × ib
for (let j = 1; j <= ib; j++) {
  for (let i = 1; i <= ic; i++) {
    const sr = rnd001(1);
    const s = sr < 0.5 ? -1 : 1;
    pesos3[i][j] = rnd001(1) * 0.3 * s;
  }
}

// ─────────────────────────────────────────────────────────────
// Bias nodes  (vector[n][l][1] = 1 for all l, both pattern types)
// ─────────────────────────────────────────────────────────────
for (let l = 1; l <= 10; l++) {
  vector[1][l][1] = 1;
  vector[2][l][1] = 1;
}

// ─────────────────────────────────────────────────────────────
// DATA LOADING
// In the original Fortran the patterns are read from files.
// Here we expose two helper functions the caller must call to
// supply the data before invoking train() / recognise().
//
//   loadPatterns(patternType, samples)
//     patternType : 1 or 2
//     samples     : array of 10 arrays, each with 30 numbers
//                   (components 1..30; component 0 is the bias = 1)
//
//   loadLabels(patternType, t1, t2)
//     target outputs for neuron 1 and neuron 2 of the output layer.
// ─────────────────────────────────────────────────────────────

/**
 * Load training patterns for one class.
 * @param {1|2}      patternType  – which class (n=1 or n=2)
 * @param {number[][]} samples    – 10-element array; each element has 30 values
 */
function loadPatterns(patternType, samples) {
  for (let l = 1; l <= 10; l++) {
    vector[patternType][l][1] = 1; // bias
    for (let i = 2; i <= ia; i++) {
      vector[patternType][l][i] = samples[l - 1][i - 2];
    }
  }
}

/**
 * Load target labels for one class.
 * @param {1|2} patternType
 * @param {number} t1  – desired output for output neuron 1
 * @param {number} t2  – desired output for output neuron 2
 */
function loadLabels(patternType, t1, t2) {
  etiqueta[patternType][1] = t1;
  etiqueta[patternType][2] = t2;
}

// ─────────────────────────────────────────────────────────────
// Forward pass  (shared by training and recognition)
// Uses pattern class n, sample l.
// ─────────────────────────────────────────────────────────────
function forwardPass(n, l) {
  // --- Layer 1 (presentation / input layer) ---
  y1[1] = 1; // bias
  for (let i = 1; i <= ia - 1; i++) {
    x1[i] = 0;
    y1[i + 1] = 0;
    for (let j = 1; j <= ia; j++) {
      x1[i] += pesos1[i][j] * vector[n][l][j];
    }
    y1[i + 1] = sigmoid(x1[i]);
  }

  // --- Hidden layer ---
  y2[1] = 1; // bias
  for (let i = 1; i <= ib; i++) {
    x2[i] = 0;
    y2[i + 1] = 0;
    for (let j = 1; j <= ia - 1; j++) {
      x2[i] += pesos2[i][j] * y1[j];
    }
    y2[i + 1] = sigmoid(x2[i]);
  }

  // --- Output layer ---
  for (let i = 1; i <= ic; i++) {
    x3[i] = 0;
    y3[i] = 0;
    for (let j = 1; j <= ib; j++) {
      x3[i] += pesos3[i][j] * y2[j];
    }
    y3[i] = sigmoid(x3[i]);
  }
}

// ─────────────────────────────────────────────────────────────
// TRAINING STAGE
// Replicates the main training loop (label 80 → goto 80)
// with the inner loop over l=1..10 for pattern class n=2.
// ─────────────────────────────────────────────────────────────
function train({ maxEpochs = 100000, targetSSE = 0.005 } = {}) {
  let epoch = 0;
  let sse = 1;

  const n = 2; // pattern class used (same as Fortran: n=2 hard-coded)

  while (epoch <= maxEpochs && sse >= targetSSE) {
    epoch++;

    // Inner loop over the 10 samples of pattern class n
    for (let l = 1; l <= 10; l++) {
      // ── Forward pass ──────────────────────────────────────────
      forwardPass(n, l);

      // ── Sum of squared errors ─────────────────────────────────
      sse = 0;
      for (let i = 1; i <= ic; i++) {
        sse += (y3[i] - etiqueta[n][i]) ** 2;
      }
      sse = 0.5 * sse;

      // ── Backpropagation ───────────────────────────────────────

      // 1. Output layer deltas
      for (let i = 1; i <= ic; i++) {
        delta3[i] = y3[i] * (1 - y3[i]) * (y3[i] - etiqueta[n][i]);
      }

      // 2. Hidden layer deltas
      for (let j = 1; j <= ib + 1; j++) {
        let sumdelta2 = 0;
        for (let i = 1; i <= ic; i++) {
          sumdelta2 += delta3[i] * pesos3[i][j];
        }
        delta2[j] = y2[j] * (1 - y2[j]) * sumdelta2;
      }

      // 3. First layer deltas
      for (let j = 1; j <= ia - 1; j++) {
        let sumdelta1 = 0;
        for (let i = 1; i <= ib; i++) {
          sumdelta1 += delta2[i] * pesos2[i][j];
        }
        delta1[j] = y1[j] * (1 - y1[j]) * sumdelta1;
      }

      // ── Weight adjustments (Δw = η · δ · y) ──────────────────

      // Adjustments pesos3  (hidden → output)
      for (let i = 1; i <= ib + 1; i++) {
        for (let j = 1; j <= ic; j++) {
          ajustew3[j][i] = eta * delta3[j] * y2[i];
        }
      }

      // Adjustments pesos2  (first → hidden)
      for (let i = 1; i <= ia - 1; i++) {
        for (let j = 1; j <= ib + 1; j++) {
          ajustew2[j][i] = eta * delta2[j] * y1[i];
        }
      }

      // Adjustments pesos1  (input → first)
      for (let j = 1; j <= ia - 1; j++) {
        for (let i = 1; i <= ia; i++) {
          ajustew1[j][i] = eta * delta1[j] * vector[n][l][i];
        }
      }

      // ── Apply adjustments ────────────────────────────────────

      // Update pesos1
      for (let j = 1; j <= ia; j++) {
        for (let i = 1; i <= ia - 1; i++) {
          pesos1[i][j] += ajustew1[i][j];
        }
      }

      // Update pesos2
      for (let j = 1; j <= ia - 1; j++) {
        for (let i = 1; i <= ib; i++) {
          pesos2[i][j] += ajustew2[i][j];
        }
      }

      // Update pesos3
      for (let j = 1; j <= ib; j++) {
        for (let i = 1; i <= ic; i++) {
          pesos3[i][j] += ajustew3[i][j];
        }
      }
    } // end l loop

    if (epoch % 100 === 0) {
      console.log(`sse=${sse.toFixed(6)}  epoch=${epoch}`);
    }
  } // end epoch loop

  return { epoch, sse };
}

// ─────────────────────────────────────────────────────────────
// RECOGNITION STAGE  (after training)
// Mirrors the Fortran section starting after label 900.
// ─────────────────────────────────────────────────────────────
function recognise({ patternType = 2, sseThreshold = 0.02 } = {}) {
  const n = patternType;
  const results = [];

  for (let l = 1; l <= 10; l++) {
    forwardPass(n, l);

    const e1 = (y3[1] - etiqueta[n][1]) ** 2;
    const e2 = (y3[2] - etiqueta[n][2]) ** 2;
    const sse = 0.5 * (e1 + e2);

    const recognised = sse < sseThreshold;
    const label = recognised ? `pattern ${n}` : "unrecognised";

    results.push({
      sample: l,
      label,
      recognised,
      y3_1: y3[1],
      y3_2: y3[2],
      target1: etiqueta[n][1],
      target2: etiqueta[n][2],
      sse,
    });

    console.log(
      `${l.toString().padStart(4)}  ${label.padEnd(14)}` +
        `  y31=${y3[1].toFixed(3)}  y32=${y3[2].toFixed(3)}` +
        `  targets=[${etiqueta[n][1].toFixed(1)}, ${etiqueta[n][2].toFixed(1)}]` +
        `  sse=${sse.toFixed(3)}`,
    );
  }

  return results;
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────
module.exports = { loadPatterns, loadLabels, train, recognise };

// ─────────────────────────────────────────────────────────────
// Quick self-test  (run with:  node back5nov2004.js)
// Generates synthetic random patterns so the network can be
// exercised without external data files.
// ─────────────────────────────────────────────────────────────
if (require.main === module) {
  // Build two toy pattern classes with random values in [0,1]
  const rand = () => Math.random();
  const makePatterns = () =>
    Array.from({ length: 10 }, () => Array.from({ length: 30 }, rand));

  loadPatterns(1, makePatterns());
  loadPatterns(2, makePatterns());
  loadLabels(1, 1, 0);
  loadLabels(2, 0, 1);

  console.log("=== Training ===");
  const { epoch, sse } = train({ maxEpochs: 100000, targetSSE: 0.005 });
  console.log(`\nTraining finished: epoch=${epoch}  sse=${sse.toFixed(6)}\n`);

  console.log("=== Recognition ===");
  recognise({ patternType: 2, sseThreshold: 0.02 });
}
