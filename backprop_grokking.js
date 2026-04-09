// CC BY-SA 4.0
// Learning representations by back-propagating errors
// David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams
// Nature volume 323, pages 533–536 (1986)
// https://www.nature.com/articles/323533a0

// ─────────────────────────────────────────────────────────────
// Generalización para N capas arbitrarias.
//
// Arquitectura definida por un array de tamaños:
//   layerSizes = [inputDim, hidden1, hidden2, ..., outputDim]
//
//   Ejemplo equivalente al original back5nov2004.js:
//     layerSizes = [30, 20, 2]   (input → hidden → output)
//
//   La primera entrada es el tamaño del vector de entrada
//   (sin contar el nodo de sesgo/bias, que se agrega automáticamente).
//
// Número de patrones de entrenamiento por clase: numSamples
// Número de clases:                              numClasses
// ─────────────────────────────────────────────────────────────

'use strict';

// ─────────────────────────────────────────────────────────────
// Utilidades
// ─────────────────────────────────────────────────────────────

/** Matriz 2D de ceros (rows × cols) */
function zeros2D(rows, cols) {
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/** Vector 1D de ceros */
function zeros1D(n) {
    return new Array(n).fill(0);
}

/** Función de activación sigmoide σ(x) = 1 / (1 + e^−x) */
function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

/** Número aleatorio en [0, 1) */
function rnd() {
    return Math.random();
}

// ─────────────────────────────────────────────────────────────
// Constructor de la red neuronal
// ─────────────────────────────────────────────────────────────

/**
 * Crea una red neuronal multicapa con backpropagation.
 *
 * @param {number[]} layerSizes  - Tamaños de cada capa, SIN contar el bias.
 *                                 Ej: [30, 20, 2] → input:30, hidden:20, output:2
 * @param {number}   numClasses  - Número de clases de patrones (≥ 1)
 * @param {number}   numSamples  - Número de muestras por clase
 * @param {number}   eta        - Tasa de aprendizaje (negativa → descenso del gradiente)
 */
function createNetwork(layerSizes, numClasses = 2, numSamples = 10, eta = -0.10) {

    // Validación mínima
    if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
        throw new Error('layerSizes debe tener al menos 2 elementos [input, ..., output].');
    }

    const L = layerSizes.length;   // número total de capas (sin contar la capa de entrada raw)
    // Renombramos para claridad:
    //   layer 0 → "capa de presentación / entrada"
    //   layer L-1 → "capa de salida"

    // ── Matrices de pesos ──────────────────────────────────────
    // pesos[k]  : pesos de capa k-1 → capa k,  tamaño (layerSizes[k]) × (layerSizes[k-1] + 1)
    //             La columna extra (+1) corresponde al bias de la capa anterior.
    //
    // Para la primera capa (k=0) los pesos conectan los vectores de entrada
    // (de dimensión inputDim + 1 con bias) hacia la primera capa oculta/salida.
    // Aquí decidimos tratar las capas ocultas como en el original:
    //   pesos[k]  tiene dimensión  layerSizes[k+1] × (layerSizes[k] + 1)
    // donde el +1 es el bias de la capa k.

    const numWeightMatrices = L - 1;   // una matriz por cada par de capas adyacentes

    // pesos[k]: (layerSizes[k+1]) × (layerSizes[k] + 1)   k = 0..L-2
    const pesos   = [];
    const ajuste  = [];   // matrices de ajuste (mismas dimensiones)

    for (let k = 0; k < numWeightMatrices; k++) {
        const rows = layerSizes[k + 1];
        const cols = layerSizes[k] + 1;   // +1 por el bias
        pesos[k]  = zeros2D(rows, cols);
        ajuste[k] = zeros2D(rows, cols);
    }

    // ── Activaciones por capa ───────────────────────────────────
    // y[k]: salida de la capa k, tamaño layerSizes[k] + 1
    //       y[k][0] = 1 (nodo de bias que alimenta la siguiente capa)
    //       y[k][1..layerSizes[k]]  = salidas de las neuronas
    //
    // Para la capa de entrada (k=0) y[0][0..inputDim] viene del patrón.

    const y     = [];   // activaciones
    const delta = [];   // gradientes de error

    for (let k = 0; k < L; k++) {
        y[k]     = zeros1D(layerSizes[k] + 1);  // índice 0 = bias
        delta[k] = zeros1D(layerSizes[k] + 1);
    }

    // ── Patrones y etiquetas ────────────────────────────────────
    // patterns[n][l][j]  n∈[0..numClasses-1], l∈[0..numSamples-1], j∈[0..inputDim]
    //   j=0 → bias (siempre 1); j=1..inputDim → componentes del patrón
    const inputDim = layerSizes[0];
    const patterns = Array.from({ length: numClasses },
        () => Array.from({ length: numSamples },
            () => zeros1D(inputDim + 1)));

    // Inicializar bias en todos los patrones
    for (let n = 0; n < numClasses; n++) {
        for (let l = 0; l < numSamples; l++) {
            patterns[n][l][0] = 1;
        }
    }

    // labels[n][i]  n∈[0..numClasses-1], i∈[0..outputDim-1]
    const outputDim = layerSizes[L - 1];
    const labels = Array.from({ length: numClasses },
        () => zeros1D(outputDim));

    // ── Inicialización de pesos aleatorios en (−0.3, 0.3) ──────
    for (let k = 0; k < numWeightMatrices; k++) {
        const rows = layerSizes[k + 1];
        const cols = layerSizes[k] + 1;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const sign = rnd() < 0.5 ? -1 : 1;
                pesos[k][i][j] = rnd() * 0.3 * sign;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Forward pass (basado en almacenamiento patterns[])
    // ─────────────────────────────────────────────────────────────
    function forwardPass(n, l) {
        const input = patterns[n][l];
        y[0][0] = 1;   // bias
        for (let j = 1; j <= inputDim; j++) {
            y[0][j] = input[j];
        }
        for (let k = 1; k < L; k++) {
            y[k][0] = 1;
            const prevSize = layerSizes[k - 1];
            const currSize = layerSizes[k];
            for (let i = 1; i <= currSize; i++) {
                let net = 0;
                for (let j = 0; j <= prevSize; j++) {
                    net += pesos[k - 1][i - 1][j] * y[k - 1][j];
                }
                y[k][i] = sigmoid(net);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Forward pass con vector de entrada arbitrario (para trainOnDataset)
    // input: array de inputDim valores (sin bias)
    // ─────────────────────────────────────────────────────────────
    function forwardPassRaw(input) {
        y[0][0] = 1;   // bias
        for (let j = 1; j <= inputDim; j++) {
            y[0][j] = input[j - 1];
        }
        for (let k = 1; k < L; k++) {
            y[k][0] = 1;
            const prevSize = layerSizes[k - 1];
            const currSize = layerSizes[k];
            for (let i = 1; i <= currSize; i++) {
                let net = 0;
                for (let j = 0; j <= prevSize; j++) {
                    net += pesos[k - 1][i - 1][j] * y[k - 1][j];
                }
                y[k][i] = sigmoid(net);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Backpropagation + actualización de pesos (basado en labels[])
    // ─────────────────────────────────────────────────────────────
    function backwardPass(n, l) {
        const target = labels[n];
        const outK   = L - 1;
        const outSize = layerSizes[outK];

        // 1. Deltas de la capa de salida
        for (let i = 1; i <= outSize; i++) {
            const yi = y[outK][i];
            delta[outK][i] = yi * (1 - yi) * (yi - target[i - 1]);
        }

        // 2. Deltas de las capas ocultas
        for (let k = outK - 1; k >= 1; k--) {
            const currSize = layerSizes[k];
            const nextSize = layerSizes[k + 1];
            for (let j = 0; j <= currSize; j++) {
                let sum = 0;
                for (let i = 1; i <= nextSize; i++) {
                    sum += delta[k + 1][i] * pesos[k][i - 1][j];
                }
                delta[k][j] = y[k][j] * (1 - y[k][j]) * sum;
            }
        }

        // 3. Ajuste de pesos en línea
        for (let k = 0; k < numWeightMatrices; k++) {
            const srcSize  = layerSizes[k];
            const dstSize  = layerSizes[k + 1];
            for (let i = 0; i < dstSize; i++) {
                const di = delta[k + 1][i + 1];
                for (let j = 0; j <= srcSize; j++) {
                    pesos[k][i][j] += eta * di * y[k][j];
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Backpropagation con target arbitrario (para trainOnDataset)
    // target: array de outputDim valores deseados
    // ─────────────────────────────────────────────────────────────
    function backwardPassRaw(target) {
        const outK    = L - 1;
        const outSize = layerSizes[outK];

        // 1. Deltas de la capa de salida
        for (let i = 1; i <= outSize; i++) {
            const yi = y[outK][i];
            delta[outK][i] = yi * (1 - yi) * (yi - target[i - 1]);
        }

        // 2. Deltas de las capas ocultas
        for (let k = outK - 1; k >= 1; k--) {
            const currSize = layerSizes[k];
            const nextSize = layerSizes[k + 1];
            for (let j = 0; j <= currSize; j++) {
                let sum = 0;
                for (let i = 1; i <= nextSize; i++) {
                    sum += delta[k + 1][i] * pesos[k][i - 1][j];
                }
                delta[k][j] = y[k][j] * (1 - y[k][j]) * sum;
            }
        }

        // 3. Ajuste de pesos en línea
        for (let k = 0; k < numWeightMatrices; k++) {
            const srcSize  = layerSizes[k];
            const dstSize  = layerSizes[k + 1];
            for (let i = 0; i < dstSize; i++) {
                const di = delta[k + 1][i + 1];
                for (let j = 0; j <= srcSize; j++) {
                    pesos[k][i][j] += eta * di * y[k][j];
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Suma de errores cuadráticos sobre todos los patrones de clase n
    // ─────────────────────────────────────────────────────────────
    function computeSSE(n) {
        let sse = 0;
        const outK   = L - 1;
        const outSize = layerSizes[outK];
        for (let l = 0; l < numSamples; l++) {
            forwardPass(n, l);
            for (let i = 1; i <= outSize; i++) {
                sse += (y[outK][i] - labels[n][i - 1]) ** 2;
            }
        }
        return 0.5 * sse;
    }

    // ─────────────────────────────────────────────────────────────
    // API pública
    // ─────────────────────────────────────────────────────────────

    /**
     * Carga los patrones de entrenamiento para una clase.
     * @param {number}     classIndex  - Índice de clase (0-based)
     * @param {number[][]} samples     - Array de numSamples vectores, cada uno con inputDim valores
     */
    function loadPatterns(classIndex, samples) {
        if (classIndex < 0 || classIndex >= numClasses) {
            throw new RangeError(`classIndex debe estar en [0, ${numClasses - 1}]`);
        }
        for (let l = 0; l < numSamples; l++) {
            patterns[classIndex][l][0] = 1;   // bias
            for (let j = 1; j <= inputDim; j++) {
                patterns[classIndex][l][j] = samples[l][j - 1];
            }
        }
    }

    /**
     * Define los targets (etiquetas) para una clase.
     * @param {number}   classIndex  - Índice de clase (0-based)
     * @param {number[]} targets     - Array de outputDim valores deseados
     */
    function loadLabels(classIndex, targets) {
        if (classIndex < 0 || classIndex >= numClasses) {
            throw new RangeError(`classIndex debe estar en [0, ${numClasses - 1}]`);
        }
        if (targets.length !== outputDim) {
            throw new Error(`targets debe tener ${outputDim} elementos`);
        }
        for (let i = 0; i < outputDim; i++) {
            labels[classIndex][i] = targets[i];
        }
    }

    /**
     * Entrena la red.
     * @param {object} opts
     * @param {number} opts.maxEpochs   - Máximo de épocas (default 100000)
     * @param {number} opts.targetSSE   - SSE objetivo para detener (default 0.005)
     * @param {number} opts.trainClass  - Índice de clase a usar para entrenamiento (default 0)
     * @param {number} opts.logEvery    - Cada cuántas épocas imprimir progreso (default 100)
     * @returns {{ epoch: number, sse: number }}
     */
    function train({ maxEpochs = 100000, targetSSE = 0.005, trainClass = 0, logEvery = 100 } = {}) {
        let epoch = 0;
        let sse   = Infinity;
        const n   = trainClass;

        while (epoch <= maxEpochs && sse >= targetSSE) {
            epoch++;

            for (let l = 0; l < numSamples; l++) {
                forwardPass(n, l);
                backwardPass(n, l);
            }

            // Calcular SSE total al final de cada época
            sse = computeSSE(n);

            if (epoch % logEvery === 0) {
                console.log(`epoch=${epoch}  sse=${sse.toFixed(6)}`);
            }
        }

        return { epoch, sse };
    }

    /**
     * Reconocimiento: evalúa todos los patrones de una clase y reporta.
     * @param {object} opts
     * @param {number} opts.classIndex   - Índice de clase (default 0)
     * @param {number} opts.sseThreshold - Umbral de SSE para considerar reconocido (default 0.02)
     * @returns {object[]}  Array con resultados por muestra
     */
    function recognise({ classIndex = 0, sseThreshold = 0.02 } = {}) {
        const n      = classIndex;
        const outK   = L - 1;
        const outSize = layerSizes[outK];
        const results = [];

        for (let l = 0; l < numSamples; l++) {
            forwardPass(n, l);

            let sse = 0;
            const outputs = [];
            const targets = [];
            for (let i = 1; i <= outSize; i++) {
                const err = y[outK][i] - labels[n][i - 1];
                sse += err * err;
                outputs.push(y[outK][i]);
                targets.push(labels[n][i - 1]);
            }
            sse = 0.5 * sse;

            const recognised = sse < sseThreshold;
            const label = recognised ? `class ${n}` : 'unrecognised';

            results.push({ sample: l, label, recognised, outputs, targets, sse });

            console.log(
                `sample=${String(l).padStart(3)}  ${label.padEnd(16)}` +
                `  out=[${outputs.map(v => v.toFixed(3)).join(', ')}]` +
                `  tgt=[${targets.map(v => v.toFixed(1)).join(', ')}]` +
                `  sse=${sse.toFixed(4)}`
            );
        }

        return results;
    }

    // ─────────────────────────────────────────────────────────────
    // trainOnDataset: entrena con pares {input, target} individuales.
    // Útil cuando cada muestra tiene un target diferente (ej: (a+b) mod 5).
    //
    // @param {Array<{input: number[], target: number[]}>} dataset
    // @param {object} opts
    // @param {number} opts.maxEpochs  - Máximo de épocas (default 50000)
    // @param {number} opts.targetSSE  - SSE total para detener (default 0.05)
    // @param {number} opts.logEvery   - Frecuencia de log (default 1000)
    // @returns {{ epoch: number, sse: number }}
    // ─────────────────────────────────────────────────────────────
    function trainOnDataset(dataset, { maxEpochs = 50000, targetSSE = 0.05, logEvery = 1000 } = {}) {
        const outK    = L - 1;
        const outSize = layerSizes[outK];
        let epoch = 0;
        let sse   = Infinity;

        while (epoch < maxEpochs && sse >= targetSSE) {
            epoch++;

            // Una pasada completa por todo el dataset (online)
            for (const { input, target } of dataset) {
                forwardPassRaw(input);
                backwardPassRaw(target);
            }

            // Calcular SSE total al final de cada época
            sse = 0;
            for (const { input, target } of dataset) {
                forwardPassRaw(input);
                for (let i = 1; i <= outSize; i++) {
                    sse += (y[outK][i] - target[i - 1]) ** 2;
                }
            }
            sse *= 0.5;

            if (epoch % logEvery === 0) {
                console.log(`epoch=${String(epoch).padStart(7)}  SSE=${sse.toFixed(6)}`);
            }
        }

        return { epoch, sse };
    }

    /**
     * Evalúa la red sobre un dataset de pares {input, target}.
     * @param {Array<{input: number[], target: number[], label: *}>} dataset
     * @returns {object[]}  Array con resultados por muestra
     */
    function evaluateDataset(dataset) {
        const outK    = L - 1;
        const outSize = layerSizes[outK];
        const results = [];

        for (const item of dataset) {
            forwardPassRaw(item.input);
            const outputs = [];
            let sse = 0;
            for (let i = 1; i <= outSize; i++) {
                outputs.push(y[outK][i]);
                sse += (y[outK][i] - item.target[i - 1]) ** 2;
            }
            results.push({ label: item.label, outputs, target: item.target, sse: 0.5 * sse });
        }

        return results;
    }

    // Exponer también acceso de lectura a pesos (útil para depuración/serialización)
    function getWeights() {
        return pesos.map(m => m.map(row => row.slice()));
    }

    function setWeights(newPesos) {
        for (let k = 0; k < numWeightMatrices; k++) {
            for (let i = 0; i < pesos[k].length; i++) {
                for (let j = 0; j < pesos[k][i].length; j++) {
                    pesos[k][i][j] = newPesos[k][i][j];
                }
            }
        }
    }

    return { loadPatterns, loadLabels, train, recognise, trainOnDataset, evaluateDataset, getWeights, setWeights };
}

// ─────────────────────────────────────────────────────────────
// Exportación
// ─────────────────────────────────────────────────────────────
module.exports = { createNetwork };

// ─────────────────────────────────────────────────────────────
// Self-test:  node backprop_n_layers.js
//
// Entrena la red con la tabla completa de (a + b) mod 5
// para a, b ∈ {0, 1, 2, 3, 4, 5}  → 36 combinaciones en total.
//
// Codificación:
//   Entrada (12 neuronas): one-hot de a (6 bits) + one-hot de b (6 bits)
//     Ejemplo: a=2 → [0,0,1,0,0,0]   b=3 → [0,0,0,1,0,0]
//     Entrada total: [0,0,1,0,0,0, 0,0,0,1,0,0]
//   Salida  (5 neuronas): one-hot del resultado (a+b) mod 5
//     clase 0 → [1,0,0,0,0]  ...  clase 4 → [0,0,0,0,1]
//
// Arquitectura: 12 → 20 → 10 → 5
// ─────────────────────────────────────────────────────────────
if (require.main === module) {

    const MAX_VAL     = 5;           // a, b ∈ {0 .. MAX_VAL}  (inclusive → 6 valores)
    const NUM_VALS    = MAX_VAL + 1; // 6 valores posibles por operando
    const NUM_CLASSES = 5;           // resultados posibles de mod 5: 0,1,2,3,4
    const INPUT_DIM   = NUM_VALS * 2;// 12 entradas (one-hot a + one-hot b)

    // ── Funciones de codificación ───────────────────────────────

    /** one-hot vector de longitud n con un 1 en posición k */
    const oneHot = (k, n) => Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));

    /** Codificación de entrada para el par (a, b) en one-hot */
    const encodeInput = (a, b) => [
        ...Array.from({ length: NUM_VALS }, (_, i) => (i === a ? 1 : 0)),
        ...Array.from({ length: NUM_VALS }, (_, i) => (i === b ? 1 : 0)),
    ];

    // ── 1. Construir el dataset: 36 pares {input, target, label} ─
    // Cada muestra tiene su propio target one-hot según (a+b) mod 5.

    const dataset = [];
    for (let a = 0; a <= MAX_VAL; a++) {
        for (let b = 0; b <= MAX_VAL; b++) {
            const resultado = (a + b) % NUM_CLASSES;
            dataset.push({
                input:  encodeInput(a, b),
                target: oneHot(resultado, NUM_CLASSES),
                label:  { a, b, resultado },
            });
        }
    }

    // Imprimir resumen del dataset (tabla de multiplicación)
    console.log('═══════════════════════════════════════════════════════');
    console.log(` Tabla (a + b) mod 5  —  a,b ∈ {0..${MAX_VAL}}  (${dataset.length} combinaciones)`);
    console.log('═══════════════════════════════════════════════════════');
    console.log('\n   (a+b)%5 │  b=0  b=1  b=2  b=3  b=4  b=5');
    console.log('  ─────────┼──────────────────────────────────');
    for (let a = 0; a <= MAX_VAL; a++) {
        const fila = Array.from({ length: NUM_VALS }, (_, b) => (a + b) % NUM_CLASSES);
        console.log(`  a=${a}      │   ${fila.join('    ')}`);
    }
    console.log(`\n Codificación: one-hot×2 → ${INPUT_DIM} entradas | salida one-hot → ${NUM_CLASSES} neuronas\n`);

    // ── 2. Crear la red ────────────────────────────────────────
    //   Input:   12 neuronas (one-hot a ++ one-hot b)
    //   Hidden1: 20 neuronas
    //   Hidden2: 10 neuronas
    //   Output:  5 neuronas (one-hot resultado)
    //
    // numClasses=1, numSamples=1: no usamos el sistema de clases interno;
    // el entrenamiento se hace completamente via trainOnDataset().
    const net = createNetwork([INPUT_DIM, 20, 10, NUM_CLASSES], 1, 1, -0.25);

    // ── 3. Entrenar con todos los 36 pares simultáneamente ─────
    console.log('═══════════════════════════════════════════════════════');
    console.log(' Entrenamiento  (maxEpochs=50000, targetSSE=0.05)');
    console.log('═══════════════════════════════════════════════════════');

    const { epoch, sse } = net.trainOnDataset(dataset, {
        maxEpochs: 50000,
        targetSSE: 0.05,
        logEvery:  2000,
    });

    // ── 4. Resultados finales ──────────────────────────────────
    console.log(`\n Fin: epoch=${epoch}  SSE=${sse.toFixed(6)}\n`);

    const results = net.evaluateDataset(dataset);

    console.log('═══════════════════════════════════════════════════════');
    console.log(' Resultados finales');
    console.log('═══════════════════════════════════════════════════════\n');

    console.log('  a   b │ esperado │ predicción │ correcto');
    console.log('────────┼──────────┼────────────┼─────────');

    let aciertos = 0;
    for (const { label: { a, b, resultado }, outputs } of results) {
        const predicho = outputs.indexOf(Math.max(...outputs));
        const correcto = predicho === resultado;
        if (correcto) aciertos++;
        console.log(
            `  ${String(a).padStart(2)}  ${String(b).padStart(2)} │` +
            `    ${resultado}     │` +
            `     ${predicho}      │  ${correcto ? '✓' : '✗'}`
        );
    }

    console.log('────────┴──────────┴────────────┴─────────');
    console.log(`\n Precisión: ${aciertos}/${results.length} = ${(100 * aciertos / results.length).toFixed(1)}%\n`);
}

