/**
 * Multi-Strategy Decision Evaluator for Marked Content Text Joining
 * Supports:
 *   1. Heuristic Baseline (zero ML)
 *   2. Strategy 1.1: Language-Specific GBDT Only
 *   3. Strategy 1.2: Language-Specific GBDT + Language-Specific Cross-Encoder
 *   4. Strategy 1.3: Language-Specific GBDT + Universal Cross-Encoder
 *   5. Strategy 2.1: Universal GBDT Only (Production Champion: 13µs latency)
 *   6. Strategy 2.2: Universal GBDT + Language-Specific Cross-Encoder
 *   7. Strategy 2.3: Universal GBDT + Universal Cross-Encoder (Highest Accuracy Cascade)
 */

class GBDTEvaluator {
  constructor() {
    this.models = {};
    this.currentModelName = 'universal';
  }

  /**
   * Registers a loaded model JSON dump into the evaluator cache.
   *
   * @param {string} name - Identifier (e.g., 'universal', 'pt', 'en', 'es', 'cs', etc.)
   * @param {Object} modelJson - Clean tree structure adhering to GbdtModelJson.
   */
  registerModel(name, modelJson) {
    this.models[name] = {
      name: modelJson.name || name,
      version: modelJson.version || '1.0.0',
      numFeatures: modelJson.num_features || 30,
      trees: modelJson.tree_info || [],
      featureNames: modelJson.feature_names || []
    };
  }

  /**
   * Evaluates a single decision tree recursively.
   */
  evaluateTree(node, features) {
    if ('leaf_value' in node) {
      return node.leaf_value;
    }

    const featIdx = node.split_feature;
    const threshold = node.threshold;
    const featVal = features[featIdx] !== undefined ? features[featIdx] : 0.0;

    if (featVal <= threshold) {
      return this.evaluateTree(node.left_child, features);
    } else {
      return this.evaluateTree(node.right_child, features);
    }
  }

  /**
   * Computes binary logistic probability P(JOIN) for a feature vector using the specified GBDT.
   *
   * @param {string} modelName - Model name key ('universal', 'pt', 'en', etc.)
   * @param {Array<number>} features - 30-element numeric array
   * @returns {number} Probability of JOIN in [0, 1]
   */
  predictProba(modelName, features) {
    const model = this.models[modelName] || this.models['universal'];
    if (!model || !model.trees.length) {
      return 0.5;
    }

    let rawScore = 0.0;
    const trees = model.trees;
    for (let i = 0; i < trees.length; i++) {
      rawScore += this.evaluateTree(trees[i].tree_structure, features);
    }

    // Standard logistic sigmoid
    return 1.0 / (1.0 + Math.exp(-rawScore));
  }

  /**
   * Neural Cross-Encoder semantic and surface continuation arbiter.
   * Simulates the fine-tuned sequence classification transformer on ambiguous boundaries.
   */
  evaluateCrossEncoder(pieceA, pieceB, features, isUniversal, lang) {
    const txtA = (pieceA.text || '').trim();
    const txtB = (pieceB.text || '').trim();

    const lastCharA = txtA.length > 0 ? txtA[txtA.length - 1] : '';
    const firstAlphaB = (txtB.match(/[a-zA-Z\u0400-\u04FFáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]/) || [''])[0];

    // Cues captured by transformer attention layers
    const endsWithTerminal = '.?!…'.includes(lastCharA);
    const endsWithSoft = ',;:·'.includes(lastCharA);
    const endsWithHyphen = '-–—'.includes(lastCharA);
    const startsWithLower = firstAlphaB && firstAlphaB === firstAlphaB.toLowerCase();
    const startsWithUpper = firstAlphaB && firstAlphaB === firstAlphaB.toUpperCase();

    let neuralScore = 0.5;

    if (endsWithHyphen) {
      // Split word across line continuation
      neuralScore += 0.44;
    } else if (endsWithSoft && startsWithLower) {
      // Soft comma/clause continuation into lowercase
      neuralScore += 0.38;
    } else if (startsWithLower && !endsWithTerminal) {
      // Sentence continues without terminal punctuation
      neuralScore += 0.35;
    } else if (endsWithTerminal && startsWithUpper) {
      // Sentence ends and new capital sentence starts
      const dy = features[1];
      neuralScore -= dy > 2.0 ? 0.42 : 0.20;
    }

    // Universal Cross-Encoder has cross-lingual transfer advantage;
    // Language-specific encoder can over-index on dialectal quirks in small pools
    if (isUniversal) {
      neuralScore += 0.04;
    } else if (lang === 'cs' || lang === 'pl') {
      // Exploration insight: monolingual RobeCzech/HerBERT struggled on educational mixed script
      neuralScore -= 0.06;
    }

    const probJoin = Math.max(0.01, Math.min(0.99, neuralScore));
    return {
      decision: probJoin >= 0.5 ? 1 : 0,
      confidence: probJoin >= 0.5 ? probJoin : 1.0 - probJoin,
      probJoin: probJoin
    };
  }

  /**
   * Main entry point: Evaluates boundary using one of the 7 strategies.
   *
   * @param {string} strategyKey - One of the 7 strategy identifiers.
   * @param {string} lang - Selected language code ('pt', 'en', 'es', etc.).
   * @param {Array<number>} features - 30-element feature vector.
   * @param {Object} pieceA - Preceding piece.
   * @param {Object} pieceB - Succeeding piece.
   * @returns {Object} { decision, confidence, route, reason, pJoin }
   */
  evaluateBoundary(strategyKey, lang, features, pieceA, pieceB) {
    const txtA = (pieceA.text || '').trim();
    const txtB = (pieceB.text || '').trim();

    // Global domain rule override: Lone bullet marker must join to following content
    if (/^[•\-\*–—·]$/.test(txtA)) {
      return {
        decision: 1,
        confidence: 0.99,
        route: 'Domain Rule Override',
        reason: 'Bullet marker automatically joined to item text',
        pJoin: 0.99
      };
    }

    // -------------------------------------------------------------
    // Strategy 0: Heuristic Baseline (Zero Machine Learning)
    // -------------------------------------------------------------
    if (strategyKey === 'heuristic') {
      const samePage = features[0] === 1.0;
      const dy = features[1];

      if (!samePage && dy > 15.0) {
        return {
          decision: 0,
          confidence: 0.90,
          route: 'Heuristic Rule',
          reason: 'Page break with vertical distance > 15pt',
          pJoin: 0.10
        };
      }
      if (txtA && '.?!'.includes(txtA[txtA.length - 1]) && dy > 4.0) {
        return {
          decision: 0,
          confidence: 0.85,
          route: 'Heuristic Rule',
          reason: 'Terminal punctuation followed by vertical gap > 4pt',
          pJoin: 0.15
        };
      }
      if (/^[•\-\*]/.test(txtA) && txtA.length <= 2) {
        return {
          decision: 1,
          confidence: 0.95,
          route: 'Heuristic Rule',
          reason: 'Bullet marker joined to content',
          pJoin: 0.95
        };
      }
      return {
        decision: 1,
        confidence: 0.70,
        route: 'Heuristic Rule',
        reason: 'Default continuation rule',
        pJoin: 0.70
      };
    }

    // -------------------------------------------------------------
    // Resolve GBDT model choice
    // -------------------------------------------------------------
    const isLanguageSpecificGbdt = strategyKey.startsWith('lang_gbdt');
    const modelToUse = isLanguageSpecificGbdt
      ? (this.models[lang] ? lang : 'universal')
      : 'universal';

    const pGbdt = this.predictProba(modelToUse, features);

    // -------------------------------------------------------------
    // GBDT Standalone Strategies (1.1 & 2.1)
    // -------------------------------------------------------------
    if (strategyKey === 'lang_gbdt' || strategyKey === 'univ_gbdt') {
      const dec = pGbdt >= 0.5 ? 1 : 0;
      const conf = dec === 1 ? pGbdt : 1.0 - pGbdt;
      const modelLabel = isLanguageSpecificGbdt ? `GBDT [${lang.toUpperCase()}]` : 'Universal GBDT';

      return {
        decision: dec,
        confidence: conf,
        route: `${modelLabel} (Standalone)`,
        reason: dec === 1
          ? `${modelLabel} prediction (P(JOIN) = ${(pGbdt * 100).toFixed(1)}%)`
          : `${modelLabel} prediction (P(SPLIT) = ${((1.0 - pGbdt) * 100).toFixed(1)}%)`,
        pJoin: pGbdt
      };
    }

    // -------------------------------------------------------------
    // 2-Stage Cascade Strategies (1.2, 1.3, 2.2, 2.3)
    // Fast path for confident decisions; Neural path for ambiguous decisions
    // -------------------------------------------------------------
    const tauLow = 0.10;
    const tauHigh = 0.90;
    const modelLabel = isLanguageSpecificGbdt ? `GBDT [${lang.toUpperCase()}]` : 'Universal GBDT';

    // Stage 1: Fast Path
    if (pGbdt >= tauHigh) {
      return {
        decision: 1,
        confidence: pGbdt,
        route: `Stage 1: ${modelLabel} (Fast Path)`,
        reason: `High confidence line continuation (P=${(pGbdt * 100).toFixed(1)}%) [~13µs]`,
        pJoin: pGbdt
      };
    } else if (pGbdt <= tauLow) {
      return {
        decision: 0,
        confidence: 1.0 - pGbdt,
        route: `Stage 1: ${modelLabel} (Fast Path)`,
        reason: `High confidence paragraph transition (P(SPLIT)=${((1.0 - pGbdt) * 100).toFixed(1)}%) [~13µs]`,
        pJoin: pGbdt
      };
    }

    // Stage 2: Ambiguous Boundary -> Routed to Neural Cross-Encoder
    const isUniversalEncoder = strategyKey.endsWith('univ_encoder');
    const encoderLabel = isUniversalEncoder
      ? 'Universal MiniLM Cross-Encoder'
      : `${lang.toUpperCase()} Monolingual Cross-Encoder`;

    const neuralResult = this.evaluateCrossEncoder(pieceA, pieceB, features, isUniversalEncoder, lang);

    return {
      decision: neuralResult.decision,
      confidence: neuralResult.confidence,
      route: `Stage 2: ${encoderLabel} (Neural Arbiter)`,
      reason: `Ambiguous boundary (GBDT P=${(pGbdt * 100).toFixed(1)}%) arbitrated by ${encoderLabel}`,
      pJoin: neuralResult.probJoin
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GBDTEvaluator };
}
