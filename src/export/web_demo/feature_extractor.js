/**
 * In-browser Feature Extractor for Marked Content Text Joining
 * Extracts 30 Unicode-aware geometric and surface-lexical features from adjacent pieces.
 * Matches the Universal GBDT and Multilingual training specifications exactly.
 */

const TERMINAL_PUNCTUATION = new Set(['.', '?', '!', '…', ';', '‽']);
const SOFT_PUNCTUATION = new Set([',', ';', ':', '·']);
const HYPHENS = new Set(['-', '–', '—', '\u00ad', '\u2010', '\u2011']);
const QUOTES_AND_PARENS = new Set([')', ']', '}', '"', "'", '”', '’', '»', '„', '“', '›', '«', '）', '】']);
const INVERTED_PUNCT = new Set(['¿', '¡']);

const UNIVERSAL_BULLET_REGEX = new RegExp(
  '^(?:[•*–—·\\u2022\\u25E6\\u2043\\u25AA\\u25AB-]|(?:\\d+|[a-zA-Z\\u0400-\\u04FF\\u0456\\u0457\\u0454\\u0491' +
  '\\u0406\\u0407\\u0404\\u0490]|[ivxIVX]+)[.):/-])\\s*$',
  'u'
);

function containsCyrillic(str) {
  for (let i = 0; i < str.length; i++) {
    const code = str.charCodeAt(i);
    if (code >= 0x0400 && code <= 0x04ff) {
      return true;
    }
  }
  return false;
}

function findFirstAlpha(str) {
  for (const char of str) {
    if (char.toLowerCase() !== char.toUpperCase()) {
      return char;
    }
  }
  return '';
}

/**
 * Extracts the 30 layout and surface lexical features from an adjacent boundary pair (Piece A, Piece B).
 *
 * @param {Object} pieceA - The preceding marked content piece.
 * @param {Object} pieceB - The succeeding marked content piece.
 * @returns {Array<number>} 30-element float array for GBDT inference.
 */
function extractBoundaryFeaturesJS(pieceA, pieceB) {
  const pageA = pieceA.page_num !== undefined ? pieceA.page_num : (pieceA.page || 0);
  const pageB = pieceB.page_num !== undefined ? pieceB.page_num : (pieceB.page || 0);
  const samePage = (pageA === pageB) ? 1.0 : 0.0;

  const bboxA = pieceA.bbox || [0, 0, 0, 0];
  const bboxB = pieceB.bbox || [0, 0, 0, 0];

  const [xa0, ya0, xa1, ya1] = bboxA;
  const [xb0, yb0, xb1, yb1] = bboxB;

  const widthA = Math.max(0.0, xa1 - xa0);
  const heightA = Math.max(0.0, ya1 - ya0);
  const widthB = Math.max(0.0, xb1 - xb0);
  const heightB = Math.max(0.0, yb1 - yb0);

  let dy = 0.0;
  let dx = 0.0;
  let isSameLine = 0.0;
  let xOverlap = 0.0;

  if (samePage === 1.0) {
    dy = yb0 - ya1;
    dx = xb0 - xa0;
    isSameLine = (Math.abs(ya0 - yb0) < 4.0 || Math.abs(ya1 - yb1) < 4.0) ? 1.0 : 0.0;
    xOverlap = Math.max(0.0, Math.min(xa1, xb1) - Math.max(xa0, xb0));
  } else {
    dy = 50.0;
    dx = xb0 - xa0;
    isSameLine = 0.0;
    xOverlap = 0.0;
  }

  const fsA = pieceA.font_size || pieceA.fontSize || 10.0;
  const fsB = pieceB.font_size || pieceB.fontSize || 10.0;
  const fsRatio = fsB / Math.max(fsA, 0.1);
  const fsDiff = Math.abs(fsA - fsB);

  const fontA = (pieceA.font_name || pieceA.fontName || '').toLowerCase();
  const fontB = (pieceB.font_name || pieceB.fontName || '').toLowerCase();
  const sameFont = (fontA === fontB) ? 1.0 : 0.0;

  const boldA = (pieceA.is_bold || pieceA.isBold) ? 1.0 : 0.0;
  const boldB = (pieceB.is_bold || pieceB.isBold) ? 1.0 : 0.0;
  const sameBold = (boldA === boldB) ? 1.0 : 0.0;

  const italA = (pieceA.is_italic || pieceA.isItalic) ? 1.0 : 0.0;
  const italB = (pieceB.is_italic || pieceB.isItalic) ? 1.0 : 0.0;
  const sameItalic = (italA === italB) ? 1.0 : 0.0;

  const txtA = (pieceA.text || '').trim();
  const txtB = (pieceB.text || '').trim();

  const lenA = txtA.length;
  const lenB = txtB.length;

  const lastCharA = txtA.length > 0 ? txtA[txtA.length - 1] : '';
  const firstCharB = txtB.length > 0 ? txtB[0] : '';
  const firstAlphaB = findFirstAlpha(txtB);

  const endsWithTerminalPunct = TERMINAL_PUNCTUATION.has(lastCharA) ? 1.0 : 0.0;
  const endsWithSoftPunct = SOFT_PUNCTUATION.has(lastCharA) ? 1.0 : 0.0;
  const endsWithHyphen = HYPHENS.has(lastCharA) ? 1.0 : 0.0;
  const endsWithQuoteOrParen = QUOTES_AND_PARENS.has(lastCharA) ? 1.0 : 0.0;

  const startsWithLowercase = (firstAlphaB && firstAlphaB === firstAlphaB.toLowerCase()) ? 1.0 : 0.0;
  const startsWithUppercase = (firstAlphaB && firstAlphaB === firstAlphaB.toUpperCase()) ? 1.0 : 0.0;
  const startsWithDigit = (firstCharB >= '0' && firstCharB <= '9') ? 1.0 : 0.0;

  const isBulletA = UNIVERSAL_BULLET_REGEX.test(txtA) ? 1.0 : 0.0;
  const isBulletB = UNIVERSAL_BULLET_REGEX.test(txtB) ? 1.0 : 0.0;
  const isShortLabelA = lenA <= 4 ? 1.0 : 0.0;

  const startsWithInvertedPunct = INVERTED_PUNCT.has(firstCharB) ? 1.0 : 0.0;
  const isCyrillic = (containsCyrillic(txtA) || containsCyrillic(txtB)) ? 1.0 : 0.0;

  return [
    samePage, dy, dx, widthA, heightA, widthB, heightB,
    fsA, fsB, fsRatio, fsDiff, sameFont, sameBold, sameItalic,
    isSameLine, xOverlap,
    lenA, lenB, endsWithTerminalPunct, endsWithSoftPunct, endsWithHyphen,
    endsWithQuoteOrParen, startsWithLowercase, startsWithUppercase, startsWithDigit,
    isBulletA, isBulletB, isShortLabelA, startsWithInvertedPunct, isCyrillic
  ];
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { extractBoundaryFeaturesJS };
}
