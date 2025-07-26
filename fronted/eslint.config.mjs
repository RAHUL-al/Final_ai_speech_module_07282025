import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    // Global rule overrides
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/ban-ts-comment": "off",
    },
  },
  {
    // File-specific rule overrides
    files: [
      "src/components/Practice/PracticePage.tsx",
      "src/components/services/api.tsx",
      "src/store/slices/**/*.tsx",
    ],
    rules: {
      "@typescript-eslint/no-unused-vars": "off",
      "prefer-const": "off",
    },
  },
];

export default eslintConfig;
