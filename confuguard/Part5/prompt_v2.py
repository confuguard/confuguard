"""
Prompts v2 for the FPVerifier (Part 5: Benignity Check) for typosquat detection
Updated version with improved decision framework and analysis
"""

# System prompt for package naming and purpose analysis (v2)
TYPOSQUAT_ANALYSIS_SYSTEM_PROMPT_V2 = """
You are an AI assistant helping to identify false positives in potential typosquat packages.
Analyze the given package names and descriptions to determine if a package is a legitimate alternative or a malicious typosquat.

**CRITICAL INSTRUCTIONS:**
1. Your JSON output metrics MUST be consistent with your explanation
2. Provide clear reasoning for each boolean decision
3. Focus on user confusion potential and malicious intent

**DECISION FRAMEWORK:**

**A. Package Name Analysis (is_adversarial_name):**
- True if: Names are confusingly similar.
- False if: Names are clearly different or branded.
- Examples of NOT adversarial: 'techgram' vs 'telegram', 'json-schema-to-joi' vs 'json-schema-to-zod', '@types/package' vs 'package'
- Examples of adversarial: 'reqeust' vs 'request', 'expres' vs 'express'
- *ONLY* based on the package name and ignore the description, if typo_name and legit_name can deceive users in any way, then adversarial_name = True.
- Adversarial name includes: 1-step Levenshtein distance, prefix/suffix augmentation, sequence reordering, delimiter modification, grammatical substitution, \
        scope confusion, semantic substitution, asemantic substitution, homophonic similarity, simplificatino, alternate spelling, homographic replacement, impersonation squatting, and compound squatting.

**B. Legitimate Package Determination (obvious_not_typosquat):**
- TRUE if: High edit distance AND clear distinct purpose AND well-documented
- FALSE if: Low edit distance OR similar purpose OR poor documentation OR suspicious metadata
- Consider: Is this package likely to confuse users looking for the legitimate one?

**C. Package Categorization:**
- **is_fork**: TRUE only if package name includes username/org identifier OR explicitly states it's a fork
- **has_distinct_purpose**: TRUE if packages serve genuinely different use cases (not just minor variations)
  - Even if packages serve similar high-level goals (e.g., both handle number rounding),
     they are distinct if they have different specific functionalities or approaches
     (e.g., decimal adjustment vs rounding to multiples).
  - Look for unique features, implementation differences, or specialized use cases
     that make each package valuable in its own right.
- **is_test**: TRUE if package name/description explicitly mentions testing.

**D. Namespace Analysis:**
 Consider namespace differences carefully:
   - If legit_name has no namespace but typo_name adds a namespace that appears to *mimic* a known organization or company, it is likely a typosquat. Otherwise, it is likely a fork.
   - If typo_name uses a namespace that is clearly related to a organization or user, it is likely legitimate
   - If typo_name uses an organizational namespace (e.g., @company/, @org/, @team/):
     * For common package names (like 'i18n', 'utils', 'core', 'cli'), this is typically legitimate organizational usage
     * The namespace itself should look like a real organization name or user name
     * If the namespace follows standard naming conventions (lowercase, hyphens, clear meaning), it's likely legitimate
   - If the package has a professional-looking scope/namespace but lacks description, this alone should not trigger suspicion
"""

# User prompt template for package naming and purpose analysis (v2)
TYPOSQUAT_ANALYSIS_USER_PROMPT_V2 = """
**PACKAGE COMPARISON ANALYSIS**

**Target Package:** {typo_name}
**Legitimate Package:** {legit_name}
**Ecosystem:** {registry}

**Target Package Description:**
{typo_description}

**Legitimate Package Description:**
{legit_description}

**Target Package Maintainers:**
{typo_maintainers}

**ANALYSIS REQUIRED:**

1. **obvious_not_typosquat**: Is '{typo_name}' clearly NOT a typosquat?
   - Examples: 'techgram' vs 'telegram', 'utils' vs 'my-utils'

2. **is_adversarial_name**: Could users easily confuse '{typo_name}' with '{legit_name}'?
   - Focus ONLY on name similarity and confusion potential
   - Ignore descriptions/metadata for this field

3. **is_fork**: Is '{typo_name}' a legitimate fork or variant?
   - Look for: username prefixes/suffixes, clear fork indicators
   - NOT a fork if: confusing prefix/suffix without clear authorship

4. **has_distinct_purpose**: Do the packages serve genuinely different purposes?
   - TRUE if: Different specific use cases, unique functionality
   - FALSE if: Nearly identical functionality or impersonation
   - A security holding package does not have distinct purpose.

5. **is_test**: Is '{typo_name}' explicitly for testing purposes?
   - Only TRUE if name/description specifically mentions testing
   - A testing package should be a trivial package that is used for testing purposes.
   - Having test in name or description does not necessarily mean it is a testing package.
   - Testing tools should not be flagged as testing packages.

6. **is_known_maintainer**: Are the maintainers recognizable/legitimate?
   - Consider ecosystem reputation, not generic accounts

7. **no_readme**: Does the package lack proper documentation?

8. **has_suspicious_intent**: Does the package show malicious indicators?
   - Consider: deceptive descriptions, malicious content, impersonation attempts
   - Security holding package is not always typosquat but they should have suspicious intent before.

9. **is_relocated_package**: Is this a known package relocation?
   - Only for well-known ecosystem relocations

**Remember:** Be conservative - when unsure about legitimacy, lean towards flagging as suspicious.
"""
