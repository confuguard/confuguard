"""
Prompts for the FPVerifier (Part 5: Benignity Check) for typosquat detection
"""

# System prompt for package naming and purpose analysis
TYPOSQUAT_ANALYSIS_SYSTEM_PROMPT = """
You are an AI assistant helping to identify false positives in potential typosquat packages.
Analyze the given package names and respond in the specified JSON format.
Provide your response as a raw JSON object, without any Markdown formatting or code block syntax.
**Please always make sure the output metrics and your explanation are consistent with each other!!!**

Important:
- If either {typo_name} or {legit_name} is deprecated or archived, and the other is its update, do not consider it an obvious non-typosquat.
- If the namespaces or scopes of the package are trivially different, it *should* be considered a potential typosquat.
- If the namespace is a transformed version of the legitimate package name, it *should* be considered an adversarial name.
- If the namespaces and scopes of the package are obviously from the same author or organization, it *should not* be considered a typosquat.
  For example: @oxc-parser and @oxc-resolver.
- If the edit distance between {typo_name} and {legit_name} is very high, it *should not* be considered suspicious.
- Package {typo_name} with description saying it is a security holding package *SHOULD* be considered a suspicious package.
  if it has very similar metadata to '{legit_name}'.
*VERYIMPORTANT:
- Typosquats typically involve minor character changes, omissions, or additions that could confuse users.
- If it is not confusing enough or the names are substantially different, it is not suspicious.
- A package with out README does not mean it is a typosquat, it could be a expenrimental package or fork.
- The 'is_adversarial_name' field should be determined SOLELY by analyzing the package name similarity and potential for confusion.
- The 'has_suspicious_intent' field should consider BOTH the package name AND its description/metadata to determine malicious content.
- {typo_name} should be considered a fork package if the package name includes user name as prefix/suffix, such as @username/package-name or package-name-username, or username-package-name.
  The package SHOULD NOT be considered a fork if the package name include confusing prefix/suffix instead of username, such as boltdb-go in golang ecosystem which is a adversarial name of boltdb.
- {typo_name} should *ONLY* be considered a test package if the package name or description specifically mentions testing or used for testing purposes.
- For npm packages, common development namespaces like @types/, @testing/, @dev/ should not be considered suspicious
- For all packages, if the package name is a *common development components*, such as "common", "utils", "core", "cli", "lib", "eslint-plugin", etc.,
    or *common naming patterns*, such as "react-*", "*.io", "eslint-config-*", "*-api-sdk", "*-http", "*-sdk", "*-lib", "*-utils", etc.,
    it *SHOULD NOT* be considered suspicious or adversarial name.
- If the package description includes malicious content (e.g. decrypted content), or confusing content (e.g. very short and vague description with out details), it is likely a typosquat.
- If the package description is ALMOST IDENTICAL DESCRIPTION TO THE LEGITIMATE PACKAGE and the name does not include an username as indication of fork, it is likely a typosquat with malicious intent instead of a fork.
- Only mark the package as obvious_not_typosquat if the package name has high edit distance OR the package description is well-written and detailed and obviously not possible to confuse the users.
- A package with obvious malicious intent should return False for `is_fork`, False for `is_test`, False for `has_distinct_purpose`, and False for `is_known_maintainer`.
- You should always provide explanation for your decision on each metric and be consistent with your explanation.
"""

# User prompt template for package naming and purpose analysis
TYPOSQUAT_ANALYSIS_USER_PROMPT = """
Compare the package name '{typo_name}' with the legitimate package name '{legit_name}' which are from the same {registry} ecosystem.
Determine if:
1. If '{typo_name}' and '{legit_name}' are very similar that {typo_name} is an intentional mimic of '{legit_name}', then obvious_not_typosquat = False.
2. *ONLY* based on the package name and ignore the description, if '{typo_name}' and '{legit_name}' can deceive users in any way, then adversarial_name = True.
    - Adversarial name includes: 1-step Levenshtein distance, prefix/suffix augmentation, sequence reordering, delimiter modification, grammatical substitution, \
        scope confusion, semantic substitution, asemantic substitution, homophonic similarity, simplificatino, alternate spelling, homographic replacement, impersonation squatting, and compound squatting.
3. '{typo_name}' is obviously a fork of '{legit_name}'.
4. '{typo_name}' has a distinct purpose from '{legit_name}'. Consider:
   - Even if packages serve similar high-level goals (e.g., both handle number rounding),
     they are distinct if they have different specific functionalities or approaches
     (e.g., decimal adjustment vs rounding to multiples).
   - Look for unique features, implementation differences, or specialized use cases
     that make each package valuable in its own right.
   - Packages can be distinct even if they overlap in some functionality,
     as long as each offers unique value propositions.
5. '{typo_name}' appears to be a test package or used for testing purposes.
6. Consider namespace differences carefully:
   - If {legit_name} has no namespace but {typo_name} adds a namespace that appears to *mimic* a known organization or company, it is likely a typosquat. Otherwise, it is likely a fork.
   - If {typo_name} uses a namespace that is clearly related to a organization or user, it is likely legitimate
   - If {typo_name} uses an organizational namespace (e.g., @company/, @org/, @team/):
     * For common package names (like 'i18n', 'utils', 'core', 'cli'), this is typically legitimate organizational usage
     * The namespace itself should look like a real organization name or user name
     * If the namespace follows standard naming conventions (lowercase, hyphens, clear meaning), it's likely legitimate
   - If the package has a professional-looking scope/namespace but lacks description, this alone should not trigger suspicion
7. If the maintainers of {typo_name} have legitimate maintainers or developers in the {registry} community, it is likely not a typosquat.
  The maintainers of {typo_name} are: {typo_maintainers}. Note that npm should not be considered as a legitimate maintainer if it's a security holding package.
8. If the package description includes suspicious intent (e.g. decrypted content, malicious content, obvious deceptive name), it is likely a typosquat.
9. If {legit_name} or {typo_name} is a relocated package of the other.
  For example, in Maven, `org.apache.jclouds.api:s3` is a relocated package of `org.jclouds.api:s3`.
  In Go, `github.com/redis/go-redis` is a relocated package of `github.com/go-redis/redis`.
  Please only consider the package as relocated if the relocation is well known in the {registry} community.
10. If the package does not have description or README, then no_readme = True. Else no_readme = False.
The potentially suspicious package is {typo_name}. Its description is:
{typo_description}

The legit package is {legit_name}. Its description is:
{legit_description}
"""


