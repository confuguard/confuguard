from openai import OpenAI
from typing import Tuple, Dict, List
from datetime import datetime, timezone
from Levenshtein import damerau_levenshtein_distance
import json
from loguru import logger
from .prompt_v1 import (
    TYPOSQUAT_ANALYSIS_SYSTEM_PROMPT,
    TYPOSQUAT_ANALYSIS_USER_PROMPT
)

class FPVerifier:
    """False Positive Verifier for typosquat detection"""

    def __init__(self, openai_api_key: str):
        self.logger = logger
        logger.critical(f"Initializing FPVerifier with openai_api_key: {openai_api_key}")
        self.client = OpenAI(api_key=openai_api_key)

    def _extract_name(self, doc: dict, registry: str) -> str:
        """Extract the package name from the document based on the registry."""
        if registry in ['maven', 'repo1.maven.org']:
            return doc.get('name') or doc.get('doc', {}).get('name') or doc.get('n') or doc.get('info', {}).get('name')
        elif registry in ['pypi', 'pypi.org']:
            return doc.get('name') or doc.get('info', {}).get('name')
        elif registry == 'nuget':
            return doc.get('name')
        else:
            return doc.get('name') or doc.get('info', {}).get('name')

    def _extract_description(self, doc: dict, registry: str) -> str:
        """Extract the package description from the document based on the registry."""
        if registry in ['maven', 'repo1.maven.org']:
            description = doc.get('description') or doc.get('doc', {}).get('description') or doc.get('d')
        elif registry in ['pypi', 'pypi.org']:
            info = doc.get('info', {})
            description = info.get('description')
            summary = info.get('summary')

            # Combine summary with description
            description = f"{summary}\n\n{description}" if summary and description else description or summary
        elif registry == 'nuget':
            description = doc.get('description')
            summary = doc.get('summary')

            # Combine summary with description
            description = f"{summary}\n\n{description}" if summary and description else description or summary
        elif registry == 'npm':
            description = ('Description:' + (doc.get('description') or '')) + ('\n\nREADME:' + (doc.get('readme') or ''))
        else:
            description = doc.get('readme') or doc.get('description') or doc.get('doc') or doc.get('payload')

        # Ensure description is a string and handle truncation
        if description:
            if isinstance(description, dict):
                # Handle cases where description might be a dictionary
                description = str(description)
            elif not isinstance(description, str):
                # Convert any other non-string type to string
                description = str(description)

            words = description.split()[:100]
            description = ' '.join(words)
            if len(words) == 100 and len(description.split()) > 100:
                description += "..."
        return description or ""

    def _extract_maintainers(self, doc: dict, registry: str) -> List[str]:
        """Extract the maintainers from the document based on the registry."""
        maintainers = []
        if registry == 'npm':
            maintainers = [m.get('name') for m in doc.get('maintainers', []) if m and isinstance(m, dict)]
            if doc.get('authors') and isinstance(doc.get('authors'), dict):
                maintainers.append(doc.get('authors', {}).get('name'))
        elif registry in ['pypi', 'pypi.org']:
            maintainers = doc.get('maintainers', [])
        elif registry in ['maven', 'repo1.maven.org']:
            doc_info = doc.get('doc', {})
            if 'maintainers' in doc_info and isinstance(doc_info['maintainers'], list):
                maintainers.extend([m.get('name') for m in doc_info.get('maintainers', []) if m])
            if 'a' in doc_info:
                maintainers.append(doc_info.get('a'))
            repo_metadata = doc.get('repo_metadata', {})
            if repo_metadata and isinstance(repo_metadata, dict) and 'owner' in repo_metadata:
                maintainers.append(repo_metadata.get('owner'))
        elif registry == 'nuget':
            # Handle case where owners is None
            owners = doc.get('owners', [])
            if owners is not None:
                maintainers = owners if isinstance(owners, list) else [owners]
        else:
            maintainers = [m.get('login') or m.get('uuid') for m in doc.get('maintainers', []) if m]
            repo_metadata = doc.get('repo_metadata', {})
            if repo_metadata and isinstance(repo_metadata, dict) and 'owner' in repo_metadata:
                maintainers.append(repo_metadata.get('owner'))
        # Filter out None and empty strings
        return [m for m in maintainers if m]

    # TODO: Change distinct purpose to "whether the impersonation"
    def _check_package_naming_and_purpose(self, typo_doc: dict, legit_doc: dict) -> Tuple[bool, bool, bool, bool, bool, bool, bool, str]:
        """
        Check if the package name indicates intentional naming, forking, or testing purposes using OpenAI's API

        Returns:
            Tuple[bool, bool, bool, bool, bool, bool, bool, str]: (is_obvious_not_typosquat, is_adversarial_name,
            is_intentional, is_fork, has_distinct_purpose, is_test, is_known_maintainer, has_suspicious_intent, explanation)
        """
        registry = typo_doc.get('registry')
        typo_name = self._extract_name(typo_doc, registry)
        legit_name = self._extract_name(legit_doc, registry)
        typo_description = self._extract_description(typo_doc, registry)
        legit_description = self._extract_description(legit_doc, registry)
        typo_maintainers = self._extract_maintainers(typo_doc, registry)

        # logger.debug(f"Typo name: {typo_name}, Legit name: {legit_name}")
        # logger.debug(f"Typo description: {typo_description}, Legit description: {legit_description}")
        # logger.debug(f"Typo maintainers: {typo_maintainers}")

        edit_distance = damerau_levenshtein_distance(typo_name, legit_name)

        # TODO: Why a package is a typosquat and why it is not a typosquat.
        system_prompt = TYPOSQUAT_ANALYSIS_SYSTEM_PROMPT

        user_prompt = TYPOSQUAT_ANALYSIS_USER_PROMPT.format(
            typo_name=typo_name,
            legit_name=legit_name,
            registry=registry,
            typo_maintainers=typo_maintainers,
            typo_description=typo_description,
            legit_description=legit_description
        )

        json_schema = {
            "name": "TyposquatAnalysisResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "obvious_not_typosquat": {"type": "boolean"},
                    "is_adversarial_name": {"type": "boolean"},
                    "is_fork": {"type": "boolean"},
                    "has_distinct_purpose": {"type": "boolean"},
                    "is_test": {"type": "boolean"},
                    "is_known_maintainer": {"type": "boolean"},
                    "no_readme": {"type": "boolean"},
                    "has_suspicious_intent": {"type": "boolean"},
                    "is_relocated_package": {"type": "boolean"},
                    "explanation": {"type": "string"}
                },
                "required": [
                    "obvious_not_typosquat",
                    "is_adversarial_name",
                    "is_fork",
                    "has_distinct_purpose",
                    "is_test",
                    "is_known_maintainer",
                    "no_readme",
                    "has_suspicious_intent",
                    "is_relocated_package",
                    "explanation"
                ],
                "additionalProperties": False
            }
        }

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="o3-mini",
                reasoning_effort="low",
                response_format={"type": "json_schema", "json_schema": json_schema}
            )

            json_content = response.choices[0].message.content.strip()
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse JSON: {json_error}")
                logger.error(f"Cleaned response content: {json_content}")
                return False, False, False, False, False, False, False, ""

            is_obvious_not_typosquat = result.get('obvious_not_typosquat', False)
            is_adversarial_name = result.get('is_adversarial_name', False)
            # is_intentional = result.get('is_intentional', False)
            is_fork = result.get('is_fork', False)
            has_distinct_purpose = result.get('has_distinct_purpose', False)
            is_test = result.get('is_test', False)
            is_known_maintainer = result.get('is_known_maintainer', False)
            has_suspicious_intent = result.get('has_suspicious_intent', False)
            no_readme = result.get('no_readme', False)
            is_relocated_package = result.get('is_relocated_package', False)
            explanation = result.get('explanation', '')
            logger.debug(f"OpenAI API response for {typo_name} -> {legit_name}: Adversarial: {is_adversarial_name}, Fork: {is_fork}, Distinct Purpose: {has_distinct_purpose}, Test: {is_test}, Known Maintainer: {is_known_maintainer}, Suspicious: {has_suspicious_intent}")
            logger.debug(f"Explanation: {explanation}")

            return is_obvious_not_typosquat, is_adversarial_name, is_fork, has_distinct_purpose, is_test, is_known_maintainer, has_suspicious_intent, no_readme, is_relocated_package, explanation

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return False, False, False, False, False, False, False, False, False, ""

    def _is_actively_developed(self, repo_metadata: dict) -> bool:
        """Check if the package is actively maintained"""
        pushed_at = repo_metadata.get('pushed_at')
        versions_count = repo_metadata.get('versions_count', 1)

        if not pushed_at:
            return False

        last_push_date = datetime.strptime(pushed_at, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
        days_since_last_push = (datetime.now(timezone.utc) - last_push_date).days
        return days_since_last_push < 365 and versions_count > 5

    def _has_overlapped_maintainers(self, typo_doc: dict, legit_doc: dict, registry: str) -> bool:
        """Check if the package has overlapping maintainers to detect false positives."""
        typo_maintainers = self._extract_maintainers(typo_doc, registry)
        legit_maintainers = self._extract_maintainers(legit_doc, registry)
        # logger.debug(f"Typo maintainers: {typo_maintainers}, Legit maintainers: {legit_maintainers}")

        # If typo package has no maintainers, it's suspicious - not a false positive
        if not typo_maintainers:
            logger.debug("Typo package has no maintainers - suspicious")
            return False

        # If legitimate package has no maintainers, we can't make a determination
        if not legit_maintainers:
            return False

        # Convert to sets for O(1) lookups and unique comparisons
        overlapping = len(set(typo_maintainers) & set(legit_maintainers))
        legit_count = len(legit_maintainers)

        # Decision logic
        if legit_count > 3:
            # logger.debug("Legitimate package has many maintainers (>3), requiring >=2 overlaps or >50% ratio")
            return overlapping >= 2 or (overlapping / legit_count) > 0.5
        return overlapping >= 1

    def _has_comprehensive_metadata(self, doc: dict) -> bool:
        """Check if the package has comprehensive metadata"""
        required_fields = ['license', 'maintainers', 'repository_url', 'webpage']
        missing_fields = [field for field in required_fields if not doc.get(field)]
        present_fields = len(required_fields) - len(missing_fields)

        if present_fields < 3 or 'description' not in doc or 'readme' not in doc:
            # self.logger.debug(f"Only {present_fields} required metadata fields present, missing: {', '.join(missing_fields)}")
            return False
        return True

    # def _is_maven_relocation(self, doc: dict) -> bool:
    #     """Check if the package is a Maven relocation"""
    #     return doc.get('info', {}).get('summary') == 'Maven relocation'

    def verify_command_squatting(self, typo_name: str, legit_command: str, typo_doc: dict, registry: str) -> Tuple[bool, Dict[str, float], str, str]:
        """
        Uses the LLM to decide whether a suspicious package (potential command squat)
        is a false positive based on package name similarity and metadata.

        Returns:
            Tuple[bool, Dict[str, float], str, str]: (
                is_false_positive,
                metrics (including confidence and edit_distance),
                explanation,
                detection_type
            )
        """
        edit_distance = damerau_levenshtein_distance(typo_name.lower(), legit_command.lower())
        typo_maintainers = self._extract_maintainers(typo_doc, registry)
        typo_description = self._extract_description(typo_doc, registry)

        system_prompt = """
        You are an AI assistant specialized in detecting command squatting.
        Analyze the given suspicious package name and metadata to determine if this is a legitimate package or a potential threat targetting the give ncommand.
        Please make sure the output metrics and your explanation are consistent with each other.
        """

        user_prompt = f"""
          Compare the suspicious package name "{typo_name}" with the legitimate command name "{legit_command}".
          If the maintainers of {typo_name} have legitimate maintainers or developers in the {registry} community, it is likely not a false positive typosquat alarm.
          If the description of {typo_name} is distinctly described in the pacakge description, it is likely not a false positive typosquat alarm.

          The maintainers of {typo_name} are: {typo_maintainers}

          The description of potentially suspicious package {typo_name}  is:
          {typo_description}

          Decide whether "{typo_name}" is a command squat (i.e. a true threat) or a false positive based on the maintainers and description.
        """

        # Define the JSON schema for the expected output
        json_schema = {
            "name": "CommandSquattingResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "is_false_positive": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "description": "A confidence score between 0 and 1"
                    },
                    "explanation": {"type": "string"}
                },
                "required": ["is_false_positive", "confidence", "explanation"],
                "additionalProperties": False
            }
        }

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="o3-mini",
                reasoning_effort="low",
                response_format={"type": "json_schema", "json_schema": json_schema}
            )

            result = json.loads(response.choices[0].message.content.strip())

            metrics = {
                "confidence": result.get("confidence", 0.0),
                "edit_distance": edit_distance
            }
            if metrics["confidence"] > 0.8:
                return (
                    result.get("is_false_positive", False),
                    metrics,
                    result.get("explanation", ""),
                    "command_squatting"
                )
            else:
                return (
                    True,
                    metrics,
                    result.get("explanation", "AI is not confident enough to tell if it's a command squat"),
                    "command_squatting"
                )

        except Exception as e:
            logger.error(f"LLM API error in verify_command_squatting: {str(e)}")
            raise

    def verify(self, typo_doc: dict, legit_doc: dict, registry: str) -> Tuple[bool, Dict[str, bool], str]:
        """
        Verify if a package is a false positive typosquat

        Args:
            typo_doc: Metadata for the potential typosquat package
            legit_doc: Metadata for the legitimate package
            registry: Package registry name

        Returns:
            Tuple of (is_false_positive: bool, metrics: Dict[str, bool])
        """
        # Add registry information to typo_doc for the _check_package_naming_and_purpose method
        typo_doc['registry'] = registry
        legit_doc['registry'] = registry

        # Extract name at the beginning using the existing helper method
        typo_name = self._extract_name(typo_doc, registry)

        metrics = {
            "obvious_not_typosquat": None,
            "is_adversarial_name": None,
            "has_distinct_purpose": None,
            "is_fork": None,
            "active_development": None,
            "comprehensive_metadata": None,
            "overlapped_maintainers": None,
            "is_known_maintainer": None,
            "has_suspicious_intent": None,
            "no_readme": None,
            "is_relocated_package": None
        }

        # Check naming patterns and purpose
        metrics["obvious_not_typosquat"], metrics["is_adversarial_name"], \
        metrics["is_fork"], metrics["has_distinct_purpose"], metrics["is_test"], \
        metrics["is_known_maintainer"], metrics["has_suspicious_intent"], \
        metrics["no_readme"], metrics["is_relocated_package"], explanation = \
            self._check_package_naming_and_purpose(typo_doc, legit_doc)

        metrics["overlapped_maintainers"] = self._has_overlapped_maintainers(typo_doc, legit_doc, registry)
        metrics["comprehensive_metadata"] = self._has_comprehensive_metadata(typo_doc)
        metrics["active_development"] = self._is_actively_developed(typo_doc)

        logger.debug(f"Metrics for {typo_name}: {metrics}")
        # --- Scoring logic ---
        risk_score = 0
        # Increase risk for suspicious factors
        if metrics["is_adversarial_name"]:
            risk_score += 8
        if metrics["has_suspicious_intent"] and not metrics["no_readme"]:
            risk_score += 10
        elif metrics["no_readme"]:
            risk_score += 3
        if not metrics["comprehensive_metadata"]:
            risk_score += 2
        if metrics["active_development"]:
            risk_score -= 1

        # Mitigate risk for factors that point to a legitimate package
        if metrics["obvious_not_typosquat"]:
            risk_score -= 18
        if metrics["is_fork"]:
            risk_score -= 2
        if metrics["is_test"]:
            risk_score -= 5
        if metrics["has_distinct_purpose"]:
            risk_score -= 18
        if metrics["is_known_maintainer"] and not metrics["has_suspicious_intent"]:
            risk_score -= 18
        if metrics["overlapped_maintainers"]:
            risk_score -= 18
        if metrics["is_relocated_package"]:
            risk_score -= 18

        # Clamp the risk score between 0 and 10
        risk_score = max(0, min(10, risk_score))

        # Determine risk level based on thresholds
        if risk_score < 5:
            risk_level = "Low"
        elif risk_score < 8:
            risk_level = "Medium"
        else:
            risk_level = "High"
        logger.debug(f"Risk score for {typo_name}: {risk_score} (Risk level: {risk_level}).")
        explanation += f" Risk level: {risk_level})."

        # For example, if the risk is low, consider it a false positive.
        is_false_positive = risk_score < 5
        logger.debug(f"Metrics: {metrics}")
        logger.debug(f"Explanation: {explanation}")
        return is_false_positive, metrics, explanation, f"risk_level: {risk_level}"
