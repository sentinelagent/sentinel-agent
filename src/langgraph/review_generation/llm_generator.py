"""
LLM Generator Node Implementation

Node 4 in the Review Generation workflow.
Executes LLM calls with structured output parsing (LLM-agnostic).
"""

import ast
import json
import logging
import re
from typing import Dict, List, Any, Optional

from pydantic import ValidationError

from src.langgraph.review_generation.base_node import BaseReviewGenerationNode
from src.langgraph.review_generation.circuit_breaker import CircuitBreaker
from src.langgraph.review_generation.schema import (
    RawLLMReviewOutput,
    RawLLMFinding,
    StructuredPrompt,
)
from src.langgraph.review_generation.exceptions import (
    LLMGenerationError,
    LLMResponseParseError,
    LLMTimeoutError,
    LLMRateLimitError,
)
from src.services.llm.llm_factory import get_llm_client, LLMFactory
from src.services.llm.base_client import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMGeneratorNode(BaseReviewGenerationNode):
    """
    Node 4: Execute LLM calls with structured output parsing.
    
    This node:
    - Accepts any LLM client via dependency injection (defaults to env-based)
    - Calls LLM with structured prompt (system + user)
    - Robustly extracts JSON from various response formats
    - Maps PromptBuilder output schema to internal RawLLMReviewOutput schema
    - Validates output against Pydantic models
    - Tracks token usage and costs
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize LLMGeneratorNode.
        
        Args:
            llm_client: Optional LLM client instance. If None, uses get_llm_client().
            circuit_breaker: Optional circuit breaker for fault tolerance.
            timeout_seconds: Timeout for LLM API calls (default 120s for large prompts).
            max_retries: Number of retries on failure.
        """
        super().__init__(
            name="llm_generator",
            timeout_seconds=timeout_seconds,
            circuit_breaker=circuit_breaker,
            max_retries=max_retries,
        )
        
        # LLM client - lazy initialization if not provided
        self._llm_client = llm_client
        self._cost_tracker = None
        
        # Track token usage for metrics
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        self.logger = logger

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
            self._cost_tracker = LLMFactory.create_cost_tracker(self._llm_client)
            # Log LLM client type
            self.logger.info(
                f"Initialized LLM client: type={type(self._llm_client).__name__}, "
                f"provider={self._llm_client.provider_name}, model={self._llm_client.model}"
            )
        return self._llm_client

    async def _execute_node_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate review using LLM.
        
        Args:
            state: Workflow state containing structured_prompt
            
        Returns:
            Dict with raw_llm_output key containing RawLLMReviewOutput
        """
        self.logger.info("Generating review using LLM")
        
        # Extract structured prompt from state
        structured_prompt = self._extract_structured_prompt(state)
        
        # Call LLM
        response = await self._call_llm(structured_prompt)
        response = await self._retry_if_truncated(response, structured_prompt)

        # Log response metadata for debugging truncation issues
        stop_reason = response.get("stop_reason", "unknown")
        output_tokens = response.get("usage", {}).get("output_tokens", 0)
        content_length = len(response.get("content", ""))

        self.logger.info(
            f"[LLM_RESPONSE] stop_reason={stop_reason}, "
            f"output_tokens={output_tokens}, content_length={content_length} chars"
        )

        # Warn if response appears truncated
        if stop_reason not in ["stop", "STOP", "end_turn", "FinishReason.STOP"]:
            if self._is_truncated_stop_reason(stop_reason):
                self.logger.warning(
                    f"[LLM_TRUNCATED] Response may be incomplete! "
                    f"stop_reason={stop_reason}, expected one of: stop, STOP, end_turn, FinishReason.STOP"
                )
            else:
                self.logger.warning(
                    f"[LLM_NONSTOP] LLM returned non-stop finish reason: {stop_reason}"
                )

        # Extract and parse JSON from response
        raw_content = response.get("content", "")
        parsed_json = self._extract_json_from_response(raw_content)
        
        # Normalize to internal schema (severity/category enums, line_hint, etc.)
        normalized_data = self._normalize_to_internal_schema(parsed_json)
        
        # Validate with Pydantic
        raw_llm_output = self._validate_output(normalized_data)
        
        # Track token usage with validation
        usage = response.get("usage", {})
        if usage and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
        else:
            self.logger.warning("LLM response missing usage data")
            input_tokens = 0
            output_tokens = 0
        
        self._total_requests += 1
        
        if self._cost_tracker:
            self._cost_tracker.record_usage(input_tokens, output_tokens)
        
        self.logger.info(
            f"LLM generation complete: {len(raw_llm_output.findings)} findings, "
            f"{input_tokens} input tokens, {output_tokens} output tokens"
        )
        
        # Return state update
        return {
            "raw_llm_output": raw_llm_output.model_dump(),
            "llm_token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "model": response.get("model", "unknown"),
            }
        }

    def _get_required_state_keys(self) -> List[str]:
        return ["structured_prompt"]

    def _get_state_type_requirements(self) -> Dict[str, type]:
        # structured_prompt can be dict (from model_dump) or StructuredPrompt
        return {"structured_prompt": (dict, StructuredPrompt)}

    # ========================================================================
    # PROMPT EXTRACTION
    # ========================================================================

    def _extract_structured_prompt(self, state: Dict[str, Any]) -> StructuredPrompt:
        """Extract and validate structured prompt from state."""
        prompt_data = state.get("structured_prompt")
        
        if prompt_data is None:
            available_keys = list(state.keys())
            raise LLMGenerationError(
                f"structured_prompt is missing from state. Available keys: {available_keys}",
                provider="unknown"
            )
        
        # Handle both dict and StructuredPrompt
        if isinstance(prompt_data, dict):
            try:
                return StructuredPrompt.model_validate(prompt_data)
            except ValidationError as e:
                raise LLMGenerationError(
                    f"Invalid structured_prompt format: {e}",
                    provider="unknown",
                    cause=e
                )
        elif isinstance(prompt_data, StructuredPrompt):
            return prompt_data
        else:
            raise LLMGenerationError(
                f"Unexpected structured_prompt type: {type(prompt_data)}",
                provider="unknown"
            )

    # ========================================================================
    # LLM CALLING
    # ========================================================================

    async def _call_llm(self, prompt: StructuredPrompt) -> Dict[str, Any]:
        """
        Call LLM with the structured prompt.
        
        Args:
            prompt: StructuredPrompt with system_prompt and user_prompt
            
        Returns:
            LLM response dict with content, usage, model, stop_reason
        """
        try:
            response = await self.llm_client.generate_completion(
                prompt=prompt.user_prompt,
                system_prompt=prompt.system_prompt,
                max_tokens=self._get_max_tokens(prompt),
                temperature=0.1,  # Low temperature for consistent structured output
                response_mime_type="application/json",
            )
            return response
            
        except TimeoutError as e:
            raise LLMTimeoutError(
                f"LLM request timed out: {e}",
                timeout_seconds=self.timeout_seconds,
                provider=self.llm_client.provider_name if self._llm_client else "unknown"
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit errors
            if "rate" in error_msg and "limit" in error_msg:
                raise LLMRateLimitError(
                    f"LLM rate limit exceeded: {e}",
                    provider=self.llm_client.provider_name if self._llm_client else "unknown"
                )
            
            # Generic LLM error
            raise LLMGenerationError(
                f"LLM API call failed: {e}",
                provider=self.llm_client.provider_name if self._llm_client else "unknown",
                cause=e
            )

    def _get_max_tokens(self, prompt: StructuredPrompt) -> int:
        """Pick a safe max token budget for completions."""
        prompt_budget = max(prompt.estimated_max_completion_tokens, 1024)
        client_budget = getattr(self.llm_client, "max_tokens", prompt_budget)
        return min(prompt_budget, client_budget)

    def _is_truncated_stop_reason(self, stop_reason: Any) -> bool:
        """Check for stop reasons that indicate truncation."""
        normalized = str(stop_reason).lower()
        return normalized in {
            "max_tokens",
            "finishreason.max_tokens",
            "finish_reason_max_tokens",
            "stop_reason_max_tokens",
            "2",
        }

    async def _retry_if_truncated(
        self,
        response: Dict[str, Any],
        prompt: StructuredPrompt
    ) -> Dict[str, Any]:
        """Retry once with a higher token budget if the response was truncated."""
        stop_reason = response.get("stop_reason", "unknown")
        if not self._is_truncated_stop_reason(stop_reason):
            return response

        current_budget = self._get_max_tokens(prompt)
        client_budget = getattr(self.llm_client, "max_tokens", current_budget)
        if current_budget >= client_budget:
            return response

        retry_budget = min(int(current_budget * 1.5), client_budget)
        self.logger.warning(
            f"[LLM_TRUNCATED] Retrying with higher max_tokens={retry_budget} "
            f"(previous={current_budget})."
        )

        return await self.llm_client.generate_completion(
            prompt=prompt.user_prompt,
            system_prompt=prompt.system_prompt,
            max_tokens=retry_budget,
            temperature=0.1,
            response_mime_type="application/json",
        )

    # ========================================================================
    # JSON EXTRACTION
    # ========================================================================

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response using optimized strategy order.

        NEW ORDER (most reliable first):
        1. Direct JSON parse
        2. Extract from first { to last } (PROMOTED - most reliable)
        3. Extract from ```json ... ``` fenced block (DEMOTED but improved regex)
        4. Try ast.literal_eval for python-dict syntax
        5. Attempt JSON repair

        Args:
            content: Raw LLM response content

        Returns:
            Parsed JSON as dict

        Raises:
            LLMResponseParseError: If JSON cannot be extracted
        """
        if not content or not content.strip():
            raise LLMResponseParseError(
                "Empty response from LLM",
                raw_response=content,
                parse_error="Response is empty"
            )

        content = content.strip()
        strategy_log = []

        # Handle empty context response format
        # When context_items is empty, LLM may return explanatory text instead of JSON
        if "no code changes" in content.lower() or "no context items" in content.lower():
            self.logger.warning(
                "[LLM_JSON] LLM returned empty context response. Constructing valid JSON structure."
            )
            # Construct valid JSON structure for empty context case
            return {
                "findings": [],
                "summary": content[:500] if len(content) > 500 else content,
                "patterns": None,
                "recommendations": None
            }

        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(content)
            self.logger.info("[LLM_JSON] Strategy 1 succeeded: Direct parse")
            return result
        except json.JSONDecodeError as e:
            strategy_log.append(f"Strategy 1 (Direct parse) failed: {e}")
            self.logger.debug(f"[LLM_JSON] Strategy 1 failed: {e}")

        # Strategy 2: Extract from first { to last } (PROMOTED - most reliable)
        first_brace = content.find('{')
        last_brace = content.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_substring = content[first_brace:last_brace + 1]
            try:
                result = json.loads(json_substring)
                self.logger.info(f"[LLM_JSON] Strategy 2 succeeded: Brace extraction ({len(json_substring)} chars)")
                return result
            except json.JSONDecodeError as e:
                strategy_log.append(f"Strategy 2 (Brace extraction) failed: {e}")
                self.logger.debug(f"[LLM_JSON] Strategy 2 failed: {e}")
        else:
            strategy_log.append("Strategy 2 (Brace extraction) failed: No valid brace pair found")

        # Strategy 3: Extract from markdown fence (DEMOTED but improved regex)
        # Handles: ```json\n{...}\n```, ```json{...}```, ``` json\n{...}```
        json_block_match = re.search(
            r'```(?:json)?\s*(.*?)\s*```',  # Removed \n requirements
            content,
            re.DOTALL | re.IGNORECASE
        )
        if json_block_match:
            try:
                json_content = json_block_match.group(1).strip()
                result = json.loads(json_content)
                self.logger.info("[LLM_JSON] Strategy 3 succeeded: Markdown fence")
                return result
            except json.JSONDecodeError as e:
                strategy_log.append(f"Strategy 3 (Markdown fence) failed: {e}")
                self.logger.debug(f"[LLM_JSON] Strategy 3 failed: {e}")
        else:
            strategy_log.append("Strategy 3 (Markdown fence) failed: No markdown fence found")

        # Strategy 4: Try ast.literal_eval for python-dict syntax
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                dict_str = content[first_brace:last_brace + 1]
                result = ast.literal_eval(dict_str)
                if isinstance(result, dict):
                    self.logger.info("[LLM_JSON] Strategy 4 succeeded: ast.literal_eval")
                    return result
                else:
                    strategy_log.append(f"Strategy 4 (ast.literal_eval) failed: Result is {type(result).__name__}, not dict")
            except (ValueError, SyntaxError) as e:
                strategy_log.append(f"Strategy 4 (ast.literal_eval) failed: {e}")
                self.logger.debug(f"[LLM_JSON] Strategy 4 failed: {e}")
        else:
            strategy_log.append("Strategy 4 (ast.literal_eval) failed: No valid brace pair found")

        # Strategy 5: Try to repair common JSON issues
        repaired = self._attempt_json_repair(content)
        if repaired is not None:
            self.logger.info("[LLM_JSON] Strategy 5 succeeded: JSON repair")
            return repaired
        else:
            strategy_log.append("Strategy 5 (JSON repair) failed: Could not repair JSON")

        # All strategies failed - log comprehensive summary
        self.logger.error(
            f"[LLM_JSON] All strategies failed:\n" +
            "\n".join(f"  - {log}" for log in strategy_log) +
            f"\nResponse length: {len(content)} chars. First 1000 chars:\n{content[:1000]}"
        )

        raise LLMResponseParseError(
            "All 5 JSON extraction strategies failed",
            raw_response=content[:2000] if len(content) > 2000 else content,
            parse_error="; ".join(strategy_log)
        )

    def _attempt_json_repair(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair common JSON issues.
        
        Common issues:
        - Trailing commas
        - Single quotes instead of double quotes
        - Unquoted keys
        
        Args:
            content: Raw content that failed JSON parsing
            
        Returns:
            Parsed dict if repair successful, None otherwise
        """
        # Find potential JSON substring
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace == -1 or last_brace == -1:
            return None
        
        json_str = content[first_brace:last_brace + 1]
        
        # Try fixing trailing commas (,] or ,})
        repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        
        # Try replacing single quotes with double quotes (risky but sometimes works)
        # Only if no double quotes present
        if '"' not in json_str and "'" in json_str:
            repaired = json_str.replace("'", '"')
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        
        return None

    # ========================================================================
    # SCHEMA NORMALIZATION
    # ========================================================================

    def _normalize_to_internal_schema(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parsed JSON to match internal RawLLMReviewOutput schema.
        
        Normalizations:
        - For each finding:
          - line_in_hunk or line -> line_hint
          - Normalize severity/category to valid enums
          - Ensure evidence structure exists
        
        Args:
            parsed: Parsed JSON from LLM
            
        Returns:
            Normalized dict matching RawLLMReviewOutput schema
        """
        normalized = {}
        
        # Get findings list
        findings = parsed.get("findings") or []
        normalized["findings"] = [
            self._normalize_finding(f) for f in findings
        ]
        
        # Copy summary (required)
        normalized["summary"] = parsed.get("summary", "No summary provided")
        
        # Copy optional fields if present
        if "patterns" in parsed:
            normalized["patterns"] = parsed["patterns"]
        if "recommendations" in parsed:
            normalized["recommendations"] = parsed["recommendations"]
        
        return normalized

    def _normalize_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a single finding to match RawLLMFinding schema.
        
        Args:
            finding: Raw finding dict from LLM
            
        Returns:
            Normalized finding dict
        """
        normalized = {}
        
        # Required fields - copy directly
        normalized["title"] = finding.get("title", "Untitled Finding")
        normalized["message"] = finding.get("message") or finding.get("description", "")
        normalized["severity"] = self._normalize_severity(finding.get("severity", "medium"))
        normalized["category"] = self._normalize_category(finding.get("category", "style"))
        normalized["file_path"] = finding.get("file_path", "")
        normalized["suggested_fix"] = finding.get("suggested_fix", "")
        normalized["confidence"] = self._normalize_confidence(finding.get("confidence", 0.5))
        
        # Copy hunk_id if present (LLM's suggestion, validated during anchoring)
        if "hunk_id" in finding:
            normalized["hunk_id"] = finding["hunk_id"]
        
        # Map line_in_hunk or line -> line_hint with explicit None checks (to handle 0 correctly)
        line_hint = finding.get("line_in_hunk")
        if line_hint is None:
            line_hint = finding.get("line")
        if line_hint is None:
            line_hint = finding.get("line_hint")
        
        if line_hint is not None:
            try:
                normalized["line_hint"] = int(line_hint)
            except (ValueError, TypeError):
                normalized["line_hint"] = None
        
        # Normalize evidence
        evidence = finding.get("evidence")
        if evidence and isinstance(evidence, dict):
            normalized["evidence"] = self._normalize_evidence(evidence)
        
        # Copy optional arrays
        if "related_symbols" in finding:
            normalized["related_symbols"] = finding["related_symbols"][:10]  # Limit
        if "code_examples" in finding:
            normalized["code_examples"] = finding["code_examples"][:3]  # Limit
        
        return normalized

    def _normalize_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evidence citation with type validation."""
        snippet_range = evidence.get("snippet_line_range", [])
        if not isinstance(snippet_range, list):
            snippet_range = []
        
        return {
            "context_item_id": evidence.get("context_item_id"),
            "snippet_line_range": snippet_range,
            "quote": evidence.get("quote"),
        }

    def _normalize_severity(self, severity: Any) -> str:
        """Normalize severity to valid enum value."""
        if not severity:
            return "medium"
        
        severity_str = str(severity).lower().strip()
        valid_severities = {"blocker", "high", "medium", "low", "nit"}
        
        if severity_str in valid_severities:
            return severity_str
        
        # Try to map common variations
        severity_map = {
            "critical": "blocker",
            "error": "high",
            "warning": "medium",
            "info": "low",
            "suggestion": "nit",
            "minor": "nit",
        }
        
        return severity_map.get(severity_str, "medium")

    def _normalize_category(self, category: Any) -> str:
        """Normalize category to valid enum value."""
        if not category:
            return "style"
        
        category_str = str(category).lower().strip()
        valid_categories = {
            "bug", "security", "performance", "style", "design",
            "docs", "observability", "maintainability"
        }
        
        if category_str in valid_categories:
            return category_str
        
        # Try to map common variations
        category_map = {
            "error": "bug",
            "vulnerability": "security",
            "efficiency": "performance",
            "formatting": "style",
            "architecture": "design",
            "documentation": "docs",
            "logging": "observability",
            "readability": "maintainability",
            "code_quality": "maintainability",
        }
        
        return category_map.get(category_str, "style")

    def _normalize_confidence(self, confidence: Any) -> float:
        """Normalize confidence to float in [0.0, 1.0]."""
        if confidence is None:
            return 0.5
        
        try:
            value = float(confidence)
            return max(0.0, min(1.0, value))
        except (ValueError, TypeError):
            return 0.5

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_output(self, normalized: Dict[str, Any]) -> RawLLMReviewOutput:
        """
        Validate normalized output against Pydantic schema.
        
        Args:
            normalized: Normalized dict
            
        Returns:
            Validated RawLLMReviewOutput
            
        Raises:
            LLMResponseParseError: If validation fails
        """
        try:
            return RawLLMReviewOutput.model_validate(normalized)
        except ValidationError as e:
            # Try to salvage what we can
            salvaged = self._salvage_partial_output(normalized, e)
            if salvaged:
                return salvaged
            
            raise LLMResponseParseError(
                f"LLM output failed schema validation: {e}",
                raw_response=json.dumps(normalized)[:500],
                parse_error=str(e)
            )

    def _fix_finding_validation_issues(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix common validation issues in findings before validation.
        
        Fixes:
        - Truncate evidence.quote if > 500 chars
        - Ensure required fields have defaults
        """
        fixed = finding.copy()
        
        # Fix evidence quote length
        if "evidence" in fixed and isinstance(fixed["evidence"], dict):
            evidence = fixed["evidence"].copy()
            quote = evidence.get("quote")
            if quote and len(quote) > 500:
                self.logger.debug(
                    f"Truncating evidence quote from {len(quote)} to 500 chars for finding: "
                    f"{finding.get('title', 'unknown')}"
                )
                evidence["quote"] = quote[:497] + "..."
                fixed["evidence"] = evidence
        
        return fixed

    def _salvage_partial_output(
        self,
        normalized: Dict[str, Any],
        error: ValidationError
    ) -> Optional[RawLLMReviewOutput]:
        """
        Attempt to salvage partial output when validation fails.
        
        Strategy: Remove invalid findings and try again.
        """
        self.logger.warning(f"Attempting to salvage partial output after validation error: {error}")
        
        findings = normalized.get("findings", [])
        valid_findings = []
        
        for finding in findings:
            try:
                # Try to fix common validation issues before dropping
                fixed_finding = self._fix_finding_validation_issues(finding)
                
                # Try to validate each finding individually
                RawLLMFinding.model_validate(fixed_finding)
                valid_findings.append(fixed_finding)
            except ValidationError as ve:
                self.logger.warning(
                    f"Dropping invalid finding: {finding.get('title', 'unknown')} - {ve}"
                )
        
        if not valid_findings and not normalized.get("summary"):
            return None
        
        # Rebuild with valid findings only
        salvaged_data = {
            "findings": valid_findings,
            "summary": normalized.get("summary", "Review completed with partial results."),
            "patterns": normalized.get("patterns", []),
            "recommendations": normalized.get("recommendations", []),
        }
        
        try:
            return RawLLMReviewOutput.model_validate(salvaged_data)
        except ValidationError:
            return None

    # ========================================================================
    # GRACEFUL DEGRADATION
    # ========================================================================

    async def _attempt_graceful_degradation(
        self,
        state: Dict[str, Any],
        error: Exception,
        metrics: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Provide fallback when LLM generation fails completely.
        
        Returns an empty review output so the workflow can continue.
        """
        self.logger.warning(f"Using graceful degradation for LLM generation: {error}")
        
        try:
            # Return empty but valid output
            fallback_output = RawLLMReviewOutput(
                findings=[],
                summary=f"Review generation failed: {str(error)[:100]}. No findings generated.",
                patterns=[],
                recommendations=[]
            )
            
            return {
                "raw_llm_output": fallback_output.model_dump(),
                "llm_token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "model": "fallback",
                },
                "llm_generation_error": str(error)
            }
            
        except Exception as fallback_error:
            self.logger.error(f"Graceful degradation also failed: {fallback_error}")
            return None

    # ========================================================================
    # METRICS AND HEALTH
    # ========================================================================

    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM-specific metrics."""
        base_metrics = self.get_performance_metrics()
        
        llm_metrics = {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_llm_requests": self._total_requests,
            "avg_input_tokens": (
                self._total_input_tokens / self._total_requests
                if self._total_requests > 0 else 0
            ),
            "avg_output_tokens": (
                self._total_output_tokens / self._total_requests
                if self._total_requests > 0 else 0
            ),
        }
        
        if self._cost_tracker:
            llm_metrics["cost_stats"] = self._cost_tracker.get_stats()
        
        if self._llm_client:
            llm_metrics["provider"] = self._llm_client.provider_name
            llm_metrics["model"] = self._llm_client.model
        
        base_metrics["llm_metrics"] = llm_metrics
        return base_metrics
