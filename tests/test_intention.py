"""Tests for IntentionDocumentManager, IntentionRefiner, and _extract_json."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from imaro.intention.document import IntentionDocumentManager
from imaro.intention.refiner import IntentionRefiner, _extract_json
from imaro.models.schemas import (
    IntentionDocument,
    LLMResponse,
    RefinementQuestion,
    RefinementRound,
)


# ── IntentionDocumentManager ────────────────────────────────────────────────


class TestIntentionDocumentManager:
    def test_save_load_roundtrip(self, sample_intention, tmp_project):
        mgr = IntentionDocumentManager()
        mgr.save(sample_intention, tmp_project)
        loaded = mgr.load(tmp_project)
        assert loaded == sample_intention

    def test_save_creates_imaro_dir(self, sample_intention, tmp_path):
        """save() creates .imaro/ if it doesn't exist."""
        mgr = IntentionDocumentManager()
        mgr.save(sample_intention, tmp_path)
        assert (tmp_path / ".imaro" / "intention.json").exists()

    def test_to_prompt_context_contains_all_fields(self, sample_intention):
        ctx = IntentionDocumentManager.to_prompt_context(sample_intention)
        assert "PROJECT INTENTION" in ctx
        assert sample_intention.purpose in ctx
        assert sample_intention.target_users in ctx
        assert sample_intention.core_value in ctx
        for req in sample_intention.functional_requirements:
            assert req in ctx
        for nfr in sample_intention.non_functional_requirements:
            assert nfr in ctx
        for oos in sample_intention.out_of_scope:
            assert oos in ctx
        for c in sample_intention.constraints:
            assert c in ctx
        for s in sample_intention.success_criteria:
            assert s in ctx

    def test_to_milestone_context_includes_milestone_info(
        self, sample_intention, sample_milestone
    ):
        ctx = IntentionDocumentManager.to_milestone_context(
            sample_intention, sample_milestone
        )
        assert sample_milestone.name in ctx
        assert sample_milestone.description in ctx
        assert "milestone context" in ctx
        for s in sample_milestone.scope:
            assert s in ctx
        for c in sample_milestone.acceptance_criteria:
            assert c in ctx
        assert "Serves Purpose: True" in ctx


# ── _extract_json ───────────────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        data = {"status": "ready", "value": 42}
        result = _extract_json(json.dumps(data))
        assert result == data

    def test_json_in_code_fence_with_lang(self):
        data = {"status": "questions"}
        text = f"```json\n{json.dumps(data)}\n```"
        result = _extract_json(text)
        assert result == data

    def test_json_in_code_fence_no_lang(self):
        data = {"key": "value"}
        text = f"```\n{json.dumps(data)}\n```"
        result = _extract_json(text)
        assert result == data

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _extract_json("this is not json at all")

    def test_json_with_surrounding_whitespace(self):
        data = {"foo": "bar"}
        text = f"  \n  {json.dumps(data)}  \n  "
        result = _extract_json(text)
        assert result == data


# ── IntentionRefiner ────────────────────────────────────────────────────────


class TestIntentionRefiner:
    @pytest.mark.asyncio
    async def test_refine_returns_document_when_ready(self, mock_provider):
        doc_data = {
            "status": "ready",
            "document": {
                "purpose": "Build a CLI tool",
                "target_users": "Developers",
                "core_value": "Speed",
                "functional_requirements": ["req1"],
                "non_functional_requirements": [],
                "out_of_scope": [],
                "constraints": [],
                "success_criteria": ["crit1"],
            },
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(doc_data), model="mock"
        )

        refiner = IntentionRefiner()
        result, is_ready = await refiner.refine("Build a CLI", mock_provider)

        assert is_ready is True
        assert isinstance(result, IntentionDocument)
        assert result.purpose == "Build a CLI tool"
        assert result.raw_input == "Build a CLI"

    @pytest.mark.asyncio
    async def test_refine_returns_questions_when_needed(self, mock_provider):
        questions_data = {
            "status": "questions",
            "questions": [
                {"question": "What language?", "context": "Need to know tech stack"},
                {"question": "What DB?", "context": "Storage choice"},
            ],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(questions_data), model="mock"
        )

        refiner = IntentionRefiner()
        result, is_ready = await refiner.refine("Build something", mock_provider)

        assert is_ready is False
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(q, RefinementQuestion) for q in result)
        assert result[0].question == "What language?"

    @pytest.mark.asyncio
    async def test_refine_with_answers_includes_history(self, mock_provider):
        doc_data = {
            "status": "ready",
            "document": {
                "purpose": "A tool",
                "target_users": "Users",
                "core_value": "Value",
                "functional_requirements": [],
                "non_functional_requirements": [],
                "out_of_scope": [],
                "constraints": [],
                "success_criteria": [],
            },
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(doc_data), model="mock"
        )

        history = [
            RefinementRound(
                questions=[RefinementQuestion(question="What language?")],
                answers=["Python"],
            )
        ]

        refiner = IntentionRefiner()
        result, is_ready = await refiner.refine_with_answers(
            "Build something", history, mock_provider
        )

        assert is_ready is True
        # Verify the prompt included history
        call_args = mock_provider.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "Previous Q&A Rounds" in prompt
        assert "What language?" in prompt
        assert "Python" in prompt

    @pytest.mark.asyncio
    async def test_refine_parses_json_in_code_fences(self, mock_provider):
        doc_data = {
            "status": "ready",
            "document": {
                "purpose": "Fenced JSON",
                "target_users": "Test",
                "core_value": "Test",
                "functional_requirements": [],
                "non_functional_requirements": [],
                "out_of_scope": [],
                "constraints": [],
                "success_criteria": [],
            },
        }
        fenced = f"```json\n{json.dumps(doc_data)}\n```"
        mock_provider.generate.return_value = LLMResponse(
            content=fenced, model="mock"
        )

        refiner = IntentionRefiner()
        result, is_ready = await refiner.refine("test", mock_provider)
        assert is_ready is True
        assert isinstance(result, IntentionDocument)
        assert result.purpose == "Fenced JSON"

    @pytest.mark.asyncio
    async def test_refine_retries_on_invalid_json(self, mock_provider):
        """First call returns bad JSON, second returns good JSON."""
        doc_data = {
            "status": "ready",
            "document": {
                "purpose": "Retry success",
                "target_users": "T",
                "core_value": "V",
                "functional_requirements": [],
                "non_functional_requirements": [],
                "out_of_scope": [],
                "constraints": [],
                "success_criteria": [],
            },
        }
        mock_provider.generate.side_effect = [
            LLMResponse(content="not json {{{", model="mock"),
            LLMResponse(content=json.dumps(doc_data), model="mock"),
        ]

        refiner = IntentionRefiner()
        result, is_ready = await refiner.refine("test", mock_provider)

        assert is_ready is True
        assert isinstance(result, IntentionDocument)
        assert result.purpose == "Retry success"
        assert mock_provider.generate.call_count == 2
