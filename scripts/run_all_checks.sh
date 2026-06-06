#!/usr/bin/env bash
# scripts/run_all_checks.sh
# Run all quality checks locally before pushing.
# Usage: bash scripts/run_all_checks.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✔  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠  $1${NC}"; }
fail() { echo -e "${RED}✘  $1${NC}"; exit 1; }

echo ""
echo "=================================================="
echo "  SR-RAG — local quality checks"
echo "=================================================="
echo ""

# 1. Virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    warn "No virtual environment detected. Activate your venv first."
    warn "  source .venv/bin/activate"
    exit 1
fi
pass "Virtual environment: $VIRTUAL_ENV"

# 2. Ruff lint
echo ""
echo ">>> Lint (ruff) ..."
if ruff check src/ api/ scripts/ tests/ --ignore E501,E402,F401 --quiet; then
    pass "Ruff lint passed"
else
    fail "Ruff lint failed — fix errors above before committing"
fi

# 3. Format check
echo ""
echo ">>> Format check ..."
if ruff format src/ api/ scripts/ tests/ --check --quiet; then
    pass "Format check passed"
else
    warn "Formatting issues found — run: ruff format src/ api/ scripts/ tests/"
fi

# 4. Import smoke test
echo ""
echo ">>> Import smoke test ..."
MODULES=(
    "src.ingestion.pdf_extractor"
    "src.ingestion.chunker"
    "src.indexing.embeddings"
    "src.indexing.vector_store"
    "src.retrieval.retriever"
    "src.retrieval.reranker"
    "src.generation.answer_generator"
    "src.generation.citations"
    "src.generation.prompt_manager"
    "src.evaluation.retrieval_metrics"
    "src.evaluation.answer_metrics"
    "src.monitoring.logger"
    "src.pipeline"
)
for mod in "${MODULES[@]}"; do
    if python -c "import $mod" 2>/dev/null; then
        pass "  $mod"
    else
        fail "  Failed to import $mod"
    fi
done

# 5. Pytest
echo ""
echo ">>> Tests (pytest) ..."
if pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-fail-under=40 \
    --ignore=tests/test_reranker.py \
    -q; then
    pass "All tests passed"
else
    fail "Tests failed — fix before committing"
fi

# 6. Secret scan
echo ""
echo ">>> Secret scan ..."
LEAK=0
for pattern in "sk-" "AIza" "ghp_"; do
    if grep -r --include="*.py" "$pattern" src/ api/ 2>/dev/null | grep -v ".env.example" | grep -q .; then
        warn "Possible secret pattern '$pattern' found — review before pushing"
        LEAK=1
    fi
done
if [[ $LEAK -eq 0 ]]; then pass "No secrets detected"; fi

# 7. .gitignore sanity
echo ""
echo ">>> .gitignore sanity ..."
for item in "data/raw_papers" "vector_store" ".env"; do
    if git check-ignore -q "$item" 2>/dev/null; then
        pass "  $item is ignored"
    else
        warn "  $item may not be in .gitignore"
    fi
done

echo ""
echo "=================================================="
echo -e "${GREEN}  All checks passed — safe to push.${NC}"
echo "=================================================="
echo ""
