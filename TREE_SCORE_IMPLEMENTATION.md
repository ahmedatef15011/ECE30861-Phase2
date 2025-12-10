# Dynamic Tree Score Implementation

## Overview
The tree score metric now uses LLM-based dependency analysis to track artifact lineage and automatically calculates scores as the average of linked artifacts' net scores. When dependencies are uploaded, tree scores cascade update across all connected artifacts.

## Architecture

### 1. LLM Dependency Analysis
**File**: `src/llm/analyzer.py`

- Uses AWS Bedrock Claude 3 Haiku to analyze `config.json` and README content
- Extracts parent models, datasets, and code repositories
- Returns structured JSON with dependency information
- Low temperature (0.1) ensures factual extraction

**Example Response**:
```json
{
  "parent_models": [
    {
      "id": "meta-llama/Llama-2-7b-hf",
      "relationship": "fine_tuned_from"
    }
  ],
  "datasets": ["squad_v2"],
  "code_repositories": [],
  "analysis_metadata": {
    "confidence": "high",
    "model_used": "anthropic.claude-3-haiku-20240307-v1:0"
  }
}
```

### 2. Lineage Storage
**File**: `src/database/models.py`

Added to `Package` model:
- `lineage_metadata` (JSON column): Caches LLM analysis results to avoid repeated API calls
- Stores parent models, datasets, code repos, and analysis metadata

**Migration**: `migrate_add_lineage_metadata.py`
- Adds JSON column to existing database
- Compatible with SQLite (JSON) and PostgreSQL (JSONB)
- Idempotent - checks if column exists before adding

### 3. Tree Score Calculation
**File**: `src/metrics/treescore.py`

**Logic**:
1. Query `PackageLineage` table for parent artifacts
2. Fetch net scores for all parent packages
3. Calculate average of parent net scores
4. **Fallback**: If no parents found, return artifact's own `net_score`

**Parameters**:
- `current_net_score`: Used as fallback when no parents exist
- `db_session`: For querying parent scores from database
- `score_fetcher`: Function to retrieve scores (legacy support)

### 4. Cascade Score Updates
**File**: `src/database/crud.py` - `update_dependent_tree_scores()`

When an artifact is uploaded/scored:
1. Find all children via `PackageLineage` table
2. For each child:
   - Recalculate tree score using updated parent scores
   - Recursively update their children (max depth: 3 levels)
   - Track visited nodes to prevent cycles
3. Return count of updated packages

**Safety Features**:
- Max recursion depth of 3 to prevent long request times
- Cycle detection using visited set
- Atomic database updates with rollback on failure

### 5. Integration Point
**File**: `src/api/main.py` - `/artifact` POST endpoint

**After package ingestion** (lines 1344-1450):
1. **Extract lineage** using LLM analyzer
2. **Store metadata** in `package.lineage_metadata`
3. **Create relationships** in `PackageLineage` table
4. **Trigger cascade** with `update_dependent_tree_scores()`

**Workflow**:
```
Upload artifact → Score → Store scores →
  ↓
Extract dependencies (LLM) →
  ↓
Store lineage metadata →
  ↓
Create PackageLineage entries →
  ↓
Cascade update tree scores for children
```

## Database Schema Changes

### PackageLineage Table
Existing relationship table:
- `parent_package_id` → Links to parent artifact
- `child_package_id` → Links to child artifact  
- `relationship_type` → Type of dependency (fine_tuned_from, trained_on, etc.)

### Package Table
New column:
- `lineage_metadata` (JSON/JSONB) → Caches LLM analysis results

## Configuration

### LLM Settings
**Model**: `anthropic.claude-3-haiku-20240307-v1:0`
- Temperature: 0.1 (factual extraction)
- Max tokens: 2000
- Cost-effective for dependency analysis

### Performance Settings
- **Max cascade depth**: 3 levels (prevents deep recursion)
- **Cycle detection**: Yes (prevents infinite loops)
- **LLM caching**: Yes (via `lineage_metadata` column)

## Usage Example

### 1. Upload Parent Model
```bash
POST /artifact
{
  "url": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
  "type": "model"
}
```
**Tree Score**: = net_score (no parents)

### 2. Upload Fine-tuned Model
```bash
POST /artifact
{
  "url": "https://huggingface.co/my-org/llama-2-7b-finetuned",
  "type": "model"
}
```

**LLM Analysis**:
- Detects `base_model: meta-llama/Llama-2-7b-hf` in config.json
- Creates lineage relationship

**Tree Score**: = average([Llama-2-7b-hf.net_score])

### 3. Upload Another Fine-tune
```bash
POST /artifact
{
  "url": "https://huggingface.co/another-org/llama-2-7b-qa",
  "type": "model"
}
```

**Cascade Update**:
- Llama-2-7b-hf tree score updates to average of both children
- Both fine-tunes get updated tree scores

## Monitoring

### Logs
```
🌳 Extracting artifact lineage...
   ✓ Linked parent: Llama-2-7b-hf (fine_tuned_from)
✅ Lineage extracted: 1 parent(s)
🔄 Updating dependent tree scores...
✅ Updated tree scores for 2 dependent artifact(s)
```

### Error Handling
- LLM failures → Tree score falls back to net_score
- Cascade failures → Logged as warning, doesn't block upload
- Database errors → Transaction rollback

## Testing

Run migration:
```bash
python migrate_add_lineage_metadata.py
```

Test tree score calculation:
```python
from src.metrics.treescore import TreeScore
from src.database.connection import SessionLocal

db = SessionLocal()
calculator = TreeScore()

# With parents
score = calculator.calculate(
    db_session=db,
    current_net_score=0.85
)

# Without parents (fallback)
score_fallback = calculator.calculate(
    db_session=db,
    current_net_score=0.75
)
```

## Performance Considerations

### LLM API Calls
- **Cached**: Results stored in `lineage_metadata`
- **Re-analysis**: Only if metadata is NULL or outdated
- **Cost**: ~$0.001 per analysis (Haiku pricing)

### Cascade Updates
- **Synchronous**: Happens during upload request
- **Depth limit**: Max 3 levels (prevents long requests)
- **Typical time**: <100ms for small dependency graphs

### Database Queries
- **Indexed**: `PackageLineage` has FK indexes on both IDs
- **Batched**: Cascade updates use single transaction
- **Scalable**: Handles 100s of artifacts efficiently

## Future Enhancements

1. **Async Processing**: Move cascade updates to background queue (SQS)
2. **Confidence Scores**: Use LLM confidence for weighted averages
3. **Version Tracking**: Track which version of parent was used
4. **Circular Dependencies**: Better handling of bidirectional relationships
5. **Manual Override**: Allow users to add/remove lineage links

## Troubleshooting

### Tree score not updating
- Check `lineage_metadata` column exists (run migration)
- Verify parent packages exist in database
- Check logs for LLM analysis errors

### Cascade not triggering
- Confirm `update_dependent_tree_scores()` called after score storage
- Check for cycle detection warnings in logs
- Verify max_depth not exceeded

### LLM extraction failing
- Ensure AWS Bedrock credentials configured
- Check Claude 3 Haiku model access in region
- Verify config.json and README content fetched correctly
