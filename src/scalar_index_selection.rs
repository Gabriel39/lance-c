// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded runtime BTree cost comparison. Estimates affect only acceleration;
//! the caller always retains its explicit fragment domain and complete filter.

use std::{borrow::Cow, cmp::Ordering, ops::Bound};

use arrow_array::{Array, RecordBatch, UInt32Array};
use datafusion::common::ScalarValue;
use lance::{
    Dataset,
    dataset::{index::LanceIndexStoreExt, scanner::ExecutionSummaryCounts},
};
use lance_core::{
    Error,
    cache::CacheKey,
    deepsize::{Context, DeepSizeOf},
};
use lance_index::scalar::{
    IndexStore, SargableQuery,
    btree::{BTREE_LOOKUP_NAME, DEFAULT_BTREE_BATCH_SIZE},
    expression::ScalarIndexExpr,
    lance_format::LanceIndexStore,
};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::scalar_segment::scoped_expression;

const MAX_CANDIDATES: usize = 8;
const MAX_LOOKUP_BYTES: u64 = 8 * 1024 * 1024;
const MAX_LOOKUP_PAGES: usize = 65_536;
const MAX_IN_VALUES: usize = 256;
const MAX_ESTIMATE_COMPARISONS: usize = 1_000_000;

struct Lookup {
    batch: RecordBatch,
    rows_per_page: u64,
}

impl DeepSizeOf for Lookup {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

struct LookupKey(String);
impl CacheKey for LookupKey {
    type ValueType = Lookup;
    fn key(&self) -> Cow<'_, str> {
        self.0.as_str().into()
    }
    fn type_name() -> &'static str {
        "ScalarSegmentBTreeCostLookup"
    }
}

fn is_btree_expression(expr: &ScalarIndexExpr) -> bool {
    match expr {
        ScalarIndexExpr::Query(search) => search.index_type.eq_ignore_ascii_case("btree"),
        ScalarIndexExpr::And(a, b) | ScalarIndexExpr::Or(a, b) => {
            is_btree_expression(a) && is_btree_expression(b)
        }
        ScalarIndexExpr::Not(_) => false,
    }
}

fn overlaps(
    min: &ScalarValue,
    max: &ScalarValue,
    query: &SargableQuery,
    nulls: u32,
) -> Option<bool> {
    let contains = |value: &ScalarValue| -> Option<bool> {
        if value.is_null() {
            return Some(nulls != 0);
        }
        if min.is_null() || max.is_null() {
            return Some(false);
        }
        Some(
            min.partial_cmp(value)? != Ordering::Greater
                && max.partial_cmp(value)? != Ordering::Less,
        )
    };
    match query {
        SargableQuery::Equals(value) => contains(value),
        SargableQuery::IsNull() => Some(nulls != 0),
        SargableQuery::IsIn(values) if values.len() <= MAX_IN_VALUES => {
            for value in values {
                if contains(value)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        SargableQuery::Range(start, end) => {
            if min.is_null() || max.is_null() {
                return Some(false);
            }
            let above_lower = match start {
                Bound::Included(v) if !v.is_null() => max.partial_cmp(v)? != Ordering::Less,
                Bound::Excluded(v) if !v.is_null() => max.partial_cmp(v)? == Ordering::Greater,
                Bound::Unbounded => true,
                _ => return None,
            };
            let below_upper = match end {
                Bound::Included(v) if !v.is_null() => min.partial_cmp(v)? != Ordering::Greater,
                Bound::Excluded(v) if !v.is_null() => min.partial_cmp(v)? == Ordering::Less,
                Bound::Unbounded => true,
                _ => return None,
            };
            Some(above_lower && below_upper)
        }
        _ => None,
    }
}

fn matches_page(
    expr: &ScalarIndexExpr,
    min: &ScalarValue,
    max: &ScalarValue,
    nulls: u32,
) -> Option<bool> {
    match expr {
        ScalarIndexExpr::Query(search) => overlaps(
            min,
            max,
            search.query.as_any().downcast_ref::<SargableQuery>()?,
            nulls,
        ),
        ScalarIndexExpr::And(a, b) => {
            Some(matches_page(a, min, max, nulls)? && matches_page(b, min, max, nulls)?)
        }
        ScalarIndexExpr::Or(a, b) => {
            Some(matches_page(a, min, max, nulls)? || matches_page(b, min, max, nulls)?)
        }
        // A page's min/max cannot determine which rows survive a complement.
        ScalarIndexExpr::Not(_) => None,
    }
}

fn comparisons_per_page(expr: &ScalarIndexExpr) -> Option<usize> {
    match expr {
        ScalarIndexExpr::Query(search) => {
            match search.query.as_any().downcast_ref::<SargableQuery>()? {
                SargableQuery::IsIn(values) if values.len() <= MAX_IN_VALUES => {
                    Some(values.len().max(1))
                }
                SargableQuery::Equals(_) | SargableQuery::Range(_, _) | SargableQuery::IsNull() => {
                    Some(1)
                }
                _ => None,
            }
        }
        ScalarIndexExpr::And(a, b) | ScalarIndexExpr::Or(a, b) => {
            comparisons_per_page(a)?.checked_add(comparisons_per_page(b)?)
        }
        ScalarIndexExpr::Not(_) => None,
    }
}

fn estimate_pages(batch: &RecordBatch, expr: &ScalarIndexExpr) -> Option<usize> {
    if batch.num_columns() < 3 || batch.num_rows() > MAX_LOOKUP_PAGES {
        return None;
    }
    // Node and byte limits alone still allow pages * IN-values * leaves work.
    if batch.num_rows().checked_mul(comparisons_per_page(expr)?)? > MAX_ESTIMATE_COMPARISONS {
        return None;
    }
    let nulls = batch.column(2).as_any().downcast_ref::<UInt32Array>()?;
    let mut pages = 0;
    for row in 0..batch.num_rows() {
        let min = ScalarValue::try_from_array(batch.column(0), row).ok()?;
        let max = ScalarValue::try_from_array(batch.column(1), row).ok()?;
        pages += usize::from(matches_page(expr, &min, &max, nulls.value(row))?);
    }
    Some(pages)
}

async fn estimate_cost(
    dataset: &Dataset,
    index: &IndexMetadata,
    expr: &ScalarIndexExpr,
    remaining_bytes: &mut u64,
) -> Option<(u64, usize)> {
    if !is_btree_expression(expr) {
        return None;
    }
    // Inspect immutable manifest sizes before opening a file. Older/partition-only
    // layouts without this root are deliberately left to the existing search path.
    let size = index
        .files
        .as_ref()?
        .iter()
        .find(|f| f.path == BTREE_LOOKUP_NAME)?
        .size_bytes;
    if size > *remaining_bytes {
        return None;
    }
    *remaining_bytes -= size;
    let key = LookupKey(format!("{}/{}", dataset.uri(), index.uuid));
    let session = dataset.session();
    let lookup = session
        .file_metadata_cache()
        .get_or_insert_with_key(key, || async {
            let store = LanceIndexStore::from_dataset_for_existing(dataset, index).await?;
            let reader = store.open_index_file(BTREE_LOOKUP_NAME).await?;
            if reader.num_rows() > MAX_LOOKUP_PAGES {
                return Err(Error::not_supported(
                    "BTree cost lookup exceeds the page budget",
                ));
            }
            let rows_per_page = reader
                .schema()
                .metadata
                .get("batch_size")
                .map(|s| s.parse::<u64>())
                .transpose()
                .map_err(|_| Error::not_supported("invalid BTree page size"))?
                .unwrap_or(DEFAULT_BTREE_BATCH_SIZE);
            if rows_per_page == 0 {
                return Err(Error::not_supported("zero BTree page size"));
            }
            let batch = reader.read_range(0..reader.num_rows(), None).await?;
            Ok(Lookup {
                batch,
                rows_per_page,
            })
        })
        .await
        .ok()?;
    let pages = estimate_pages(&lookup.batch, expr)?;
    Some((lookup.rows_per_page.saturating_mul(pages as u64), pages))
}

pub(crate) struct ReadDomain<'a> {
    pub fragment_ids: &'a [u64],
    pub physical_rows: u64,
}

/// None selects a scoped full scan. Unknown estimates preserve the grouping index.
pub(crate) async fn choose(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    grouping: &IndexMetadata,
    grouping_expr: ScalarIndexExpr,
    index_query: &ScalarIndexExpr,
    domain: ReadDomain<'_>,
    stats: &mut ExecutionSummaryCounts,
) -> Option<(Uuid, ScalarIndexExpr)> {
    if !is_btree_expression(&grouping_expr) {
        return Some((grouping.uuid, grouping_expr));
    }
    let mut remaining_bytes = MAX_LOOKUP_BYTES;
    stats
        .all_counts
        .insert("scalar_index_auto_candidates".into(), 1);
    let Some((mut cost, mut pages)) =
        estimate_cost(dataset, grouping, &grouping_expr, &mut remaining_bytes).await
    else {
        return Some((grouping.uuid, grouping_expr));
    };
    let mut selected = (grouping.uuid, grouping_expr);
    let mut candidates = 1;
    let mut alternatives: Vec<_> = indices.iter().filter(|i| i.uuid != grouping.uuid).collect();
    alternatives.sort_by(|a, b| (&a.name, a.uuid).cmp(&(&b.name, b.uuid)));
    for candidate in alternatives {
        if candidates >= MAX_CANDIDATES {
            break;
        }
        let Some(field_id) = candidate.keyed_field() else {
            continue;
        };
        let Some(field) = dataset.schema().fields.iter().find(|f| f.id == field_id) else {
            continue;
        };
        let Some(coverage) = &candidate.fragment_bitmap else {
            continue;
        };
        if domain
            .fragment_ids
            .iter()
            .any(|id| u32::try_from(*id).map_or(true, |id| !coverage.contains(id)))
        {
            continue;
        }
        let Ok(Some(expr)) = scoped_expression(
            index_query,
            &candidate.name,
            &field.name,
            false,
            0,
            &mut 128,
        ) else {
            continue;
        };
        if !is_btree_expression(&expr) {
            continue;
        }
        candidates += 1;
        if let Some((candidate_cost, candidate_pages)) =
            estimate_cost(dataset, candidate, &expr, &mut remaining_bytes).await
            && candidate_cost < cost
        {
            cost = candidate_cost;
            pages = candidate_pages;
            selected = (candidate.uuid, expr);
        }
    }
    stats
        .all_counts
        .insert("scalar_index_auto_candidates".into(), candidates);
    stats
        .all_counts
        .insert("scalar_index_auto_estimated_pages".into(), pages);
    stats.all_counts.insert(
        "scalar_index_auto_estimated_rows".into(),
        cost.min(usize::MAX as u64) as usize,
    );
    if cost >= domain.physical_rows {
        stats
            .all_counts
            .insert("scalar_index_auto_full_scans".into(), 1);
        return None;
    }
    stats.all_counts.insert(
        "scalar_index_auto_changed".into(),
        usize::from(selected.0 != grouping.uuid),
    );
    Some(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use lance_index::scalar::{SargableQuery, expression::ScalarIndexSearch};
    use std::sync::Arc;

    fn lookup() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("min", DataType::Int32, true),
                Field::new("max", DataType::Int32, true),
                Field::new("null_count", DataType::UInt32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(0), Some(10), Some(20)])),
                Arc::new(Int32Array::from(vec![None, Some(9), Some(19), Some(29)])),
                Arc::new(UInt32Array::from(vec![4, 0, 1, 0])),
            ],
        )
        .unwrap()
    }
    fn eq(value: i32) -> ScalarIndexExpr {
        ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "key".into(),
            index_name: "key_idx".into(),
            index_type: "BTree".into(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::Int32(Some(value)))),
            needs_recheck: false,
            fragment_bitmap: None,
        })
    }
    #[test]
    fn runtime_lookup_distinguishes_narrow_and_broad_predicates() {
        assert_eq!(estimate_pages(&lookup(), &eq(15)), Some(1));
        assert_eq!(estimate_pages(&lookup(), &eq(99)), Some(0));
        let broad = ScalarIndexExpr::Or(Box::new(eq(5)), Box::new(eq(15)));
        assert_eq!(estimate_pages(&lookup(), &broad), Some(2));
    }
    #[test]
    fn intersect_page_sets_without_counting_or_duplicates_twice() {
        let repeated = ScalarIndexExpr::Or(Box::new(eq(15)), Box::new(eq(15)));
        assert_eq!(estimate_pages(&lookup(), &repeated), Some(1));
        let disjoint = ScalarIndexExpr::And(Box::new(eq(5)), Box::new(eq(15)));
        assert_eq!(estimate_pages(&lookup(), &disjoint), Some(0));
        assert_eq!(
            estimate_pages(&lookup(), &ScalarIndexExpr::Not(Box::new(eq(15)))),
            None
        );
    }
    #[test]
    fn runtime_lookup_respects_nulls_and_exclusive_range_bounds() {
        let mut expression = eq(0);
        if let ScalarIndexExpr::Query(search) = &mut expression {
            search.query = Arc::new(SargableQuery::IsNull());
        }
        assert_eq!(estimate_pages(&lookup(), &expression), Some(2));
        if let ScalarIndexExpr::Query(search) = &mut expression {
            search.query = Arc::new(SargableQuery::Range(
                std::ops::Bound::Excluded(ScalarValue::Int32(Some(9))),
                std::ops::Bound::Excluded(ScalarValue::Int32(Some(20))),
            ));
        }
        assert_eq!(estimate_pages(&lookup(), &expression), Some(1));
    }
    #[test]
    fn expensive_in_cost_probes_fall_back_without_searching() {
        let mut expression = eq(0);
        if let ScalarIndexExpr::Query(search) = &mut expression {
            search.query = Arc::new(SargableQuery::IsIn(
                (0..257).map(|v| ScalarValue::Int32(Some(v))).collect(),
            ));
        }
        assert_eq!(estimate_pages(&lookup(), &expression), None);
    }
}
