// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::RecordBatchIterator;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance_c::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}
fn ok(code: i32) {
    if code != 0 {
        unsafe {
            let msg = lance_last_error_message();
            let message = CStr::from_ptr(msg).to_string_lossy().into_owned();
            lance_free_string(msg);
            panic!("{message}");
        }
    }
}

fn fixture() -> (tempfile::TempDir, CString) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch.lance");
    lance_c::runtime::block_on(async {
        for fragment in 0..2 {
            let ids: Vec<i32> = (fragment * 64..(fragment + 1) * 64).collect();
            let vectors = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                2,
                Arc::new(Float32Array::from(
                    ids.iter().flat_map(|i| [*i as f32, 1.]).collect::<Vec<_>>(),
                )),
                None,
            )
            .unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("embedding", vectors.data_type().clone(), false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(vectors)],
            )
            .unwrap();
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                path.to_str().unwrap(),
                Some(WriteParams {
                    mode: if fragment == 0 {
                        WriteMode::Create
                    } else {
                        WriteMode::Append
                    },
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        }
    });
    (dir, c(path.to_str().unwrap()))
}

struct Scan(*mut LanceScanner);
impl Drop for Scan {
    fn drop(&mut self) {
        unsafe { lance_scanner_close(self.0) }
    }
}
struct Data(*mut LanceDataset);
impl Drop for Data {
    fn drop(&mut self) {
        unsafe { lance_dataset_close(self.0) }
    }
}
impl Data {
    fn open(uri: &CString) -> Self {
        let ds = unsafe { lance_dataset_open(uri.as_ptr(), ptr::null(), 0) };
        assert!(!ds.is_null());
        Self(ds)
    }
    fn scan(&self) -> Scan {
        let scan = unsafe { lance_scanner_new(self.0, ptr::null(), ptr::null()) };
        assert!(!scan.is_null());
        Scan(scan)
    }
}

fn configure(scan: &Scan, q: &[f32], k: u32) {
    ok(unsafe {
        lance_scanner_nearest_batch(
            scan.0,
            c("embedding").as_ptr(),
            q.as_ptr().cast(),
            2,
            q.len() / 2,
            0,
            k,
        )
    });
}
fn read(scan: &Scan) -> Vec<(i32, i32, f32)> {
    let mut stream = FFI_ArrowArrayStream::empty();
    ok(unsafe { lance_scanner_to_arrow_stream(scan.0, &mut stream) });
    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut stream) }.unwrap();
    assert_eq!(
        reader
            .schema()
            .field_with_name("query_index")
            .unwrap()
            .data_type(),
        &DataType::Int32
    );
    let mut result = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        let qs = batch
            .column_by_name("query_index")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let ds = batch
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            result.push((qs.value(i), ids.value(i), ds.value(i)));
        }
    }
    result.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.total_cmp(&b.2)).then(a.1.cmp(&b.1)));
    result
}
fn expected(q: &[f32], k: usize, minimum_id: i32) -> Vec<(i32, i32, f32)> {
    let mut result = Vec::new();
    for (index, q) in q.as_chunks::<2>().0.iter().enumerate() {
        let mut rows = (minimum_id..128)
            .map(|id| {
                (
                    index as i32,
                    id,
                    (id as f32 - q[0]).powi(2) + (1. - q[1]).powi(2),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.1.cmp(&b.1)));
        result.extend(rows.into_iter().take(k));
    }
    result
}

#[test]
fn batch_nearest_flat_preserves_independent_queries_and_copies_input() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    for count in [1, 3] {
        let scan = data.scan();
        let mut q = vec![2.25, 1., 100.25, 1., 2.25, 1.];
        q.truncate(count * 2);
        let want = expected(&q, 3, 0);
        configure(&scan, &q, 3);
        q.fill(f32::NAN);
        ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
        ok(unsafe { lance_scanner_set_batch_size(scan.0, 2) });
        ok(unsafe { lance_scanner_set_strict_batch_size(scan.0, true) });
        assert_eq!(read(&scan), want);
    }
}

#[test]
fn batch_nearest_prefilter_empty_results_and_projection() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let q = [2.25, 1., 100.25, 1.];
    for minimum in [80, 128] {
        let filter = c(&format!("id >= {minimum}"));
        let column = c("id");
        let projection = [column.as_ptr(), ptr::null()];
        let scan = Scan(unsafe { lance_scanner_new(data.0, projection.as_ptr(), filter.as_ptr()) });
        configure(&scan, &q, 3);
        ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
        ok(unsafe { lance_scanner_set_prefilter(scan.0, true) });
        assert_eq!(read(&scan), expected(&q, 3, minimum));
    }
}

#[test]
fn batch_nearest_indexed_shared_and_refined_fallback() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let params = LanceVectorIndexParams {
        index_type: LanceVectorIndexType::IvfFlat,
        metric: LanceMetricType::L2,
        num_partitions: 2,
        num_sub_vectors: 0,
        num_bits: 0,
        max_iterations: 0,
        hnsw_m: 0,
        hnsw_ef_construction: 0,
        sample_rate: 0,
    };
    ok(unsafe {
        lance_dataset_create_vector_index(
            data.0,
            c("embedding").as_ptr(),
            ptr::null(),
            &params,
            true,
        )
    });
    let q = [2.25, 1., 100.25, 1., 2.25, 1.];
    for refine in [None, Some(2)] {
        let scan = data.scan();
        configure(&scan, &q, 3);
        ok(unsafe { lance_scanner_set_nprobes(scan.0, 2) });
        if let Some(r) = refine {
            ok(unsafe { lance_scanner_set_refine_factor(scan.0, r) });
        }
        assert_eq!(read(&scan), expected(&q, 3, 0));
    }
}

#[test]
fn batch_nearest_rejects_invalid_input_atomically() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let scan = data.scan();
    let q = [2.25f32, 1., 100.25, 1.];
    configure(&scan, &q, 2);
    for (dim, count, dtype, k) in [
        (0, 2, 0, 2),
        (2, 0, 0, 2),
        (3, 1, 0, 2),
        (2, 2, 1, 2),
        (2, 2, 4, 2),
        (2, 2, 99, 2),
        (2, 2, 0, 0),
        (usize::MAX, 2, 0, 2),
        (2, usize::MAX, 0, 2),
        (2, 129, 0, 2),
        (i32::MAX as usize, 1, 0, 2),
        (2, 2, 0, 100001),
    ] {
        assert_ne!(
            unsafe {
                lance_scanner_nearest_batch(
                    scan.0,
                    c("embedding").as_ptr(),
                    q.as_ptr().cast(),
                    dim,
                    count,
                    dtype,
                    k,
                )
            },
            0
        );
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_ne!(
            unsafe {
                lance_scanner_nearest_batch(
                    scan.0,
                    c("embedding").as_ptr(),
                    [bad, 1.].as_ptr().cast(),
                    2,
                    1,
                    0,
                    2,
                )
            },
            0
        );
    }
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(
                scan.0,
                c("missing").as_ptr(),
                q.as_ptr().cast(),
                2,
                2,
                0,
                2,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(scan.0, c("id").as_ptr(), q.as_ptr().cast(), 2, 2, 0, 2)
        },
        0
    );
    ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
    assert_eq!(read(&scan), expected(&q, 2, 0));
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(
                scan.0,
                c("embedding").as_ptr(),
                q.as_ptr().cast(),
                2,
                2,
                0,
                2,
            )
        },
        0
    );
}

#[test]
fn batch_nearest_rejects_global_windows_in_both_configuration_orders() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let q = [2.25f32, 1., 100.25, 1.];
    for offset in [false, true] {
        let scan = data.scan();
        configure(&scan, &q, 2);
        assert_ne!(
            unsafe {
                if offset {
                    lance_scanner_set_offset(scan.0, 1)
                } else {
                    lance_scanner_set_limit(scan.0, 1)
                }
            },
            0
        );
        ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
        assert_eq!(read(&scan), expected(&q, 2, 0));
        let scan = data.scan();
        ok(unsafe {
            if offset {
                lance_scanner_set_offset(scan.0, 1)
            } else {
                lance_scanner_set_limit(scan.0, 1)
            }
        });
        assert_ne!(
            unsafe {
                lance_scanner_nearest_batch(
                    scan.0,
                    c("embedding").as_ptr(),
                    q.as_ptr().cast(),
                    2,
                    2,
                    0,
                    2,
                )
            },
            0
        );
    }
}

#[test]
fn batch_nearest_refinement_budget_and_query_replacement() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let q = [2.25f32, 1., 100.25, 1.];
    let scan = data.scan();
    configure(&scan, &q, 2);
    assert_ne!(unsafe { lance_scanner_set_refine_factor(scan.0, 0) }, 0);
    assert_ne!(unsafe { lance_scanner_set_refine_factor(scan.0, 25001) }, 0);
    ok(unsafe {
        lance_scanner_nearest(scan.0, c("embedding").as_ptr(), q.as_ptr().cast(), 2, 0, 2)
    });
    ok(unsafe { lance_scanner_set_limit(scan.0, 1) });
    let scan = data.scan();
    ok(unsafe { lance_scanner_set_refine_factor(scan.0, 25001) });
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(
                scan.0,
                c("embedding").as_ptr(),
                q.as_ptr().cast(),
                2,
                2,
                0,
                2,
            )
        },
        0
    );
}

fn typed_fixture(values: Arc<dyn Array>, reserved_name: bool) -> (tempfile::TempDir, CString) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("typed.lance");
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", values.data_type().clone(), false)),
        2,
        values,
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            if reserved_name { "query_index" } else { "id" },
            DataType::Int32,
            false,
        ),
        Field::new("embedding", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    lance_c::runtime::block_on(Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        path.to_str().unwrap(),
        None,
    ))
    .unwrap();
    (dir, c(path.to_str().unwrap()))
}

#[test]
fn batch_nearest_supports_float16_float64_and_binary_queries() {
    use arrow_array::{Float16Array, Float64Array, UInt8Array};
    let f16s: Vec<half::f16> = [0., 0., 1., 1., 2., 2., 3., 3.]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let f64s = [0f64, 0., 1., 1., 2., 2., 3., 3.];
    let bytes = [0u8, 0, 1, 1, 15, 15, 255, 255];
    for (values, query, dtype, metric) in [
        (
            Arc::new(Float16Array::from(f16s.clone())) as Arc<dyn Array>,
            f16s.as_ptr().cast(),
            1,
            0,
        ),
        (
            Arc::new(Float64Array::from(f64s.to_vec())) as Arc<dyn Array>,
            f64s.as_ptr().cast(),
            2,
            0,
        ),
        (
            Arc::new(UInt8Array::from(bytes.to_vec())) as Arc<dyn Array>,
            bytes.as_ptr().cast(),
            3,
            3,
        ),
    ] {
        let (_dir, uri) = typed_fixture(values, false);
        let data = Data::open(&uri);
        let scan = data.scan();
        ok(unsafe {
            lance_scanner_nearest_batch(scan.0, c("embedding").as_ptr(), query, 2, 4, dtype, 1)
        });
        ok(unsafe { lance_scanner_set_metric(scan.0, metric) });
        ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
        assert_eq!(
            read(&scan),
            vec![(0, 0, 0.), (1, 1, 0.), (2, 2, 0.), (3, 3, 0.)]
        );
    }
}

#[test]
fn batch_nearest_rejects_reserved_query_index_column() {
    let (_dir, uri) = typed_fixture(Arc::new(Float32Array::from(vec![0.; 8])), true);
    let data = Data::open(&uri);
    let scan = data.scan();
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(
                scan.0,
                c("embedding").as_ptr(),
                [0f32, 0.].as_ptr().cast(),
                2,
                1,
                0,
                1,
            )
        },
        0
    );
}

#[test]
fn batch_nearest_exported_stream_owns_query_and_survives_handle_close() {
    let (_dir, uri) = fixture();
    let mut stream = FFI_ArrowArrayStream::empty();
    {
        let data = Data::open(&uri);
        let scan = data.scan();
        configure(&scan, &[2.25, 1., 100.25, 1.], 16);
        ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
        ok(unsafe { lance_scanner_set_batch_size(scan.0, 1) });
        ok(unsafe { lance_scanner_to_arrow_stream(scan.0, &mut stream) });
    }
    let mut reader = unsafe { ArrowArrayStreamReader::from_raw(&mut stream) }.unwrap();
    assert!(reader.next().unwrap().unwrap().num_rows() > 0);
    // Releasing an unfinished exported stream exercises the existing cancellation path.
    drop(reader);
}

#[test]
fn batch_nearest_and_full_text_search_are_mutually_exclusive() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let q = [2.25f32, 1.];
    let scan = data.scan();
    ok(unsafe { lance_scanner_full_text_search(scan.0, c("term").as_ptr(), ptr::null(), 0) });
    assert_ne!(
        unsafe {
            lance_scanner_nearest_batch(
                scan.0,
                c("embedding").as_ptr(),
                q.as_ptr().cast(),
                2,
                1,
                0,
                1,
            )
        },
        0
    );
    let scan = data.scan();
    configure(&scan, &q, 1);
    assert_ne!(
        unsafe { lance_scanner_full_text_search(scan.0, c("term").as_ptr(), ptr::null(), 0) },
        0
    );
}

#[test]
fn batch_nearest_respects_fragment_scope() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let scan = data.scan();
    let q = [2.25, 1., 100.25, 1.];
    configure(&scan, &q, 3);
    ok(unsafe { lance_scanner_set_fragment_ids(scan.0, [1u64].as_ptr(), 1) });
    ok(unsafe { lance_scanner_set_prefilter(scan.0, true) });
    ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
    assert_eq!(read(&scan), expected(&q, 3, 64));
}

#[test]
fn batch_replacement_clears_single_query_range_and_rejects_new_bounds() {
    let (_dir, uri) = fixture();
    let data = Data::open(&uri);
    let scan = data.scan();
    let q = [2.25f32, 1., 100.25, 1.];
    ok(unsafe {
        lance_scanner_nearest(scan.0, c("embedding").as_ptr(), q.as_ptr().cast(), 2, 0, 3)
    });
    // This range excludes every L2 result; it must not leak into the replacement batch.
    ok(unsafe { lance_scanner_set_distance_range(scan.0, ptr::null(), &0.0) });
    configure(&scan, &q, 3);
    assert_eq!(
        unsafe { lance_scanner_set_distance_range(scan.0, ptr::null(), &0.0) },
        -1
    );
    assert_eq!(lance_last_error_code(), LanceErrorCode::InvalidArgument);
    ok(unsafe { lance_scanner_set_use_index(scan.0, false) });
    assert_eq!(read(&scan), expected(&q, 3, 0));
}
