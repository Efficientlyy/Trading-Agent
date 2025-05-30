use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn parse_message_benchmark(c: &mut Criterion) {
    let message = r#"{"c":"spot@public.depth.v3.api@BTCUSDT","s":"BTCUSDT","d":{"s":"BTCUSDT","t":1622185591123,"v":12345,"b":[["39000.5","1.25"],["39000.0","2.5"]],"a":[["39001.0","0.5"],["39001.5","1.0"]]},"t":1622185591123}"#;
    
    c.bench_function("parse_message", |b| {
        b.iter(|| {
            let parsed: serde_json::Value = serde_json::from_str(black_box(message)).unwrap();
            black_box(parsed);
        })
    });
}

fn order_book_update_benchmark(c: &mut Criterion) {
    c.bench_function("order_book_update", |b| {
        b.iter(|| {
            let mut book = std::collections::BTreeMap::new();
            for i in 0..100 {
                let price = 40000.0 + (i as f64) * 0.5;
                book.insert(price, 1.0);
            }
            black_box(book);
        })
    });
}

criterion_group!(benches, parse_message_benchmark, order_book_update_benchmark);
criterion_main!(benches);
